import os
import sys
import time
import json
import wandb
import warnings
import pandas as pd
import pyarrow as pa

from tqdm import tqdm
from datasets import Dataset

import torch
import torch.distributed as dist

from transformers import TrainingArguments, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

import peft
from peft import LoraConfig

sys.path.append(os.environ["BMOCA_HOME"])


def load_hfmodel(ckpt=None):
    if ckpt == None:
        path = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        path = ckpt

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
        attn_implementation="flash_attention_2",
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    base_model = base_model.float()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = True
    print("Loaded Model and Tokenizer")

    return base_model, tokenizer


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def main(cfg):
    # load dataset for training and validation
    train_df = pd.read_csv(
        f'{os.environ["BMOCA_HOME"]}/asset/dataset/bmoca_demo_llama.csv'
    )
    train_dataset = Dataset(pa.Table.from_pandas(train_df))

    # load model and tokenizer
    base_model, tokenizer = load_hfmodel(cfg.model_name)
    tokenizer.padding_side = "right"

    training_args = SFTConfig(
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        output_dir=cfg.ckpt_dir,
        per_device_train_batch_size=cfg.per_gpu_bsz,
        per_device_eval_batch_size=cfg.per_gpu_bsz,
        fp16=True,
        bf16=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        num_train_epochs=cfg.n_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        save_strategy="epoch",
        # eval_on_start = True,
        # evaluation_strategy="epoch",
        seed=cfg.seed,
        group_by_length=True,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sep_tokens = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")[1:]
    # sep_tokens = tokenizer.encode('<|assistant|>')[:-1]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=sep_tokens, tokenizer=tokenizer
    )
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2**15,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        callbacks=[PeftSavingCallback],
        peft_config=peft_config,
    )

    print("Set Trainer")
    print("Start Training!")
    trainer.train()


if __name__ == "__main__":
    run = wandb.init(
        project="BMoCA-Llama3",
        name=f"Llama-3",
    )

    # config
    cfg = wandb.config
    cfg.seed = 1
    cfg.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    cfg.ckpt_dir = f'{os.environ["BMOCA_HOME"]}/logs/ckpt_llama3'
    cfg.per_gpu_bsz = 1
    cfg.gradient_accumulation_steps = 8
    cfg.lr = 1e-6
    cfg.logging_steps = 1
    cfg.n_epochs = 15
    cfg.weight_decay = 1.0
    cfg.warmup_ratio = 0.01
    cfg.eval_steps = 50
    cfg.save_steps = 100

    main(cfg)
