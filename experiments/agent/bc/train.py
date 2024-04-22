import os
import ast
import math
import numpy as np
import wandb
import random
import traceback
import pathlib
import logging
import datetime
import argparse

from glob import glob
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

import torch
from torch import optim, nn

from bmoca.dataset.utils import episode_len, load_episode
from bmoca.dataset.replay_buffer import ReplayBuffer

from bmoca.agent.vlui_agent import utils
from bmoca.agent.vlui_agent.bc_policy import BCPolicy
from bmoca.agent.vlui_agent.utils import EfficientNetEncoder

_WORK_PATH = os.environ['BMOCA_HOME']

_TASK_NAME_LIST = [
    "turn_on_airplane_mode",
    "turn_on_alarm_at_9_am",
    "create_alarm_at_10:30_am",
    "decrease_the_screen_brightness_in_setting",
    "call_911",
    "go_to_the_'add_a_language'_page_in_setting",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dist(normalized_touch_yx, normalized_lift_yx):
  touch_y, touch_x = normalized_touch_yx
  lift_y, lift_x = normalized_lift_yx
  
  renormalized_touch_yx = [touch_y * 2, touch_x]
  renormalized_lift_yx = [lift_y * 2, lift_x]
  
  distance = np.linalg.norm(np.array(renormalized_touch_yx) - np.array(renormalized_lift_yx))
  return distance


class BCReplayBuffer(ReplayBuffer):
    def __init__(self, 
                 replay_dir,
                 task_instructions,
                 ):
        self._replay_dir = replay_dir
        self.task_instructions = task_instructions

        # loading
        self._size = 0
        self._episode_fns = []
        self._episode_dict = dict()
        self._episode_size = dict()

        # preprocess
        self.tokenizer = AutoTokenizer.from_pretrained(f"{_WORK_PATH}/asset/agent/Auto-UI-Base")

    def _load(self):
        print(f"ReplayBuffer loading... (from {self._replay_dir})")
        for task_instruction in self.task_instructions:
            self._episode_fns += sorted(Path(self._replay_dir).expanduser().glob(f"train_env_*/{task_instruction}/*.npz"))
        self._episode_fns = sorted(self._episode_fns)
        for eps_fn in self._episode_fns:
            
            # load
            try:
                episode = load_episode(eps_fn)
            except:
                traceback.print_exc()    
                logging.error("Exception while loading in replay buffer")
                break
            
            self._episode_dict[eps_fn] = episode
            
            # get size info
            eps_len = episode_len(episode)
            self._episode_size[eps_fn] = eps_len
            self._size += eps_len

        print(f"ReplayBuffer loaded!!! ({len(self._episode_fns)} demos, {self._size} steps)")

    def _sample(self, sample_episode=False):   
        
        # sample & load episode
        ep_fn = random.choice(self._episode_fns)

        episode = self._episode_dict[ep_fn]
        
        if sample_episode: 
            return episode
        
        # sample index
        ep_len = self._episode_size[ep_fn]
        idx = np.random.randint(0, ep_len)
        # sample transition
        instruction = episode['instruction'][idx]
        observation = episode['prev_obs'][idx]
        action      = (episode['prev_act'][idx] * 2.0 - 1.0)
        
        query_input = self.tokenizer(
                        instruction,
                        max_length=32,
                        pad_to_max_length=True,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
        )

        return (query_input["input_ids"][0], query_input["attention_mask"][0], \
                observation, action)


class EfficientNetBCPolicy(BCPolicy):
    def __init__(self, finetune_vis_encoder=True, version="b0"):
        super().__init__()
        self.finetune_vis_encoder = finetune_vis_encoder
        # encoder is overwritten
        self.img_enc = EfficientNetEncoder(finetune_vis_encoder=self.finetune_vis_encoder, version=version)        


def run(args):
    ## Replay Buffer
    replay_buffer = BCReplayBuffer(
                        replay_dir=args.demo_dir, 
                        task_instructions=_TASK_NAME_LIST,
                    )
    replay_buffer._load() 
    
    replay_loader = torch.utils.data.DataLoader(
                                replay_buffer,
                                batch_size=args.batch_size,
                                pin_memory=True,
                                num_workers=4,
                                prefetch_factor=2,
                                persistent_workers=True,
                            )
    replay_iter = iter(replay_loader)

    ## Model
    model = EfficientNetBCPolicy(finetune_vis_encoder=args.finetune_vis_encoder, version=args.efficient_net_version)
    model = model.to(args.device)

    optimizer = optim.Adam(model.get_train_parameters(), lr=0)
    scheduler = utils.CosineAnnealingWarmUpRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_max=args.eta_max,  T_up=args.T_up, gamma=args.gamma)
    criterion = nn.MSELoss()

    use_data_parallel=False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        use_data_parallel = True

    # wandb
    if args.log_wandb:
        wandb_run_name = args.wandb_name
        wandb_project = 'bmoca'
        wandb_run_name += f"_seed{args.seed}"
        if args.output_message is not None: wandb_run_name += f"_{args.output_message}"

        run_id = wandb.util.generate_id()
        run_name = f'{wandb_run_name}.{run_id}'
        logging.info(f"wandb running with {run_name}")

        wandb.init(project=wandb_project, 
                config=args)
        wandb.run.name = run_name
        wandb.run.save()

    # main loop
    print("START OF MAIN LOOP")

    tap_acc = 0
    for e in tqdm(range(args.train_steps), desc='Training'):
        train_loss = 0

        # sample batch
        batch = next(replay_iter)

        # train
        model.train()
        model.zero_grad()

        instruction_input_ids, instruction_attention_mask,\
            observation, label_gesture = utils.to_torch(batch, args.device)
        label_gesture = label_gesture.float()
        pred_gesture = model(instruction_input_ids, instruction_attention_mask, observation)

        loss = criterion(pred_gesture, label_gesture)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        batch_num = pred_gesture.shape[0]
        train_loss += loss.detach().item() * batch_num
        
        scheduler.step()
        if (e + 1) % args.log_period == 0:
            metrics = {'train_loss': train_loss, 'lr': scheduler.get_lr()[0], 'tap_acc': tap_acc}
            if args.log_wandb:
                wandb.log(metrics)
                
        if (e + 1) % args.eval_period == 0:
            model.eval()
            
            # get tap accuracy
            tap_num = 0
            threshold = 0.1
            
            label_gesture_cpu = label_gesture.cpu().detach().numpy()
            pred_gesture_cpu = pred_gesture.cpu().detach().numpy()
            for i in range(batch_num):
                if dist(label_gesture_cpu[i, :2], pred_gesture_cpu[i, :2]) < threshold \
                    and dist(label_gesture_cpu[i, 2:], pred_gesture_cpu[i, 2:]) < threshold:
                    tap_num += 1
            tap_acc = tap_num / batch_num
            
            if args.log_wandb:
                # update metrics to include tap_acc
                metrics.update({'tap_acc': tap_acc})
                wandb.log(metrics)
            print(f"tap_acc: {tap_acc}, train_loss: {train_loss}, lr: {scheduler.get_lr()[0]}")

        # save model
        if (e + 1) % args.save_period == 0:
            if use_data_parallel:
                model_ = model.module
            else:
                model_ = model
            model_.save(f'{args.output_dir}/bc_policy_{e}.pt')
            model_.save(f'{args.output_dir}/bc_policy_latest.pt')
    
    # last save when the training is over
    if use_data_parallel:
        model_ = model.module
    else:
        model_ = model
    model_.save(f'{args.output_dir}/bc_policy_latest.pt')


def arg_as_list(s:str):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--finetune_vis_encoder', type=bool, default=True)
    parser.add_argument('--text_encoder', type=str, default=f"{_WORK_PATH}/asset/Auto-UI-Base")
    parser.add_argument('--input_len', type=int, default=32)
    parser.add_argument('--action_history', default=False, action='store_true')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--efficient_net_version', type=str, default = "b0")    
    # training
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--train_with_eval', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_steps', type=int, default=4000)
    parser.add_argument('--log_period', type=int, default=1, help = 'every log_period loss is computed and logged in wandb')
    parser.add_argument('--eval_period', type=int, default=50, help= 'every eval_period tap_acc is computed and logged in wandb')
    parser.add_argument('--save_period', type=int, default=2000, help='every save_period model is saved')
    # scheduler
    parser.add_argument('--T_0', type=int, default=2000)
    parser.add_argument('--T_mult', type=int, default=1)
    parser.add_argument('--eta_max', type=float, default=3e-4, help='max learning rate for 1st T_0')
    parser.add_argument('--T_up',type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.8)
    # output
    parser.add_argument('--demo_dir', type=str, default=f"{_WORK_PATH}/asset/demonstration/")
    parser.add_argument('--env_num', type=int, default=35)
    parser.add_argument('--output_dir', type=str, default=f"{_WORK_PATH}/logs/bc")
    parser.add_argument('--output_message', type=str, default=None)
    parser.add_argument('--debug', default=False, action='store_true')
    # misc.
    parser.add_argument('--log_wandb', default=True, action='store_true')
    parser.add_argument('--wandb_name', type=str, default='bc_main')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ## Log directory
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    print("output_dir:", args.output_dir)
    
    ## main
    run(args)
