import os
import json
import regex as re
import numpy as np
import transformers
import torch

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from bmoca.agent.foundation_model import base_prompt
from bmoca.agent.foundation_model.utils import load_config


LLAMA2_PROMPT_FORMAT="""
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_prompt }} [/INST]
"""

LLAMA3_PROMPT_FORMAT="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

FEW_SHOT_PROMPT = """
Below illustrates the example of human experts.
Each example is sampled from a full trajectory from the beginning to the end of the task completion.
An example is a pair of observation from the environment and corresponding action taken by the expert is descripted as:
- Observation: <An observation from the environment>
- Action: <An action taken by the human expert>

<expert_demonstration>"""


def load_model_tokenizer(model_name = "meta-llama/Llama-2-7b-chat-hf"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        # load_in_8bit=True,
        # load_in_4bit=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def build_prompt(system_prompt, user_prompt, llama_version=3):
    if llama_version==2:    
        prompt = LLAMA2_PROMPT_FORMAT
    elif llama_version==3:
        prompt = LLAMA3_PROMPT_FORMAT
    else:
        raise ValueError('llama_version should be either 2 or 3')
    
    prompt = prompt.replace("{{ system_prompt }}", system_prompt)
    
    prompt = prompt.replace("{{ user_prompt }}", user_prompt)
    
    return prompt


def build_user_prompt(instruction, obs, 
                      few_shot_prompt=None,
                      action_history=None):
    prompt = ""
    
    # instruction
    prompt += base_prompt.INSTRUCTION_PROMPT
    
    # goal
    goal_prompt = re.sub(r"<task_instruction>", instruction, base_prompt.GOAL_PROMPT)
    prompt += goal_prompt 
    
    # demonstration
    if not (few_shot_prompt is None):
        prompt += few_shot_prompt
    
    # observation
    if not (action_history is None):
        prompt += "\nPrevious actions: " + str(action_history) 
    prompt += "\nCurrent observation: " + str(obs)
    prompt += "\nAnswer: "

    return prompt


def build_few_shot_prompt(few_shot_examples, k=1,
                          trajectory_type='trajectory'):
    expert_demonstration = ""

    # random select k
    selected_few_shot_examples = np.random.choice(few_shot_examples, k)

    # build prompt
    if trajectory_type == 'trajectory':
        expert_demonstration = ""
        for few_shot_example in selected_few_shot_examples:
            expert_demonstration += "Demonstration Example:\n" + few_shot_example

        return re.sub(r"<expert_demonstration>", 
                      expert_demonstration, 
                      base_prompt.FEW_SHOT_PROMPT)
        
    elif trajectory_type == 'transition':
        expert_demonstration = "Transition Example:\n"
        for few_shot_example in selected_few_shot_examples:
            expert_demonstration += few_shot_example + "\n\n"

        return re.sub(r"<expert_demonstration>", 
                      expert_demonstration, 
                      FEW_SHOT_PROMPT)
    
    else:
        err_message = "few shot example type should be either trajectory or transition"
        raise NotImplementedError(err_message)
