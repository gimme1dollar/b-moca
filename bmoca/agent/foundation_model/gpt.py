import os
import time
import json
import regex as re
import numpy as np
from pathlib import Path
from itertools import tee

from openai import OpenAI

from bmoca.agent.foundation_model import base_prompt
from bmoca.agent.foundation_model.utils import load_config


def load_model(config_path):
    agent_config = load_config(config_path)
    client = OpenAI(api_key = agent_config["OPENAI_API_KEY"])
    return client, agent_config


def poll_run(client, thread, run):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(run.status)
        time.sleep(0.5)
    return run


def build_prompt(instruction, obs,
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
        
    # # final prompt
    # prompt += base_prompt.FINAL_PROMPT

    # observation
    if not (action_history is None):
        prompt += "\nPrevious actions: " + str(action_history) 
    prompt += "\nCurrent observation: " + str(obs)
    prompt += "\nAnswer: "

    return prompt


def build_few_shot_prompt(few_shot_examples, k=1):
    expert_demonstration = ""

    # random select k
    selected_few_shot_examples = np.random.choice(few_shot_examples, k)

    # build prompt
    for few_shot_example in selected_few_shot_examples:
        expert_demonstration += "Demonstration Example:\n" + few_shot_example

    return re.sub(r"<expert_demonstration>", 
                  expert_demonstration, 
                  base_prompt.FEW_SHOT_PROMPT)
