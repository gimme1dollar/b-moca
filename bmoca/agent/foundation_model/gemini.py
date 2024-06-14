import os
import json
import regex as re
import numpy as np
from pathlib import Path

import google.generativeai as genai

from bmoca.agent.foundation_model import base_prompt
from bmoca.agent.foundation_model.utils import load_config


def load_model(config_path):
    agent_config = load_config(config_path)
    
    genai.configure(api_key=agent_config["GEMINI_API_KEY"])
    generation_config = {
        "temperature": agent_config["TEMPERATURE"],
        "top_p": agent_config["TOP_P"],
        # "top_k": agent_config["TOP_K"],
        "max_output_tokens": agent_config["MAX_TOKENS"],
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-001",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    return model


def build_prompt(instruction, obs,
                 few_shot_prompt=None,
                 action_history=None):
    prompt = base_prompt.SYSTEM_PROMPT
    
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

