import os
import json
import regex as re
import numpy as np
from pathlib import Path

from bmoca.agent.llm_agent import base_prompt
from bmoca.agent.llm_agent.utils import load_config

from openai import OpenAI


def load_model(config_path):
    agent_config = load_config(config_path)
    client = OpenAI(api_key = agent_config["OPENAI_API_KEY"])
    return client, agent_config


def get_id_from_element(elem):
    if "resource-id" in elem.attrib and elem.attrib["resource-id"]:
        elem_id = elem.attrib["resource-id"].replace(":", ".").replace("/", "_")
    else:
        elem_id = f"{elem.attrib['class']}"
            
    if "content-desc" in elem.attrib and (elem.attrib['content-desc'] != "") and len(elem.attrib["content-desc"]) < 20:
        content_desc = elem.attrib['content-desc'].replace("/", "_").replace(" ", "").replace(":", "_")
        elem_id += f"_{content_desc}"
    elif "text" in elem.attrib and (elem.attrib['text'] != ""):
        text = elem.attrib['text'].lower().replace("/", "_").replace(",","").replace(" ", "_").replace(":", "_")
        elem_id += f"_{text}"
        
    return elem_id


def parse_obs(obs, height_width=None):
    parsed_obs = []
    parsed_bbox = []
    for _, elem in obs:
        if not "bounds" in elem.attrib: continue
        
        elem_id = get_id_from_element(elem)
        
        bounds = elem.attrib["bounds"][1:-1].split("][")
        x1, y1 = map(int, bounds[0].split(","))
        x2, y2 = map(int, bounds[1].split(","))
        
        parsed_elem = {'numeric_tag': len(parsed_obs), 'description': elem_id}
        if ("slider" in elem_id) or ("settings.id_label" in elem_id):
            height, width = height_width
            parsed_elem['location bounding box ((x1, y1), (x2, y2))'] = \
                f"(({x1/width:0.2f}, {y1/height:0.2f}), ({x2/width:0.2f}, {y2/height:0.2f}))"

        parsed_obs.append(parsed_elem)
        parsed_bbox.append(((x1, y1), (x2, y2)))

    return parsed_obs, parsed_bbox


def build_prompt(instruction, obs, few_shot_prompt=None):
    # base instruction
    system_prompt = base_prompt.INSTRUCTION_PROMPT 

    # goal
    goal_prompt = re.sub(r"<task_instruction>", instruction, base_prompt.GOAL_PROMPT)
    system_prompt += goal_prompt 
    
    # demonstration
    if not (few_shot_prompt is None):
        system_prompt += few_shot_prompt

    # final prompt
    user_prompt = base_prompt.FINAL_PROMPT

    # observation
    user_prompt += "\nObservation: " + str(obs)

    return system_prompt, user_prompt


def parse_act(act):
    output_start = act.find("Action: ")
    
    if output_start != -1:
        func_start = output_start + 8
        func_end = act.find(")", func_start) + 1
        parsed_action = act[func_start:func_end].strip()
    else:
        parsed_action = act
    return parsed_action
