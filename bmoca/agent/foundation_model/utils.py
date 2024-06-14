import io
import os
import cv2
import yaml
import base64
import regex as re
import numpy as np
import pyshine as ps
import xml.etree.ElementTree as xml_element_tree

from PIL import Image
from glob import glob
from pathlib import Path
from itertools import tee

from bmoca.agent.foundation_model import base_prompt

_WORK_PATH = os.environ["BMOCA_HOME"]


def load_config(config_path=f'{_WORK_PATH}/config/foundation_model_config.yaml'):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs


def parse_obs(obs, height_width=None, 
              skip_non_leaf=False,
              attribute_check=False):
    parsed_obs = []
    parsed_bbox = []
    
    raw_obs, obs = tee(obs)
    for _, elem in obs:
        parsed_elem = {}
        
        if not "bounds" in elem.attrib: continue
        
        # category description
        resource_id = ""
        if "resource-id" in elem.attrib and elem.attrib["resource-id"]:
            resource_desc = elem.attrib["resource-id"].replace(":", ".").replace("/", "_")
            resource_id = resource_desc.split(".")[-1]
        if skip_non_leaf and resource_id == "": continue

        # class
        class_desc = f"{elem.attrib['class']}"
        class_name = class_desc.split(".")[-1]
        
        # detailed description
        description = ""
        if "content-desc" in elem.attrib and (elem.attrib['content-desc'] != ""):
            content_desc = elem.attrib['content-desc'].replace("/", "_").replace(" ", "").replace(":", "_")
            description = f"{content_desc}"
        elif "text" in elem.attrib and (elem.attrib['text'] != ""):
            text = elem.attrib['text'].lower().replace("/", "_").replace(",","").replace(" ", "_").replace(":", "_")
            description = f"{text}"

        # checked
        checked_or_selected = 'false'
        if "checkable" in elem.attrib and (elem.attrib['checkable'] != ""):
            if elem.attrib['checkable'] == "true":
                if "checked" in elem.attrib and (elem.attrib['checked'] != ""):
                    if elem.attrib['checked'] == "true":
                        checked_or_selected = 'true'
        if "selected" in elem.attrib and (elem.attrib['selected'] != ""):
            if elem.attrib['selected'] == "true":
                checked_or_selected = 'true'

        # bbox location
        bounds = elem.attrib["bounds"][1:-1].split("][")
        x1, y1 = map(int, bounds[0].split(","))
        x2, y2 = map(int, bounds[1].split(","))
        parsed_bbox.append(((x1, y1), (x2, y2)))
        
        # parsed_elem
        parsed_elem['numeric_tag'] = len(parsed_obs)
        parsed_elem['resource_id'] = resource_id
        parsed_elem['class'] = class_name
        parsed_elem['description'] = description
        if attribute_check:
            parsed_elem['checked'] = checked_or_selected
        if ("slider" in description) or ("settings.id_label" in description):
            height, width = height_width
            parsed_elem['bbox location'] = \
                f"(({x1/width:0.2f}, {y1/height:0.2f}), ({x2/width:0.2f}, {y2/height:0.2f}))"
        parsed_obs.append(parsed_elem)

    return raw_obs, parsed_obs, parsed_bbox


def load_few_shot_examples(replay_dir, 
                           few_shot_example_type="trajectory",
                           trim_trajectory=False):
    """ Few shot example loader for LLM agents """
    print(f"few shot examples loading... (from {replay_dir})")
    few_shot_examples = []

    trajectory_fns = [Path(p) for p in glob(str(replay_dir))]

    if few_shot_example_type == "trajectory":
        for trajectory_fn in trajectory_fns:
            trajectory = open(trajectory_fn, 'r')     
            trajectory = trajectory.read()
            
            if trim_trajectory:
                target = ""
                for step_cnt, transition in enumerate(trajectory.split("\n\n")):
                    if (not ("- Observation: " in transition)) or \
                        (not ("- Action: " in transition)): continue
                    if step_cnt >= 5: continue
                    
                    target += transition + "\n"
                trajectory = target
                            
            # add to few shot examples pool
            few_shot_examples.append(trajectory)    
        
    elif few_shot_example_type == "transition":
        for trajectory_fn in trajectory_fns:
            trajectory = open(trajectory_fn, 'r')     
            trajectory = trajectory.read()
            
            for transition in trajectory.split("\n\n"):
                if (not ("- Observation: " in transition)) or \
                    (not ("- Action: " in transition)): continue
                        
                # add to few shot examples pool
                few_shot_examples.append(transition)    
    else:
        err_message = "few shot example type should be either trajectory or transition"
        raise NotImplementedError(err_message)

    print(f"few shot examples loaded!!! ({len(few_shot_examples)} examples)")
    return few_shot_examples


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image