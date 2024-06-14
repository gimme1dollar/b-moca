import os
import io
import cv2
import json
import time
import torch
import random
import datetime
import argparse
import traceback
import itertools
import numpy as np
import regex as re
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from pathlib import Path

from bmoca.utils import set_random_seed

from bmoca.environment.environment import BMocaEnv
from bmoca.environment.wrapper import LlamaActionParsingWrapper

from bmoca.agent.foundation_model import base_prompt

from bmoca.agent.foundation_model.llama import load_model_tokenizer
from bmoca.agent.foundation_model.llama import build_prompt, build_user_prompt, build_few_shot_prompt

from bmoca.agent.foundation_model.utils import load_config, parse_obs
from bmoca.agent.foundation_model.utils import PIL2OpenCV, OpenCV2PIL
from bmoca.agent.foundation_model.utils import load_few_shot_examples

from huggingface_hub import login


_HOME_PATH = Path.home()
_WORK_PATH = os.environ['BMOCA_HOME']
_TASK_PATH = f"{_WORK_PATH}/asset/tasks"
_DEMO_PATH = f"{_WORK_PATH}/asset/dataset/demonstrations_text_action"

_ENV_ID_AVD_NAME_DICT = {
    '100': 'pixel_3',
    '101': 'pixel_3',
    '102': 'pixel_3',
    '103': 'pixel_3',
    '104': 'pixel_3',
    '105': 'pixel_3',
    '106': 'pixel_4',
    '107': 'pixel_5',
    '108': 'pixel_6',
    '109': 'WXGA_Tablet',
}

_TASK_NAME_LIST = [
    "clock/create_alarm_at_10:30_am",
    "clock/create_alarm_at_10:30_am_on_every_weekday",
    "settings/go_to_the_'add_a_language'_page_in_setting",
    "phone/call_the_white_house_(202-456-1111)",
    "calculator/input_'cos(180)'_in_Calculator",
    "wikipedia/disable_the_top_2_and_'randomizer'_topics_in_the_feed_customization_settings_on_Wikipedia_and_go_back_to_the_feed",
]

_AGENT_CONFIG = load_config()
login(_AGENT_CONFIG['HUGGINGFACE_TOKEN'])


def step(args, env,
         model, tokenizer, # Llama model 
         timestep,
         few_shot_examples=None,
         action_history=None):
    
    # build prompt
    if "call" in env.instruction:
        attribute_check = False
    else:
        attribute_check = True
    
    raw_obs = timestep.curr_obs['text']
    timestep.curr_obs['text'], parsed_obs, parsed_bbox =\
        parse_obs(raw_obs, env._coordinator._screen_size, 
                  skip_non_leaf=False, 
                  attribute_check=attribute_check)
        
    if not (few_shot_examples is None):
        few_shot_prompt = build_few_shot_prompt(few_shot_examples, 
                                                k=args.num_few_shot)
    else:
        few_shot_prompt = None

    user_prompt = build_user_prompt(env.instruction, parsed_obs, 
                                    few_shot_prompt=few_shot_prompt,
                                    action_history=action_history)
    prompt = build_prompt(
        system_prompt=base_prompt.SYSTEM_PROMPT,
        user_prompt=user_prompt,
        llama_version=3
    )
    log_result = prompt + "\n\n"
    
    # image
    imgpil = Image.fromarray(timestep.curr_obs['pixel'])
    imgcv = PIL2OpenCV(imgpil)
    imgcv = cv2.resize(imgcv, 
                        dsize=(1024//4, 2048//4), 
                        interpolation=cv2.INTER_AREA)
    imgpil = OpenCV2PIL(imgcv)

    # predict action
    model_inputs = tokenizer([prompt], 
                            return_tensors="pt",
                            max_length=2**15, # 32K
                            padding=True,
                            truncation=True,
                            ).to('cuda')
    tokenized_chat = model_inputs.input_ids
    
    if _AGENT_CONFIG["TEMPERATURE"] > 0:
        generated_ids = model.generate(
                            tokenized_chat,
                            do_sample=True,
                            max_new_tokens=_AGENT_CONFIG["MAX_TOKENS"],
                            temperature=_AGENT_CONFIG["TEMPERATURE"],
                            top_p=_AGENT_CONFIG["TOP_P"],
                            top_k=_AGENT_CONFIG["TOP_K"],
                            num_return_sequences=_AGENT_CONFIG["NUM_RETURN_SEQUENCES"],
                            repetition_penalty=_AGENT_CONFIG["REPETITION_PENALTY"],
                            # no_repeat_ngram_size=_AGENT_CONFIG["NO_REPEAT_NGRAM_SIZE"],
                        )
    else:
        generated_ids = model.generate(
                            tokenized_chat,
                            do_sample=False,
                            max_new_tokens=_AGENT_CONFIG["MAX_TOKENS"],
                            temperature=_AGENT_CONFIG["TEMPERATURE"],
                            num_return_sequences=_AGENT_CONFIG["NUM_RETURN_SEQUENCES"],
                            repetition_penalty=_AGENT_CONFIG["REPETITION_PENALTY"],
                            # no_repeat_ngram_size=_AGENT_CONFIG["NO_REPEAT_NGRAM_SIZE"],
                        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in \
                                        zip(tokenized_chat, generated_ids)
    ]

    raw_act = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if args.verbose:
        while True:
            try:
                print(raw_act)
                break 
            except BlockingIOError:
                time.sleep(0.1)
    log_result += raw_act + "\n\n"

    # step env
    try:
        timestep = env.step(raw_act, 
                            parsed_obs, parsed_bbox)
        return timestep, log_result, imgpil
    except:
        traceback.print_exc()
        print("Error while evaluation")
        return None, log_result, imgpil


def evaluate_one_env(args, \
        avd_name , env_id, # env
        task_name,
        model, tokenizer):
    task_path = f"{_TASK_PATH}/{task_name}.textproto"
    if args.verbose: print(env_id)
    
    # env creation
    env = BMocaEnv(
        task_path=task_path,
        avd_name=avd_name,
        state_type='text',
        action_tanh = False,
        adjusting_freq = args.adjusting_freq,
        run_headless=(not args.run_with_head)
    )
    env = LlamaActionParsingWrapper(env)
    
    if args.num_few_shot > 0:
        replay_dir = Path(f"{_DEMO_PATH}/{task_name.split('/')[-1]}/train_env_*/*.txt")
        replay_dir = replay_dir.expanduser()
        few_shot_examples = load_few_shot_examples(replay_dir,
                                                   trim_trajectory=True)
    else:
        few_shot_examples = None

    # main loop
    timestep = env.reset(target_env_id=env_id)
    action_history = []
    log_text, log_image = "", []
    if args.verbose:
        plt.imshow(timestep.curr_obs['pixel'])
        plt.show()
        
    while True:
        timestep_new, step_text, step_image \
            = step(args, env, 
                   model, tokenizer, 
                   timestep,
                   few_shot_examples=few_shot_examples,
                   action_history=action_history) 
            
        # logging
        log_text += step_text
        log_image.append(step_image)
        
        with open(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.txt', "w") as file:
            while True:
                try:
                    file.write(log_text)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
        if len(log_image) > 1:
            log_image[0].save(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.gif', 
                        save_all=True, append_images=log_image[1:], optimize=False, duration=500, loop=0)
        else:
            log_image[0].save(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.gif')

        # check proper transition
        if (timestep_new is None):
            if env._task_manager._stats['episode_steps'] >=\
                env._task_manager._task.max_episode_steps:
                    break
            continue
        timestep = timestep_new
        
        # action history
        action_history.append(timestep.prev_act)
        if len(action_history) > 4: action_history.pop(0)
            
        # check end of timestep
        if timestep.last(): break

    env.close()

    if timestep.curr_rew > 0:
        return True # success
    else:
        return False # failure
    

def run(args):
    success_rates = {}
    
    # load model
    model, tokenizer = load_model_tokenizer(model_name=args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
            
    for task_name in _TASK_NAME_LIST:
        success_rates[task_name] = {}

        if args.verbose: print(task_name)

        # iterate over environments
        for env_id, avd_name in _ENV_ID_AVD_NAME_DICT.items():
            success_rates[task_name][env_id] = 0
            
            # evaluate
            try:
                is_success = evaluate_one_env(args,\
                                            avd_name, f"test_env_{env_id}", \
                                            task_name, \
                                            model, tokenizer)
            except:
                traceback.print_exc()
                continue

            if is_success:
                success_rates[task_name][env_id] += 1
                if args.verbose: print('Success')
            else:
                if args.verbose: print('Failure')

            print(success_rates)
            with open(f'{args.log_dir}/_success_rate.json', 'w') as f:
                json.dump(success_rates, f)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--avd_id', type=int, default=0)
    parser.add_argument('--android_avd_home', type=str, 
                        default=f'{_HOME_PATH}/.android/avd')
    parser.add_argument('--android_sdk_root', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk')
    parser.add_argument('--emulator_path', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator')
    parser.add_argument('--adb_path', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb')
    parser.add_argument('--run_with_head', default=False, action='store_true')   
    parser.add_argument('--adjusting_freq', type=float, default=1.0/3.0)   
    # agent setting
    parser.add_argument('--model_name', type=str, 
                        default=f'meta-llama/Meta-Llama-3-70B-Instruct')
    parser.add_argument('--config_path', type=str, 
                        default=f'{_WORK_PATH}/config/model_config.yaml')
    parser.add_argument('--num_few_shot', type=int, default=1) 
    # logging
    parser.add_argument('--log_dir', type=str, default=f"{_WORK_PATH}/logs/llama3")
    parser.add_argument('--log_message', type=str, default=None)
    # misc
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', type=bool, default=True)
     
    args = parser.parse_args()

    # set avd name
    for k, _ in _ENV_ID_AVD_NAME_DICT.items():
        _ENV_ID_AVD_NAME_DICT[k] += f"_test_{args.avd_id:02d}"

    # set log dir
    if not os.path.isdir(args.log_dir): os.mkdir(args.log_dir)

    if args.num_few_shot > 0:
        args.log_message = f"few_shot_{args.num_few_shot}"
    else:
        args.log_message = "zero_shot"
    args.log_dir += f"/{args.log_message}_{args.seed}"

    args.time_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    args.log_dir += f"_{args.time_now}"
    if not os.path.isdir(args.log_dir): os.mkdir(args.log_dir)

    return args


if __name__ == "__main__":
    args = parse_args() 
    set_random_seed(args.seed)

    run(args)
