import os
import io
import cv2
import copy
import time
import json
import torch
import wandb
import random
import logging
import argparse
import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, EfficientNetModel

from bmoca.agent.vlui_agent import utils
from bmoca.environment.environment import BMocaEnv

from experiments.agent.bc.train import BCPolicyEfficientNet

_HOME_PATH = Path.home()
_WORK_PATH = os.environ["BMOCA_HOME"]
_TASK_PATH = f"{_WORK_PATH}/asset/tasks"

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


def step(args, env, model, tokenizer, timestep):
    # preprocess inputs 
    instruction = timestep.instruction
    query_input = tokenizer(
                    instruction,
                    max_length=32,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

    instruction_input_ids = query_input["input_ids"] 
    instruction_attention_mask = query_input["attention_mask"]
    observation = timestep.curr_obs['pixel']
    if args.verbose:
        temp = Image.fromarray((observation * 255).astype(np.uint8))
        temp.save(f"{_WORK_PATH}/logs/" + \
                  f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")

    observation = observation.copy()
    observation = cv2.resize(observation, dsize=(128, 256), interpolation=cv2.INTER_AREA)
    observation = observation.astype(np.float32) 
    observation = np.expand_dims(observation, axis=0) 
    observation = np.transpose(observation, (0, 3, 1, 2))

    instruction_input_ids, instruction_attention_mask, observation = \
        utils.to_torch([instruction_input_ids, instruction_attention_mask, observation], 
                       args.device)

    pred_gesture = model(instruction_input_ids, instruction_attention_mask, observation)
    pred_gesture = pred_gesture.detach().cpu().numpy()
    if args.verbose:print((pred_gesture[0] + 1.0) / 2.0)

    # step env
    timestep = env.step(pred_gesture[0])
    return timestep


def evaluate(args, env, env_id, model, tokenizer):
    # main loop
    timestep = env.reset(target_env_id=env_id)

    while True:
        timestep = step(args, env, model, tokenizer, timestep)
        if (timestep is None): return False
        if timestep.last(): break

    if timestep.curr_rew > 0:
        return True # success
    else:
        return False # failure


def run(args):
    # main loop
    success_rates = {}
    for task_name in _TASK_NAME_LIST:
        success_rates[task_name] = {} 

        task_path = f"{_TASK_PATH}/{task_name}.textproto" 
        if args.verbose: print(task_name)

        env = None
        create_flag = False
        prev_avd_name = None

        # iterate over environments
        for env_id, avd_name in _ENV_ID_AVD_NAME_DICT.items():
            success_rates[task_name][env_id] = 0

            # env creation
            if prev_avd_name != avd_name:
                create_flag = True

                if not (env is None):
                    env.close()

                prev_avd_name = avd_name
            else:
                create_flag = False

            if create_flag:
                env = BMocaEnv(
                    task_path=task_path,
                    avd_name=avd_name,
                    state_type='pixel',
                    action_tanh=True,
                    adjusting_freq = args.adjusting_freq,
                    run_headless=(not args.run_with_head)
                )

            # load model
            model = BCPolicyEfficientNet(finetune_vis_encoder = args.finetune_vis_encoder)
            model.load(filename = args.model_path, device = args.device)
            model.to(args.device)

            tokenizer = AutoTokenizer.from_pretrained(f"{_WORK_PATH}/asset/agent/Auto-UI-Base")   

            # test in target_env
            target_env_id = f"test_env_{env_id}"
            if args.verbose: print(target_env_id)

            is_success = evaluate(args, env, target_env_id, model, tokenizer)
            if is_success:
                success_rates[task_name][env_id] += 1
                if args.verbose: print('Success')
            else:
                if args.verbose: print('Failure')

    print(success_rates)


def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--avd_id', type=int, default=0)
    parser.add_argument('--android_avd_home', type=str, default=f'{_HOME_PATH}/.android/avd')
    parser.add_argument('--android_sdk_root', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk')
    parser.add_argument('--emulator_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator')
    parser.add_argument('--adb_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb')
    parser.add_argument('--run_with_head', default=False, type=bool)
    parser.add_argument('--adjusting_freq', type=float, default=1.0/3)   
    # model
    parser.add_argument('--finetune_vis_encoder', type=bool, default=True)
    # test
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, default=None)
    # misc
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert not (args.model_path is None), "args.model_path should be specified!"
    set_random_seed(args.seed)

    args.time_now = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    # update AVD_NAME
    for k, _ in _ENV_ID_AVD_NAME_DICT.items():
        _ENV_ID_AVD_NAME_DICT[k] += f"_test_{args.avd_id:02d}"

    # run main
    run(args)
