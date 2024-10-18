import os
import cv2
import json
import datetime
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from itertools import tee

from bmoca.utils import set_random_seed

from bmoca.environment.environment import BMocaEnv
from bmoca.environment.wrapper import GPTActionParsingWrapper

from bmoca.agent.foundation_model import base_prompt

from bmoca.agent.foundation_model.gpt import load_model
from bmoca.agent.foundation_model.gpt import build_prompt, build_few_shot_prompt

from bmoca.agent.foundation_model.utils import parse_obs
from bmoca.agent.foundation_model.utils import PIL2OpenCV, OpenCV2PIL, encode_image
from bmoca.agent.foundation_model.utils import load_few_shot_examples


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


def step(args, env,
         model, agent_config, # GPT model 
         timestep, 
         few_shot_examples=None,
         action_history=None):
    
    # build prompt
    raw_obs = timestep.curr_obs['text']
    timestep.curr_obs['text'], parsed_obs, parsed_bbox =\
        parse_obs(raw_obs, env._coordinator._screen_size,
                  attribute_check=True)
        # parse_obs(raw_obs, env._coordinator._screen_size)

    if not (few_shot_examples is None):
        few_shot_prompt = build_few_shot_prompt(few_shot_examples, k=args.num_few_shot)
    else:
        few_shot_prompt = None
    prompt = build_prompt(env.instruction, parsed_obs, 
                          few_shot_prompt=few_shot_prompt,
                          action_history=action_history)
    log_result = prompt + "\n\n"
    
    # image
    img_obs = timestep.curr_obs['pixel']
    imgpil = Image.fromarray(img_obs)
    imgcv = PIL2OpenCV(imgpil)
    imgcv = cv2.resize(imgcv, 
                        dsize=(1024//4, 2048//4), 
                        interpolation=cv2.INTER_AREA)
    imgpil = OpenCV2PIL(imgcv)

    # predict action
    if args.multi_modal:
        imgpil_path = args.log_dir + "/tmp.png"
        imgpil.save(imgpil_path)
        completion = model.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                "role": "system", 
                "content": base_prompt.SYSTEM_PROMPT
                },
                {
                "role": "user", 
                "content": [
                    {
                    "type": "text", 
                    "text": prompt},
                    {
                    "type": "image_url", 
                    "image_url": 
                        {"url": f"data:image/jpeg;base64,{encode_image(imgpil_path)}"}
                    }]
                },
            ],
            temperature=agent_config["TEMPERATURE"],
            max_tokens=agent_config["MAX_TOKENS"],
            top_p=agent_config["TOP_P"],
            seed=args.seed,
        )
    else:
        completion = model.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": base_prompt.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=agent_config["TEMPERATURE"],
            max_tokens=agent_config["MAX_TOKENS"],
            top_p=agent_config["TOP_P"],
            seed=args.seed,
        )
        
    raw_act = completion.choices[0].message.content
    log_result += raw_act + "\n\n"
    
    # printing
    if args.verbose: 
        plt.imshow(timestep.curr_obs['pixel'])
        plt.show()
        print(prompt)
        print(raw_act)

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
        model, agent_config):
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
    env = GPTActionParsingWrapper(env)

    # load few shot examples
    if args.num_few_shot > 0:
        replay_dir = Path(f"{_DEMO_PATH}/{task_name.split('/')[-1]}/train_env_*/*.txt")
        replay_dir = replay_dir.expanduser()
        few_shot_examples = load_few_shot_examples(replay_dir)
    else:
        few_shot_examples = None

    # main loop
    timestep = env.reset(target_env_id=env_id)
    action_history = []
    log_text, log_image = "", []
    
    while True:
        timestep_new, step_text, step_image \
            = step(args, env, 
                   model, agent_config,
                   timestep, 
                   few_shot_examples=few_shot_examples,
                   action_history=action_history) 

        # logging
        log_text += step_text
        log_image.append(step_image)
        
        with open(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.txt', "w") as file:
            file.write(log_text)
        if len(log_image) > 1:
            log_image[0].save(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.gif', 
                                save_all=True, append_images=log_image[1:], 
                                optimize=False, duration=500, loop=0)
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
    for task_name in _TASK_NAME_LIST:
        success_rates[task_name] = {}

        if args.verbose: print(task_name)

        # iterate over environments
        for env_id, avd_name in _ENV_ID_AVD_NAME_DICT.items():
            success_rates[task_name][env_id] = 0

            # load model
            model, agent_config = load_model(args.config_path)

            # evaluate
            try:
                is_success = evaluate_one_env(args,\
                                            avd_name, f"test_env_{env_id}", \
                                            task_name, \
                                            model, agent_config)
            except:
                traceback.print_exc()
                continue

            if is_success:
                success_rates[task_name][env_id] += 1
                if args.verbose: print('Success')
            else:
                if args.verbose: print('Failure')
            
            # success_rates logging
            with open(f'{args.log_dir}/_success_rate.json', 'w') as f:
                json.dump(success_rates, f)
            print(success_rates)


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
    parser.add_argument('--run_with_head', 
                        default=False, action='store_true')   
    parser.add_argument('--adjusting_freq', type=float, default=1.0/3.0)   
    # agent setting
    parser.add_argument('--config_path', type=str, 
                        default=f'{_WORK_PATH}/config/foundation_model_config.yaml')
    parser.add_argument('--multi_modal', default=False, action='store_true')
    parser.add_argument('--num_few_shot', type=int, default=0)
    # logging
    parser.add_argument('--log_dir', type=str, default=f"{_WORK_PATH}/logs/gpt4o")
    parser.add_argument('--log_message', type=str, default=None)
    # misc
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', default=False, action='store_true')
    
    args = parser.parse_args()

    # set avd name
    for k, _ in _ENV_ID_AVD_NAME_DICT.items():
        _ENV_ID_AVD_NAME_DICT[k] += f"_test_{args.avd_id:02d}"

    # set log directory
    if args.multi_modal:
        args.log_dir += "_multimodal"
    else:
        args.log_dir += "_textonly"
    if not os.path.isdir(args.log_dir): os.mkdir(args.log_dir)

    if args.num_few_shot > 0:
        args.log_message = "few_shot"
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
