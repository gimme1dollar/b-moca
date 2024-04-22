import os
import json
import random
import argparse
import traceback
import numpy as np

from pathlib import Path

from bmoca.environment.environment import BMocaEnv
from bmoca.environment.wrapper import GeminiTextActionParsingWrapper

from bmoca.agent.llm_agent.gemini import load_model, parse_obs, build_prompt

_HOME_PATH = Path.home()
_WORK_PATH = os.environ['BMOCA_HOME']
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
    # torch.manual_seed(seed)


def step(args, env, model, timestep):
    # build prompt
    raw_obs = timestep.curr_obs['text']
    parsed_obs, parsed_bbox = parse_obs(raw_obs, env._coordinator._screen_size)
    
    prompt = build_prompt(env.instruction, parsed_obs, few_shot_prompt=None)
    if args.verbose: print(prompt)

    # predict action
    raw_act = model.generate_content(prompt).text
    if args.verbose: print(raw_act)

    # step env
    try:
        timestep = env.step(raw_act, parsed_bbox)
        return timestep
    except:
        traceback.print_exc()
        print("Error while evaluation")
        return None


def evaluate(args, env, env_id, model):
    # main loop
    timestep = env.reset(target_env_id=env_id)

    while True:
        timestep_new = step(args, env, model, timestep)

        if (timestep_new is None):
            if env._task_manager._stats['episode_steps'] >=\
                env._task_manager._task.max_episode_steps:
                    return False
            continue
        timestep = timestep_new
        if timestep.last(): break

    if timestep.curr_rew > 0:
        return True # success
    else:
        return False # failure


def run(args):
    success_rates = {}
    for task_name in _TASK_NAME_LIST:
        success_rates[task_name] = {}

        task_path = f"{_TASK_PATH}/{task_name}.textproto"
        if args.verbose: print(task_name)

        # iterate over environments
        for env_id, avd_name in _ENV_ID_AVD_NAME_DICT.items():
            success_rates[task_name][env_id] = 0

            # env creation
            env = BMocaEnv(
                task_path=task_path,
                avd_name=avd_name,
                state_type='text',
                action_tanh = False,
                adjusting_freq = args.adjusting_freq,
                run_headless=(not args.run_with_head)
            )
            env = GeminiTextActionParsingWrapper(env)

            # load model
            model = load_model(args.config_path)

            # test in target_env
            target_env_id = f"test_env_{env_id}"
            if args.verbose: print(target_env_id)

            # evaluate
            is_success = evaluate(args, env, target_env_id, model)    
            
            if is_success:
                success_rates[task_name][env_id] += 1
                if args.verbose: print('Success')
            else:
                if args.verbose: print('Failure')
            env.close()

    print(success_rates)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--avd_id', type=int, default=0)
    parser.add_argument('--android_avd_home', type=str, default=f'{_HOME_PATH}/.android/avd')
    parser.add_argument('--android_sdk_root', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk')
    parser.add_argument('--emulator_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator')
    parser.add_argument('--adb_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb')
    parser.add_argument('--run_with_head', default=False, action='store_true')   
    parser.add_argument('--adjusting_freq', type=float, default=1/3.0)   
    # agent
    parser.add_argument('--config_path', type=str, default=f'{_WORK_PATH}/config/model_config.yaml')
    # misc
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', default=bool, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args() 
    set_random_seed(args.seed)

    # update AVD_NAME
    for k, _ in _ENV_ID_AVD_NAME_DICT.items():
        _ENV_ID_AVD_NAME_DICT[k] += f"_test_{args.avd_id:02d}"

    # main run
    run(args)