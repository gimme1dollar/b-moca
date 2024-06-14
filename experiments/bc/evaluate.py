import os
import cv2
import argparse
import datetime
import numpy as np

from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer

from bmoca.utils import set_random_seed

from bmoca.agent.custom import utils
from bmoca.agent.custom.utils import TOUCH, SWIPE, BUTTON
from bmoca.agent.custom.bc.bc_policy import BCContPolicy, BCDiscPolicy
from bmoca.agent.foundation_model.utils import PIL2OpenCV, OpenCV2PIL

from bmoca.environment.environment import BMocaEnv


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
    "clock/create_alarm_at_10:30_am",
    "clock/create_alarm_at_10:30_am_on_every_weekday",
    "settings/go_to_the_'add_a_language'_page_in_setting",
    "phone/call_the_white_house_(202-456-1111)",
    "calculator/input_'cos(180)'_in_Calculator",
    "wikipedia/disable_the_top_2_and_'randomizer'_topics_in_the_feed_customization_settings_on_Wikipedia_and_go_back_to_the_feed",
]


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

    observation = observation.copy()
    observation = cv2.resize(observation, dsize=(128, 256), interpolation=cv2.INTER_AREA)
    observation = observation.astype(np.float32) 
    observation = np.expand_dims(observation, axis=0) 
    observation = np.transpose(observation, (0, 3, 1, 2))

    instruction_input_ids, instruction_attention_mask, observation = \
        utils.to_torch([instruction_input_ids, instruction_attention_mask, observation], 
                       args.device)

    if args.action_space == 'continuous':
        pred_gesture = model(instruction_input_ids, instruction_attention_mask, observation)
        pred_gesture = pred_gesture.detach().cpu().numpy()
        
        if args.verbose:print((pred_gesture[0] + 1.0) / 2.0)
        pred_gesture = pred_gesture[0]
        
    elif args.action_space == 'discrete':
        actions = model(instruction_input_ids, instruction_attention_mask, observation)
        actions = actions.argmax().item()
        if actions >= len(TOUCH) + len(SWIPE):
            pred_gesture = BUTTON[actions - len(TOUCH) - len(SWIPE)]
        elif actions >= len(TOUCH):
            pred_gesture = SWIPE[actions - len(TOUCH)]
        else:
            pred_gesture = TOUCH[actions]
        pred_gesture = np.array(pred_gesture)

    timestep = env.step(pred_gesture)
    return timestep


def evaluate(args, env, env_id, task_name, model, tokenizer):
    imgs = []
    
    # main loop
    timestep = env.reset(target_env_id=env_id)
    print('reset done')
    while True:
        timestep = step(args, env, model, tokenizer, timestep)
        
        pixel_obs = Image.fromarray((timestep.prev_obs['pixel'] * 255).astype(np.uint8))
        
        imgcv = PIL2OpenCV(pixel_obs)
        imgcv = cv2.resize(imgcv, 
                            dsize=(1024//4, 2048//4), 
                            interpolation=cv2.INTER_AREA)
        imgpil = OpenCV2PIL(imgcv)
        
        imgpil_path = args.log_dir + "/tmp.png"
        imgpil.save(imgpil_path)
        imgs.append(imgpil)
        
        if (timestep is None): return False
        if timestep.last(): break

    if timestep.curr_rew > 0:
        success = True # success
    else:
        success = False # failure
        
    if len(imgs) > 1:
        imgs[0].save(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.gif', 
                        save_all=True, append_images=imgs[1:], optimize=False, duration=500, loop=0)
    else:
        imgs[0].save(f'{args.log_dir}/{task_name.split("/")[-1]}_{env_id}.png')
    return success


def run(args):
    # main loop
    success_rates = {}
    env = None
    for task_name in _TASK_NAME_LIST:
        success_rates[task_name] = {} 

        task_path = f"{_TASK_PATH}/{task_name}.textproto" 
        if args.verbose: print(task_name)

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
                    print("environment successfuly closed")

                prev_avd_name = avd_name
            else:
                create_flag = False

            if create_flag:
                print(f'creating env {avd_name}')
                print(args.adjusting_freq)
                env = BMocaEnv(
                    task_path=task_path,
                    avd_name=avd_name,
                    state_type='pixel',
                    action_tanh=True,
                    adjusting_freq = (args.adjusting_freq),
                    run_headless=(not args.run_with_head)
                )
                print('env has been created')

            # load model
            if args.action_space == 'continuous':
                model = BCContPolicy()
            elif args.action_space == 'discrete':
                model = BCDiscPolicy()
                
            model.load(filename = args.model_path, device = args.device)
            model.to(args.device)

            tokenizer = AutoTokenizer.from_pretrained(f"{_WORK_PATH}/asset/agent/Auto-UI-Base")   

            # test in target_env
            target_env_id = f"test_env_{env_id}"
            if args.verbose: print(target_env_id)

            is_success = evaluate(args, env, target_env_id, task_name, model, tokenizer)
                
            if is_success:
                success_rates[task_name][env_id] += 1
                if args.verbose: print('Success')
            else:
                if args.verbose: print('Failure')

    print(success_rates)
    env.close()
    print("env finally closed")


def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--avd_id', type=int, default=0) # set to be zero
    parser.add_argument('--android_avd_home', type=str, 
                        default=f'{_HOME_PATH}/.android/avd')
    parser.add_argument('--android_sdk_root', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk')
    parser.add_argument('--emulator_path', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator')
    parser.add_argument('--adb_path', type=str, 
                        default=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb')
    parser.add_argument('--run_with_head', default=False, type=bool)
    parser.add_argument('--adjusting_freq', type=float, default=0.33)   
    # model
    parser.add_argument('--finetune_vis_encoder', type=bool, default=True)
    # test
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, 
                        default=f'{_WORK_PATH}/logs/bc/bc_policy_latest.pt')
    parser.add_argument('--log_dir', type=str, default=f"{_WORK_PATH}/logs/bc")

    # misc
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--action_space', type=str, 
                        default='discrete', choices=['continuous', 'discrete'], help='action space to be used')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert not (args.model_path is None), "args.model_path should be specified!"
    set_random_seed(args.seed)

    args.time_now = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    for k, _ in _ENV_ID_AVD_NAME_DICT.items():
        _ENV_ID_AVD_NAME_DICT[k] += f"_test_{args.avd_id:02d}"

    # run main
    run(args)