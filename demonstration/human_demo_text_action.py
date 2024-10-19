import os
import re
import cv2
import sys
import time
import copy
import datetime
import argparse
import itertools
import traceback
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from pathlib import Path
from itertools import tee
from collections import defaultdict

import dm_env
from android_env.wrappers import base_wrapper

from bmoca.environment.environment import BMocaEnv, BMocaTimeStep

from bmoca.agent.foundation_model.utils import parse_obs
from bmoca.agent.custom.utils import episode_len, save_episode

_HOME_PATH = Path.home()
_WORK_PATH = os.environ["BMOCA_HOME"]

_ENV_ID_LIST = ["train_env_000"]

_TASK_NAME_LIST = ["phone/call_911"]

scroll_map = {
    "up": [0.8000, 0.5000, 0.2000, 0.5000],
    "down": [0.2000, 0.5000, 0.8000, 0.5000],
    "left": [0.5000, 0.2000, 0.5000, 0.8000],
    "right": [0.5000, 0.8000, 0.5000, 0.2000],
}


class RolloutStorer:
    def __init__(self, store_dir):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(exist_ok=True)

        self._current_episode = defaultdict(list)

    def add(self, time_step):
        """add in self._current_episode"""

        for k in time_step._fields:

            # get value
            v = time_step[k]
            if "obs" in k and (not (v is None)):
                v = v["pixel"]

            self._current_episode[k].append(copy.deepcopy(v))

    def store(self, timestamp):
        """store episode as .npz file in memory"""

        # dict values into np.array
        episode = dict()
        for k, v in self._current_episode.items():
            episode[k] = np.array(copy.deepcopy(v), dtype=np.object_)

        # store episode
        eps_len = episode_len(episode)
        eps_fn = f"{timestamp}_{eps_len}.npz"  # order for loaindg latest episodes first
        save_episode(episode, self._store_dir / eps_fn)

        # reset
        self._current_episode = defaultdict(list)
        return eps_len


class DemonstrationStorer(RolloutStorer):
    def __init__(self, store_dir):
        self._store_dir = store_dir
        if not os.path.isdir(store_dir):
            os.mkdir(store_dir)

        self._current_episode = ""
        self._prev_obs = None

    def add(self, timestep, demo_act=None, vh_file_name=None):
        if self._current_episode != "":
            self._current_episode += "\n"

        # o_t
        self._current_episode += f"- Observation: {self._prev_obs}\n"

        # a_t
        self._current_episode += f"- Action: {demo_act}\n"

        # o_t+1
        timestep.curr_obs["text"], parsed_curr_obs, _ = parse_obs(
            timestep.curr_obs["text"], skip_non_leaf=False, attribute_check=True
        )
        self._current_episode += f"- Next observation: {parsed_curr_obs}\n"

        # # store VH
        # if not (vh_file_name is None):
        #     timestep.curr_obs['text'], obs = tee(timestep.curr_obs['text'])

        #     xml_str = ""
        #     for event, elem in obs:
        #         xml_str += ET.tostring(elem, encoding="unicode")

        #     with open(vh_file_name, "a") as f:
        #         f.write(xml_str)

        # r_t
        self._current_episode += f"- Reward: {timestep.curr_rew}\n"

        # copy curr_obs to self.prev_obs
        self._prev_obs = copy.deepcopy(parsed_curr_obs)
        return

    def store(self, fn=None):
        if fn is None:
            fn = (
                self._store_dir
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                + ".txt"
            )

        with open(fn, "a") as f:
            f.write(self._current_episode)

        self._current_episode = ""
        return


class TextActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with TextAction"""

    def __init__(
        self,
        env: dm_env.Environment,
    ):
        super().__init__(env)
        self._env = env
        self._is_tablet = env._coordinator._is_tablet

        self.screen_h, self.screen_w = self._env._coordinator._screen_size

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, raw_action, bbox_list):
        try:
            if "dual-gesture" in raw_action:
                touch_y, touch_x, lift_y, lift_x = re.findall(
                    r"dual-gesture\((.*?)\)", raw_action
                )[0].split(",")

                final_action = np.array(
                    [float(touch_y), float(touch_x), float(lift_y), float(lift_x)]
                )

            elif "tap" in raw_action:
                area = int(re.findall(r"tap\((.*?)\)", raw_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

                final_action = np.array(
                    [
                        tar_y / self.screen_h,
                        tar_x / self.screen_w,
                        tar_y / self.screen_h,
                        tar_x / self.screen_w,
                    ]
                )

            elif "swipe" in raw_action:
                scroll_direction = re.findall(r"swipe\((.*?)\)", raw_action)[0][1:-1]

                final_action = np.array(scroll_map[scroll_direction])

            elif "press" in raw_action:
                key = re.findall(r"press\((.*?)\)", raw_action)[0][1:-1]

                if key == "BACK":
                    final_action = np.array([252 / 256, 43 / 128, 252 / 256, 43 / 128])
                if key == "HOME":
                    final_action = np.array([252 / 256, 64 / 128, 252 / 256, 64 / 128])
                if key == "OVERVIEW":
                    final_action = np.array([252 / 256, 85 / 128, 252 / 256, 85 / 128])

                if self._is_tablet:
                    # in arabic language, screen is x-axis inverted
                    final_action[1], final_action[3] = (
                        1 - final_action[1],
                        1 - final_action[3],
                    )
            else:
                raise ValueError
        except:
            print("** wrong format **")
            self._task_manager._stats["episode_steps"] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(
            env_id=self._env.curr_env_id,
            step_type=timestep.step_type,
            instruction=timestep.instruction,
            prev_obs=timestep.prev_obs,
            prev_act=raw_action,  # store raw action
            curr_obs=timestep.curr_obs,
            curr_rew=timestep.curr_rew,
        )
        # view_hierarchy=timestep.view_hierarchy)


def _collect_reward(timestep) -> bool:
    """Accumulates rewards collected over the course of an episode."""

    if timestep.curr_rew and timestep.curr_rew != 0:
        return True
    return False


def build_file_name(args, target_env_id, task_instruction, timenow):
    # timenow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    log_dir = f"{args.log_dir}/{task_instruction}"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_dir += f"/{target_env_id}"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    file_name = f"{log_dir}/{timenow}.txt"
    return file_name


def load_env(args, task_instruction=None):
    if task_instruction is None:
        task_path = args.task_path + "/dummy.textproto"
    else:
        task_path = args.task_path + f"/{task_instruction}.textproto"

    env = BMocaEnv(
        task_path=task_path,
        avd_name=args.avd_name,
        run_headless=(not args.run_with_head),
        state_type="text",
        action_tanh=False,
        adjusting_freq=(1.0),
    )
    env = TextActionParsingWrapper(env)  # parser_fn = parse_act

    return env


def run(args):
    """Iterate over tasks & train_envs"""

    storer = DemonstrationStorer(store_dir=args.log_dir)

    for task_name in _TASK_NAME_LIST:
        task_instruction = task_name.split("/")[-1]
        print(f"\n\n{task_instruction}\n\n")

        with load_env(args, task_name) as env:

            for target_env_id in _ENV_ID_LIST:
                print(f"\n{target_env_id}\n")
                timenow = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                # Reset environment.
                timestep = env.reset(target_env_id=target_env_id)
                storer.add(timestep=timestep, demo_act=None)

                while True:

                    # process raw_obs
                    raw_obs = timestep.curr_obs["text"]
                    _, parsed_obs, parsed_bbox = parse_obs(
                        raw_obs, skip_non_leaf=False, attribute_check=True
                    )

                    # print observation
                    try:
                        time.sleep(0.1)
                        print("Observation: \n" + str(parsed_obs) + "\n" + "Action:")
                    except:
                        traceback.print_exc()

                    # get action
                    """
                    The action input should be in the form of "(action)//".
                    For example, to tap the UI element with numeric tag of 5, please input "tap(5)//"
                    """
                    while True:
                        demo_act = sys.stdin.readline()
                        if "//" in demo_act:
                            break
                    demo_act = str(demo_act).split("//")[0]
                    raw_act = "Action: " + demo_act

                    # parse action
                    timestep = env.step(raw_act, parsed_bbox)
                    storer.add(timestep=timestep, demo_act=demo_act)

                    if _collect_reward(timestep):
                        break

                # store
                storer.store(
                    fn=build_file_name(args, target_env_id, task_instruction, timenow)
                )

    return


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--avd_name", type=str, default="pixel_3_test_00")
    parser.add_argument(
        "--android_avd_home", type=str, default=f"{_HOME_PATH}/.android/avd"
    )
    parser.add_argument(
        "--android_sdk_root", type=str, default=f"{_HOME_PATH}/.local/share/android/sdk"
    )
    parser.add_argument(
        "--emulator_path",
        type=str,
        default=f"{_HOME_PATH}/.local/share/android/sdk/emulator/emulator",
    )
    parser.add_argument(
        "--adb_path",
        type=str,
        default=f"{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb",
    )
    parser.add_argument("--task_path", type=str, default=f"{_WORK_PATH}/asset/tasks")
    parser.add_argument("--run_with_head", type=bool, default=True)
    # log
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f"{_WORK_PATH}/asset/dataset/demonstrations_text_action",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # main run
    run(args)
