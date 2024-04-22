import os
import os
import cv2
import time
import copy
import pygame
import socket
import logging
import argparse 
import numpy as np
import datetime
from typing import Any
from pathlib import Path
from collections import defaultdict

from absl import logging

import dm_env
from dm_env import specs, StepType

from android_env import loader
from android_env.wrappers import base_wrapper

from android_env.components import action_type
from android_env.components import utils

from android_env.proto import adb_pb2
from android_env.proto import state_pb2, snapshot_service_pb2
from android_env.components import errors

from bmoca.dataset.utils import episode_len, save_episode, load_episode
from bmoca.dataset.replay_buffer import RolloutStorer
from bmoca.environment.environment import BMocaEnv
from bmoca.environment.environment import BMocaTimeStep
from bmoca.environment.wrapper import RawActionWrapper


_HOME_PATH = Path.home()
_WORK_PATH = os.environ['BMOCA_HOME']


_ENV_ID_LIST = [
    "train_env_000", "train_env_001", "train_env_002", 
    "train_env_003", "train_env_004", "train_env_005", 
    "train_env_006", "train_env_007", "train_env_008", 
    "train_env_009", "train_env_010", "train_env_011", 
    "train_env_012", "train_env_013", "train_env_014", 
    "train_env_015", "train_env_016", "train_env_017", 
    "train_env_018", "train_env_019", "train_env_020", 
    "train_env_021", "train_env_022", "train_env_023", 
    "train_env_024", "train_env_025", "train_env_026", 
    "train_env_027", "train_env_028", "train_env_029", 
    "train_env_030", "train_env_031", "train_env_032", 
    "train_env_033", "train_env_034",
]

class VLUIRolloutStorer(RolloutStorer):
    def __init__(self, 
                 store_dir
    ):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(exist_ok=True)

        self._current_episode = defaultdict(list)

    def store(self, timestamp):
            '''store episode as .npz file in memory'''
            # dict values into np.array
            episode = dict()
            for k, v in self._current_episode.items():
                episode[k] = np.array(copy.deepcopy(v), dtype=np.object_)
            # store episode
            eps_len = episode_len(episode)
            # order for loaindg latest episodes first
            eps_fn = f'{timestamp}_{eps_len}.npz' 
            save_episode(episode, self._store_dir / eps_fn)
            # reset
            self._current_episode = defaultdict(list)
            return eps_len


class DemoCollectionWrapper(base_wrapper.BaseWrapper):
    def __init__(
        self,
        env: dm_env.Environment,
    ):
        super().__init__(env)
        self._env = env

    def reset(self, *args, **kwargs):        
        self._curr_act_value = np.array([0.5, 0.5, 0.5, 0.5])
        self._prev_act_type = 1
      
        timestep = self._env.reset(*args, **kwargs)
        self._prev_obs = copy.deepcopy(timestep.curr_obs)
        return timestep

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._env.step(action)
        curr_obs = timestep.curr_obs
        prev_obs = timestep.prev_obs

        if action['action_type'] == 0 and self._prev_act_type == 1:
          self._curr_act_value[:2] = action['touch_position'][:]
        elif action['action_type'] == 1 and self._prev_act_type == 0:
          self._curr_act_value[2:] = action['touch_position'][:] 
          # get obs from the simulator again
          try:
            time.sleep(3.0)
            curr_obs = copy.deepcopy(self._env._coordinator._get_simulator_state())
            prev_obs = copy.deepcopy(self._prev_obs)
          except (errors.ReadObservationError, socket.error):
            logging.exception('Unable to fetch observation. Restarting simulator.')
            self._env._coordinator._simulator_healthy = False
          
          print(self._curr_act_value) 
          self._prev_obs = copy.deepcopy(curr_obs)

        self._prev_act_type = action['action_type']
        timestep = BMocaTimeStep(
                    step_type=StepType.LAST if timestep.curr_rew > 0 else timestep.step_type,
                    instruction=timestep.instruction,
                    prev_obs=prev_obs,
                    prev_act=self._curr_act_value,
                    curr_obs=curr_obs,
                    curr_rew=timestep.curr_rew)
        return timestep 


def _render_pygame_frame(surface: pygame.Surface, screen: pygame.Surface, frame) -> None:
  """Displays latest observation on pygame surface."""

  # frame = timestep.observation['pixels'][:, :, :3]  # (H x W x C) (RGB)
  # frame = timestep.observation[:, :, -3:]
  frame = utils.transpose_pixels(frame)  # (W x H x C)

  pygame.surfarray.blit_array(surface, frame)
  pygame.transform.smoothscale(surface, screen.get_size(), screen)
  pygame.display.flip()
  return np.transpose(frame, axes=(2, 1, 0)) # C x H x W


def _scale_position(position: np.ndarray, screen: pygame.Surface) -> np.ndarray:
  """AndroidEnv accepts mouse inputs as floats so we need to scale it."""

  scaled_pos = np.divide(position, screen.get_size(), dtype=np.float32)
  return np.array([scaled_pos[1], scaled_pos[0]])


def _get_action_from_event(event: pygame.event.Event, screen: pygame.Surface) -> dict[str, Any]:
  """Returns the current action by reading data from a pygame Event object."""

  act_type = action_type.ActionType.LIFT
  if event.type == pygame.MOUSEBUTTONDOWN:
    act_type = action_type.ActionType.TOUCH

  return {
      'action_type':
          np.array(act_type, dtype=np.int32),
      'touch_position':
          _scale_position(event.pos, screen),
  }


def _get_action_from_mouse(screen: pygame.Surface) -> dict[str, Any]:
  """Returns the current action by reading data from the mouse."""

  act_type = action_type.ActionType.LIFT
  if pygame.mouse.get_pressed()[0]:
    act_type = action_type.ActionType.TOUCH

  return {
      'action_type':
          np.array(act_type, dtype=np.int32),
      'touch_position':
          _scale_position(pygame.mouse.get_pos(), screen),
  }


def _collect_reward(timestep) -> bool:
  """Accumulates rewards collected over the course of an episode."""
  if timestep.curr_rew and timestep.curr_rew != 0:
    return True 
  return False


def load_env(args):
    env = BMocaEnv(
        task_path=args.task_path,
        avd_name=args.avd_name,
        run_headless=(not args.run_with_head),
        state_type='pixel',
        action_tanh = False,
        adjusting_freq = (30.0),
    )
    env = RawActionWrapper(env)
    env = DemoCollectionWrapper(env)

    return env


def run(args):
  print("START OF MAIN PROGRAM")

  print("env loading...")
  with load_env(args) as env:
    print("env loaded!!!")            

    pygame.init()
    pygame.display.set_caption('test_reward')

    # Reset environment.
    for target_env_id in _ENV_ID_LIST:

      # make target_env dir and task dir
      store_dir = f"{args.log_dir}/{target_env_id}"
      task_name = args.task_path.split('/')[-1].split('.')[0]
      store_dir += f"/{task_name}"
      os.makedirs(store_dir, exist_ok=True)

      storer = VLUIRolloutStorer(store_dir=store_dir)
      
      for num_collection in range(args.num_of_collection):
        print(f"Collection {num_collection}/{args.num_of_collection}")
        
        timestep = env.reset(target_env_id=target_env_id)
        storer.add(timestep)

        # Create pygame canvas.
        screen_size = [512, 1024]  # (W x H)

        screen = pygame.display.set_mode(screen_size)  # takes (W x H)
        surface = pygame.Surface(screen_size)  # takes (W x H)

        # Start game loop.
        break_flag = False
        while True:
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                break

            all_events = pygame.event.get()
            for event in all_events:
                if event.type == pygame.QUIT:
                    break

            # Filter event queue for mouse click events.
            mouse_click_events = [
                event for event in all_events
                if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP]
            ]

            # Process all mouse click events.
            for event in mouse_click_events:
                action = _get_action_from_event(event, screen)

                timestep = env.step(action)
                if action['action_type'] == 1:
                  storer.add(timestep)

                frame = timestep.curr_obs['pixel'][:,:,-3:].copy() * 255
                frame = cv2.resize(frame, dsize=screen_size, interpolation=cv2.INTER_AREA)
                _ = _render_pygame_frame(surface, screen, frame)

                if _collect_reward(timestep):
                    print("Rewarded")
                    break_flag = True

            # Sample the current position of the mouse either way.
            action = _get_action_from_mouse(screen)

            timestep = env.step(action)

            if _collect_reward(timestep):
                print("Rewarded")
                break_flag = True

            frame = timestep.curr_obs['pixel'][:,:,-3:].copy() * 255
            frame = cv2.resize(frame, dsize=screen_size, interpolation=cv2.INTER_AREA)
            _ = _render_pygame_frame(surface, screen, frame)

            if break_flag:
                #eps_fn should be the format of timestamp_steps.npz
                storer.store(timestamp=datetime.datetime.now().strftime(('%Y%m%dT%H%M%S')))
                pygame.quit()
                break
          
  print("END OF MAIN PROGRAM")
  return


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--avd_name', type=str, default='pixel_3_train_00')
    # parser.add_argument('--target_env', type=str, default='train_env_000')
    parser.add_argument('--android_avd_home', type=str, default=f'{_HOME_PATH}/.android/avd')
    parser.add_argument('--android_sdk_root', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk')
    parser.add_argument('--emulator_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator')
    parser.add_argument('--adb_path', type=str, default=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb')
    parser.add_argument('--run_with_head', default=False, action='store_true')   
    # task
    parser.add_argument('--task_path', type=str, default=f'{_WORK_PATH}/asset/tasks/call_911.textproto') 
    # log
    parser.add_argument('--log_dir', type=str, default=f'{_WORK_PATH}/asset/demonstration') 
    parser.add_argument('--num_of_collection', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args() 
    run(args)
