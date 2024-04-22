"""Wraps the AndroidEnv environment"""
import os
import re
import cv2
import time
import copy
import socket
import logging
import datetime
import itertools
import traceback
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from android_env.proto import adb_pb2
from android_env.proto import state_pb2, snapshot_service_pb2
from android_env.wrappers import base_wrapper
from android_env.components import errors

import dm_env
from dm_env import specs, StepType

from bmoca.agent.llm_agent.gpt import parse_act as parse_act_gpt
from bmoca.agent.llm_agent.gemini import parse_act as parse_act_gemini

from bmoca.environment.environment import BMocaTimeStep

scroll_map = {
    "up": [0.8000, 0.5000, 0.2000, 0.5000],
    "down": [0.2000, 0.5000, 0.8000, 0.5000],
    "left": [0.5000, 0.2000, 0.5000, 0.8000],
    "right": [0.5000, 0.8000, 0.5000, 0.2000],
}


class RawActionWrapper(base_wrapper.BaseWrapper):
    def __init__(
        self,
        env: dm_env.Environment,
    ):
        super().__init__(env)
        self._env = env
        self._coordinator = env._coordinator

    def reset(self, *args, **kwargs) -> dm_env.TimeStep:
        return self._env.reset(*args, **kwargs)

    def step(self, action) -> dm_env.TimeStep:
        # send action
        self._coordinator._send_simulator_action(action)

        # get state from the simulator
        try:
            simulator_signals = self._coordinator._get_simulator_state()
        except (errors.ReadObservationError, socket.error):
            logging.exception('Unable to fetch observation. Restarting simulator.')
            self._coordinator._simulator_healthy = False

        # return transition
        if not self._coordinator._simulator_healthy:
            return dm_env.truncation(reward=0.0, observation=None)
        timestep = self._coordinator._task_manager.rl_step(simulator_signals)

        timestep = BMocaTimeStep(step_type=StepType.LAST if timestep.reward > 0 else timestep.step_type,
                                        instruction=self.instruction,
                                        prev_obs=self.prev_obs,
                                        prev_act=action,
                                        curr_obs=timestep.observation,
                                        curr_rew=timestep.reward) 

        self.prev_obs = copy.deepcopy(timestep.curr_obs)
        return timestep


class GeminiTextActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with Gemini"""
    def __init__(
        self,
        env: dm_env.Environment,
    ):
        super().__init__(env)
        self._env = env
        self._is_tablet = env._coordinator._is_tablet

        if self._is_tablet: 
            self.screen_w, self.screen_h = self._env._coordinator._screen_size 
        else: 
            self.screen_h, self.screen_w = self._env._coordinator._screen_size 

    def reset(self, *args, **kwargs):        
        return self._env.reset(*args, **kwargs)
    
    def step(self, raw_action, bbox_list):
        try:
            parsed_action = parse_act_gemini(raw_action)

            action_type = parsed_action.split("(")[0]

            if action_type == "dual-gesture":
                touch_y, touch_x, lift_y, lift_x = re.findall(r"dual-gesture\((.*?)\)", \
                                                              parsed_action)[0].split(",")

                final_action = np.array([float(touch_y), float(touch_x), float(lift_y), float(lift_x)])

            elif action_type == "tap":
                area = int(re.findall(r"tap\((.*?)\)", parsed_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2
                
                final_action = np.array([tar_y / self.screen_h, tar_x / self.screen_w, \
                                         tar_y / self.screen_h, tar_x / self.screen_w])

            elif action_type == "swipe":
                scroll_direction = re.findall(r"swipe\((.*?)\)", parsed_action)[0][1:-1]

                final_action = np.array(scroll_map[scroll_direction])

            elif action_type == "press":
                key = re.findall(r"press\((.*?)\)", parsed_action)[0][1:-1]

                if key == "BACK":
                    if self._is_tablet:
                        final_action = np.array([252/256, 85/128, 252/256, 85/128])
                    else:
                        final_action = np.array([252/256, 43/128, 252/256, 43/128])
                if key == "HOME":
                    final_action = np.array([252/256, 64/128, 252/256, 64/128])
                if key == "OVERVIEW":
                    if self._is_tablet:
                        final_action = np.array([252/256, 43/128, 252/256, 43/128])
                    else:
                        final_action = np.array([252/256, 85/128, 252/256, 85/128])
            else:
                raise ValueError
        except:
            print("** wrong format **")
            self._task_manager._stats['episode_steps'] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(step_type=timestep.step_type,
                                        instruction=timestep.instruction,
                                        prev_obs=timestep.prev_obs,
                                        prev_act=raw_action, # store raw action
                                        curr_obs=timestep.curr_obs,
                                        curr_rew=timestep.curr_rew) 
        

class GPTActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with GPT"""
    def __init__(
        self,
        env: dm_env.Environment,
    ):
        super().__init__(env)
        self._env = env
        self._is_tablet = env._coordinator._is_tablet
        
        if self._is_tablet: 
            self.screen_w, self.screen_h = self._env._coordinator._screen_size 
        else: 
            self.screen_h, self.screen_w = self._env._coordinator._screen_size 
        
    def reset(self, *args, **kwargs):        
        return self._env.reset(*args, **kwargs)

    def step(self, raw_action, bbox_list):
        try:
            try:
                parsed_action = parse_act_gpt(raw_action)
            except:
                raise ValueError

            action_type = parsed_action.split("(")[0]

            if action_type == "dual-gesture":
                touch_y, touch_x, lift_y, lift_x = re.findall(r"dual-gesture\((.*?)\)", \
                                                              parsed_action)[0].split(",")

                final_action = np.array([float(touch_y), float(touch_x), float(lift_y), float(lift_x)])

            elif action_type == "tap":
                area = int(re.findall(r"tap\((.*?)\)", parsed_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

                final_action = np.array([tar_y / self.screen_h, tar_x / self.screen_w, \
                                         tar_y / self.screen_h, tar_x / self.screen_w])

            elif action_type == "swipe":
                scroll_direction = re.findall(r"swipe\((.*?)\)", parsed_action)[0][1:-1]

                final_action = np.array(scroll_map[scroll_direction])

            elif action_type == "press":
                key = re.findall(r"press\((.*?)\)", parsed_action)[0][1:-1]

                if key == "BACK":
                    if self._is_tablet:
                        final_action = np.array([252/256, 85/128, 252/256, 85/128])
                    else:
                        final_action = np.array([252/256, 43/128, 252/256, 43/128])
                if key == "HOME":
                    final_action = np.array([252/256, 64/128, 252/256, 64/128])
                if key == "OVERVIEW":
                    if self._is_tablet:
                        final_action = np.array([252/256, 43/128, 252/256, 43/128])
                    else:
                        final_action = np.array([252/256, 85/128, 252/256, 85/128])
            else:
                raise ValueError            
        except:
            print("** wrong format **")
            self._task_manager._stats['episode_steps'] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(step_type=timestep.step_type,
                                        instruction=timestep.instruction,
                                        prev_obs=timestep.prev_obs,
                                        prev_act=raw_action, # store raw action
                                        curr_obs=timestep.curr_obs,
                                        curr_rew=timestep.curr_rew) 
