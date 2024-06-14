"""Wraps the AndroidEnv environment"""
import re
import copy
import socket
import logging
import traceback
import numpy as np

from android_env.wrappers import base_wrapper
from android_env.components import errors

import dm_env
from dm_env import StepType

from bmoca.environment.environment import BMocaTimeStep

_SCROLL_MAP = {
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

        timestep = BMocaTimeStep(env_id=self.curr_env_id,
                                step_type=StepType.LAST if timestep.reward > 0 else timestep.step_type,
                                instruction=self.instruction,
                                prev_obs=self.prev_obs,
                                prev_act=action,
                                curr_obs=timestep.observation,
                                curr_rew=timestep.reward) 

        self.prev_obs = copy.deepcopy(timestep.curr_obs)
        return timestep


class GPTActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with GPT"""
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

    def step(self, raw_action, 
             elem_list, bbox_list):
        try:
            if "dual-gesture" in raw_action:
                touch_y, touch_x, lift_y, lift_x = re.findall(r"dual-gesture\((.*?)\)", \
                                                              raw_action)[0].split(",")

                final_action = np.array([float(touch_y), float(touch_x), float(lift_y), float(lift_x)])
                parsed_action = f'dual-gesture({touch_y}, {touch_x}, {lift_y}, {lift_x})'

            elif "tap" in raw_action:
                area = int(re.findall(r"tap\((.*?)\)", raw_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

                final_action = np.array([tar_y / self.screen_h, tar_x / self.screen_w, \
                                         tar_y / self.screen_h, tar_x / self.screen_w])
                parsed_action = f'tap({elem_list[area]})'

            elif "swipe" in raw_action:
                scroll_direction = re.findall(r"swipe\((.*?)\)", raw_action)[0][1:-1]

                final_action = np.array(_SCROLL_MAP[scroll_direction])
                parsed_action = f'swipe({scroll_direction})'

            elif "press" in raw_action:
                key = re.findall(r"press\((.*?)\)", raw_action)[0][1:-1]

                if key == "BACK":
                    final_action = np.array([252/256, 43/128, 252/256, 43/128])
                if key == "HOME":
                    final_action = np.array([252/256, 64/128, 252/256, 64/128])
                if key == "OVERVIEW":
                    final_action = np.array([252/256, 85/128, 252/256, 85/128])
                    
                if self._is_tablet:
                    # in arabic language, screen is x-axis inverted
                    final_action[1], final_action[3] = 1-final_action[1], 1-final_action[3]
                parsed_action = f'press({key})'
                
            else:
                raise ValueError           
        except:
            traceback.print_exc()
            print("** wrong format **")
            self._task_manager._stats['episode_steps'] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(env_id=timestep.env_id,
                             step_type=timestep.step_type,
                             instruction=timestep.instruction,
                             prev_obs=timestep.prev_obs,
                             prev_act=parsed_action, # store parsed action
                             curr_obs=timestep.curr_obs,
                             curr_rew=timestep.curr_rew) 


class GeminiActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with Gemini"""
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

    def step(self, raw_action, 
             elem_list, bbox_list):
        try:
            if "dual-gesture" in raw_action:
                touch_y, touch_x, lift_y, lift_x = re.findall(r"dual-gesture\((.*?)\)", \
                                                              raw_action)[0].split(",")

                final_action = np.array([float(touch_y), float(touch_x), float(lift_y), float(lift_x)])
                parsed_action = f'dual-gesture({touch_y}, {touch_x}, {lift_y}, {lift_x})'

            elif "tap" in raw_action:
                area = int(re.findall(r"tap\((.*?)\)", raw_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

                final_action = np.array([tar_y / self.screen_h, tar_x / self.screen_w, \
                                         tar_y / self.screen_h, tar_x / self.screen_w])
                parsed_action = f'tap({elem_list[area]})'

            elif "swipe" in raw_action:
                scroll_direction = re.findall(r"swipe\((.*?)\)", raw_action)[0][1:-1]

                final_action = np.array(_SCROLL_MAP[scroll_direction])
                parsed_action = f'swipe({scroll_direction})'

            elif "press" in raw_action:
                key = re.findall(r"press\((.*?)\)", raw_action)[0][1:-1]

                if key == "BACK":
                    final_action = np.array([252/256, 43/128, 252/256, 43/128])
                if key == "HOME":
                    final_action = np.array([252/256, 64/128, 252/256, 64/128])
                if key == "OVERVIEW":
                    final_action = np.array([252/256, 85/128, 252/256, 85/128])
                    
                if self._is_tablet:
                    # in arabic language, screen is x-axis inverted
                    final_action[1], final_action[3] = 1-final_action[1], 1-final_action[3]
                parsed_action = f'press({key})'
                
            else:
                raise ValueError           
        except:
            print("** wrong format **")
            self._task_manager._stats['episode_steps'] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(env_id=timestep.env_id,
                             step_type=timestep.step_type,
                             instruction=timestep.instruction,
                             prev_obs=timestep.prev_obs,
                             prev_act=parsed_action, # store parsed action
                             curr_obs=timestep.curr_obs,
                             curr_rew=timestep.curr_rew) 


class LlamaActionParsingWrapper(base_wrapper.BaseWrapper):
    """BMocaEnv with Llama"""
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

    def step(self, raw_action, 
             elem_list, bbox_list):
        try:
            if "dual-gesture" in raw_action:
                touch_y, touch_x, lift_y, lift_x = re.findall(r"dual-gesture\((.*?)\)", \
                                                              raw_action)[0].split(",")

                final_action = np.array([float(touch_y), float(touch_x), float(lift_y), float(lift_x)])
                parsed_action = f'dual-gesture({touch_y}, {touch_x}, {lift_y}, {lift_x})'

            elif "tap" in raw_action:
                area = int(re.findall(r"tap\((.*?)\)", raw_action)[0])
                gesture_position = bbox_list[area]

                tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
                tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

                final_action = np.array([tar_y / self.screen_h, tar_x / self.screen_w, \
                                         tar_y / self.screen_h, tar_x / self.screen_w])
                parsed_action = f'tap({elem_list[area]})'

            elif "swipe" in raw_action:
                scroll_direction = re.findall(r"swipe\((.*?)\)", raw_action)[0][1:-1]

                final_action = np.array(_SCROLL_MAP[scroll_direction])
                parsed_action = f'swipe({scroll_direction})'

            elif "press" in raw_action:
                key = re.findall(r"press\((.*?)\)", raw_action)[0][1:-1]

                if key == "BACK":
                    final_action = np.array([252/256, 43/128, 252/256, 43/128])
                if key == "HOME":
                    final_action = np.array([252/256, 64/128, 252/256, 64/128])
                if key == "OVERVIEW":
                    final_action = np.array([252/256, 85/128, 252/256, 85/128])
                    
                if self._is_tablet:
                    # in arabic language, screen is x-axis inverted
                    final_action[1], final_action[3] = 1-final_action[1], 1-final_action[3]
                parsed_action = f'press({key})'
                
            else:
                raise ValueError           
        except:
            traceback.print_exc()
            print("** wrong format **")
            self._task_manager._stats['episode_steps'] += 1
            return None

        # step with final action
        timestep = self._env.step(final_action)
        return BMocaTimeStep(env_id=timestep.env_id,
                             step_type=timestep.step_type,
                             instruction=timestep.instruction,
                             prev_obs=timestep.prev_obs,
                             prev_act=parsed_action, # store parsed action
                             curr_obs=timestep.curr_obs,
                             curr_rew=timestep.curr_rew) 