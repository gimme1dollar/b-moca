"""Coordinator handles interaction between internal components of AndroidEnv."""

import cv2
import time
import copy
import dm_env
import socket
import tempfile
import threading
import subprocess
import numpy as np
import xml.etree.ElementTree as xml_element_tree

from typing import Any
from absl import logging
from collections.abc import Sequence

from android_env.proto import adb_pb2
from android_env.proto import state_pb2
from android_env.proto import task_pb2
from android_env.components import action_type as action_type_lib
from android_env.components import adb_call_parser
from android_env.components import errors
from android_env.components import specs
from android_env.components import task_manager as task_manager_lib
from android_env.components import utils
from android_env.components.simulators import base_simulator

_SWIPE_DISTANCE_THRESHOLD = 0.14


def is_tap_action(normalized_touch_yx, normalized_lift_yx):
  touch_y, touch_x = normalized_touch_yx
  lift_y, lift_x = normalized_lift_yx
  
  renormalized_touch_yx = [touch_y * 2, touch_x]
  renormalized_lift_yx = [lift_y * 2, lift_x]
  
  distance = np.linalg.norm(np.array(renormalized_touch_yx) - np.array(renormalized_lift_yx))
  flag =  distance <= _SWIPE_DISTANCE_THRESHOLD

  return flag


def touch_position_to_pixel_position(
    touch_position: np.ndarray,
    height_width: Sequence[int],
) -> tuple[int, int]:
  """Maps touch position in [0,1] to the corresponding pixel on the screen."""
  touch_pixels = (touch_position * height_width).astype(np.int32)
  cap_idx = lambda v, idx_len: min(v, idx_len - 1)
  return tuple(map(cap_idx, touch_pixels, height_width))


class Coordinator:
  """Handles interaction between internal components of BMocaEnv."""

  def __init__(
      self,
      simulator: base_simulator.BaseSimulator,
      task_manager: task_manager_lib.TaskManager,
      state_type: str = 'pixel',
      show_touches: bool = False,
      show_pointer_location: bool = False,
      show_status_bar: bool = True,
      show_navigation_bar: bool = True,
      tmp_dir: str | None = None,
      adjusting_freq: float = 0, #Hz, if 0 then no adjustment
      is_tablet: bool = False,
  ):
    """Handles communication between AndroidEnv and its components.

    Args:
      simulator: A BaseSimulator instance.
      task_manager: The TaskManager, responsible for coordinating RL tasks.
      num_fingers: Number of virtual fingers of the agent.
      show_touches: Whether to show circles on the screen indicating the
        position of the current touch.
      show_pointer_location: Whether to show blue lines on the screen indicating
        the position of the current touch.
      tmp_dir: Temporary directory to write transient data.
    """
    self._simulator = simulator
    self._task_manager = task_manager

    self._show_touches = show_touches
    self._show_pointer_location = show_pointer_location
    self._show_status_bar = show_status_bar
    self._show_navigation_bar = show_navigation_bar

    self._adb_call_parser: adb_call_parser.AdbCallParser = None
    self._tmp_dir = tmp_dir or tempfile.gettempdir()
    self._adjusting_freq = adjusting_freq

    self.state_type = state_type
    self._screen_size = np.array([0, 0], dtype=np.int32) # [H, W]
    self._screen_resize = np.array([128, 256], dtype=np.int32) # [W, H]
    self._is_tablet = is_tablet

    # launch simulator
    logging.info('simulator launching...')
    self._simulator_healthy = False
    self._launch_simulator()
    logging.info('simulator launched!!!')

  def _launch_simulator(self, max_retries: int = 3):
    """Launches the simulator.

    Sets up the simulator and other task-related settings.

    Args:
      max_retries: Number of times to attempt a restart before raising an error.
    """
    self._simulator_healthy = False

    # Attempt to restart the system a given number of times.
    num_tries = 1
    latest_error = None
    while True:
      if num_tries > max_retries:
        raise errors.TooManyRestartsError(
            'Maximum number of restart attempts reached.') from latest_error
      logging.info('Simulator launch attempt %d of %d', num_tries, max_retries)

      self._task_manager.stop()

      # Launch the simulator.
      self._simulator.launch()

      # From here on, the simulator is assumed to be up and running.
      self._adb_call_parser = self._create_adb_call_parser()

      # screen settings
      try:
        screenshot_tmp = self._simulator.get_screenshot()
        self._screen_size = np.array(screenshot_tmp.shape[:2], dtype=np.int32)
        if self._is_tablet: self._screen_size = self._screen_size[::-1]
        
        self._adb_call_parser.parse(
            adb_pb2.AdbRequest(
                settings=adb_pb2.AdbRequest.SettingsRequest(
                    name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
                    put=adb_pb2.AdbRequest.SettingsRequest.Put(
                        key='show_touches',
                        value='1' if self._show_touches else '0'))))
        self._adb_call_parser.parse(
            adb_pb2.AdbRequest(
                settings=adb_pb2.AdbRequest.SettingsRequest(
                    name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
                    put=adb_pb2.AdbRequest.SettingsRequest.Put(
                        key='pointer_location',
                        value='1' if self._show_pointer_location else '0'))))
        if self._show_navigation_bar and self._show_status_bar:
          policy_control_value = 'null*'
        elif self._show_navigation_bar and not self._show_status_bar:
          policy_control_value = 'immersive.status=*'
        elif not self._show_navigation_bar and self._show_status_bar:
          policy_control_value = 'immersive.navigation=*'
        else:
          policy_control_value = 'immersive.full=*'
        self._adb_call_parser.parse(
            adb_pb2.AdbRequest(
                settings=adb_pb2.AdbRequest.SettingsRequest(
                    name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
                    put=adb_pb2.AdbRequest.SettingsRequest.Put(
                        key='policy_control', value=policy_control_value))))

      except errors.AdbControllerError as e:
        logging.exception('_update_simulator_screen_settings failed.')
        num_tries += 1
        continue

      # Start the task.
      self._task_manager.start(
          adb_call_parser_factory=self._create_adb_call_parser,
          log_stream=self._simulator.create_log_stream(),
      )
      try:
        self._task_manager.setup_task()
      except errors.StepCommandError as error:
        logging.exception('Failed to set up the task. Restarting simulator.')
        latest_error = error
        num_tries += 1
        continue

      # Restart was successful.
      self._simulator_healthy = True
      break
    return 

  def rl_reset(self) -> dm_env.TimeStep:
    """Resets the RL episode."""
    # Relaunch the simulator if necessary.
    if not self._simulator_healthy: # or self._should_periodic_relaunch():
      self._launch_simulator()

    # Execute a lift action before resetting the task.
    lift_action = {
        'action_type': np.array(action_type_lib.ActionType.LIFT),
        'touch_position': np.array([0, 0]),
    }
    self._send_simulator_action(lift_action)

    # Reset the task.
    self._task_manager.reset_task()

    # get state
    simulator_signals = self._get_simulator_state()
    return self._task_manager.rl_reset(simulator_signals)

  def rl_step(self, dual_gesture_action: np.ndarray) -> dm_env.TimeStep:
    """Executes the selected action and returns a timestep.

    Args:
      agent_action: Selected action to perform on the simulated Android device.
        If `agent_action` is `None` it means that this is an RL reset (to start
        a new episode).

    Returns:
      An RL timestep.
    """
    # send action after postprocessing dual_gesture action into AndroidEnv actions    
    touch_y, touch_x, lift_y, lift_x = dual_gesture_action

    action_touch = {'action_type':0, 'touch_position': np.array([touch_y, touch_x])}
    self._send_simulator_action(action_touch)

    if is_tap_action([touch_y, touch_x], [lift_y, lift_x]):
        action_lift = {'action_type':1, 'touch_position': np.array([touch_y, touch_x])}
        self._send_simulator_action(action_lift)
    else:
        action_layover = {'action_type':0, 'touch_position': np.array([lift_y, lift_x])}
        self._send_simulator_action(action_layover)

        action_lift = {'action_type':1, 'touch_position': np.array([lift_y, lift_x])}
        self._send_simulator_action(action_lift)

    # get state from the simulator
    try:
      simulator_signals = self._get_simulator_state()
    except (errors.ReadObservationError, socket.error):
      logging.exception('Unable to fetch observation. Restarting simulator.')
      self._simulator_healthy = False

    # return transition
    if not self._simulator_healthy:
      return dm_env.truncation(reward=0.0, observation=None)
    return self._task_manager.rl_step(simulator_signals)

  def _get_simulator_state(self, start_time=None) -> dict[str, np.ndarray]:
    """Gathers data from various sources to assemble the RL observation."""  
    # adjust frequency for robust observation acquisition
    if self._adjusting_freq > 0: 
      time.sleep(1/self._adjusting_freq)

    res_state = {}

    # pixel states
    pixel = self._simulator.get_screenshot()  # Sync mode.

    if self.state_type == 'pixel': # preprocess
      pixel = cv2.resize(pixel, dsize=self._screen_resize, interpolation=cv2.INTER_AREA)
      pixel = pixel / 255.0

    res_state['pixel'] = pixel

    # text states
    if self.state_type == 'text':
      try:
          _adb_prefix = self._simulator._adb_controller.command_prefix()

          # get xml 
          dump_command = _adb_prefix + ['shell', 'uiautomator', 'dump']
          subprocess.run(dump_command, check=True)

          cat_command = _adb_prefix + [
              'pull',
              '/sdcard/window_dump.xml',
              f'{self._tmp_dir}/ui_hierarchy.xml',
          ]
          subprocess.run(cat_command, check=True, capture_output=True, text=True)

          # build hierarchy tree
          xml_path = f'{self._tmp_dir}/ui_hierarchy.xml'
          ui_hierarchy = xml_element_tree.iterparse(xml_path)

      except subprocess.CalledProcessError as e:
          logging.error(f'Command failed: {e}')
          ui_hierarchy = None
  
      res_state['text'] = ui_hierarchy
  
    return res_state

  def _send_simulator_action(self, action):
    """Loads a state.

    Args:
      action: AndroidEnv action
        'action_type'
        'touch_position'

    Returns:
      A `LoadStateResponse` containing the status, error message (if
      applicable), and any other relevant information.
    """
    is_touch = (action['action_type'] == action_type_lib.ActionType.TOUCH)
    touch_position = action['touch_position']
    if self._is_tablet: 
      touch_position = touch_position[::-1] # w, h reversed
      touch_position[0] = 1 - touch_position[0] # x reversed
    touch_pixels = touch_position_to_pixel_position(touch_position, height_width=self._screen_size)

    try:
        self._simulator.send_touch([(touch_pixels[1], touch_pixels[0], is_touch, 0)])
    except (socket.error, errors.SendActionError):
      logging.exception('Unable to execute action. Restarting simulator.')
      self._simulator_healthy = False
      
  def load_snapshot(
      self, request: state_pb2.LoadStateRequest
  ) -> state_pb2.LoadStateResponse:
    """Loads a state.

    Args:
      request: A `LoadStateRequest` containing any parameters necessary to
        specify how/what state to load.

    Returns:
      A `LoadStateResponse` containing the status, error message (if
      applicable), and any other relevant information.
    """
    self._task_manager.stop()
    response = self._simulator.load_state(request)
    self._task_manager.start(
        adb_call_parser_factory=self._create_adb_call_parser,
        log_stream=self._simulator.create_log_stream(),
    )
    return response
  
  def _create_adb_call_parser(self):
    """Creates a new AdbCallParser instance."""
    return adb_call_parser.AdbCallParser(
        adb_controller=self._simulator.create_adb_controller(),
        tmp_dir=self._tmp_dir)

  def execute_adb_call(self, call: adb_pb2.AdbRequest) -> adb_pb2.AdbResponse:
    return self._adb_call_parser.parse(call)

  def close(self):
    """Cleans up the state of this Coordinator."""
    if hasattr(self, '_task_manager'):
      self._task_manager.stop()
    if hasattr(self, '_simulator'):
      self._simulator.close()

  def __del__(self):
    self.close()