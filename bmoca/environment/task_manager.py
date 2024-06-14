# coding=utf-8
# Reference from https://github.com/google-deepmind/android_env
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TaskManager handles all events and information related to the task."""

import ast
from collections.abc import Callable
import copy
import datetime
import json
import re
import threading
from typing import Any

import logging
from android_env.components import adb_call_parser as adb_call_parser_lib
from android_env.components import app_screen_checker
from android_env.components import dumpsys_thread
from android_env.components import log_stream as log_stream_lib
from android_env.components import logcat_thread
from android_env.components import setup_step_interpreter
from android_env.proto import task_pb2
import dm_env
import numpy as np

from bmoca.environment.evaluator_script.evaluator import Evaluator


class TaskManager:
  """Handles all events and information related to the task."""

  def __init__(
      self,
      task: task_pb2.Task,
      instruction: str,
      max_bad_states: int = 3,
      max_failed_current_activity: int = 10,
  ):
    """Controls task-relevant events and information.

    Args:
      task: A task proto defining the RL task.
      max_bad_states: How many bad states in a row are allowed before a restart
        of the simulator is triggered.
      dumpsys_check_frequency: Frequency, in steps, at which to check
        current_activity and view hierarchy
      max_failed_current_activity: The maximum number of tries for extracting
        the current activity before forcing the episode to restart.
      extras_max_buffer_size: The maximum number of extras elements to store. If
        this number is exceeded, elements are dropped in the order they were
        received.
    """
    self._task = task
    self._max_bad_states = max_bad_states
    self._max_failed_current_activity = max_failed_current_activity
    self._driver = None
    self._instruction = instruction
    self._evaluator = Evaluator(instruction)

    self._lock = threading.Lock()
    self._logcat_thread = None
    self._setup_step_interpreter = None

    # Initialize stats.
    self._stats = {
        'episode_steps': 0,
        'reset_count_step_timeout': 0,
        'reset_count_user_exited': 0,
        'reset_count_episode_end': 0,
        'reset_count_max_duration_reached': 0,
        'restart_count_max_bad_states': 0,
        'task_updates': 0,
    }

    # Initialize internal state
    self._task_start_time = None
    self._bad_state_counter = 0
    self._is_bad_episode = False

    self._latest_values = {
        'reward': 0.0,
        'score': 0.0,
        'extra': {},
        'episode_end': False,
    }

    logging.debug('Task config: %s', self._task)

  def stats(self) -> dict[str, Any]:
    """Returns a dictionary of stats.

    This method is expected to be called after setup_task() has been called.
    """
    output = copy.deepcopy(self._stats)
    if self._setup_step_interpreter is not None:
      output.update(self._setup_step_interpreter.stats())
    return output

  def setup_task(self) -> None:
    """Performs one-off task setup.."""
    self._setup_step_interpreter.interpret(self._task.setup_steps)

  def stop(self) -> None:
    """Suspends task processing."""
    self._stop_logcat_thread()

  def start(
      self,
      adb_call_parser_factory: Callable[[], adb_call_parser_lib.AdbCallParser],
      log_stream: log_stream_lib.LogStream) -> None:
    """Starts task processing."""

    self._start_logcat_thread(log_stream=log_stream)
    self._logcat_thread.resume()
    # self._start_dumpsys_thread(adb_call_parser_factory())
    self._start_setup_step_interpreter(adb_call_parser_factory())

  def reset_task(self) -> None:
    """Resets a task for a new run."""

    self._logcat_thread.pause()
    self._setup_step_interpreter.interpret(self._task.reset_steps)
    self._logcat_thread.resume()

    # Reset some other variables.
    if not self._is_bad_episode:
      self._bad_state_counter = 0
    self._is_bad_episode = False

    self._task_start_time = datetime.datetime.now()
    with self._lock:
      self._latest_values = {
          'reward': 0.0,
          'score': 0.0,
          'extra': {},
          'episode_end': False,
      }

  def rl_reset(self, observation: dict[str, Any]) -> dm_env.TimeStep:
    """Performs one RL step."""

    self._stats['episode_steps'] = 0
    
    self._logcat_thread.line_ready().wait()
    
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=0.0,
        observation=observation)

  def rl_step(self, observation: dict[str, Any]) -> dm_env.TimeStep:
    """Performs one RL step."""
    self._stats['episode_steps'] += 1

    self._logcat_thread.line_ready().wait()
    with self._lock:
      reward = self._get_current_reward()
      transition_fn = self._determine_transition_fn()
      
    if self._evaluator.success_detector(self._driver):
      return dm_env.termination(reward=1.0, observation=observation)

    return transition_fn(reward=reward, observation=observation)

  def _get_current_reward(self) -> float:
    """Returns total reward accumulated since the last step."""
    reward = self._latest_values['reward']
    self._latest_values['reward'] = 0.0
    return reward

  def _determine_transition_fn(self) -> Callable[..., dm_env.TimeStep]:
    """Determines the type of RL transition will be used."""

    # Check if episode has ended
    if self._latest_values['episode_end']:
      self._stats['reset_count_episode_end'] += 1
      logging.debug('End of episode from logcat! Ending episode.')
      return dm_env.termination

    # Check if step limit or time limit has been reached
    if self._task.max_episode_steps > 0:
      if self._stats['episode_steps'] >= self._task.max_episode_steps:
        self._stats['reset_count_max_duration_reached'] += 1
        logging.debug('Maximum task duration (%r steps) reached. '
                     'Truncating the episode.', self._task.max_episode_steps)
        return dm_env.truncation

    return dm_env.transition

  def _start_setup_step_interpreter(
      self, adb_call_parser: adb_call_parser_lib.AdbCallParser):
    self._setup_step_interpreter = setup_step_interpreter.SetupStepInterpreter(
        adb_call_parser=adb_call_parser)

  def _start_logcat_thread(self, log_stream: log_stream_lib.LogStream):
    log_stream.set_log_filters(list(self._task.log_parsing_config.filters))
    self._logcat_thread = logcat_thread.LogcatThread(log_stream=log_stream)

    for event_listener in self._logcat_listeners():
      self._logcat_thread.add_event_listener(event_listener)

  def _stop_logcat_thread(self):
    if self._logcat_thread is not None:
      self._logcat_thread.kill()
      self._logcat_thread = None

  def _increment_bad_state(self) -> None:
    """Increments the bad state counter.

    Bad states are errors that shouldn't happen and that trigger an
    episode reset. If enough bad states have been seen consecutively,
    we restart the simulation in the hope of returning the simulation
    to a good state.
    """
    logging.warning('Bad state detected.')
    if self._max_bad_states:
      self._is_bad_episode = True
      self._bad_state_counter += 1
      logging.warning('Bad state counter: %d.', self._bad_state_counter)
      if self._bad_state_counter >= self._max_bad_states:
        logging.error('Too many consecutive bad states. Restarting simulator.')
        self._stats['restart_count_max_bad_states'] += 1
        self._should_restart = True
    else:
      logging.warning('Max bad states not set, bad states will be ignored.')

  def _logcat_listeners(self):
    """Creates list of EventListeners for logcat thread."""

    # Defaults to 'a^' since that regex matches no string by definition.
    regexps = self._task.log_parsing_config.log_regexps
    listeners = []

    # Reward listeners
    def _reward_handler(event, match):
      del event
      reward = float(match.group(1))
      with self._lock:
        self._latest_values['reward'] += reward

    for regexp in regexps.reward:
      listeners.append(logcat_thread.EventListener(
          regexp=re.compile(regexp or 'a^'),
          handler_fn=_reward_handler))

    # RewardEvent listeners
    for reward_event in regexps.reward_event:
      def get_reward_event_handler(reward):
        def _reward_event_handler(event, match):
          del event, match
          with self._lock:
            self._latest_values['reward'] += reward
        return _reward_event_handler

      listeners.append(logcat_thread.EventListener(
          regexp=re.compile(reward_event.event or 'a^'),
          handler_fn=get_reward_event_handler(reward_event.reward)))
    
    return listeners
