"""BMoca environment implementation."""

import os
import copy
import time
import dm_env
import subprocess
import numpy as np

from typing import Any
from pathlib import Path
from absl import logging
from android_env.environment import AndroidEnv

from android_env.proto import state_pb2, snapshot_service_pb2
from android_env.components import config_classes
from android_env.components import coordinator as coordinator_lib
from android_env.components import task_manager as task_manager_lib
from android_env.components.simulators.emulator import emulator_simulator

from android_env.proto import task_pb2
from google.protobuf import text_format

from bmoca.environment import coordinator as coordinator_lib

import dm_env
from dm_env import specs, StepType
from typing import Any, NamedTuple

_HOME_PATH = Path.home()
_WORK_PATH = os.environ['BMOCA_HOME']


class BMocaTimeStep(NamedTuple):
    step_type: Any
    instruction: Any # \in "Goal: {task name}"
    prev_obs: Any
    prev_act: Any # \in [0, 1]^(4)
    curr_obs: Any # \in [0, 1]^(256x128x3) (xml.etree iterator)
    curr_rew: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class BMocaEnv(AndroidEnv):
  """An RL environment that interacts with Android apps, based on AndroidEnv"""
  
  def __init__(self, 
               task_path: str = f'{_WORK_PATH}/asset/tasks/home_screen.textproto',
               avd_name: str = f'pixel_3_test_00',
               android_avd_home: str = f'{_HOME_PATH}/.android/avd',
               android_sdk_root: str = f'{_HOME_PATH}/.local/share/android/sdk',
               emulator_path: str = f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator',
               adb_path: str = f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb',
               run_headless: bool = True,
               state_type: str = 'pixel', # 'ui_hierarchy'
               action_tanh: bool = True,
               adjusting_freq: float = 0,
               ):
    """Initializes the state of this BMocaEnv object."""

    # simulator
    self._simulator = emulator_simulator.EmulatorSimulator(
        emulator_launcher_args=dict(
            avd_name=avd_name,
            android_avd_home=os.path.expanduser(android_avd_home),
            android_sdk_root=os.path.expanduser(android_sdk_root),
            emulator_path=os.path.expanduser(emulator_path),
            run_headless=run_headless,
            gpu_mode='swiftshader_indirect',
        ),
        adb_controller_config=config_classes.AdbControllerConfig(
            adb_path=os.path.expanduser(adb_path),
            adb_server_port=5037,
        ),
    )

    # task manager
    task = task_pb2.Task()
    with open(task_path, 'r') as proto_file:
      text_format.Parse(proto_file.read(), task)
    self._task_manager = task_manager_lib.TaskManager(task)
    
    # coordinator
    self._coordinator = coordinator_lib.Coordinator(simulator=self._simulator, 
                                                    task_manager=self._task_manager, 
                                                    state_type=state_type,
                                                    adjusting_freq=adjusting_freq,
                                                    is_tablet="Tablet" in avd_name)

    # snapshot lists
    snapshot_list = self._coordinator._simulator._snapshot_stub.ListSnapshots(
                        snapshot_service_pb2.SnapshotFilter(
                                statusFilter=snapshot_service_pb2.SnapshotFilter.LoadStatus.All
                        )
                    )

    self.env_id_list = []
    for snapshot in snapshot_list.snapshots:
        if "env" in snapshot.snapshot_id:
            self.env_id_list.append(snapshot.snapshot_id)

    # informations
    self.instruction = "Goal: " + task_path.split("/")[-1].split(".")[0].replace("_", " ")
    self.state_type = state_type
    self.action_tanh = action_tanh
    self.curr_env_id = None

  def set_device(self, load=False):
    if load:
      time.sleep(2)
          
    if "06:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 123106272015.00"'
        _ = subprocess.run(command, text=True, shell=True)
    elif "10:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 123113302015.00"'
        _ = subprocess.run(command, text=True, shell=True)
    elif "13:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 123110302015.00"'
        _ = subprocess.run(command, text=True, shell=True)
    elif "17:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 123117332015.00"'
        _ = subprocess.run(command, text=True, shell=True)
    elif "20:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 123122292016.00"'
        _ = subprocess.run(command, text=True, shell=True)
    elif "23:30" in self.instruction:
        command = f'adb -s emulator-{self._simulator._adb_port-1} shell "su 0 toybox date 010103302016.00"'
        _ = subprocess.run(command, text=True, shell=True)
        
    time.sleep(0.1)
    return 

  def reset(self, target_env_id=None) -> dm_env.TimeStep:
    """Resets the environment for a new RL episode."""
        
    # load target snapshot
    if (target_env_id is None):
      target_env_id = np.random.choice(self.env_id_list, 1)[0]

    request = state_pb2.LoadStateRequest(
                args = {
                    'snapshot_name': target_env_id
                }
            )
  
    self.curr_env_id = target_env_id
    self._coordinator.load_snapshot(request)
    self.set_device(load=True)
  
    # get timestep
    timestep = self._coordinator.rl_reset()
    timestep = BMocaTimeStep(step_type=StepType.LAST if timestep.reward > 0 else timestep.step_type,
                                    instruction=self.instruction,
                                    prev_obs=None,
                                    prev_act=None,
                                    curr_obs=timestep.observation,
                                    curr_rew=timestep.reward) 

    # TODO: deepcopy doesn't work for iterator (obs['text'])
    self.prev_obs = copy.deepcopy(timestep.curr_obs)
    return timestep

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    """Takes a step in the environment."""

    if self.action_tanh: action = (action + 1.0) / 2.0
    timestep = self._coordinator.rl_step(action)
    timestep = BMocaTimeStep(step_type=StepType.LAST if timestep.reward > 0 else timestep.step_type,
                                    instruction=self.instruction,
                                    prev_obs=self.prev_obs,
                                    prev_act=action,
                                    curr_obs=timestep.observation,
                                    curr_rew=timestep.reward) 

    # TODO: deepcopy doesn't work for iterator (obs['text'])
    self.prev_obs = copy.deepcopy(timestep.curr_obs)
    self.set_device(load=False)

    return timestep

  def close(self) -> None:
    """Cleans up running processes, threads and local files."""
    logging.info('Cleaning up AndroidEnv...')
    if hasattr(self, '_coordinator'):
      self._coordinator.close()
    logging.info('Done cleaning up AndroidEnv.')
    
  def observation_spec(self):
    raise NotImplementedError
  
  def action_spec(self):
    raise NotImplementedError
