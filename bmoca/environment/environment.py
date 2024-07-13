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

"""BMoca environment implementation."""
import os
import copy
import time
import dm_env
import datetime
import subprocess
import numpy as np
from absl import logging
from pathlib import Path
from itertools import tee
from typing import Any, NamedTuple

from android_env.environment import AndroidEnv
from android_env.proto import state_pb2, snapshot_service_pb2
from android_env.components import config_classes
from android_env.components.simulators.emulator import emulator_simulator

from android_env.proto import task_pb2
from google.protobuf import text_format

from bmoca.environment import coordinator as coordinator_lib
from bmoca.environment import task_manager as task_manager_lib

import dm_env
from dm_env import specs, StepType

_HOME_PATH = Path.home()
_WORK_PATH = os.environ["BMOCA_HOME"]


class BMocaTimeStep(NamedTuple):
    env_id: Any
    step_type: Any
    instruction: Any  # \in "Goal: {task name}"
    prev_obs: Any
    prev_act: Any  # \in [0, 1]^(4)
    curr_obs: Any  # \in [0, 1]^(256x128x3) (xml.etree iterator)
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

    def __init__(
        self,
        task_path: str = f"{_WORK_PATH}/asset/tasks/home_screen.textproto",
        avd_name: str = f"pixel_3_test_00",
        android_avd_home: str = f"{_HOME_PATH}/.android/avd",
        android_sdk_root: str = f"{_HOME_PATH}/.local/share/android/sdk",
        emulator_path: str = f"{_HOME_PATH}/.local/share/android/sdk/emulator/emulator",
        adb_path: str = f"{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb",
        run_headless: bool = True,
        state_type: str = "pixel",  # 'text'
        action_tanh: bool = True,
        adjusting_freq: float = 0,
    ):
        """Initializes the state of this BMocaEnv object."""

        # informations
        self.avd_name = avd_name
        self.instruction = "Goal: " + task_path.split("/")[-1].split(".")[0].replace(
            "_", " "
        )
        self.state_type = state_type
        self.action_tanh = action_tanh
        self.curr_env_id = None
        self.appium_servertime = None

        # simulator
        self._simulator = emulator_simulator.EmulatorSimulator(
            emulator_launcher_args=dict(
                avd_name=avd_name,
                android_avd_home=os.path.expanduser(android_avd_home),
                android_sdk_root=os.path.expanduser(android_sdk_root),
                emulator_path=os.path.expanduser(emulator_path),
                run_headless=run_headless,
                gpu_mode="swiftshader_indirect",
            ),
            adb_controller_config=config_classes.AdbControllerConfig(
                adb_path=os.path.expanduser(adb_path),
                adb_server_port=5037,
            ),
        )

        # appium server init
        appium_command = [f"{_HOME_PATH}/.nvm/versions/node/v18.12.1/bin/appium"]
        self.appium_process = subprocess.Popen(
            appium_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.appium_servertime = datetime.datetime.now()
        time.sleep(3)

        # task manager
        task = task_pb2.Task()
        with open(task_path, "r") as proto_file:
            text_format.Parse(proto_file.read(), task)
        self._task_manager = task_manager_lib.TaskManager(task, self.instruction)

        # coordinator
        self._coordinator = coordinator_lib.Coordinator(
            avd_name=avd_name,
            simulator=self._simulator,
            task_manager=self._task_manager,
            state_type=state_type,
            adjusting_freq=adjusting_freq,
            is_tablet="Tablet" in avd_name,
            driver_attempts=20,
        )

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

        return

    def reset(self, target_env_id=None) -> dm_env.TimeStep:
        """Resets the environment for a new RL episode."""

        # set device with load
        if target_env_id is None:
            target_env_id = np.random.choice(self.env_id_list, 1)[0]
        self.set_device(load=target_env_id)

        # get timestep
        timestep = self.get_state(action=None)
        return timestep

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Takes a step in the environment."""

        # set device without load
        self.set_device(load=None)

        # get timestep
        if self.action_tanh:
            action = (action + 1.0) / 2.0
        timestep = self.get_state(action=action)
        return timestep

    def get_state(self, action=None):
        # get timestep
        if action is None:
            timestep = self._coordinator.rl_reset()
            self.prev_obs = None
        else:
            timestep = self._coordinator.rl_step(action)
        timestep = BMocaTimeStep(
            env_id=self.curr_env_id,
            step_type=StepType.LAST if timestep.reward > 0 else timestep.step_type,
            instruction=self.instruction,
            prev_obs=self.prev_obs,
            prev_act=None,
            curr_obs=timestep.observation,
            curr_rew=timestep.reward,
        )

        # copy curr_obs to self.prev_obs
        prev_obs = {}
        for k, v in timestep.curr_obs.items():
            if k == "pixel":
                prev_obs[k] = copy.deepcopy(timestep.curr_obs[k])
            elif k == "text":
                prev_obs[k], timestep.curr_obs[k] = tee(timestep.curr_obs[k])
        self.prev_obs = prev_obs

        return timestep

    def set_device(self, load=None):
        if not (load is None):
            request = state_pb2.LoadStateRequest(args={"snapshot_name": load})

            # quit driver before loading snapshot
            if self._coordinator._driver is not None:
                self._coordinator._driver.quit()
                self._coordinator._driver = None
                self._task_manager._driver = None

            # restart appium server if it has been running for more than 10 minutes
            if datetime.datetime.now() - self.appium_servertime > datetime.timedelta(
                minutes=10
            ):
                self.appium_process.terminate()
                self.appium_process.wait()
                appium_command = [
                    f"{_HOME_PATH}/.nvm/versions/node/v18.12.1/bin/appium"
                ]
                self.appium_process = subprocess.Popen(
                    appium_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                self.appium_servertime = datetime.datetime.now()
                time.sleep(3)

            # load snapshot
            self.curr_env_id = load
            self._coordinator.load_snapshot(request)
            time.sleep(2)

        # misc. settings
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

    def close(self) -> None:
        """Cleans up running processes, threads and local files."""
        logging.info("Cleaning up B-MoCA...")
        if hasattr(self, "_coordinator"):
            self._coordinator.close()
        self.appium_process.terminate()
        logging.info("Done cleaning up B-MoCA.")

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        raise NotImplementedError
