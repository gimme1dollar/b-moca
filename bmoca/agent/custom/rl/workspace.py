import os
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from collections import namedtuple

from absl import app
from absl import logging

import torch

import bmoca.agent.custom.utils as utils

from bmoca.agent.custom.rl.ppo import PPO

from bmoca.environment.environment import BMocaEnv


os.environ["TOKENIZERS_PARALLELISM"] = "false"
_HOME_PATH = Path.home()
_WORK_PATH = os.environ["BMOCA_HOME"]
_TASK_PATH = f"{_WORK_PATH}/asset/tasks"


_ENV_ID_AVD_NAME_DICT = {
    '100': 'pixel_3_test',
    '101': 'pixel_3_test',
    '102': 'pixel_3_test',
    '103': 'pixel_3_test',
    '104': 'pixel_3_test',
    '105': 'pixel_3_test',
    '106': 'pixel_4_test',
    '107': 'pixel_5_test',
    '108': 'pixel_6_test',
    '109': 'WXGA_Tablet_test',
}

TRAIN_ENV_ID_LIST = [f'train_env_{env:03d}' for env in [0]]      
                  

def make_env(env_cfg, 
             avd_name, 
             task):    
    env = BMocaEnv(
                task_path=os.path.join(_TASK_PATH, f'{task}.textproto'),
                avd_name=avd_name,
                android_avd_home=f'{_HOME_PATH}/.android/avd',
                android_sdk_root=f'{_HOME_PATH}/.local/share/android/sdk',
                emulator_path=f'{_HOME_PATH}/.local/share/android/sdk/emulator/emulator',
                adb_path=f'{_HOME_PATH}/.local/share/android/sdk/platform-tools/adb',
                run_headless=True,
                state_type='pixel',
                action_tanh=True,
                adjusting_freq=(env_cfg.adjusting_freq)) 
    return env


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup()

    def setup(self):
        # create envs
        algo_cfg, env_cfg, log_cfg = self.cfg.algo, self.cfg.env, self.cfg.log
        self.avail_tasks = algo_cfg.avail_tasks
        self.train_id = env_cfg.train_id
        self.test_id = env_cfg.test_id
        self.task = random.sample(self.avail_tasks, 1)[0]
        
        os.makedirs(f"{_WORK_PATH}/buffer", exist_ok=True)
        os.makedirs(f"{_WORK_PATH}/results/{log_cfg.checkpoint_name}", exist_ok=True)
        
        self.train_env_ids = TRAIN_ENV_ID_LIST
       
        self.train_env = make_env(env_cfg, f'pixel_3_train_{self.train_id:02d}', self.task)
                
        action_shape = [385]
        
        self.use_prob = (self.cfg.algorithm in ['ppo'])
        
        # create agent
        ALGORITHMS = {'ppo':PPO}
        try:
            self.agent = ALGORITHMS[self.cfg.algorithm](**vars(algo_cfg), 
                                                    checkpoint_name=log_cfg.checkpoint_name,
                                                    action_shape=action_shape)
        except:
            print("exception occured")
            self.close()   
        
            
    def reset(self, env_cfg, task):
        self.train_env = make_env(env_cfg=env_cfg, 
                                  avd_name=f'pixel_3_train_{self.train_id:02d}', 
                                  task=task)
       
    def collect_buffer(self):
        if self.cfg.load_buffer != 'None':
            self.agent.succ_replay_buffer.load_buffer(os.path.join(f"{_WORK_PATH}/buffer", self.cfg.load_buffer+'_succ.npz'))
            self.agent.fail_replay_buffer.load_buffer(os.path.join(f"{_WORK_PATH}/buffer", self.cfg.load_buffer+'_fail.npz'))
            print("all done successfully")
            
        for itr in range(self.cfg.expr_iters):
            print(f"expr iter {itr} start")
            episode = utils.sample_episode(self.train_env, "random", episode_len=self.cfg.episode_len, use_prob=self.use_prob,
                                            target_env_ids=random.sample(self.train_env_ids, 1)[0])
            self.agent.episode_to_buffer(episode)
        
            print(episode["env_id"], end=" ")
            print()
            del episode
            
            print("succ buffer size:", len(self.agent.succ_replay_buffer), " fail buffer size:", len(self.agent.fail_replay_buffer))
            if itr % 50 == 0:
                self.agent.succ_replay_buffer.save(f"{_WORK_PATH}/buffer", self.cfg.log.buffer_name+'_succ')
                self.agent.fail_replay_buffer.save(f"{_WORK_PATH}/buffer", self.cfg.log.buffer_name+'_fail')

        print("\n\n************* COLLECTING BUFFER DONE *************")

    
    def eval(self, itr):
        self.agent.training = False
        train_average_return = 0.0
        
        for env_ids in self.train_env_ids:
            train_episode = utils.sample_episode(self.train_env, self.agent, episode_len=self.cfg.episode_len, use_prob=self.use_prob,
                                                  target_env_ids=env_ids)          
        
            train_return, _ = utils.episode_to_return(train_episode)
            train_average_return += train_return
                    
        del train_episode
        
        print(f"{itr} step train average return: {train_average_return / len(self.train_env_ids)}")
        if not self.cfg.eval_at_train:
            self.train_env.close()
            print("train env closed")
            success_rates = {}
            test_avg_return = 0.0
            self.eval_env = None
            prev_avd_name = None

            for env_id, avd_name in _ENV_ID_AVD_NAME_DICT.items():
                print(env_id, avd_name)
                if prev_avd_name != avd_name:
                    create_flag = True
                    if not (self.eval_env is None):
                        self.eval_env.close()
                        print("environment successfuly closed")
                    prev_avd_name = avd_name
                else:
                    create_flag = False
                
                if create_flag:
                    print(f'creating {avd_name}..')
                    self.eval_env = make_env(self.cfg.env, avd_name+f'_{self.test_id:02d}', task=self.task)
                    print('env creation done')

                test_episode = utils.sample_episode(self.eval_env, self.agent, episode_len=self.cfg.episode_len, use_prob=self.use_prob,
                                                    target_env_ids=f"test_env_{env_id}")

                test_return, _ = utils.episode_to_return(test_episode)
                success_rates[f"test_env_{env_id}"] = test_return
                test_avg_return += test_return
                
                del test_episode
                
            self.eval_env.close()
            print("eval env closed")
            
            print(success_rates)
            print(f"{itr} step test average return: {test_avg_return / len(_ENV_ID_AVD_NAME_DICT)}")
        
            self.task = random.sample(self.avail_tasks, 1)[0]
            self.reset(self.cfg.env, self.task)
            
        self.agent.training = True
        if train_average_return > self.max_return:
            self.max_return = train_average_return
            self.agent.save(filename='./best.pt')
        
        
    def run(self):
        self.max_return = 0
        self.collect_buffer()
        self.train()
        self.close()


    def train(self):
        if self.cfg.load_weight != 'None':
            self.agent.load(f'{self.cfg.load_weight}')
            print('weight has been loaded!!')
        
        print("\n\n************* START TRAINING *************")
        for itr in range(self.cfg.train_iters):
            print(f"train iter {itr} start")
            episode = utils.sample_episode(self.train_env, self.agent, episode_len=self.cfg.episode_len, use_prob=self.use_prob,
                                            target_env_ids=random.sample(self.train_env_ids, 1)[0])
            
            print(episode["env_id"], end=" ")
            print()
            self.agent.episode_to_buffer(episode)
            del episode
            torch.cuda.empty_cache()

            if itr % self.cfg.update_interval == 0:
                self.agent.update(self.cfg.batch_size)
            
            if len(self.avail_tasks) > 1:
                if itr % 20 == 0 and itr != 0:
                    self.task = random.sample(self.avail_tasks, 1)[0]
                    self.train_env.close()
                    self.reset(self.cfg.env, self.task)
                    
            if (itr) % self.cfg.log.train_log_interval == 0:
                gc.collect()
                self.eval(itr)
                if itr % self.cfg.log.save_interval == 0:
                    self.agent.save(filename=f'./latest_model.pt')
        
        print("\n\n************* END OF TRAINING *************")
        return
    
    def close(self):
        self.train_env.close()
        if not self.cfg.eval_at_train:
            self.eval_env.close()
        self.agent.save(filename='./fin.pt')