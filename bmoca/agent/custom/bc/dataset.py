import os
import json
import copy
import random
import logging
import traceback
import numpy as np

from glob import glob
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer

from bmoca.agent.custom.utils import episode_len, save_episode, load_episode, dist, TOUCH, SWIPE, BUTTON
from bmoca.agent.custom.utils import closest_direction, dist, match

from bmoca.agent.foundation_model.utils import parse_obs


_WORK_PATH = os.environ['BMOCA_HOME']

class ReplayBuffer(IterableDataset):
    """ Replay loader for VLUI agents """
    def __init__(self, 
                 replay_dir, # contains task
                 n_step=3,
                 discount=0.99):
        self._replay_dir = replay_dir
        
        # misc.
        self._n_step = n_step
        self._discount = discount

    def _sample(self):
        raise NotImplementedError
    
    def __iter__(self):
        while True:
            yield self._sample()
            
            
class BCContReplayBuffer(ReplayBuffer):
    def __init__(self, 
                 replay_dir,
                 task_instructions,
                 num_env=35
                 ):
        self._replay_dir = replay_dir
        self.task_instructions = task_instructions

        # loading
        self._size = 0
        self.num_env = num_env
        self._episode_fns = []
        self._episode_dict = dict()
        self._episode_size = dict()

        self.tokenizer = AutoTokenizer.from_pretrained(f"{_WORK_PATH}/asset/agent/Auto-UI-Base")
        
    def _Preprocess(self, episode):
        return episode

    def _load(self):
        print(f"ReplayBuffer loading... (from {self._replay_dir})")
        print(f"Num env: {self.num_env}")
        if self.num_env == 35:
            envs = list(range(35))
        elif self.num_env == 10:
            envs = [0, 1, 2, 3, 4, 5, 7, 8, 21, 22]
        elif self.num_env == 7:
            envs = list(range(7))
        elif self.num_env == 21:
            envs = list(range(21))
        
        for task_instruction in self.task_instructions:
            for env in envs:
                pattern = f"{task_instruction}/train_env_{env:03d}/*.npz"
                self._episode_fns +=\
                    sorted(Path(self._replay_dir).expanduser().glob(pattern))
        self._episode_fns = sorted(self._episode_fns)
        
        for eps_fn in self._episode_fns:
            # load
            try:
                episode = load_episode(eps_fn)
                episode = self._Preprocess(episode)

            except:
                traceback.print_exc()    
                logging.error("Exception while loading in replay buffer")
                break
            
            self._episode_dict[eps_fn] = episode
            
            # get size info
            eps_len = episode_len(episode)
            self._episode_size[eps_fn] = eps_len
            self._size += eps_len

        print(f"ReplayBuffer loaded!!! ({len(self._episode_fns)} demos, {self._size} steps)")
        
        
    def _sample(self, sample_episode=False):   
        
        # sample & load episode
        ep_fn = random.choice(self._episode_fns)

        episode = self._episode_dict[ep_fn]
        
        if sample_episode: 
            return episode
        
        # sample index
        ep_len = self._episode_size[ep_fn]
        idx = np.random.randint(0, ep_len)
        # sample transition
        instruction = episode['instruction'][idx]
        observation = episode['prev_obs'][idx]
        action      = (episode['prev_act'][idx] * 2.0 - 1.0)
        
        query_input = self.tokenizer(
                        instruction,
                        max_length=32,
                        pad_to_max_length=True,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
        )

        return (query_input["input_ids"][0], query_input["attention_mask"][0], \
                observation, action)


class BCDiscReplayBuffer(BCContReplayBuffer):
    def __init__(self, 
                 replay_dir,
                 task_instructions,
                 num_env=35
                 ):
        
        super().__init__(replay_dir=replay_dir, task_instructions=task_instructions, num_env=num_env)
    
    def _Preprocess(self, episode):
        # prune None
        for k, v in episode.items():
            episode[k] = episode[k][1:]
        
        # TOUCH = [[y, x, y, x] for y in np.arange(-0.95, 1.05, 0.1) for x in np.arange(-0.95, 1.05, 0.2)]
        threshold = 0.1
        concat_action = np.zeros((len(episode['prev_act']), 1), dtype=np.int64)
        
        for idx, action in enumerate(episode['prev_act']):
            action = 2*action - 1
            is_touch = (dist(action[:2], action[2:]) < threshold)
            is_home = (dist(action[:2], 2*np.array([0.95, 0.50])-1) < 0.01)
            is_back = (dist(action[:2], 2*np.array([0.95, 0.22])-1) < 0.01)
            is_overview = (dist(action[:2], 2*np.array([0.95, 0.78])-1) < 0.01)
            
            if is_touch and is_back:
                concat_action[idx] = len(TOUCH) + 4 + 0
                continue
            elif is_touch and is_home:
                concat_action[idx] = len(TOUCH) + 4 + 1
                continue
            elif is_touch and is_overview:
                concat_action[idx] = len(TOUCH) + 4 + 2
                continue
            
            if is_touch:
                touch_action = match(TOUCH, action)
                concat_action[idx] = touch_action
            else:
                swipe_direction = closest_direction(action[2:] - action[:2])
                concat_action[idx] = len(TOUCH) + swipe_direction

        episode['prev_act'] = concat_action
        return episode
    
    
    def _sample(self, sample_episode=False):   
        
        # sample & load episode
        ep_fn = random.choice(self._episode_fns)

        episode = self._episode_dict[ep_fn]
        
        if sample_episode: 
            return episode
        
        # sample index
        ep_len = self._episode_size[ep_fn]
        idx = np.random.randint(0, ep_len)
        
        # sample transition
        instruction = episode['instruction'][idx]
        observation = episode['prev_obs'][idx]
        action      = episode['prev_act'][idx]
        
        query_input = self.tokenizer(
                        instruction,
                        max_length=32,
                        pad_to_max_length=True,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
        )

        return (query_input["input_ids"][0], query_input["attention_mask"][0], \
                observation, action)

