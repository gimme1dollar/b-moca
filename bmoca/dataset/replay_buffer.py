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

from bmoca.dataset.utils import episode_len, save_episode, load_episode

_WORK_PATH = os.environ['BMOCA_HOME']


class RolloutStorer():
    def __init__(self, 
                 store_dir
    ):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(exist_ok=True)

        self._current_episode = defaultdict(list)

    def add(self, time_step):
        """add in self._current_episode"""

        for k in time_step._fields:
            # get value
            v = time_step[k]
            if 'obs' in k and (not (v is None)): v = v['pixel']

            self._current_episode[k].append(copy.deepcopy(v))

    def store(self, episode_idx=0):
        """store episode as .npz file in memory"""

        # dict values into np.array
        episode = dict()
        for k, v in self._current_episode.items():
            episode[k] = np.array(copy.deepcopy(v), dtype=np.object_)

        # store episode
        eps_len = episode_len(episode)
        eps_fn = f"{episode_idx:09d}_{eps_len}.npz"  # order for loaindg latest episodes first
        save_episode(episode, self._store_dir / eps_fn)

        # reset
        self._current_episode = defaultdict(list)
        return eps_len


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


def load_few_shot_examples(replay_dir):
    """ Few shot example loader for LLM agents """
    print(f"few shot examples loading... (from {replay_dir})")
    few_shot_examples = ""

    episode_fns = [Path(p) for p in glob(str(replay_dir))]
    print(f"\t{len(episode_fns)} files...")
    for eps_fn in episode_fns:
        instruction = str(eps_fn).split("/")[-2].replace("_", " ")
        instruction = f"- Instruction: {instruction}\n" 
        
        with open(eps_fn, "r") as file:
            few_shot_transitions = file.read()
            few_shot_transitions = few_shot_transitions.split("\n\n")

            for few_shot_transition in few_shot_transitions:
                few_shot_example = instruction + few_shot_transition + "\n"

                if few_shot_examples != "": few_shot_examples += "\n"
                few_shot_examples += few_shot_example.replace("\n\n", "\n")

    few_shot_examples = few_shot_examples.split("\n\n")

    print(f"few shot examples loaded!!! ({len(few_shot_examples)} examples)")
    return few_shot_examples
