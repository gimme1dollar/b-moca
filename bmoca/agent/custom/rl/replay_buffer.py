import os
import random
from collections import namedtuple, deque
import numpy as np
import abc
from pathlib import Path
from bmoca.agent.custom.utils import dist, match, closest_direction, TOUCH, SWIPE, BUTTON
from collections import defaultdict
from bmoca.agent.custom.utils import episode_len, load_episode, save_episode


Transition = namedtuple('Transition', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done'))
Transition_Prob = namedtuple('Transition_Prob', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done', 'prob', 'task'))
Transition_Task = namedtuple('Transition_Task', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done', 'task'))
            
            
class RLReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.named_tuple = namedtuple('Transition', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done'))
        self.memory = deque([], maxlen=capacity)
    
    @abc.abstractmethod
    def _Preprocess(self, episode):        
        return
    
    def push(self, *args):
        self.memory.append(self.named_tuple(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class MultiTaskReplayBuffer(RLReplayBuffer):
    def __init__(self, capacity, task_instructions):
        super().__init__(capacity=capacity)
        self.task_instructions = task_instructions
        self.named_tuple = namedtuple('Transition_Task', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done', 'task'))
        self._episode_fns = []
        self._size = 0
    
    def episode_to_buffer(self, episode, ep_len):
        for t in range(ep_len):
            state = episode["prev_obs"][t].astype(np.float32)
            action = np.array([episode["prev_act"][t]], dtype=np.int64)
            # print(action)
            n_rewards_sum = np.sum(episode["curr_rew"][t:t+1])
            n_next_state = episode["curr_obs"][t].astype(np.float32)
            done = (n_rewards_sum > 0)
            task = episode["instruction"][t]
            self.push(state, action, n_rewards_sum, n_next_state, done, task)
        return
    
    def load_demo(self, filepath, duplicate=1):
        print(f"ReplayBuffer loading... (from {filepath})")
        print("Loading only 10..")
        for task_instruction in self.task_instructions:
            for env in [0, 1, 2, 3, 4, 5, 7, 8, 21 ,22]:
                self._episode_fns += sorted(Path(filepath).expanduser().glob(f"{task_instruction}/train_env_{env:03d}/*.npz"))
        
        self._episode_fns = sorted(self._episode_fns)
        # print(self._episode_fns)
        for _ in range(duplicate):
            for eps_fn in self._episode_fns:
                episode = load_episode(eps_fn)
                episode = self._Preprocess(episode)
                eps_len = episode_len(episode)
                self.episode_to_buffer(episode, eps_len)
                # get size info
                self._size += eps_len

        print(f"ReplayBuffer loaded!!! ({len(self._episode_fns)} demos, {self._size // duplicate} steps)")
        
    def load_buffer(self, filepath):
        filepath = Path(filepath).expanduser()
        episode = load_episode(filepath)
        eps_len = episode_len(episode)
        self.episode_to_buffer(episode, eps_len)
        print(f"ReplayBuffer loaded!!! ({eps_len} steps)")
        
    def save(self, filepath, name):
        filepath = Path(filepath)
        '''store episode as .npz file in memory'''
        episode = {'prev_obs':[], 'prev_act':[], 'curr_rew':[], 'curr_obs':[], 'instruction':[]}
        for idx in range(len(self.memory)):
            transition = self.memory[idx]
            episode['prev_obs'].append(transition.state)
            episode['prev_act'].append(transition.action)
            episode['curr_rew'].append(transition.n_rewards_sum)
            episode['curr_obs'].append(transition.n_next_state)
            episode['instruction'].append(transition.task)
                
        episode['prev_obs'] = np.array(episode['prev_obs'], dtype=np.float32)
        episode['prev_act'] = np.array(episode['prev_act'], dtype=np.float32)
        episode['curr_rew'] = np.array(episode['curr_rew'], dtype=np.float32)
        episode['curr_obs'] = np.array(episode['curr_obs'], dtype=np.float32)
        episode['instruction'] = np.array(episode['instruction'], dtype=np.object_)
        
        # order for loaindg latest episodes first
        eps_fn = f'{name}.npz' 
        save_episode(episode, filepath / eps_fn)
        return
    

class MultiTaskReplayBufferConcat(MultiTaskReplayBuffer):
    def __init__(self, capacity, task_instructions):
        super().__init__(capacity=capacity, task_instructions=task_instructions)
    
    def _Preprocess(self, episode):
        for k, v in episode.items():
            episode[k] = episode[k][1:]
            
        threshold = 0.1
        concat_action = np.zeros((len(episode['prev_act']), 1), dtype=np.int64)
        
        for idx, action in enumerate(episode['prev_act']):
            action = 2*action - 1
            is_touch = (dist(action[:2], action[2:]) < threshold)
            is_back = (dist(action[:2], 2*np.array([0.95, 0.22])-1) < 0.01)
            is_home = (dist(action[:2], 2*np.array([0.95, 0.50])-1) < 0.01)
            is_overview = (dist(action[:2], 2*np.array([0.95, 0.78])-1) < 0.01)
            
            if is_touch and is_back:
                concat_action[idx] = len(TOUCH) + len(SWIPE) + 0
                continue
            elif is_touch and is_home:
                concat_action[idx] = len(TOUCH) + len(SWIPE) + 1
                continue
            elif is_touch and is_overview:
                concat_action[idx] = len(TOUCH) + len(SWIPE) + 2
                continue
            
            if is_touch:
                touch_action = match(TOUCH, action)
                concat_action[idx] = touch_action
            else:
                swipe_direction = closest_direction(action[2:] - action[:2])
                concat_action[idx] = len(TOUCH) + swipe_direction

        episode['prev_act'] = concat_action.squeeze(axis=1)
        return episode
    
    
class MultiTaskReplayBufferConcatProb(MultiTaskReplayBufferConcat):
    def __init__(self, capacity, task_instructions):
        super().__init__(capacity=capacity, task_instructions=task_instructions)
        self.named_tuple = namedtuple('Transition_Prob', ('state', 'action', 'n_rewards_sum', 'n_next_state', 'done', 'prob', 'task'))
    
    def episode_to_buffer(self, episode, ep_len):
        for t in range(ep_len):
            state = episode["prev_obs"][t].astype(np.float32)
            action = episode["prev_act"][t].astype(np.int64)
            # print(action)
            n_rewards_sum = np.sum(episode["curr_rew"][t:t+1])
            n_next_state = episode["curr_obs"][t].astype(np.float32)
            prob = episode["prob"][t].astype(np.float32)
            done = (n_rewards_sum > 0)
            task = episode["instruction"][t]
            self.push(state, action, n_rewards_sum, n_next_state, done, prob, task)
        return
    