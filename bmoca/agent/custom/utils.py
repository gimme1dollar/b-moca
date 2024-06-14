import io
import enum
import numpy as np
import os
import re
import random
import traceback
import math
import torch
import torch.nn as nn
import torch.distributions as pyd
from PIL import Image
from transformers import EfficientNetModel
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer

_WORK_PATH = os.environ['BMOCA_HOME']


TOUCH = [[y, x, y, x] 
        for y in np.arange(-0.95, 1.05, 0.075) # 14 bins
        for x in np.arange(-0.95, 1.05, 0.15)] # 27 bins
SWIPE = [[0.6, 0.0, -0.6, 0.0], # up
         [-0.6, 0.0, 0.6, 0.0], # down
         [0.0, -0.6, 0.0, 0.6], # right
         [0.0, 0.6, 0.0, -0.6]] # left
BUTTON = [[ 0.9, -0.56, 0.9, -0.56], # back
          [0.9, 0.0, 0.9, 0.0], # home
          [0.9, 0.56, 0.9, 0.56]] # overview


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def episode_len(episode):
    return next(iter(episode.values())).shape[0]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
        return episode
    
    
def dist(normalized_touch_yx, normalized_lift_yx):
  touch_y, touch_x = normalized_touch_yx
  lift_y, lift_x = normalized_lift_yx
  
  renormalized_touch_yx = [touch_y * 2, touch_x] # normalization to consider the scale difference
  renormalized_lift_yx = [lift_y * 2, lift_x]
  
  distance = np.linalg.norm(np.array(renormalized_touch_yx) - np.array(renormalized_lift_yx))
  return distance


def match(LIST, query):
    query = np.array(query)
    min_dist = 99999
    min_idx = -1
    for idx, vec in enumerate(LIST):
        vec_ = np.array(vec)
        dist = np.linalg.norm(vec_ - query)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx
    return min_idx


def closest_direction(vector):
    norm_vector = vector / np.linalg.norm(vector)
    directions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1])]
    cos_similarities = np.array([np.dot(norm_vector, unit_vector) for unit_vector in directions])
    return cos_similarities.argmax()



def sample_episode(env, agent, episode_len, use_prob=False, target_env_ids=None):
    """
    return list if episode dict
    """    
    ACTION_BUFFER = np.zeros((episode_len, 1), dtype=np.int64)
        
    episodes = {"observation": np.zeros((episode_len, 256, 128, 3), dtype=np.float32),
                "action": ACTION_BUFFER,
                "reward": np.zeros(episode_len, dtype=np.float32), 
                "next_observation": np.zeros((episode_len, 256, 128, 3), dtype=np.float32), 
                "done":np.zeros(episode_len, dtype=np.int16),
                "prob":np.zeros(episode_len, dtype=np.float32)}
    
    
    ACTION_SPACE = TOUCH+SWIPE+BUTTON
    lengths = [len(TOUCH), len(SWIPE), len(BUTTON)]
    accm_lengths = [sum(lengths[:idx]) for idx in range(len(lengths))]

    timestep = env.reset(target_env_id=target_env_ids)
    for step in range(episode_len):
        # select action
        if agent == 'random':
            save_actions = np.random.randint(0, len(ACTION_SPACE), size=(1,))
            actions = np.array(ACTION_SPACE[save_actions[0]])
        else:
            # print("up to here done")
            actions = agent.get_action_from_timesteps(timestep)
            if use_prob:
                actions, prob = actions
                prob = prob.cpu()
          
            save_actions = actions
            actions = np.array(ACTION_SPACE[actions[0]])
                                       
        timestep = env.step(actions)
        all_done = 1
       
        episodes["observation"][step] = timestep.prev_obs['pixel']
        episodes["action"][step] = int(save_actions[0])
        done = int(timestep.step_type == 2)
        all_done *= done
        episodes["reward"][step] = timestep.curr_rew
        episodes["next_observation"][step] = timestep.curr_obs['pixel']
        episodes["done"][step] = done
        episodes["length"] = episode_len
        episodes["instruction"] = timestep.instruction
        episodes["env_id"] = timestep.env_id
        if use_prob:
            episodes["prob"][step] = prob[0][save_actions]
        if all_done:
            break
    return episodes


def episode_to_return(episode):
    sum_returns = 0
    returns_per_setting = {}
    rewards = episode["reward"]
    task = episode["instruction"][6:].replace(" ", "_")
    env_id = episode["env_id"]
    success = int(rewards.any())
    if env_id in returns_per_setting.keys():
        returns_per_setting[env_id].append(success)
    else:
        returns_per_setting[env_id] = [success]

    print(f'env:{episode["env_id"]}, task:{episode["instruction"]}, success:', success)
    sum_returns += success
    avg_returns_per_setting = {}
    for e, returns in returns_per_setting.items():
        avg_returns_per_setting[e] = sum(returns) / len(returns)

    del episode
    return sum_returns, avg_returns_per_setting
    

def tokenize_instruction(instruction: str):
    tokenizer = AutoTokenizer.from_pretrained(f"{_WORK_PATH}/asset/agent/Auto-UI-Base")
    query_input = tokenizer(
                        instruction,
                        max_length=32,
                        pad_to_max_length=True,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
        )
    return query_input["input_ids"], query_input["attention_mask"]


def get_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        # For each parameter tensor, calculate its norm and add it to the total norm
        param_norm = torch.norm(param.data)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5  # Take the square root to get the total norm
    return total_norm
    
    
def get_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate L2 norm of gradient
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5  # Take the square root to get the total norm
    return total_norm


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    return
