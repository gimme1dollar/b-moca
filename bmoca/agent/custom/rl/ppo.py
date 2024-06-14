import os
import torch
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim

from bmoca.agent.custom import utils
from bmoca.agent.custom.networks import CustomAgentBackbone

from bmoca.agent.custom.rl.replay_buffer import Transition_Prob 
from bmoca.agent.custom.rl.replay_buffer import MultiTaskReplayBufferConcatProb, MultiTaskReplayBufferConcat

_WORK_PATH = os.environ['BMOCA_HOME']


class CustomRLPolicy(CustomAgentBackbone):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                action_shape=[385],
                ):
        super().__init__()
        
        # redefine heads
        self.prediction_head = nn.Sequential(
            nn.Linear(self.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_shape[0]),
        )
    
    def forward(self,\
                goal_input_ids,\
                goal_attention_mask,\
                img_observation,\
                eval_mode=False,
        ):
        """
        args:
            screen_input: processed screen image with keys of (flattened_patches, attention_mask)
            query_input: tokenized query text with keys of (input_ids, attention_mask)
        """
        query_input = {
            'input_ids': goal_input_ids,
            'attention_mask': goal_attention_mask,
        }

        # textual encoding
        with torch.no_grad():
            self.text_enc.eval()
            text_features = self.text_enc.encode(**query_input)

        # visual encoding 
        img_features = self.img_enc(img_observation)
        if len(img_features.size()) == 2:
            img_features = img_features.unsqueeze(1) # [B, hidden_dim] -> [B, 1, hidden_dim]

        # attention
        feat_att, _ = self.mha_layer(text_features, img_features, img_features)

        feat_att = torch.mean(feat_att, axis=1) # [B, S, E] -> [B, E]
        
        logits = self.prediction_head(feat_att)

        action_probs = F.softmax(logits, dim=1)
        policy_dist = Categorical(probs=action_probs) # is directly passing logit here okay?
        action = policy_dist.sample()
        log_prob = F.log_softmax(logits, dim=1)
        
        return action, log_prob, action_probs


class CustomRLCriticV(CustomAgentBackbone):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base"):
        super().__init__(text_enc_name=text_enc_name)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(self.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
    
    def forward(self,\
                goal_input_ids,\
                goal_attention_mask,\
                img_observation,\
                eval_mode=False,
        ):
        """
        args:
            screen_input: processed screen image with keys of (flattened_patches, attention_mask)
            query_input: tokenized query text with keys of (input_ids, attention_mask)
        """
        query_input = {
            'input_ids': goal_input_ids,
            'attention_mask': goal_attention_mask,
        }

        # textual encoding
        with torch.no_grad():
            self.text_enc.eval()
            text_features = self.text_enc.encode(**query_input)

        # visual encoding 
        img_features = self.img_enc(img_observation)
        if len(img_features.size()) == 2:
            img_features = img_features.unsqueeze(1) # [B, hidden_dim] -> [B, 1, hidden_dim]

        # attention
        feat_att, _ = self.mha_layer(text_features, img_features, img_features)

        feat_att = torch.mean(feat_att, axis=1) # [B, S, E] -> [B, E]
        values = self.prediction_head(feat_att)
        
        # predict with heads
        return values
    
    
class PPO:
    def __init__(
        self,
        action_shape = [25],
        td_step = 1,
        # modules
        feature_dim = 768,
        actor_hidden_dim = 512,
        critic_hidden_dim = 512,
        #training
        device = "cuda",
        lr = 2e-4,
        actor_lr = 2e-4,
        lmbda=0.9,
        clip_e=0.2,
        entropy_alpha=None,
        critic_tau = 0.01,
        critic_gamma = 0.99,
        actor_warm_up_steps=-1,
        actor_use_layernorm=True,
        # save
        checkpoint_name=None,
        buffer_size=10000,
        update_steps=None,
        avail_tasks=None,
    ):
        
        self.device = device
        self.action_shape = action_shape
        self.td_step = td_step
        self.feature_dim = feature_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.lr = lr
        self.actor_lr = actor_lr
        self.critic_tau = critic_tau
        self.critic_gamma = critic_gamma
        self.actor_warm_up_steps = actor_warm_up_steps
        self.actor_use_layernorm = actor_use_layernorm
        self.entropy_alpha = entropy_alpha
        self.training = True
        self.lmbda = lmbda
        self.update_steps = update_steps
        self.avail_tasks = avail_tasks
        self.clip_e = clip_e
        
        # checkpoint
        self.checkpoint_name = checkpoint_name        
        
        # modules
        self.actor = CustomRLPolicy(action_shape=action_shape).to(device)        
        self.critic = CustomRLCriticV().to(device)
    
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = MultiTaskReplayBufferConcatProb(capacity=buffer_size, task_instructions=self.avail_tasks)

        self.global_step = 0
        self.training = True
        self.set_train()
        
    def set_train(self):
        self.actor.train()
        self.critic.train()
        
    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        
    def save(self, filename="./tmp.pt"):
        save_path = os.path.join(f"{_WORK_PATH}/results/{self.checkpoint_name}", filename)
        torch.save({
            "actor": self.actor.state_dict(), 
            "critic": self.critic.state_dict(),
            "actor_optimizer":self.actor_optimizer.state_dict(),
            "critic_optimizer":self.critic_optimizer.state_dict()}, save_path)
        return
    
    def load(self, filename="./tmp.pt", vanilla_load=False):
        self.actor.load(filename=filename, device=self.device)
        if vanilla_load:
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
        return
    
    def get_action_from_timesteps(self, timestep):
        self.set_eval()
        obs = np.stack([timestep.curr_obs['pixel']]) 
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.permute(0, 3, 1, 2)
        
        input_ids, attention_mask = utils.tokenize_instruction(timestep.instruction)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
            
        with torch.no_grad():
            action, _, prob = self.actor(img_observation=obs, goal_input_ids=input_ids, goal_attention_mask=attention_mask)
            if not self.training:
                action = torch.argmax(prob, dim=1)
            action = action.cpu().data.numpy()
        self.set_train()
        return action, prob
    
    def episode_to_buffer(self, episode):
        reward_weight = self.critic_gamma ** np.arange(self.td_step)
        
        for t in range(episode["length"]):
            state = episode["observation"][t]
            action = episode["action"][t]
            n_rewards_sum = np.sum(episode["reward"][t:t+self.td_step] * reward_weight)
            n_next_state = episode["next_observation"][t]
            done = episode["done"][t]
            prob = episode["prob"][t]
            task = episode["instruction"]
            
            self.replay_buffer.push(state, action, n_rewards_sum, n_next_state, done, prob, task)
            if done: break
            
        del episode
        return
    
    def run_update(self,
                   state_batch,
                   action_batch, 
                   reward_batch,
                   next_state_batch,
                   done_batch,
                   prob_batch,
                   input_ids_batch,
                   attention_mask_batch,
                   update_actor=True):
        self.set_train()
        
        for ustep in range(self.update_steps):
            with torch.no_grad():
                v_target = self.critic(img_observation=next_state_batch, goal_input_ids=input_ids_batch, 
                                              goal_attention_mask=attention_mask_batch)
                target_v_value = (reward_batch.unsqueeze(1) + (self.critic_gamma**self.td_step * (1 - done_batch.unsqueeze(1)) * v_target).detach()).float()

            
            current_v = self.critic(img_observation=state_batch, goal_input_ids=input_ids_batch, 
                                    goal_attention_mask=attention_mask_batch)
            critic_loss = F.mse_loss(current_v, target_v_value)
    
            td = target_v_value - current_v

            advantage, R = [], 0.0
            for i in reversed(range(len(td))):
                delta = td[i]
                term_ = (1 - done_batch[i])
                R = delta + term_ * self.critic_gamma * self.lmbda * R
                advantage.append(R)
            advantage.reverse()
            advantage = torch.tensor(advantage).to(self.device).float().unsqueeze(1)
            
            _, log_prob, probs = self.actor(img_observation=state_batch, goal_input_ids=input_ids_batch, 
                                                 goal_attention_mask=attention_mask_batch)
            
            
            entropy = -torch.sum(probs * log_prob, dim=1).mean()
            entropy_bonus = self.entropy_alpha * entropy
                                            
            log_prob_a = log_prob.gather(1, action_batch)
            old_prob = prob_batch.unsqueeze(1)
            ratio = torch.exp(log_prob_a - torch.log(old_prob).detach()) 
            
            clipped_ratio = torch.clamp(ratio, 1-self.clip_e, 1+self.clip_e)
            actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage) - entropy_bonus
            actor_loss = actor_loss.mean()
        
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
            
            if update_actor:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
    
    
    def update(self, batch_size):
        transitions = list(self.replay_buffer.memory)
        batch = Transition_Prob(*zip(*transitions))
        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(self.device)
        state_batch = state_batch.permute(0, 3, 1, 2)
        action_batch = torch.tensor(np.stack(batch.action), dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(np.stack(batch.n_rewards_sum), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.stack(batch.n_next_state), dtype=torch.float32).to(self.device)
        next_state_batch = next_state_batch.permute(0, 3, 1, 2)
        done_batch = torch.tensor(np.stack(batch.done), dtype=torch.float32).to(self.device)
        prob_batch = torch.tensor(np.stack(batch.prob), dtype=torch.float32).to(self.device)
        
        input_ids_batch, attention_mask_batch = utils.tokenize_instruction(batch.task)
        input_ids_batch = input_ids_batch.to(self.device)
        attention_mask_batch = attention_mask_batch.to(self.device)
        
        update_actor = True if self.global_step > self.actor_warm_up_steps else False

        self.run_update(state_batch, 
                        action_batch, 
                        reward_batch,
                        next_state_batch,
                        done_batch,
                        prob_batch,
                        input_ids_batch=input_ids_batch,
                        attention_mask_batch=attention_mask_batch,
                        update_actor = update_actor
                        )
            
        self.global_step += 1
        self.replay_buffer.memory.clear()
        return