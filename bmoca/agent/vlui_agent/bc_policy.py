import os
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler

from transformers import AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.image_transforms import ChannelDimension, to_channel_dimension_format

from lib.auto_ui.auto_ui.model import T5ForMultimodalGeneration

_WORK_PATH = os.environ['BMOCA_HOME']


class CNNEncoder(nn.Module):
    def __init__(self, obs_dim=[3, 256, 128], emb_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim

        self.trunk = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                            nn.ReLU(),
                    ).to(device)
        self.img_dim = self._get_img_dim()    

        self.adapter = nn.Linear(in_features=512, out_features=self.emb_dim).to(device)

    def _get_img_dim(self):
        x = torch.rand(1, *self.obs_dim).to(self.device)
        x = self.trunk(x)
        return x.view(*x.shape[:2], -1).shape[-1]

    def forward(self, x):
        if x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2).float()

        x = self.trunk(x)
        x = x.view(*x.shape[:2], -1)
        x = x.transpose(-1, -2)
        x = self.adapter(x)

        return x


class BCPolicy(nn.Module):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                ):
        super(BCPolicy, self).__init__()
        # encoders
        self.img_enc = CNNEncoder()        

        self.text_enc = T5ForMultimodalGeneration.from_pretrained(text_enc_name, img_dim=self.img_enc.img_dim, ignore_mismatched_sizes=True)
        self.text_enc.requires_grad_(False)

        self.text_config = self.text_enc.config

        # attention module
        self.mha_layer = nn.MultiheadAttention(
                            embed_dim=self.text_config.hidden_size, \
                            kdim=self.text_config.hidden_size, \
                            vdim=self.text_config.hidden_size, \
                            num_heads=1, batch_first=True)

        # heads
        self.gesture_position_head = nn.Sequential(
            nn.Linear(self.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
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

        # predict with heads
        gesture_output = self.gesture_position_head(feat_att)
        gesture_output = torch.tanh(gesture_output)

        return gesture_output

    def get_train_parameters(self):
        res = []
        res += self.text_enc.parameters()
        res += self.img_enc.parameters()
        res += self.mha_layer.parameters()
        res += self.gesture_position_head.parameters()
        return res

    def load(self, 
            filename=f'./tmp.pt',
            device='cuda'
            ):
        policy_load = torch.load(filename, map_location=device)

        self.text_enc.load_state_dict(policy_load['text_enc'])
        self.img_enc.load_state_dict(policy_load['img_enc'])
        self.mha_layer.load_state_dict(policy_load['mha_layer'])
        self.gesture_position_head.load_state_dict(policy_load['gesture_position_head'])

    def save(self, 
            filename=f'./tmp.pt'
            ):
        torch.save({
            'text_enc': self.text_enc.state_dict(),
            'img_enc': self.img_enc.state_dict(),
            'mha_layer': self.mha_layer.state_dict(),
            'gesture_position_head': self.gesture_position_head.state_dict(),
        }, filename)
        