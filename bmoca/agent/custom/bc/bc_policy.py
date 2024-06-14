import os
import torch

import torch
from torch import nn

from bmoca.agent.custom.networks import CustomAgentBackbone

_WORK_PATH = os.environ['BMOCA_HOME']

class BCContPolicy(CustomAgentBackbone):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                ):
        super(BCContPolicy, self).__init__(text_enc_name=text_enc_name)
        
        # heads
        self.prediction_head = nn.Sequential(
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
        gesture_output = self.prediction_head(feat_att)
        gesture_output = torch.tanh(gesture_output)

        return gesture_output


class BCDiscPolicy(CustomAgentBackbone):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                action_shape=[385]
                ):
        super().__init__(text_enc_name=text_enc_name)
        
        # heads
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
        actions = self.prediction_head(feat_att)
        
        # predict with heads
        return actions