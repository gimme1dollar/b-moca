import os
import torch
import torch.nn as nn

from transformers import EfficientNetModel
from lib.auto_ui.auto_ui.model import T5ForMultimodalGeneration

_WORK_PATH = os.environ['BMOCA_HOME']


class EfficientNetEncoder(nn.Module):
    def __init__(self, 
                 finetune_vis_encoder:bool=True, 
                 obs_dim=[3, 256, 128], 
                 emb_dim=768, version="b0"):
        super().__init__()
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        
        efficient_net_model_name = f"google/efficientnet-{version}"
        self.trunk_pretrained = EfficientNetModel.from_pretrained(efficient_net_model_name)
        # freeze the trunk and just use it as a feature extractor
        self.trunk_pretrained.requires_grad_(finetune_vis_encoder) 
        
        self.hidden_dim = self.trunk_pretrained.config.hidden_dim # 1536 for b3, 1280 for b0. 
        self.img_dim = self._get_img_dim()    

        # changed in_features from 512 to 1280
        self.adapter = nn.Linear(in_features=self.hidden_dim, out_features=self.emb_dim) 


    def _get_img_dim(self):
        self.obs_dim[0] = self.obs_dim[0]
        
        x = torch.rand(1, *self.obs_dim) # [1, 12, 256, 128]
        x = x.reshape(1, -1, *self.obs_dim[1:]).to(torch.float32) # [4, 3, 256, 128] 
        
        x = self.trunk_pretrained(x).last_hidden_state # [4, 1280, 8, 4]
        x = x.view(*x.shape[:2], -1).shape[-1] # [1280, 32] -> [32]
        return x


    def forward(self, x):
        if x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2).float()
        x = self.trunk_pretrained(x).last_hidden_state # [N, 3, 256, 128] -> [N, 1280, 8, 4]
        
        x = x.reshape(x.shape[0], 1, *x.shape[1:]) # [4 * N, 1280, 8, 4] -> [N, 4, 1280, 8, 4]
        x = x.mean(dim = 1) # [N, 4, 1280, 8, 4] -> [N, 1280, 8, 4]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [N, 1280, 8, 4] -> [N, 1280, 32]
        x = x.transpose(-1, -2) # [N, 32, 1280]
        
        x = self.adapter(x) # [N, 32, self.emb_dim] 
        
        return x


class CustomAgentBackbone(nn.Module):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                ):
        super(CustomAgentBackbone, self).__init__()
        # encoders
        self.img_enc = EfficientNetEncoder()        

        self.text_enc = T5ForMultimodalGeneration.from_pretrained(
                            text_enc_name, 
                            img_dim=self.img_enc.img_dim, 
                            ignore_mismatched_sizes=True
                        )        
        self.text_enc.requires_grad_(False)
        self.text_config = self.text_enc.config

        # attention module
        self.mha_layer = nn.MultiheadAttention(
                            embed_dim=self.text_config.hidden_size, \
                            kdim=self.text_config.hidden_size, \
                            vdim=self.text_config.hidden_size, \
                            num_heads=1, batch_first=True)

        # heads
        self.prediction_head = nn.Sequential(
            nn.Linear(self.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )
        
    def forward(self):
        return 

    def get_train_parameters(self):
        res = []
        res += self.text_enc.parameters()
        res += self.img_enc.parameters()
        res += self.mha_layer.parameters()
        res += self.prediction_head.parameters()
        return res

    def load(self, 
            filename=f'./tmp.pt',
            device='cuda'
            ):
        policy_load = torch.load(filename, map_location=device)

        self.text_enc.load_state_dict(policy_load['text_enc'])
        self.img_enc.load_state_dict(policy_load['img_enc'])
        self.mha_layer.load_state_dict(policy_load['mha_layer'])
        self.prediction_head.load_state_dict(policy_load['prediction_head'])

    def save(self, 
            filename=f'./tmp.pt'
            ):
        torch.save({
            'text_enc': self.text_enc.state_dict(),
            'img_enc': self.img_enc.state_dict(),
            'mha_layer': self.mha_layer.state_dict(),
            'prediction_head': self.prediction_head.state_dict(),
        }, filename)
        
        