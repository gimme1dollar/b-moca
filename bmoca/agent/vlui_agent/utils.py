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


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def convert_torch_image_to_pillow(tensor_img):
    if len(tensor_img.shape) == 4:
        assert tensor_img.shape[0] == 1
        tensor_img = tensor_img.squeeze(0)
    
    tensor_img = tensor_img.permute(1, 2, 0)

    np_img = tensor_img.detach().cpu().numpy()
    
    return Image.fromarray(np_img, mode="RGB")


def convert_np_image_to_pillow(np_img):
    return Image.fromarray(np_img, mode="RGB")


def save_pillow_images_to_gif(images, filename="tmp"):
    try:
        images[0].save(f"{filename}.gif", 
                       save_all=True, 
                       append_images=images[1:], 
                       duration=500, 
                       loop=0)
        return True
    except:
        traceback.print_exc()
        return False


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EfficientNetEncoder(nn.Module):
    def __init__(self, finetune_vis_encoder:bool=False, obs_dim=[3, 256, 128], emb_dim=768, version = "b0"):
        super().__init__()
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        efficient_net_model_name = f"google/efficientnet-{version}"
        self.trunk_pretrained = EfficientNetModel.from_pretrained(efficient_net_model_name)
        self.trunk_pretrained.requires_grad_(finetune_vis_encoder) # freeze the trunk and just use it as a feature extractor
        self.hidden_dim = self.trunk_pretrained.config.hidden_dim # 1536 for b3, 1280 for b0. 
        self.img_dim = self._get_img_dim()    

        self.adapter = nn.Linear(in_features=self.hidden_dim, out_features=self.emb_dim) # changed in_features from 512 to 1280

    def _get_img_dim(self):
        self.obs_dim[0] = self.obs_dim[0]
        x = torch.rand(1, *self.obs_dim) # [1, 12, 256, 128]
        x = x.reshape(1, -1, *self.obs_dim[1:]).to(torch.float32) # [1, 12, 256, 128] -> [4, 3, 256, 128] Efficient Net needs float32
        x = self.trunk_pretrained(x).last_hidden_state # [4, 1280, 8, 4]
        return x.view(*x.shape[:2], -1).shape[-1] # [1280, 32] -> gets 32

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
