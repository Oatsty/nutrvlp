import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import clip

class ViT_timm_custom(nn.Module):
    def __init__(self):
        super(ViT_timm_custom, self).__init__()

        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.dim_out = self.encoder.head.in_features

    def forward_features(self, x):
        """https://github.com/rwightman/pytorch-image-models/issues/657"""
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

class TFoodImageEncoder(nn.Module):

    def __init__(self, emb_dim: int, device: torch.device):
        super(TFoodImageEncoder, self).__init__()
        self.dim_emb = emb_dim
        self.device = device
        self.convnet = ViT_timm_custom()
        self.dim_out = self.convnet.dim_out
        self.dtype = self.convnet.encoder.patch_embed.proj.weight.dtype
        self.fc = nn.Linear(self.dim_out, self.dim_emb)

    def forward(self, image):
        x = self.convnet(image.type(self.dtype))
        return x

class TFoodImageEncoderClip(nn.Module):

    def __init__(self, emb_dim: int, device: torch.device):
        super(TFoodImageEncoderClip, self).__init__()
        self.dim_emb = emb_dim
        self.device = device
        model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        model.float()
        self.convnet = model.visual
        self.dim_out = self.convnet.proj.size()[0]
        self.dtype = self.convnet.conv1.weight.dtype
        self.fc = nn.Linear(self.dim_out, self.dim_emb)

    def forward(self, image):
        x = self.convnet.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.convnet.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.convnet.positional_embedding.to(x.dtype)
        x = self.convnet.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.convnet.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.convnet.ln_post(x)
        return x
