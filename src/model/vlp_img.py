import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class VLPImageEncoder(nn.Module):
    """The image encoder.

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'

    """
    def __init__(self, emb_dim: int, device: torch.device):
        super(VLPImageEncoder, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        # modules
        model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        model.float()
        self.encoder = model.visual
        self.dim_out = self.encoder.proj.size()[0]
        self.dtype = self.encoder.conv1.weight.dtype
        self.fc = nn.Linear(self.dim_out, self.emb_dim)

    def forward(self, image: torch.Tensor):
        x = self.encoder.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.encoder.ln_post(x)
        return x
