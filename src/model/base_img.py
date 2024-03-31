import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class ImageEncoder(nn.Module):
    """The image encoder.

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'

    """
    def __init__(self, emb_dim: int, device: torch.device):
        super(ImageEncoder, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        # modules
        model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        model.float()
        self.encoder = model.visual
        self.dim_out = self.encoder.output_dim
        self.dtype = self.encoder.conv1.weight.dtype
        self.fc = nn.Linear(self.dim_out, self.emb_dim)

    def forward(self, image: torch.Tensor):
        x = self.encoder(image.type(self.dtype))
        x = self.fc(x)
        return x


class ClipImageEncoder(nn.Module):
    """The CLIP image encoder.

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'

    """
    def __init__(self, device: torch.device):
        super(ClipImageEncoder, self).__init__()
        self.device = device
        model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        model.float()
        self.encoder = model.visual
        self.dtype = self.encoder.conv1.weight.dtype

    def forward(self, image: torch.Tensor):
        x = self.encoder(image.type(self.dtype))
        return x
