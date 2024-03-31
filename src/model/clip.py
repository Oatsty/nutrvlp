from typing import Dict
import torch
import torch.nn as nn

import clip

class Clip(nn.Module):
    """The CLIP image and text encoder.

    Parameters
    ---
    device: 'cpu' or 'cuda'
    """
    def __init__(self, device: torch.device):
        super(Clip, self).__init__()
        self.device = device
        self.model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        self.model.float()

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        out = {}
        out['image_embedding'] = self.model.encode_image(image)
        out['recipe_embedding'] = self.model.encode_text(text)
        return out
