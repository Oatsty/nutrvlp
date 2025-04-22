import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import clip

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

class ClipTextEncoder(nn.Module):
    """The CLIP text encoder.

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'

    """
    def __init__(self, device: torch.device):
        super(ClipTextEncoder, self).__init__()
        self.device = device
        model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
        model.float()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.dtype = model.dtype
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        # self.text_projection = model.text_projection

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class ClipLoraImageEncoder(nn.Module):
    """The CLIP image encoder with LORA

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'
    """
    def __init__(self, device: torch.device, r: int = 16, alpha: int = 16):
        super(ClipLoraImageEncoder, self).__init__()
        self.r = r
        self.alpha = alpha
        self.device = device
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["k_proj", "q_proj", "v_proj", "c_fc", "c_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model = ClipImageEncoder(device)
        self.lora_model = get_peft_model(model, config)
        self.lora_model.to(device)

    def forward(self, image: torch.Tensor):
        out = self.lora_model(image)
        return out

class ClipLoraTextEncoder(nn.Module):
    """The CLIP text encoder with LORA

    Parameters
    ---
    emb_dim (int): Output embedding size.
    device: 'cpu' or 'cuda'
    """
    def __init__(self, device: torch.device,  r: int = 16, alpha: int = 16):
        super(ClipLoraTextEncoder, self).__init__()
        self.r = r
        self.alpha = alpha
        self.device = device
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["k_proj", "q_proj", "v_proj", "c_fc", "c_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model = ClipTextEncoder(device)
        self.lora_model = get_peft_model(model, config)
        self.lora_model.to(device)

    def forward(self, image: torch.Tensor):
        out = self.lora_model(image)
        return out
