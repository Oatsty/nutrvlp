import torch
import torch.nn as nn

class NutrEncoder(nn.Module):
    """The simple nutr encoder.

    Parameters
    ---
    in_dim (int): Output embedding size.
    num_nutrs (int): Number of input nutrition types (4 for cal, fat, carb, protein).
    """
    def __init__(self, in_dim: int, num_nutrs: int):
        super(NutrEncoder, self).__init__()

        # nutrition encoder
        self.nutr_encoder = nn.Sequential(
            nn.Linear(num_nutrs,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

    def forward(self, nutr):
        out = self.nutr_encoder(nutr)
        return out

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout=0.1, layer_norm_eps=1e-5):
        super(Block, self).__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, dropout)
        self.norm = nn.LayerNorm(out_dim,eps=layer_norm_eps)
        self.proj = nn.Identity()
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim,out_dim)

    def forward(self, x):
        x = self.norm(self.proj(x) + self.mlp(x))
        return x

class DeepNutrEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout=0.1, layer_norm_eps=1e-5):
        super(DeepNutrEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(Block(in_dim, hidden_dim, out_dim, dropout=dropout, layer_norm_eps=layer_norm_eps))
        self.blocks.extend([
            Block(out_dim, hidden_dim, out_dim, dropout=dropout, layer_norm_eps=layer_norm_eps) for _ in range(num_layers - 1)
        ])
        self.proj_nutr = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.proj_nutr(x)
        return x
