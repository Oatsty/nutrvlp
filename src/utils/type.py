from typing import TypedDict

import torch


class RecipeTokens(TypedDict):
    title: torch.Tensor
    ingrs: torch.Tensor
    instrs: torch.Tensor
