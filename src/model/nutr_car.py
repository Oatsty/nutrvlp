from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .car import Car

class NutrCarDirectIngredientOld(Car):
    def __init__(self, output_size: int, hidden_size: int, n_heads: int, n_layers: int, segment_model_name: str, llm_model_name: str, num_nutrs: int, num_ingrs: int, device: torch.device, score_threshold=0.95, area_threshold=0.005, **kwargs) -> None:
        super().__init__(output_size, hidden_size, n_heads, n_layers, segment_model_name, llm_model_name, device, score_threshold, area_threshold, **kwargs)
        self.ingr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_ingrs)
        )
        self.nutr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_nutrs)
        )

    def forward(self, img: torch.Tensor, pil_img, title: torch.Tensor, ingrs: torch.Tensor, instrs: torch.Tensor, description: torch.Tensor, title_text: Optional[list[str]] = None, ingrs_text: Optional[list[list[str]]] = None, instrs_text: Optional[list[list[str]]] = None):
        outs = {}
        outs['embs'] = self.forward_features(img,pil_img,title,ingrs,instrs,description)
        outs['nutr'] = self.nutr_decoder(outs['embs']['img'])
        outs['ingrs'] = self.ingr_decoder(outs['embs']['img'])
        return outs

class NutrCarDirectIngredient(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, n_heads: int, n_layers: int, segment_model_name: str, llm_model_name: str, num_nutrs: int, num_ingrs: int, device: torch.device, score_threshold=0.95, area_threshold=0.005, **kwargs) -> None:
        super().__init__()
        self.car = Car(output_size, hidden_size, n_heads, n_layers, segment_model_name, llm_model_name, device, score_threshold, area_threshold, **kwargs)
        self.ingr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_ingrs)
        )
        self.nutr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_nutrs)
        )

    def forward(self, img: torch.Tensor, pil_img, title: torch.Tensor, ingrs: torch.Tensor, instrs: torch.Tensor, description: torch.Tensor, title_text: Optional[list[str]] = None, ingrs_text: Optional[list[list[str]]] = None, instrs_text: Optional[list[list[str]]] = None):
        outs = {}
        outs['embs'] = self.car.forward_features(img,pil_img,title,ingrs,instrs,description)
        outs['nutr'] = self.nutr_decoder(outs['embs']['img'])
        outs['ingrs'] = self.ingr_decoder(outs['embs']['img'])
        return outs

def create_nutr_car_model(config, device):
    model_name = config.MODEL.SEGMENT.NAME
    llm_model_name = config.MODEL.DESCRIPTION.NAME
    output_size = config.MODEL.EMB_DIM
    hidden_size = config.MODEL.RECIPE.HIDDEN_DIM
    n_heads = config.MODEL.RECIPE.NUM_HEADS
    n_layers = config.MODEL.RECIPE.NUM_LAYERS
    num_nutrs = config.DATA.NUM_NUTRS
    num_ingrs = config.DATA.NUM_INGRS
    return NutrCarDirectIngredient(output_size, hidden_size, n_heads, n_layers, model_name, llm_model_name, num_nutrs, num_ingrs, device).to(device)

def create_nutr_car_model_old(config, device):
    model_name = config.MODEL.SEGMENT.NAME
    llm_model_name = config.MODEL.DESCRIPTION.NAME
    output_size = config.MODEL.EMB_DIM
    hidden_size = config.MODEL.RECIPE.HIDDEN_DIM
    n_heads = config.MODEL.RECIPE.NUM_HEADS
    n_layers = config.MODEL.RECIPE.NUM_LAYERS
    num_nutrs = config.DATA.NUM_NUTRS
    num_ingrs = config.DATA.NUM_INGRS
    return NutrCarDirectIngredientOld(output_size, hidden_size, n_heads, n_layers, model_name, llm_model_name, num_nutrs, num_ingrs, device).to(device)
