import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_nutr import DeepNutrEncoder
from .tfood import TFood

class DeepNutrTFood(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, nutr_hidden_size: int, num_heads: int, num_layers: int, nutr_num_layers: int, vision_width: int, num_nutrs: int, device: torch.device):
        super(DeepNutrTFood, self).__init__()
        self.tfood = TFood(vocab_size, dim_emb, hidden_size, num_heads, num_layers, vision_width, device)
        self.nutr_embedder = DeepNutrEncoder(num_nutrs, nutr_hidden_size, dim_emb, nutr_num_layers)

    def forward_features(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out_recipe = self.tfood.recipe_embedder(recipe)
        recipe_embedding = out_recipe[0]
        out_image = self.tfood.image_embedder(img)
        image_embedding = out_image[:, 0, :]
        nutr_embedding = self.nutr_embedder(nutr)
        out = {}
        out['recipe_embedding'] = F.normalize(torch.tanh(self.tfood.proj_recipe(recipe_embedding)))
        out['image_embedding'] = F.normalize(torch.tanh(self.tfood.proj_image(image_embedding)))
        out['nutr_embedding'] = F.normalize(torch.tanh(nutr_embedding))
        return out
    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        return self.forward_features(recipe,img,nutr)

class DeepTFoodDirectIngredient(DeepNutrTFood):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, nutr_hidden_size: int, num_heads: int, num_layers: int, nutr_num_layers: int, vision_width: int, num_nutrs: int, num_ingrs: int, device: torch.device):
        super(DeepTFoodDirectIngredient, self).__init__(vocab_size, dim_emb, hidden_size, nutr_hidden_size, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, device)
        self.ingr_decoder = nn.Sequential(
            nn.Linear(dim_emb,dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb,num_ingrs)
        )
        self.nutr_decoder = nn.Sequential(
            nn.Linear(dim_emb,dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb,num_nutrs)
        )

    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out = self.forward_features(recipe,img,nutr)
        out['nutr'] = self.nutr_decoder(out['image_embedding'])
        out['ingrs'] = self.ingr_decoder(out['image_embedding'])
        return out

def create_nutr_tfood_model(config, device):
    model_name = config.MODEL.NAME
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    emb_dim = config.MODEL.EMB_DIM
    hidden_dim = config.MODEL.RECIPE.HIDDEN_DIM
    nutr_hidden_dim = config.MODEL.NUTR.HIDDEN_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    nutr_num_layers = config.MODEL.NUTR.NUM_LAYERS
    vision_width = config.MODEL.IMAGE.VISION_WIDTH
    num_nutrs = config.DATA.NUM_NUTRS
    num_ingrs = config.DATA.NUM_INGRS
    if model_name == 'deep_nutr_tfood_direct_ingrs':
        return DeepTFoodDirectIngredient(vocab_size, emb_dim, hidden_dim, nutr_hidden_dim, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, num_ingrs,device).to(device)
    else:
        return DeepTFoodDirectIngredient(vocab_size, emb_dim, hidden_dim, nutr_hidden_dim, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, num_ingrs,device).to(device)
