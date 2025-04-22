import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlp_recipe import HTransformerRecipeEmbedding
from .tfood_img import TFoodImageEncoder, TFoodImageEncoderClip

class TFood(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, vision_width: int, device: torch.device):
        super(TFood, self).__init__()
        self.dim_emb = dim_emb
        self.vision_width = vision_width
        self.recipe_embedder = HTransformerRecipeEmbedding(vocab_size,dim_emb,hidden_size,num_heads,num_layers,device)
        self.image_embedder = TFoodImageEncoder(dim_emb, device)
        self.proj_recipe = nn.Linear(self.recipe_embedder.dim_recipe, self.dim_emb)
        self.proj_image = nn.Linear(self.image_embedder.dim_out, self.dim_emb)

    def forward(self, recipe: torch.Tensor, img: torch.Tensor):
        out_recipe = self.recipe_embedder(recipe)
        recipe_embedding = out_recipe[0]
        out_image = self.image_embedder(img)
        image_embedding = out_image[:, 0, :]
        out = {}
        out['recipe_embedding'] = F.normalize(torch.tanh(self.proj_recipe(recipe_embedding)))
        out['image_embedding'] = F.normalize(torch.tanh(self.proj_image(image_embedding)))
        return out

class TFoodClip(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, vision_width: int, device: torch.device):
        super(TFoodClip, self).__init__()
        self.dim_emb = dim_emb
        self.vision_width = vision_width
        self.recipe_embedder = HTransformerRecipeEmbedding(vocab_size,dim_emb,hidden_size,num_heads,num_layers,device)
        self.image_embedder = TFoodImageEncoderClip(dim_emb, device)
        self.proj_recipe = nn.Linear(self.recipe_embedder.dim_recipe, self.dim_emb)
        self.proj_image = nn.Linear(self.image_embedder.dim_out, self.dim_emb)

    def forward(self, recipe: torch.Tensor, img: torch.Tensor):
        out_recipe = self.recipe_embedder(recipe)
        recipe_embedding = out_recipe[0]
        out_image = self.image_embedder(img)
        image_embedding = out_image[:,0,:]
        out = {}
        out['recipe_embedding'] = F.normalize(torch.tanh(self.proj_recipe(recipe_embedding)))
        out['image_embedding'] = F.normalize(torch.tanh(self.proj_image(image_embedding)))
        return out

def create_tfood_model(config, device):
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    emb_dim = config.MODEL.EMB_DIM
    hidden_dim = config.MODEL.RECIPE.HIDDEN_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    vision_width = config.MODEL.IMAGE.VISION_WIDTH
    return TFood(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,device).to(device)

def create_tfood_clip_model(config, device):
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    emb_dim = config.MODEL.EMB_DIM
    hidden_dim = config.MODEL.RECIPE.HIDDEN_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    vision_width = config.MODEL.IMAGE.VISION_WIDTH
    return TFoodClip(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,device).to(device)
