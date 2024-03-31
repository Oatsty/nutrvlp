import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_nutr import DeepNutrEncoder, NutrEncoder

from .vlp import VLPCook

class NutrVLP(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, vision_width: int, num_nutrs: int, device: torch.device):
        super(NutrVLP, self).__init__()
        self.vlp_cook = VLPCook(vocab_size, dim_emb, hidden_size, num_heads, num_layers, vision_width, device)
        self.nutr_embedder = NutrEncoder(dim_emb,num_nutrs)

    def forward_features(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out_recipe = self.vlp_cook.recipe_embedder(recipe)
        recipe_embedding = out_recipe[0]
        out_image = self.vlp_cook.image_embedder(img)
        image_embedding = torch.cat((out_image[:, 0, :],torch.zeros(out_image.shape[0],self.vlp_cook.vision_width,device=img.device,dtype=torch.float)),dim=1)
        nutr_embedding = self.nutr_embedder(nutr)
        out = {}
        out['recipe_embedding'] = F.normalize(torch.tanh(self.vlp_cook.proj_recipe(recipe_embedding)))
        out['image_embedding'] = F.normalize(torch.tanh(self.vlp_cook.proj_image(image_embedding)))
        out['nutr_embedding'] = F.normalize(torch.tanh(nutr_embedding))
        return out
    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        return self.forward_features(recipe,img,nutr)

class NutrVLPDirect(NutrVLP):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, vision_width: int, num_nutrs: int, device: torch.device):
        super(NutrVLPDirect, self).__init__(vocab_size, dim_emb, hidden_size, num_heads, num_layers, vision_width, num_nutrs, device)
        self.nutr_decoder = nn.Sequential(
            nn.Linear(dim_emb,dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb,num_nutrs)
        )

    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out = self.forward_features(recipe,img,nutr)
        out['nutr'] = self.nutr_decoder(out['image_embedding'])
        return out

class NutrVLPDirectIngredient(NutrVLP):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, vision_width: int, num_nutrs: int, num_ingrs: int, device: torch.device):
        super(NutrVLPDirectIngredient, self).__init__(vocab_size, dim_emb, hidden_size, num_heads, num_layers, vision_width, num_nutrs, device)
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

class DeepNutrVLP(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, nutr_hidden_size: int, num_heads: int, num_layers: int, nutr_num_layers: int, vision_width: int, num_nutrs: int, device: torch.device):
        super(DeepNutrVLP, self).__init__()
        self.vlp_cook = VLPCook(vocab_size, dim_emb, hidden_size, num_heads, num_layers, vision_width, device)
        self.nutr_embedder = DeepNutrEncoder(num_nutrs, nutr_hidden_size, dim_emb, nutr_num_layers)

    def forward_features(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out_recipe = self.vlp_cook.recipe_embedder(recipe)
        recipe_embedding = out_recipe[0]
        out_image = self.vlp_cook.image_embedder(img)
        image_embedding = torch.cat((out_image[:, 0, :],torch.zeros(out_image.shape[0],self.vlp_cook.vision_width,device=img.device,dtype=torch.float)),dim=1)
        nutr_embedding = self.nutr_embedder(nutr)
        out = {}
        out['recipe_embedding'] = F.normalize(torch.tanh(self.vlp_cook.proj_recipe(recipe_embedding)))
        out['image_embedding'] = F.normalize(torch.tanh(self.vlp_cook.proj_image(image_embedding)))
        out['nutr_embedding'] = F.normalize(torch.tanh(nutr_embedding))
        return out
    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        return self.forward_features(recipe,img,nutr)

class DeepNutrVLPDirect(DeepNutrVLP):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, nutr_hidden_size: int, num_heads: int, num_layers: int, nutr_num_layers: int, vision_width: int, num_nutrs: int, device: torch.device):
        super(DeepNutrVLPDirect, self).__init__(vocab_size, dim_emb, hidden_size, nutr_hidden_size, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, device)
        self.nutr_decoder = nn.Sequential(
            nn.Linear(dim_emb,dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb,num_nutrs)
        )

    def forward(self, recipe: torch.Tensor, img: torch.Tensor, nutr: torch.Tensor):
        out = self.forward_features(recipe,img,nutr)
        out['nutr'] = self.nutr_decoder(out['image_embedding'])
        return out

class DeepNutrVLPDirectIngredient(DeepNutrVLP):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, nutr_hidden_size: int, num_heads: int, num_layers: int, nutr_num_layers: int, vision_width: int, num_nutrs: int, num_ingrs: int, device: torch.device):
        super(DeepNutrVLPDirectIngredient, self).__init__(vocab_size, dim_emb, hidden_size, nutr_hidden_size, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, device)
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

def create_nutr_vlp_model(config, device):
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
    if model_name == 'nutr_vlp':
        return NutrVLP(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,num_nutrs,device).to(device)
    elif model_name == 'nutr_vlp_direct':
        return NutrVLPDirect(vocab_size, emb_dim, hidden_dim, num_heads, num_layers, vision_width, num_nutrs, device).to(device)
    elif model_name == 'nutr_vlp_direct_ingrs':
        return NutrVLPDirectIngredient(vocab_size, emb_dim, hidden_dim, num_heads, num_layers, vision_width, num_nutrs, num_ingrs,device).to(device)
    elif model_name == 'deep_nutr_vlp':
        return DeepNutrVLP(vocab_size, emb_dim, hidden_dim, nutr_hidden_dim, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, device).to(device)
    elif model_name == 'deep_nutr_vlp_direct':
        return DeepNutrVLPDirect(vocab_size, emb_dim, hidden_dim, nutr_hidden_dim, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, device).to(device)
    elif model_name == 'deep_nutr_vlp_direct_ingrs':
        return DeepNutrVLPDirectIngredient(vocab_size, emb_dim, hidden_dim, nutr_hidden_dim, num_heads, num_layers, nutr_num_layers, vision_width, num_nutrs, num_ingrs,device).to(device)
    else:
        return NutrVLP(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,num_nutrs,device).to(device)
