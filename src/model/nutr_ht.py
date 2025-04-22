from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_nutr import DeepNutrEncoder

from .ht import HT, get_image_model

class NutrHT(HT):
    def __init__(self, output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers):
        super(NutrHT, self).__init__(output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers)
        self.nutr_embedder = DeepNutrEncoder(num_nutrs, nutr_hidden_size, output_size, nutr_num_layers)

    def forward_features(self, recipe: Dict[str,torch.Tensor], img: torch.Tensor, nutr: torch.Tensor):
        out = {}
        out['image_embedding'], out['recipe_embedding'], _ = self.joint_embedding(img, recipe['title'], recipe['ingrs'], recipe['instrs'])
        out['nutr_embedding'] = nn.Tanh()(self.nutr_embedder(nutr))
        return out

    def forward(self, recipe: Dict[str,torch.Tensor], img: torch.Tensor, nutr: torch.Tensor):
        return self.forward_features(recipe,img,nutr)

class NutrHTDirect(NutrHT):
    def __init__(self, output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers):
        super(NutrHTDirect,self).__init__(output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers)
        self.nutr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_nutrs)
        )

    def forward(self, recipe: Dict[str,torch.Tensor], img: torch.Tensor, nutr: torch.Tensor):
        out = self.forward_features(recipe,img,nutr)
        out['nutr'] = self.nutr_decoder(out['image_embedding'])
        return out

class NutrHTDirectIngredient(NutrHT):
    def __init__(self, output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers, num_ingrs):
        super(NutrHTDirectIngredient,self).__init__(output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers)
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

    def forward(self, recipe: Dict[str,torch.Tensor], img: torch.Tensor, nutr: torch.Tensor):
        out = self.forward_features(recipe,img,nutr)
        out['nutr'] = self.nutr_decoder(out['image_embedding'])
        out['ingrs'] = self.ingr_decoder(out['image_embedding'])
        return out

class NutrOnlyHT(NutrHT):
    def __init__(self, output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers):
        super(NutrOnlyHT,self).__init__(output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers)
        self.nutr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_nutrs)
        )

    def forward(self, img: torch.Tensor):
        out_img = self.joint_embedding.image_encoder(img,freeze_backbone=False)
        out = self.nutr_decoder(out_img)
        return out

class NutrIngrOnlyHT(NutrHT):
    def __init__(self, output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers, num_ingrs):
        super(NutrIngrOnlyHT,self).__init__(output_size, image_model, vocab_size, hidden_recipe, n_heads, n_layers, num_nutrs, nutr_hidden_size, nutr_num_layers)
        self.nutr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_nutrs)
        )
        self.ingr_decoder = nn.Sequential(
            nn.Linear(output_size,output_size),
            nn.ReLU(),
            nn.Linear(output_size,num_ingrs)
        )

    def forward(self, img: torch.Tensor):
        out_img = self.joint_embedding.image_encoder(img,freeze_backbone=False)
        out = {}
        out['nutr'] = self.nutr_decoder(out_img)
        out['ingrs'] = self.ingr_decoder(out_img)
        return out

def create_nutr_ht_model(config,device):
    model_name = config.MODEL.NAME
    output_size = config.MODEL.EMB_DIM
    image_model = config.MODEL.IMAGE.IMAGE_MODEL
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    hidden_recipe = config.MODEL.RECIPE.HIDDEN_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    num_nutrs = config.DATA.NUM_NUTRS
    hidden_nutr = config.MODEL.NUTR.HIDDEN_DIM
    nutr_num_layers = config.MODEL.NUTR.NUM_LAYERS
    num_ingrs = config.DATA.NUM_INGRS
    if model_name == 'nutr_ht':
        return NutrHT(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers).to(device)
    elif model_name == 'nutr_ht_direct':
        return NutrHTDirect(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers).to(device)
    elif model_name == 'nutr_ht_direct_ingrs':
        return NutrHTDirectIngredient(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers,num_ingrs).to(device)
    elif model_name == 'nutr_only_ht':
        return NutrOnlyHT(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers).to(device)
    elif model_name == 'nutr_ingr_only_ht':
        return NutrIngrOnlyHT(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers, num_ingrs).to(device)
    else:
        return NutrHT(output_size,image_model,vocab_size,hidden_recipe,num_heads,num_layers,num_nutrs,hidden_nutr,nutr_num_layers).to(device)
