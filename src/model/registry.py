from model.vlp import VLPCook
from .bert_recipe import BertOnlyRecipeTransformer, BertRecipeTransformer
from .base_recipe import RecipeOnlyTransformer, RecipeTransformer
from .base_img import ImageEncoder
from .base_nutr import NutrEncoder
from .nutr_vlp import NutrVLP, NutrVLPDirect

def create_models(config, device):
    recipe_model_name = config.MODEL.RECIPE.NAME
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    emb_dim = config.MODEL.EMB_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    num_nutrs = config.DATA.NUM_NUTRS
    pretrained = config.MODEL.RECIPE.PRETRAINED
    if recipe_model_name == 'ht':
        recipe_model = RecipeTransformer(vocab_size, emb_dim, num_heads, num_layers, num_nutrs)
    elif recipe_model_name == 'BERT':
        recipe_model = BertRecipeTransformer(emb_dim, num_heads, num_layers, num_nutrs, pretrained)
    elif recipe_model_name == 'BERTOnly':
        recipe_model = BertOnlyRecipeTransformer(emb_dim, num_nutrs, pretrained)
    elif recipe_model_name == 'RecipeOnly':
        recipe_model = RecipeOnlyTransformer(vocab_size, emb_dim, num_heads, num_layers)
    else:
        ValueError(f'unimplement recipe model: {recipe_model_name}')
    img_model = ImageEncoder(emb_dim,device)
    nutr_model = NutrEncoder(emb_dim, num_nutrs)
    return recipe_model.to(device), img_model.to(device), nutr_model.to(device)

def create_whole_model(config,device):
    model_name = config.MODEL.NAME
    vocab_size = config.MODEL.RECIPE.VOCAB_SIZE
    emb_dim = config.MODEL.EMB_DIM
    hidden_dim = config.MODEL.RECIPE.HIDDEN_DIM
    num_heads = config.MODEL.RECIPE.NUM_HEADS
    num_layers = config.MODEL.RECIPE.NUM_LAYERS
    vision_width = config.MODEL.IMAGE.VISION_WIDTH
    num_nutrs = config.DATA.NUM_NUTRS
    if model_name == 'nutr_vlp':
        return NutrVLP(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,num_nutrs,device).to(device)
    elif model_name == 'nutr_vlp_direct':
        return NutrVLPDirect(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,num_nutrs,device).to(device)
    elif model_name == 'vlp':
        return VLPCook(vocab_size,emb_dim,hidden_dim,num_heads,num_layers,vision_width,device).to(device)
