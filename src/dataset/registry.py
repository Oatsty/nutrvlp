from .base import make_data_loader
import torchvision.transforms as transforms

def create_data(config, split):
    data_dir = config.DATA.DIR
    path_tokenized_recipes = config.DATA.PATH_TOKENIZED_RECIPES
    path_nutrs = config.DATA.PATH_NUTRS
    path_ingrs = config.DATA.PATH_INGRS
    num_ingrs = config.DATA.NUM_INGRS
    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.TRAIN.NUM_WORKERS
    max_instrs_len = config.DATA.MAX_INSTRS_LEN
    max_ingrs_len = config.DATA.MAX_INGRS_LEN
    max_instrs = config.DATA.MAX_INSTRS
    max_ingrs = config.DATA.MAX_INGRS
    nutr_names = config.DATA.NUTRS
    nutr_scale_list = config.DATA.NUTR_STDS
    nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
    return make_data_loader(
        data_dir,
        split,
        path_tokenized_recipes,
        path_nutrs,
        path_ingrs,
        num_ingrs,
        nutr_names,
        nutr_scale,
        batch_size,
        num_workers,
        shuffle=True,
        # shuffle=True if split == 'train' else False,
        max_instrs_len=max_instrs_len,
        max_ingrs_len=max_ingrs_len,
        max_instrs=max_instrs,
        max_ingrs=max_ingrs,
    )

def create_recipe_retrieval_data(config, split):
    data_dir = config.DATA.DIR
    path_tokenized_recipes = config.DATA.PATH_TOKENIZED_RECIPES
    path_nutrs = config.DATA.PATH_NUTRS
    path_ingrs = config.DATA.PATH_INGRS
    num_ingrs = config.DATA.NUM_INGRS
    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.TRAIN.NUM_WORKERS
    max_instrs_len = config.DATA.MAX_INSTRS_LEN
    max_ingrs_len = config.DATA.MAX_INGRS_LEN
    max_instrs = config.DATA.MAX_INSTRS
    max_ingrs = config.DATA.MAX_INGRS
    nutr_names = config.DATA.NUTRS
    nutr_scale_list = config.DATA.NUTR_STDS
    nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
    scale_size = 256
    crop_size = 224
    transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.RandomCrop(crop_size),
        # transforms.CenterCrop(crop_size),
        # transforms.ToTensor(),  # divide by 255 automatically
        # transforms.Normalize(mean=mean, std=std)
    ])
    # pil_transform = transforms.Compose([
    #     transforms.Resize(scale_size),
    #     transforms.RandomCrop(crop_size),
    # ])
    path_layer1 = config.DATA.PATH_LAYER1
    path_description = config.DATA.PATH_DESCRIPTION
    path_mask_embed = config.DATA.PATH_MASK_EMBED
    return make_data_loader(
        data_dir,
        split,
        path_tokenized_recipes,
        path_nutrs,
        path_ingrs,
        num_ingrs,
        nutr_names,
        nutr_scale,
        batch_size,
        num_workers,
        shuffle=True if split == 'train' else False,
        max_instrs_len=max_instrs_len,
        max_ingrs_len=max_ingrs_len,
        max_instrs=max_instrs,
        max_ingrs=max_ingrs,
        transform=transform,
        # pil_transform=pil_transform,
        path_layer1 = path_layer1,
        path_description = path_description,
        path_mask_embed = path_mask_embed,
        # do_normalize = False,
    )
