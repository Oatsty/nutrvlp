import argparse
import logging
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = 'food.com'
_C.DATA.DIR = '/srv/datasets2/recipe1m+/food.com_data_rescaled'
_C.DATA.PATH_TOKENIZED_RECIPES = '/srv/datasets2/recipe1m+/food.com_data_rescaled/text/tokenized_recipe.json'
_C.DATA.PATH_NUTRS = '/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/food.com_nutr_g_per_recipe.json'
_C.DATA.PATH_INGRS = '/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/simplified_food_ids_per_recipes.json'
# _C.DATA.SPLIT = 'train'
_C.DATA.MAX_INSTRS_LEN = 20
_C.DATA.MAX_INGRS_LEN = 15
_C.DATA.MAX_INSTRS = 20
_C.DATA.MAX_INGRS = 20
_C.DATA.NUM_NUTRS = 4
_C.DATA.NUM_INGRS = 533
_C.DATA.NUTRS = ['energy','fat','carb','protein']
_C.DATA.NUTR_STDS = [123.50, 10.20,18.95,4.87]
# _C.DATA.NO_LABEL = False
# _C.DATA.TYPE = 'lmdb'

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'nutr_vlp'
_C.MODEL.EMB_DIM = 768
_C.MODEL.RECIPE = CN()
_C.MODEL.RECIPE.NAME = 'ht'
_C.MODEL.RECIPE.PRETRAINED = 'google-bert/bert-base-uncased'
_C.MODEL.RECIPE.VOCAB_SIZE = 200000
_C.MODEL.RECIPE.NUM_HEADS = 4
_C.MODEL.RECIPE.NUM_LAYERS = 2
_C.MODEL.RECIPE.HIDDEN_DIM = 512
_C.MODEL.IMAGE = CN()
_C.MODEL.IMAGE.VISION_WIDTH = 768
_C.MODEL.IMAGE.IMAGE_MODEL = 'vit_base_patch16_224'
_C.MODEL.NUTR = CN()
_C.MODEL.NUTR.HIDDEN_DIM = 2048
_C.MODEL.NUTR.NUM_LAYERS = 6

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.SEED = 12345
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.LOSS = 'triplet'
_C.TRAIN.LR = 1e-5
_C.TRAIN.SCALE_LR = 1.0
_C.TRAIN.WEIGHT_DECAY = 1e-5
_C.TRAIN.FINETUNE = False
_C.TRAIN.FINETUNE_MODEL = 'all'
_C.TRAIN.RECIPE_LOSS_WEIGHT = 1.0
_C.TRAIN.MAE_WEIGHT = 0.01
_C.TRAIN.INGRS_WEIGHT = 0.025
_C.TRAIN.WARMUP_EPOCHS = 10

# -----------------------------------------------------------------------------
# eval
# -----------------------------------------------------------------------------
_C.EVAL = CN()

# -----------------------------------------------------------------------------
# infer
# -----------------------------------------------------------------------------
_C.INFER = CN()
# _C.INFER.INGR_INDEX_PATH = '/srv/datasets2/recipe1m+/ingr_index.json'
# _C.INFER.OUTPUT_PATH = '/srv/datasets2/recipe1m+/recipes_with_ingr_no_nutr.json'


# -----------------------------------------------------------------------------
# misc
# -----------------------------------------------------------------------------
_C.SAVE_PATH = "models/temp_w_nutr/"
_C.IMG_MODEL_PATH = ""
_C.RECIPE_MODEL_PATH = ""
_C.OUT_DIR = "out"
_C.RETRIEVAL_DIR = "/tmp/food.com_net"
_C.TITLE = ["test"]
_C.VLP_PATH = "vlpcook_checkpoints/ckpt_best_eval_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar"
_C.HT_PATH = "ht_checkpoints/model-best.ckpt"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    if _check_args("cfg"):
        _update_config_from_file(config, args.cfg)

    config.defrost()

    # merge from specific arguments
    if _check_args('dir_data'):
        # config.DATA.DIR_DATA = args.dir_data
        pass
    if _check_args('img_model_path'):
        config.IMG_MODEL_PATH = args.img_model_path
    if _check_args('recipe_model_path'):
        config.RECIPE_MODEL_PATH = args.recipe_model_path

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger()
    format_str = r"[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, datefmt=r"%Y/%m/%d %H:%M:%S", format=format_str
    )
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir_path / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)
    return logger


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--img-model-path',type=str)
    parser.add_argument('--recipe-model-path',type=str)
    # parser.add_argument('--dir-data', type=str)
    return parser


def get_arguments() -> Tuple[argparse.Namespace, CN]:
    parser = get_parser()
    args = parser.parse_args()
    config = get_config(args)
    return args, config


if __name__ == "__main__":
    _, config = get_arguments()
    # logger = init_logger(".", "log.txt")
    set_random_seed(config.TRAIN.SEED)
    dump_path = os.path.join("config", os.path.splitext(config.SAVE_PATH)[0] + ".yaml")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w") as f:
        f.write(config.dump())  # type: ignore
    # logger.info(f"Config Path: {dump_path}")
