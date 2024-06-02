import copy
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import sys

from tqdm import tqdm

sys.path.append('/home/parinayok/food.com_net')

from dataset import create_data
import init_config
from model.clip import Clip
from utils.evaluator import Evaluator

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, model: nn.Module, device):
    model.eval()
    logger.info(f'running test phase')
    retrieval_dir = "/tmp/food.com_net"
    out_dir = "out/clip/img/"
    evaluator_img = Evaluator(out_dir,retrieval_dir)
    out_dir = "out/clip/text/"
    evaluator_text = Evaluator(out_dir,retrieval_dir)
    out_dir = "out/clip/text_img/"
    evaluator_text_img = Evaluator(out_dir,retrieval_dir)
    for batch in tqdm(dataloader):
        text_input = batch['recipe']['texts'].to(device)
        img_input = batch['image']['data'].to(device)
        out = model(img_input,text_input)
        out_img = out['image_embedding']
        out_recipe = out['recipe_embedding']
        out_img_recipe = (out_img + out_recipe) / 2
        evaluator_img(out_img, out_img, batch)
        evaluator_text(out_img, out_recipe, batch)
        evaluator_text_img(out_img, out_img_recipe, batch)
    evaluator_img.calculate_similarity()
    evaluator_text.calculate_similarity()
    evaluator_text_img.calculate_similarity()

def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    config.defrost()
    config.DATA.PATH_TOKENIZED_RECIPES = '/srv/datasets2/recipe1m+/food.com_data_rescaled/clip_text/tokenized_recipe.json'
    config.DATA.MAX_INSTRS_LEN = 10
    config.DATA.MAX_INGRS_LEN = 6
    config.DATA.MAX_INSTRS = 5
    config.DATA.MAX_INGRS = 5
    config.freeze()
    logger = init_config.init_logger('.','log.txt')
    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(device)
    logger.info('creating data loader')
    dataloader = create_data(config, 'test')
    logger.info('creating models')
    model = Clip(device)

    evaluate(config, dataloader, model, device)

if __name__ == '__main__':
    main()
