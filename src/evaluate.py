import copy
import logging
import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import sys

from tqdm import tqdm

sys.path.append('/home/parinayok/food.com_net')

from dataset import create_data
import init_config
from model import create_models
from utils.loss import TripletLoss, MultiTripletLoss
from utils.evaluator import Evaluator

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, recipe_model: nn.Module, img_model: nn.Module, im2rec_loss_function: nn.Module, recipe_loss_function: nn.Module, device):
    # logger.info('running demo')
    recipe_loss_weight = config.TRAIN.RECIPE_LOSS_WEIGHT
    running_loss = 0.
    instance_count = 0
    recipe_model.eval()
    img_model.eval()
    logger.info(f'running test phase')
    retrieval_dir = config.RETRIEVAL_DIR
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0])
    evaluator = Evaluator(out_dir,retrieval_dir)
    for batch in tqdm(dataloader):
        recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
        nutr = batch['recipe']['nutr'].to(device)
        img_input = batch['image']['data'].to(device)
        out_recipe, out_comp_embs = recipe_model(recipe_input,nutr=nutr)
        out_img = img_model(img_input)
        loss = im2rec_loss_function(out_img,out_recipe) + recipe_loss_weight * recipe_loss_function(out_comp_embs)
        running_loss += loss.item()
        instance_count += len(out_img)
        evaluator(out_img, out_recipe, batch)
    avg_loss = running_loss / instance_count
    logger.info(f'loss = {avg_loss}')
    evaluator.calculate_similarity()
    return avg_loss

def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    logger = init_config.init_logger('.','log.txt')
    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(device)
    logger.info('creating data loader')
    dataloader = create_data(config, 'test')
    logger.info('creating models')
    recipe_model, img_model, _ = create_models(config,device)
    recipe_model_path = os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH)
    recipe_model.load_state_dict(torch.load(recipe_model_path,map_location=device))
    img_model_path = os.path.join(config.SAVE_PATH,config.IMG_MODEL_PATH)
    img_model.load_state_dict(torch.load(img_model_path,map_location=device))
    loss_function = TripletLoss()
    recipe_loss_function = MultiTripletLoss()

    evaluate(config, dataloader, recipe_model, img_model, loss_function, recipe_loss_function, device)

if __name__ == '__main__':
    main()
