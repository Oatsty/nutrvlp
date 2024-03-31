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
from utils.loss import get_loss

logger = logging.getLogger()

def train_one_epoch(config, dataloaders: Dict[str, DataLoader], recipe_model: nn.Module, img_model: nn.Module, nutr_model: nn.Module, recipe_loss_function: nn.Module, optim, device):
    # logger.info('running demo')
    recipe_loss_weight = config.TRAIN.RECIPE_LOSS_WEIGHT
    avg_loss_dict = {}
    for phase in ['train','val']:
        running_loss = 0.
        instance_count = 0
        if phase == 'train':
            recipe_model.train()
            img_model.train()
            nutr_model.train()
        else:
            recipe_model.eval()
            img_model.eval()
            nutr_model.eval()
        logger.info(f'running {phase} phase')
        for batch in tqdm(dataloaders[phase]):
            optim.zero_grad()
            recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
            nutr = batch['recipe']['nutr'].to(device)
            img_input = batch['image']['data'].to(device)
            out_recipe, out_comp_embs = recipe_model(recipe_input)
            out_img = img_model(img_input)
            out_nutr  = nutr_model(nutr)
            loss = recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr) + recipe_loss_weight * recipe_loss_function(out_comp_embs,nutrs=nutr)
            if phase == 'train':
                loss.backward()
                optim.step()
            running_loss += loss.item()
            instance_count += len(out_img)
        avg_loss = running_loss / instance_count
        avg_loss_dict[phase] = avg_loss
        logger.info(f'loss = {avg_loss}')
    return avg_loss_dict

def train(config, dataloaders: Dict[str, DataLoader], recipe_model: nn.Module, img_model: nn.Module, nutr_model: nn.Module, recipe_loss_function: nn.Module, optim, device):
    num_epochs = config.TRAIN.NUM_EPOCHS
    logger.info(f'training for {num_epochs} epochs')
    min_val_loss = np.inf
    best_recipe_model = copy.deepcopy(recipe_model)
    best_img_model = copy.deepcopy(img_model)
    best_nutr_model = copy.deepcopy(nutr_model)
    best_epoch = 0
    for epoch in range(num_epochs):
        logger.info(f'epoch {epoch + 1}')
        avg_loss_dict = train_one_epoch(config, dataloaders, recipe_model, img_model, nutr_model, recipe_loss_function, optim, device)
        val_loss = avg_loss_dict['val']
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_recipe_model = copy.deepcopy(recipe_model)
            best_img_model = copy.deepcopy(img_model)
            best_nutr_model = copy.deepcopy(nutr_model)
            best_epoch = epoch + 1
        if (epoch + 1) % 10 == 0:
            torch.save(recipe_model.state_dict(),os.path.join(config.SAVE_PATH,'checkpoints',f'recipe_ep{epoch+1}.pt'))
            torch.save(img_model.state_dict(),os.path.join(config.SAVE_PATH,'checkpoints',f'img_ep{epoch+1}.pt'))
            torch.save(nutr_model.state_dict(),os.path.join(config.SAVE_PATH,'checkpoints',f'nutr_ep{epoch+1}.pt'))
    logger.info(f'best epoch: {best_epoch}')
    logger.info(f'min val loss: {min_val_loss}')
    torch.save(best_recipe_model.state_dict(),os.path.join(config.SAVE_PATH,'recipe_test.pt'))
    torch.save(best_img_model.state_dict(),os.path.join(config.SAVE_PATH,'img_test.pt'))
    torch.save(best_nutr_model.state_dict(),os.path.join(config.SAVE_PATH,'nutr_test.pt'))

def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(config.SAVE_PATH,'checkpoints'), exist_ok=True)

    #init logger and dump config
    log_path = os.path.join("log", config.SAVE_PATH + ".txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = init_config.init_logger(
        os.path.dirname(log_path), os.path.basename(log_path)
    )
    logger.info(config.dump())

    # dump current config
    dump_path = os.path.join("config", config.SAVE_PATH + ".yaml")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w") as f:
        f.write(config.dump())  # type: ignore

    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(device)
    phases = ['train', 'val']
    logger.info('creating data loaders')
    dataloaders = {phase: create_data(config, phase) for phase in phases}
    logger.info('creating models')
    recipe_model, img_model, nutr_model = create_models(config,device)
    _, all_loss_function = get_loss(config)
    params_backbone = list(img_model.encoder.parameters())
    params_fc = list(img_model.fc.parameters()) + list(recipe_model.parameters())
    optim = torch.optim.Adam(
        [
            {'params': params_fc},
            {'params': params_backbone,'lr': config.TRAIN.LR*config.TRAIN.SCALE_LR},
        ],
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY)

    train(config, dataloaders, recipe_model, img_model, nutr_model, all_loss_function, optim, device)

if __name__ == '__main__':
    main()
