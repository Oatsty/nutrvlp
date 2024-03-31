import copy
import logging
import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from timm.scheduler import CosineLRScheduler
import sys

from tqdm import tqdm

sys.path.append('/home/parinayok/food.com_net')

from dataset import create_data
import init_config
from model.nutr_vlp import create_nutr_vlp_model
from utils.loss import get_loss
from utils.metrics import cal_mae
from utils.misc import convert_state_dict

logger = logging.getLogger()

def train_one_epoch(config, dataloaders: Dict[str, DataLoader], model: nn.Module, recipe_loss_function: nn.Module, optim, device):
    # logger.info('running demo')
    avg_loss_dict = {}
    for phase in ['train','val']:
        running_loss = 0.
        running_retrieval_loss = 0.
        running_ingrs_loss = 0.
        instance_count = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        logger.info(f'running {phase} phase')
        for batch in tqdm(dataloaders[phase]):
            optim.zero_grad()
            recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
            nutr = batch['recipe']['nutr'].to(device)
            img_input = batch['image']['data'].to(device)
            ingr_clss = batch['recipe']['ingr_clss'].to(device)
            out = model(recipe_input, img_input, nutr)
            out_recipe  = out['recipe_embedding']
            out_img = out['image_embedding']
            out_nutr = out['nutr_embedding']
            pred_ingrs = out['ingrs']
            retrieval_loss = recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr)
            ingrs_loss = config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
            loss = retrieval_loss + ingrs_loss
            if phase == 'train':
                loss.backward()
                optim.step()
            running_loss += loss.item()
            running_retrieval_loss += retrieval_loss.item()
            running_ingrs_loss += ingrs_loss.item()
            instance_count += len(out_img)
        avg_loss = running_loss / instance_count
        avg_retrieval_loss = running_retrieval_loss / instance_count
        avg_ingrs_loss = running_ingrs_loss / instance_count
        avg_loss_dict[phase] = avg_loss
        logger.info(f'loss = {avg_loss}')
        logger.info(f'retrieval loss = {avg_retrieval_loss}')
        logger.info(f'ingrs loss = {avg_ingrs_loss}')

    return avg_loss_dict

def train(config, dataloaders: Dict[str, DataLoader], model: nn.Module, recipe_loss_function: nn.Module, optim, scheduler, device):
    num_epochs = config.TRAIN.NUM_EPOCHS
    logger.info(f'training for {num_epochs} epochs')
    min_val_loss = np.inf
    best_model = copy.deepcopy(model)
    best_epoch = 0
    for epoch in range(num_epochs):
        logger.info(f'epoch {epoch + 1}')
        avg_loss_dict = train_one_epoch(config, dataloaders, model, recipe_loss_function, optim, device)
        scheduler.step(epoch+1)
        val_loss = avg_loss_dict['val']
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(),os.path.join(config.SAVE_PATH,'checkpoints',f'food_ep{epoch+1}.pt'))
    logger.info(f'best epoch: {best_epoch}')
    logger.info(f'min val loss: {min_val_loss}')
    torch.save(best_model.state_dict(),os.path.join(config.SAVE_PATH,'food_test.pt'))

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
    model = create_nutr_vlp_model(config, device)
    vlp_path = config.VLP_PATH
    org_state_dict = torch.load(vlp_path,map_location=device)
    new_state_dict = convert_state_dict(org_state_dict)
    model.vlp_cook.load_state_dict(new_state_dict)
    if not config.TRAIN.FINETUNE:
        model.vlp_cook.recipe_embedder.requires_grad_(False)
        model.vlp_cook.image_embedder.requires_grad_(False)
    elif config.TRAIN.FINETUNE_MODEL == 'clip':
        model.vlp_cook.recipe_embedder.requires_grad_(False)
    elif config.TRAIN.FINETUNE_MODEL == 'ht':
        model.vlp_cook.image_embedder.requires_grad_(False)
    _, recipe_loss_function = get_loss(config)
    if not config.TRAIN.FINETUNE:
        params = list(model.vlp_cook.proj_image.parameters()) + \
            list(model.vlp_cook.proj_recipe.parameters()) + \
            list(model.nutr_decoder.parameters()) + \
            list(model.ingr_decoder.parameters()) + \
            list(model.nutr_embedder.parameters())
    elif config.TRAIN.FINETUNE_MODEL == 'clip':
        params = list(model.vlp_cook.proj_image.parameters()) + \
            list(model.vlp_cook.proj_recipe.parameters()) + \
            list(model.nutr_embedder.parameters()) +\
            list(model.nutr_decoder.parameters()) + \
            list(model.ingr_decoder.parameters()) +\
            list(model.vlp_cook.image_embedder.parameters())
    elif config.TRAIN.FINETUNE_MODEL == 'ht':
        params = list(model.vlp_cook.proj_image.parameters()) + \
            list(model.vlp_cook.proj_recipe.parameters()) + \
            list(model.nutr_embedder.parameters()) +\
            list(model.nutr_decoder.parameters()) + \
            list(model.ingr_decoder.parameters()) +\
            list(model.vlp_cook.recipe_embedder.parameters())
    else:
        params = list(model.parameters())
    optim = torch.optim.Adam(
        [
            {'params': params},
        ],
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = CosineLRScheduler(
            optim,
            t_initial=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
            lr_min=1e-7,
            warmup_t=config.TRAIN.WARMUP_EPOCHS,
            warmup_prefix=True,
        )

    train(config, dataloaders, model, recipe_loss_function, optim, scheduler, device)

if __name__ == '__main__':
    main()
