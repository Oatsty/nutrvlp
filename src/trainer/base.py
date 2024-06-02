from abc import ABC
import copy
import logging
import os
import numpy as np
import torch

from ..dataset import create_data
from ..model import create_models
from ..model.nutr_vlp import create_nutr_vlp_model
from ..model.nutr_ht import create_nutr_ht_model
from ..utils.loss import get_loss
from ..utils.misc import convert_state_dict

logger = logging.getLogger()

class BaseTrainer(ABC):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        phases = ['train', 'val']
        logger.info('creating data loaders')
        self.dataloaders = {phase: create_data(config, phase) for phase in phases}
        self.loss_function, self.recipe_loss_function = get_loss(config)
        logger.info('creating models')
        self.recipe_model, self.img_model, _ = create_models(config,device)

    def train_one_epoch(self):
        pass

    def train(self):
        num_epochs = self.config.TRAIN.NUM_EPOCHS
        logger.info(f'training for {num_epochs} epochs')
        min_val_loss = np.inf
        best_recipe_model = copy.deepcopy(self.recipe_model)
        best_img_model = copy.deepcopy(self.img_model)
        best_epoch = 0
        for epoch in range(num_epochs):
            logger.info(f'epoch {epoch + 1}')
            avg_loss_dict = self.train_one_epoch()
            self.scheduler.step(epoch+1)
            val_loss = avg_loss_dict['val']
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_recipe_model = copy.deepcopy(self.recipe_model)
                best_img_model = copy.deepcopy(self.img_model)
                best_epoch = epoch + 1
            if (epoch + 1) % 10 == 0:
                torch.save(self.recipe_model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'recipe_ep{epoch+1}.pt'))
                torch.save(self.img_model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'img_ep{epoch+1}.pt'))
        logger.info(f'best epoch: {best_epoch}')
        logger.info(f'min val loss: {min_val_loss}')
        torch.save(best_recipe_model.state_dict(),os.path.join(self.config.SAVE_PATH,'recipe_test.pt'))
        torch.save(best_img_model.state_dict(),os.path.join(self.config.SAVE_PATH,'img_test.pt'))

class BaseNutrTrainer(ABC):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        phases = ['train', 'val']
        logger.info('creating data loaders')
        self.dataloaders = {phase: create_data(config, phase) for phase in phases}
        logger.info('creating models')
        self.recipe_model, self.img_model, self.nutr_model = create_models(config,device)
        self.loss_function, self.recipe_loss_function = get_loss(config)

    def train_one_epoch(self):
        pass

    def train(self):
        num_epochs = self.config.TRAIN.NUM_EPOCHS
        logger.info(f'training for {num_epochs} epochs')
        min_val_loss = np.inf
        best_recipe_model = copy.deepcopy(self.recipe_model)
        best_img_model = copy.deepcopy(self.img_model)
        best_nutr_model = copy.deepcopy(self.nutr_model)
        best_epoch = 0
        for epoch in range(num_epochs):
            logger.info(f'epoch {epoch + 1}')
            avg_loss_dict = self.train_one_epoch()
            val_loss = avg_loss_dict['val']
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_recipe_model = copy.deepcopy(self.recipe_model)
                best_img_model = copy.deepcopy(self.img_model)
                best_nutr_model = copy.deepcopy(self.nutr_model)
                best_epoch = epoch + 1
            if (epoch + 1) % 10 == 0:
                torch.save(self.recipe_model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'recipe_ep{epoch+1}.pt'))
                torch.save(self.img_model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'img_ep{epoch+1}.pt'))
                torch.save(self.nutr_model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'nutr_ep{epoch+1}.pt'))
        logger.info(f'best epoch: {best_epoch}')
        logger.info(f'min val loss: {min_val_loss}')
        torch.save(best_recipe_model.state_dict(),os.path.join(self.config.SAVE_PATH,'recipe_test.pt'))
        torch.save(best_img_model.state_dict(),os.path.join(self.config.SAVE_PATH,'img_test.pt'))
        torch.save(best_nutr_model.state_dict(),os.path.join(self.config.SAVE_PATH,'nutr_test.pt'))


class BaseVLPTrainer(ABC):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        phases = ['train', 'val']
        logger.info('creating data loaders')
        self.dataloaders = {phase: create_data(config, phase) for phase in phases}
        logger.info('creating models')
        self.model = create_nutr_vlp_model(config, device)
        vlp_path = config.VLP_PATH
        org_state_dict = torch.load(vlp_path,map_location=device)
        new_state_dict = convert_state_dict(org_state_dict)
        self.model.vlp_cook.load_state_dict(new_state_dict)
        self.loss_function, self.recipe_loss_function = get_loss(config)

    def train_one_epoch(self):
        pass

    def train(self):
        num_epochs = self.config.TRAIN.NUM_EPOCHS
        logger.info(f'training for {num_epochs} epochs')
        min_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        best_epoch = 0
        for epoch in range(num_epochs):
            logger.info(f'epoch {epoch + 1}')
            avg_loss_dict = self.train_one_epoch()
            self.scheduler.step(epoch+1)
            val_loss = avg_loss_dict['val']
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch + 1
            # if (epoch + 1) % 10 == 0:
            #     torch.save(self.model.state_dict(),os.path.join(self.config.SAVE_PATH,'checkpoints',f'food_ep{epoch+1}.pt'))
        logger.info(f'best epoch: {best_epoch}')
        logger.info(f'min val loss: {min_val_loss}')
        torch.save(best_model.state_dict(),os.path.join(self.config.SAVE_PATH,'food_test.pt'))

class BaseHTTrainer(ABC):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        phases = ['train', 'val']
        logger.info('creating data loaders')
        self.dataloaders = {phase: create_data(config, phase) for phase in phases}
        self.loss_function, self.recipe_loss_function = get_loss(config)
        logger.info('creating models')
        self.model = create_nutr_ht_model(config,device)
        ht_path = config.HT_PATH
        state_dict = torch.load(ht_path,map_location=device)
        self.model.joint_embedding.load_state_dict(state_dict)

    def train_one_epoch(self):
        pass

    def train(self):
        num_epochs = self.config.TRAIN.NUM_EPOCHS
        warmup_epochs = self.config.TRAIN.WARMUP_EPOCHS
        logger.info(f'training for {num_epochs} epochs')
        min_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        best_epoch = 0
        for epoch in range(num_epochs):
            logger.info(f'epoch {epoch + 1}')
            if epoch < warmup_epochs:
                avg_loss_dict = self.train_one_epoch(self.optim1)
            else:
                avg_loss_dict = self.train_one_epoch(self.optim2)
            self.scheduler.step(epoch+1)
            val_loss = avg_loss_dict['val']
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch + 1
            # if (epoch + 1) % 10 == 0:
            #     torch.save(model.state_dict(),os.path.join(config.SAVE_PATH,'checkpoints',f'food_ep{epoch+1}.pt'))
        logger.info(f'best epoch: {best_epoch}')
        logger.info(f'min val loss: {min_val_loss}')
        torch.save(best_model.state_dict(),os.path.join(self.config.SAVE_PATH,'food_test.pt'))
