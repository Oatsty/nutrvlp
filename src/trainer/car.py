import logging
import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from tqdm import tqdm
from .base import BaseCarTrainer

logger = logging.getLogger()

class CarTrainer(BaseCarTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        params = list(self.model.parameters())
        self.optim = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim,
                t_initial=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                lr_min=1e-7,
                warmup_t=config.TRAIN.WARMUP_EPOCHS,
                warmup_prefix=True,
            )

    def train_one_epoch(self, optim):
        # logger.info('running demo')
        avg_loss_dict = {}
        for phase in ['train','val']:
            running_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                img_input = batch['image']['data'].to(self.device)
                pil_image_input = batch['image']['pil_image']
                # text_input = batch['recipe']['raw']
                description = batch['recipe']['description'].to(self.device)
                out = self.model(
                    img_input,
                    pil_image_input,
                    recipe_input['title'],
                    recipe_input['ingrs'],
                    recipe_input['instrs'],
                    description,
                    # text_input['title'],
                    # text_input['ingredients'],
                    # text_input['instructions'],
                )
                out['embs']['segment'] = batch['recipe']['mask_embed'].to(self.device)
                loss = self.recipe_loss_function(out)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out['img'])
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class CarNoSegmentTrainer(BaseCarTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        params = list(self.model.parameters())
        self.optim = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim,
                t_initial=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                lr_min=1e-7,
                warmup_t=config.TRAIN.WARMUP_EPOCHS,
                warmup_prefix=True,
            )

    def train_one_epoch(self, optim):
        # logger.info('running demo')
        avg_loss_dict = {}
        # for phase in ['train','val']:
        for phase in ['train']:
            running_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                img_input = batch['image']['data'].to(self.device)
                pil_image_input = batch['image']['pil_image']
                # text_input = batch['recipe']['raw']
                description = batch['recipe']['description'].to(self.device)
                out = self.model(
                    img_input,
                    pil_image_input,
                    recipe_input['title'],
                    recipe_input['ingrs'],
                    recipe_input['instrs'],
                    description,
                    # text_input['title'],
                    # text_input['ingredients'],
                    # text_input['instructions'],
                )
                loss = self.recipe_loss_function(out)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out['img'])
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict
