import logging
import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from tqdm import tqdm
from utils.metrics import cal_mae
from .base import BaseNutrCarTrainer

logger = logging.getLogger()


class NutrCarTrainer(BaseNutrCarTrainer):
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
            running_retrieval_loss = 0.
            running_mae_loss = 0.
            running_ingrs_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                nutr = batch['recipe']['nutr'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
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
                pred_nutr = out['nutr']
                pred_ingrs = out['ingrs']
                out['embs']['segment'] = batch['recipe']['mask_embed'].to(self.device)
                retrieval_loss = self.recipe_loss_function(out['embs'],nutrs=nutr)
                mae_loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                ingrs_loss = self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
                loss = retrieval_loss + mae_loss + ingrs_loss
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                running_retrieval_loss += retrieval_loss.item()
                running_mae_loss += mae_loss.item()
                running_ingrs_loss += ingrs_loss.item()
                instance_count += len(pred_nutr)
            avg_loss = running_loss / instance_count
            avg_retrieval_loss = running_retrieval_loss / instance_count
            avg_mae_loss = running_mae_loss / instance_count
            avg_ingrs_loss = running_ingrs_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')
            logger.info(f'retrieval loss = {avg_retrieval_loss}')
            logger.info(f'mae loss = {avg_mae_loss}')
            logger.info(f'ingrs loss = {avg_ingrs_loss}')

        return avg_loss_dict
