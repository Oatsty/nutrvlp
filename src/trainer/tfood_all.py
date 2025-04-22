import logging
import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from utils.metrics import cal_mae
from .base import BaseTFoodTrainer

logger = logging.getLogger()

class TFoodDirectIngrs3BranchesTrainer(BaseTFoodTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.tfood.recipe_embedder.requires_grad_(False)
            self.model.tfood.image_embedder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.tfood.recipe_embedder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            self.model.tfood.image_embedder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.tfood.proj_image.parameters()) + \
                list(self.model.tfood.proj_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.tfood.proj_image.parameters()) + \
                list(self.model.tfood.proj_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.tfood.image_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.tfood.proj_image.parameters()) + \
                list(self.model.tfood.proj_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.tfood.recipe_embedder.parameters())
        else:
            params = list(self.model.parameters())
        self.optim = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim,
                t_initial=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                lr_min=1e-7,
                warmup_t=config.TRAIN.WARMUP_EPOCHS,
                warmup_prefix=True,
            )

    def train_one_epoch(self):
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
                self.optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                pred_nutr = out['nutr']
                pred_ingrs = out['ingrs']
                retrieval_loss = self.recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr)
                mae_loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                ingrs_loss = self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
                loss = retrieval_loss + mae_loss + ingrs_loss
                if phase == 'train':
                    loss.backward()
                    self.optim.step()
                running_loss += loss.item()
                running_retrieval_loss += retrieval_loss.item()
                running_mae_loss += mae_loss.item()
                running_ingrs_loss += ingrs_loss.item()
                instance_count += len(out_img)
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
