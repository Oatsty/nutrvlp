import logging
import torch
from tqdm import tqdm
from .base import BaseNutrTrainer

logger = logging.getLogger()

class NutrTrainer(BaseNutrTrainer):
    def __init__(self, config, device) -> None:
        super(NutrTrainer, self).__init__(config, device)
        params_backbone = list(self.img_model.encoder.parameters())
        params_fc = list(self.img_model.fc.parameters()) + list(self.recipe_model.parameters())
        self.optim = torch.optim.Adam(
            [
                {'params': params_fc},
                {'params': params_backbone,'lr': config.TRAIN.LR*config.TRAIN.SCALE_LR},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)

    def train_one_epoch(self):
        # logger.info('running demo')
        recipe_loss_weight = self.config.TRAIN.RECIPE_LOSS_WEIGHT
        avg_loss_dict = {}
        for phase in ['train','val']:
            running_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.recipe_model.train()
                self.img_model.train()
                self.nutr_model.train()
            else:
                self.recipe_model.eval()
                self.img_model.eval()
                self.nutr_model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                self.optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out_recipe, out_comp_embs = self.recipe_model(recipe_input)
                out_img = self.img_model(img_input)
                out_nutr  = self.nutr_model(nutr)
                loss = self.recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr) + recipe_loss_weight * self.recipe_loss_function(out_comp_embs,nutrs=nutr)
                if phase == 'train':
                    loss.backward()
                    self.optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')
        return avg_loss_dict
