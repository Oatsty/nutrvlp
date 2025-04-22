import logging
import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from tqdm import tqdm
from utils.metrics import cal_mae
from .base import BaseHTTrainer

logger = logging.getLogger()

class HTDirectIngrs3BranchesTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.text_encoder.parameters())
        else:
            params = list(self.model.parameters())

        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters()) \
            + list(self.model.ingr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                    optim.step()
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

class HTTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                out_text = (out_recipe + out_nutr) / 2
                loss = self.loss_function(out_img,out_text,nutrs=nutr)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class HT3BranchesTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        else:
            params = list(self.model.parameters())

        warmup_params = list(self.model.nutr_embedder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                loss = self.recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class HTNoNutrTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                loss = self.loss_function(out_img,out_recipe,nutrs=nutr)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class HTDirectTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                pred_nutr = out['nutr']
                out_text = (out_recipe + out_nutr) / 2
                loss = self.loss_function(out_img,out_text,nutrs=nutr) + self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class HTDirect3BranchesTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                pred_nutr = out['nutr']
                loss = self.recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr) + self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                instance_count += len(out_img)
            avg_loss = running_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')

        return avg_loss_dict

class HTIngrsOnly3BranchesTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.text_encoder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters()) \
            + list(self.model.ingr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
            running_ingrs_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                out_nutr = out['nutr_embedding']
                pred_ingrs = out['ingrs']
                retrieval_loss = self.recipe_loss_function({'img': out_img, 'rec': out_recipe, 'nutr': out_nutr},nutrs=nutr)
                ingrs_loss = self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
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

class HTIngrsOnlyNoNutrTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.text_encoder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters()) \
            + list(self.model.ingr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
            running_ingrs_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                pred_ingrs = out['ingrs']
                retrieval_loss = self.loss_function(out_img,out_recipe,nutrs=nutr)
                ingrs_loss = self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
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

class HTDirectIngrsTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.text_encoder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters()) \
            + list(self.model.ingr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                out_text = (out_recipe + out_nutr) / 2
                retrieval_loss = self.loss_function(out_img,out_text,nutrs=nutr)
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

class HTDirectIngrsNoNutrTrainer(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            self.model.joint_embedding.text_encoder.requires_grad_(False)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_embedder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.joint_embedding.merger_recipe.parameters()) + \
                list(self.model.nutr_embedder.parameters()) +\
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) +\
                list(self.model.joint_embedding.text_encoder.parameters())
        else:
            params = list(self.model.parameters())
        warmup_params = list(self.model.nutr_embedder.parameters()) \
            + list(self.model.nutr_decoder.parameters()) \
            + list(self.model.ingr_decoder.parameters())
        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            [
                {'params': params},
            ],
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
                out = self.model(recipe_input, img_input, nutr)
                out_recipe  = out['recipe_embedding']
                out_img = out['image_embedding']
                pred_nutr = out['nutr']
                pred_ingrs = out['ingrs']
                retrieval_loss = self.loss_function(out_img,out_recipe,nutrs=nutr)
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

class NutrOnlyHT(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.nutr_decoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.nutr_decoder.parameters()) + \
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.nutr_decoder.parameters())
        else:
            params = list(self.model.parameters())

        warmup_params = list(self.model.nutr_decoder.parameters())

        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
                t_initial=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                lr_min=1e-7,
                warmup_t=config.TRAIN.WARMUP_EPOCHS,
                warmup_prefix=True,
            )

    def train_one_epoch(self, optim):
        # logger.info('running demo')
        avg_loss_dict = {}
        for phase in ['train','val']:
            running_mae_loss = 0.
            instance_count = 0
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            logger.info(f'running {phase} phase')
            for batch in tqdm(self.dataloaders[phase]):
                optim.zero_grad()
                nutr = batch['recipe']['nutr'].to(self.device)
                img_input = batch['image']['data'].to(self.device)
                out = self.model(img_input)
                pred_nutr = out
                mae_loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                if phase == 'train':
                    mae_loss.backward()
                    optim.step()
                running_mae_loss += mae_loss.item()
                instance_count += len(out)
            avg_mae_loss = running_mae_loss / instance_count
            avg_loss_dict[phase] = avg_mae_loss
            logger.info(f'mae loss = {avg_mae_loss}')

        return avg_loss_dict

class NutrIngrOnlyHT(BaseHTTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        if not config.TRAIN.FINETUNE:
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'clip':
            params = list(self.model.nutr_decoder.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.joint_embedding.image_encoder.parameters())
        elif config.TRAIN.FINETUNE_MODEL == 'ht':
            params = list(self.model.joint_embedding.image_encoder.fc.parameters()) + \
                list(self.model.ingr_decoder.parameters()) + \
                list(self.model.nutr_decoder.parameters())
        else:
            params = list(self.model.parameters())

        warmup_params = list(self.model.nutr_decoder.parameters()) + list(self.model.ingr_decoder.parameters())

        self.optim1 = torch.optim.Adam(
            warmup_params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.optim2 = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = CosineLRScheduler(
                self.optim2,
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
                img_input = batch['image']['data'].to(self.device)
                ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
                out = self.model(img_input)
                pred_nutr = out['nutr']
                pred_ingrs = out['ingrs']
                mae_loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
                ingrs_loss = self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
                loss = mae_loss + ingrs_loss
                if phase == 'train':
                    loss.backward()
                    optim.step()
                running_loss += loss.item()
                running_mae_loss += mae_loss.item()
                running_ingrs_loss += ingrs_loss.item()
                instance_count += len(pred_nutr)
            avg_loss = running_loss / instance_count
            avg_mae_loss = running_mae_loss / instance_count
            avg_ingrs_loss = running_ingrs_loss / instance_count
            avg_loss_dict[phase] = avg_loss
            logger.info(f'loss = {avg_loss}')
            logger.info(f'mae loss = {avg_mae_loss}')
            logger.info(f'ingrs loss = {avg_ingrs_loss}')

        return avg_loss_dict
