import logging
import os
import torch
import torch.nn.functional as F
import json

from tqdm import tqdm

from ..dataset import create_data
from ..model.nutr_ht import create_nutr_ht_model
from ..model.nutr_vlp import create_nutr_vlp_model
from ..model import create_models
from ..utils.loss import TripletLoss, MultiTripletLoss
from ..utils.evaluator import Evaluator
from ..utils.metrics import cal_mae

logger = logging.getLogger()

class NutrHTAllEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_ht_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        logger.info(f'running test phase')
        retrieval_dir = self.config.RETRIEVAL_DIR
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_all_domains')
        evaluator_all_domains = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_recipe')
        evaluator_nutr_recipe = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_img')
        evaluator_nutr_img = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_recipe_img')
        evaluator_recipe_img = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_only')
        evaluator_nutr = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_recipe_only')
        evaluator_recipe = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_img_only')
        evaluator_img = Evaluator(out_dir,retrieval_dir)
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            out_recipe = out['recipe_embedding']
            out_img = out['image_embedding']
            out_nutr = out['nutr_embedding']
            out_all_domains = (out_recipe + out_nutr + out_img) / 3
            out_nutr_recipe = (out_nutr + out_recipe) / 2
            out_nutr_img = (out_nutr + out_img) / 2
            out_recipe_img = (out_recipe + out_img) / 2
            loss = self.loss_function(out_img,out_all_domains)
            running_loss += loss.item()
            instance_count += len(out_img)
            evaluator_all_domains(out_img, out_all_domains, batch)
            evaluator_nutr_recipe(out_img, out_nutr_recipe, batch)
            evaluator_nutr_img(out_img, out_nutr_img, batch)
            evaluator_recipe_img(out_img, out_recipe_img, batch)
            evaluator_nutr(out_img, out_nutr, batch)
            evaluator_recipe(out_img, out_recipe, batch)
            evaluator_img(out_img, out_img, batch)
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        evaluator_all_domains.calculate_similarity()
        evaluator_nutr_recipe.calculate_similarity()
        evaluator_nutr_img.calculate_similarity()
        evaluator_recipe_img.calculate_similarity()
        evaluator_nutr.calculate_similarity()
        evaluator_recipe.calculate_similarity()
        evaluator_img.calculate_similarity()
        return avg_loss

class NutrHTDirectEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_ht_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        nutr_names = self.config.DATA.NUTRS
        nutr_scale_list = self.config.DATA.NUTR_STDS
        nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        os.makedirs(out_dir,exist_ok=True)
        out_path = os.path.join(out_dir,'out_nutr.json')
        out_dict = {}
        logger.info(f'running test phase')
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            pred_nutr = out['nutr']
            loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
            running_loss += loss.item()
            instance_count += len(img_input)
            ids = batch['recipe']['ids']
            for id, pred_n in zip(ids, pred_nutr):
                out_dict[id] = {name: pn.item()*nutr_scale[name] for name,pn in zip(nutr_names,pred_n)}
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        with open(out_path,'w') as f:
            json.dump(out_dict,f,indent=2)
        return avg_loss

class NutrHTDirectIngrsEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_ht_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        nutr_names = self.config.DATA.NUTRS
        nutr_scale_list = self.config.DATA.NUTR_STDS
        nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        os.makedirs(out_dir,exist_ok=True)
        out_nutr_path = os.path.join(out_dir,'out_nutr.json')
        out_ingrs_path = os.path.join(out_dir,'out_ingrs.json')
        out_nutr_dict = {}
        out_ingrs_dict = {}
        logger.info(f'running test phase')
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            pred_nutr = out['nutr']
            pred_ingrs = out['ingrs']
            loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr) + self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
            running_loss += loss.item()
            instance_count += len(img_input)
            ids = batch['recipe']['ids']
            for id, pred_n in zip(ids, pred_nutr):
                out_nutr_dict[id] = {name: pn.item()*nutr_scale[name] for name,pn in zip(nutr_names,pred_n)}
            pred_ingr_indexes = [torch.nonzero(pred_ingr.bool(),as_tuple=True)[0].tolist() for pred_ingr in F.sigmoid(pred_ingrs) > 0.2]
            for id, pred_i in zip(ids,pred_ingr_indexes):
                out_ingrs_dict[id] = pred_i
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        with open(out_nutr_path,'w') as f:
            json.dump(out_nutr_dict,f,indent=2)
        with open(out_ingrs_path,'w') as f:
            json.dump(out_ingrs_dict,f,indent=2)
        return avg_loss

class NutrVLPAllEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_vlp_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        logger.info(f'running test phase')
        retrieval_dir = self.config.RETRIEVAL_DIR
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_all_domains')
        evaluator_all_domains = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_recipe')
        evaluator_nutr_recipe = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_img')
        evaluator_nutr_img = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_recipe_img')
        evaluator_recipe_img = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_nutr_only')
        evaluator_nutr = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_recipe_only')
        evaluator_recipe = Evaluator(out_dir,retrieval_dir)
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0] + '_img_only')
        evaluator_img = Evaluator(out_dir,retrieval_dir)
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            out_recipe = out['recipe_embedding']
            out_img = out['image_embedding']
            out_nutr = out['nutr_embedding']
            out_all_domains = (out_recipe + out_nutr + out_img) / 3
            out_nutr_recipe = (out_nutr + out_recipe) / 2
            out_nutr_img = (out_nutr + out_img) / 2
            out_recipe_img = (out_recipe + out_img) / 2
            loss = self.loss_function(out_img,out_all_domains)
            running_loss += loss.item()
            instance_count += len(out_img)
            evaluator_all_domains(out_img, out_all_domains, batch)
            evaluator_nutr_recipe(out_img, out_nutr_recipe, batch)
            evaluator_nutr_img(out_img, out_nutr_img, batch)
            evaluator_recipe_img(out_img, out_recipe_img, batch)
            evaluator_nutr(out_img, out_nutr, batch)
            evaluator_recipe(out_img, out_recipe, batch)
            evaluator_img(out_img, out_img, batch)
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        evaluator_all_domains.calculate_similarity()
        evaluator_nutr_recipe.calculate_similarity()
        evaluator_nutr_img.calculate_similarity()
        evaluator_recipe_img.calculate_similarity()
        evaluator_nutr.calculate_similarity()
        evaluator_recipe.calculate_similarity()
        evaluator_img.calculate_similarity()
        return avg_loss


class NutrVLPDirectEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_vlp_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        nutr_names = self.config.DATA.NUTRS
        nutr_scale_list = self.config.DATA.NUTR_STDS
        nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        os.makedirs(out_dir,exist_ok=True)
        out_path = os.path.join(out_dir,'out_nutr.json')
        out_dict = {}
        logger.info(f'running test phase')
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            pred_nutr = out['nutr']
            loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr)
            running_loss += loss.item()
            instance_count += len(img_input)
            ids = batch['recipe']['ids']
            for id, pred_n in zip(ids, pred_nutr):
                out_dict[id] = {name: pn.item()*nutr_scale[name] for name,pn in zip(nutr_names,pred_n)}
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        with open(out_path,'w') as f:
            json.dump(out_dict,f,indent=2)
        return avg_loss


class NutrVLPDirectIngrsEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.model = create_nutr_vlp_model(config,device)
        self.model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
        self.loss_function = TripletLoss()

    def evaluate(self):
        nutr_names = self.config.DATA.NUTRS
        nutr_scale_list = self.config.DATA.NUTR_STDS
        nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
        running_loss = 0.
        instance_count = 0
        self.model.eval()
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        os.makedirs(out_dir,exist_ok=True)
        out_nutr_path = os.path.join(out_dir,'out_nutr.json')
        out_ingrs_path = os.path.join(out_dir,'out_ingrs.json')
        out_nutr_dict = {}
        out_ingrs_dict = {}
        logger.info(f'running test phase')
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            img_input = batch['image']['data'].to(self.device)
            nutr = batch['recipe']['nutr'].to(self.device)
            ingr_clss = batch['recipe']['ingr_clss'].to(self.device)
            out = self.model(recipe_input,img_input,nutr)
            pred_nutr = out['nutr']
            pred_ingrs = out['ingrs']
            loss = self.config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr) + self.config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
            running_loss += loss.item()
            instance_count += len(img_input)
            ids = batch['recipe']['ids']
            for id, pred_n in zip(ids, pred_nutr):
                out_nutr_dict[id] = {name: pn.item()*nutr_scale[name] for name,pn in zip(nutr_names,pred_n)}
            pred_ingr_indexes = [torch.nonzero(pred_ingr.bool(),as_tuple=True)[0].tolist() for pred_ingr in F.sigmoid(pred_ingrs) > 0.2]
            for id, pred_i in zip(ids,pred_ingr_indexes):
                out_ingrs_dict[id] = pred_i
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        with open(out_nutr_path,'w') as f:
            json.dump(out_nutr_dict,f,indent=2)
        with open(out_ingrs_path,'w') as f:
            json.dump(out_ingrs_dict,f,indent=2)
        return avg_loss


class OldEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.recipe_model, self.img_model, _ = create_models(config,device)
        recipe_model_path = os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH)
        self.recipe_model.load_state_dict(torch.load(recipe_model_path,map_location=device))
        img_model_path = os.path.join(config.SAVE_PATH,config.IMG_MODEL_PATH)
        self.img_model.load_state_dict(torch.load(img_model_path,map_location=device))
        self.loss_function = TripletLoss()
        self.recipe_loss_function = MultiTripletLoss()

    def evaluate(self):
        # logger.info('running demo')
        recipe_loss_weight = self.config.TRAIN.RECIPE_LOSS_WEIGHT
        running_loss = 0.
        instance_count = 0
        self.recipe_model.eval()
        self.img_model.eval()
        logger.info(f'running test phase')
        retrieval_dir = self.config.RETRIEVAL_DIR
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        evaluator = Evaluator(out_dir,retrieval_dir)
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            nutr = batch['recipe']['nutr'].to(self.device)
            img_input = batch['image']['data'].to(self.device)
            out_recipe, out_comp_embs = self.recipe_model(recipe_input,nutr=nutr)
            out_img = self.img_model(img_input)
            loss = self.loss_function(out_img,out_recipe) + recipe_loss_weight * self.recipe_loss_function(out_comp_embs)
            running_loss += loss.item()
            instance_count += len(out_img)
            evaluator(out_img, out_recipe, batch)
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        evaluator.calculate_similarity()
        return avg_loss


class NutrEvaluator(object):
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        logger.info('creating data loader')
        self.dataloader = create_data(config, 'test')
        logger.info('creating models')
        self.recipe_model, self.img_model, self.nutr_model = create_models(config,device)
        recipe_model_path = os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH)
        self.recipe_model.load_state_dict(torch.load(recipe_model_path,map_location=device))
        img_model_path = os.path.join(config.SAVE_PATH,config.IMG_MODEL_PATH)
        self.img_model.load_state_dict(torch.load(img_model_path,map_location=device))
        nutr_model_path = os.path.join(config.SAVE_PATH,config.NUTR_MODEL_PATH)
        self.nutr_model.load_state_dict(torch.load(nutr_model_path,map_location=device))
        self.loss_function = TripletLoss()
        self.recipe_loss_function = MultiTripletLoss()

    def evaluate(self):
        # logger.info('running demo')
        recipe_loss_weight = self.config.TRAIN.RECIPE_LOSS_WEIGHT
        running_loss = 0.
        instance_count = 0
        self.recipe_model.eval()
        self.img_model.eval()
        self.nutr_model.eval()
        logger.info(f'running test phase')
        retrieval_dir = self.config.RETRIEVAL_DIR
        out_dir = os.path.join(self.config.OUT_DIR,self.config.SAVE_PATH,os.path.splitext(self.config.RECIPE_MODEL_PATH)[0])
        evaluator = Evaluator(out_dir,retrieval_dir)
        for batch in tqdm(self.dataloader):
            recipe_input = {x: batch['recipe'][x].to(self.device) for x in ['title','ingrs','instrs']}
            nutr = batch['recipe']['nutr'].to(self.device)
            img_input = batch['image']['data'].to(self.device)
            _, out_comp_embs = self.recipe_model(recipe_input,nutr=nutr)
            out_img = self.img_model(img_input)
            out_nutr = self.nutr_model(nutr)
            loss = self.loss_function(out_img,out_nutr) + recipe_loss_weight * self.recipe_loss_function(out_comp_embs)
            running_loss += loss.item()
            instance_count += len(out_img)
            evaluator(out_img, out_nutr, batch)
        avg_loss = running_loss / instance_count
        logger.info(f'loss = {avg_loss}')
        evaluator.calculate_similarity()
        return avg_loss
