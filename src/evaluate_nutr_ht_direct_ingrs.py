import json
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import sys

from tqdm import tqdm

from model.nutr_ht import create_nutr_ht_model


sys.path.append('/home/parinayok/food.com_net')

from dataset import create_data
import init_config
from model.nutr_vlp import create_nutr_vlp_model
from utils.loss import TripletLoss
from utils.evaluator import Evaluator
from utils.metrics import cal_mae

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, model: nn.Module, im2rec_loss_function: nn.Module, device):
    nutr_names = config.DATA.NUTRS
    nutr_scale_list = config.DATA.NUTR_STDS
    nutr_scale = {name: scale for (name,scale) in zip(nutr_names,nutr_scale_list)}
    running_loss = 0.
    instance_count = 0
    model.eval()
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0])
    os.makedirs(out_dir,exist_ok=True)
    out_nutr_path = os.path.join(out_dir,'out_nutr.json')
    out_ingrs_path = os.path.join(out_dir,'out_ingrs.json')
    out_nutr_dict = {}
    out_ingrs_dict = {}
    logger.info(f'running test phase')
    for batch in tqdm(dataloader):
        recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
        img_input = batch['image']['data'].to(device)
        nutr = batch['recipe']['nutr'].to(device)
        ingr_clss = batch['recipe']['ingr_clss'].to(device)
        out = model(recipe_input,img_input,nutr)
        pred_nutr = out['nutr']
        pred_ingrs = out['ingrs']
        loss = config.TRAIN.MAE_WEIGHT * cal_mae(nutr,pred_nutr) + config.TRAIN.INGRS_WEIGHT * F.binary_cross_entropy_with_logits(pred_ingrs,ingr_clss,reduction='none').mean(-1).sum()
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
    model = create_nutr_ht_model(config,device)
    model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
    loss_function = TripletLoss()
    evaluate(config, dataloader, model, loss_function, device)

if __name__ == '__main__':
    main()
