import logging
import os
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import sys

from tqdm import tqdm

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp


sys.path.append('/home/parinayok/food.com_net/src')

from dataset import create_data
import init_config
from model.ht import create_ht_model
from utils.loss import TripletLoss
from utils.evaluator import Evaluator
from utils.misc import convert_state_dict

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, model: nn.Module, im2rec_loss_function: nn.Module, device):
    running_loss = 0.
    instance_count = 0
    model.eval()
    logger.info(f'running test phase')
    heatmap_pp_list = {}
    heatmap_list = {}
    mask_pp_list = {}
    mask_list = {}
    print(model.joint_embedding.image_encoder.backbone)
    target_layer = model.joint_embedding.image_encoder.backbone.head # type: ignore
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)
    for batch in tqdm(dataloader):
        recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
        img_input = batch['image']['data'].to(device)
        out = model(recipe_input,img_input)
        out_recipe = out['recipe_embedding']
        out_img = out['image_embedding']
        mask_pp, _ = gradcam_pp(recipe_input,img_input)
        mask, _ = gradcam(recipe_input,img_input)
        for recipe_id, img in zip(batch['recipe']['ids'],img_input):
            heatmap_pp, _ = visualize_cam(mask_pp, img)
            heatmap, _ = visualize_cam(mask, img)
            key = recipe_id
            heatmap_pp_list[key] = heatmap_pp
            heatmap_list[key] = heatmap
            mask_pp_list[key] = mask_pp
            mask_list[key] = mask
        loss = im2rec_loss_function(out_img,out_recipe)
        running_loss += loss.item()
        instance_count += len(out_img)
        break
    # avg_loss = running_loss / instance_count
    # logger.info(f'loss = {avg_loss}')
    # return avg_loss
    dir_name = f'infer_w_gradcam/'
    os.makedirs(dir_name, exist_ok=True)
    pickle.dump(heatmap_pp_list, open(os.path.join(dir_name,config.SAVE_PATH + '_heatmap_pp.obj'),'wb'))
    pickle.dump(heatmap_list, open(os.path.join(dir_name,config.SAVE_PATH + '_heatmap.obj'),'wb'))
    pickle.dump(mask_pp_list, open(os.path.join(dir_name,config.SAVE_PATH + '_mask_pp.obj'),'wb'))
    pickle.dump(mask_list, open(os.path.join(dir_name,config.SAVE_PATH + '_mask.obj'),'wb'))

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
    model = create_ht_model(config,device)
    ht_path = config.HT_PATH
    state_dict = torch.load(ht_path,map_location=device)
    model.joint_embedding.load_state_dict(state_dict)
    loss_function = TripletLoss()
    evaluate(config, dataloader, model, loss_function, device)

if __name__ == '__main__':
    main()
