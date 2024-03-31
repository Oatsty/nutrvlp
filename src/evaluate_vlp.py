import logging
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import sys

from tqdm import tqdm


sys.path.append('/home/parinayok/food.com_net')

from dataset import create_data
import init_config
from model.vlp import create_vlp_model
from utils.loss import TripletLoss
from utils.evaluator import Evaluator
from utils.misc import convert_state_dict

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, model: nn.Module, im2rec_loss_function: nn.Module, device):
    running_loss = 0.
    instance_count = 0
    model.eval()
    logger.info(f'running test phase')
    retrieval_dir = config.RETRIEVAL_DIR
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0])
    evaluator = Evaluator(out_dir,retrieval_dir)
    for batch in tqdm(dataloader):
        recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
        img_input = batch['image']['data'].to(device)
        out = model(recipe_input,img_input)
        out_recipe = out['recipe_embedding']
        out_img = out['image_embedding']
        loss = im2rec_loss_function(out_img,out_recipe)
        running_loss += loss.item()
        instance_count += len(out_img)
        evaluator(out_img, out_recipe, batch)
    avg_loss = running_loss / instance_count
    logger.info(f'loss = {avg_loss}')
    evaluator.calculate_similarity()
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
    model = create_vlp_model(config,device)
    vlp_path = config.VLP_PATH
    org_state_dict = torch.load(vlp_path,map_location=device)
    new_state_dict = convert_state_dict(org_state_dict)
    model.load_state_dict(new_state_dict)
    loss_function = TripletLoss()
    evaluate(config, dataloader, model, loss_function, device)

if __name__ == '__main__':
    main()
