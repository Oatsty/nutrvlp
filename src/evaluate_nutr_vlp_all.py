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
from model.nutr_vlp import create_nutr_vlp_model
from utils.loss import TripletLoss
from utils.evaluator import Evaluator

logger = logging.getLogger()

def evaluate(config, dataloader: DataLoader, model: nn.Module, im2rec_loss_function: nn.Module, device):
    running_loss = 0.
    instance_count = 0
    model.eval()
    logger.info(f'running test phase')
    retrieval_dir = config.RETRIEVAL_DIR
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_all_domains')
    evaluator_all_domains = Evaluator(out_dir,retrieval_dir)
    # out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_nutr_recipe')
    # evaluator_nutr_recipe = Evaluator(out_dir,retrieval_dir)
    # out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_nutr_img')
    # evaluator_nutr_img = Evaluator(out_dir,retrieval_dir)
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_recipe_img')
    evaluator_recipe_img = Evaluator(out_dir,retrieval_dir)
    # out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_nutr_only')
    # evaluator_nutr = Evaluator(out_dir,retrieval_dir)
    out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_recipe_only')
    evaluator_recipe = Evaluator(out_dir,retrieval_dir)
    # out_dir = os.path.join(config.OUT_DIR,config.SAVE_PATH,os.path.splitext(config.RECIPE_MODEL_PATH)[0] + '_img_only')
    # evaluator_img = Evaluator(out_dir,retrieval_dir)
    for batch in tqdm(dataloader):
        recipe_input = {x: batch['recipe'][x].to(device) for x in ['title','ingrs','instrs']}
        img_input = batch['image']['data'].to(device)
        nutr = batch['recipe']['nutr'].to(device)
        out = model(recipe_input,img_input,nutr)
        out_recipe = out['recipe_embedding']
        out_img = out['image_embedding']
        out_nutr = out['nutr_embedding']
        out_all_domains = (out_recipe + out_nutr + out_img) / 3
        out_nutr_recipe = (out_nutr + out_recipe) / 2
        out_nutr_img = (out_nutr + out_img) / 2
        out_recipe_img = (out_recipe + out_img) / 2
        loss = im2rec_loss_function(out_img,out_all_domains)
        running_loss += loss.item()
        instance_count += len(out_img)
        evaluator_all_domains(out_img, out_all_domains, batch)
        # evaluator_nutr_recipe(out_img, out_nutr_recipe, batch)
        # evaluator_nutr_img(out_img, out_nutr_img, batch)
        evaluator_recipe_img(out_img, out_recipe_img, batch)
        # evaluator_nutr(out_img, out_nutr, batch)
        evaluator_recipe(out_img, out_recipe, batch)
        # evaluator_img(out_img, out_img, batch)
    avg_loss = running_loss / instance_count
    logger.info(f'loss = {avg_loss}')
    evaluator_all_domains.calculate_similarity()
    # evaluator_nutr_recipe.calculate_similarity()
    # evaluator_nutr_img.calculate_similarity()
    evaluator_recipe_img.calculate_similarity()
    # evaluator_nutr.calculate_similarity()
    evaluator_recipe.calculate_similarity()
    # evaluator_img.calculate_similarity()
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
    model = create_nutr_vlp_model(config,device)
    model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH,config.RECIPE_MODEL_PATH),map_location=device))
    loss_function = TripletLoss()
    evaluate(config, dataloader, model, loss_function, device)

if __name__ == '__main__':
    main()
