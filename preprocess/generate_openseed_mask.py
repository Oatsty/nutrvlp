import pickle
import sys
import os
from pathlib import Path

# sys.path.append('/home/parinayok/food.com_net/src')
sys.path.append('/home/parinayok/food.com_net/src/')

import torch
from tqdm import tqdm
import init_config as init_config
from dataset import create_recipe_retrieval_data

sys.path.append('/home/parinayok/food.com_net/src/OpenSeeD')
from seg_openseed import OpenSeeDSeg


def main():
    save_path = '/srv/datasets2/recipe1m+/utfood_3.10/masks.pt'

    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    _, config = init_config.get_arguments()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    openseed_seg = OpenSeeDSeg(device)
    phases = ['train', 'val', 'test']
    dataloader = {phase: create_recipe_retrieval_data(config, phase) for phase in phases}
    masks_dict = {}
    for phase in ["train", 'val', "test"]:
        for batch in tqdm(dataloader[phase]):
            img_input = batch['image']['data'].to(device)
            ids = batch['recipe']['ids']
            masks, _, _ = openseed_seg.get_mask(img_input)
            masks = masks.bool().cpu()
            for id, m in zip(ids, masks):
                masks_dict[id] = m
            break
        break

    with open(save_path, 'wb') as f:
        pickle.dump(masks_dict, f)


if __name__ == "__main__":
    main()
