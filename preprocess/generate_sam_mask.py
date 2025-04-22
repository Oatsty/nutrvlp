import pickle
import sys
import os
import json

sys.path.append('/home/parinayok/food.com_net/src/')

import torch
from tqdm import tqdm
import init_config as init_config
from dataset import create_recipe_retrieval_data
from model.car import SegmentEncoder

def main():
    save_dir = '/srv/datasets2/recipe1m+/recipe1m_3.10/masks/'

    os.makedirs(os.path.dirname(save_dir),exist_ok=True)

    _, config = init_config.get_arguments()
    config.defrost()
    config.TRAIN.BATCH_SIZE = 1
    config.freeze()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SegmentEncoder(config.MODEL.SEGMENT.NAME,device)
    model.requires_grad_(False)
    phases = ['train', 'val', 'test']
    dataloader = {phase: create_recipe_retrieval_data(config, phase) for phase in phases}
    embs_dict = {}
    # masks_dict = {}
    invalid_ids = []
    for phase in ["train", 'val', "test"]:
        for batch in tqdm(dataloader[phase]):
            img_input = batch['image']['data'].to(device)
            pil_img = batch['image']['pil_image']
            ids = batch['recipe']['ids']
            try:
                embs, masks = model.forward_feature(pil_img, img_input)
                for id, e, m in zip(ids, embs, masks):
                    embs_dict[id] = e
                    # masks_dict[id] = m
            except:
                print(f'invalid id: {ids[0]}')
                invalid_ids.append(ids[0])

    with open(os.path.join(save_dir, 'mask_embs.pt'), 'wb') as f:
        pickle.dump(embs_dict, f)
    # with open(os.path.join(save_dir, 'masks.pt'), 'wb') as f:
    #     pickle.dump(masks_dict, f)
    with open(os.path.join(save_dir, 'invalid_ids.json'),'w') as f:
        json.dump(invalid_ids, f)

if __name__ == "__main__":
    main()
