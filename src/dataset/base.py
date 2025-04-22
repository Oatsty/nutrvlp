import json
import lmdb
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.common import encode, decode
from .transforms import Compose, ListDictsToDictLists, PadTensors, StackTensors

class DatasetLMDB(Dataset):
    def __init__(self, dir_data, split):
        super(DatasetLMDB, self).__init__()
        self.dir_data = dir_data
        self.dir_lmdb = os.path.join(self.dir_data, 'data_lmdb')

        self.path_envs = {}
        self.path_envs['ids'] = os.path.join(self.dir_lmdb, split, 'ids.lmdb')
        self.envs = {}
        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True, lock=False)
        self.txns = {}
        self.txns['ids'] = self.envs['ids'].begin(write=False, buffers=True)

    def get(self, index, env_name):
        return decode(bytes(self.txns[env_name].get(encode(index))))

    def __len__(self):
        return self.envs['ids'].stat()['entries']


class Images(DatasetLMDB):

    def __init__(self, dir_data, split, transform=None, do_normalize=True, **kwargs):
        super(Images, self).__init__(dir_data, split)

        self.path_envs['numims'] = os.path.join(self.dir_lmdb, split, 'numims.lmdb')
        self.path_envs['impos'] = os.path.join(self.dir_lmdb, split, 'impos.lmdb')
        self.path_envs['imnames'] = os.path.join(self.dir_lmdb, split, 'imnames.lmdb')
        self.path_envs['ims'] = os.path.join(self.dir_lmdb, split, 'ims.lmdb')

        self.envs['numims'] = lmdb.open(self.path_envs['numims'], readonly=True, lock=False)
        self.envs['impos'] = lmdb.open(self.path_envs['impos'], readonly=True, lock=False)
        self.envs['imnames'] = lmdb.open(self.path_envs['imnames'], readonly=True, lock=False)
        self.envs['ims'] = lmdb.open(self.path_envs['ims'], readonly=True, lock=False)

        self.txns['numims'] = self.envs['numims'].begin(write=False, buffers=True)
        self.txns['impos'] = self.envs['impos'].begin(write=False, buffers=True)
        self.txns['imnames'] = self.envs['imnames'].begin(write=False, buffers=True)
        self.txns['ims'] = self.envs['ims'].begin(write=False, buffers=True)

        if not transform:
            scale_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                # transforms.Resize(scale_size),
                transforms.RandomCrop(crop_size),
                # transforms.CenterCrop(size),
                # transforms.ToTensor(),  # divide by 255 automatically
                # transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transform
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.do_normalize = do_normalize
        self.normalize = transforms.Normalize(mean=mean, std=std)


    def format_path_img(self, raw_path):
        sitemap = os.path.basename(os.path.dirname(raw_path))
        basename = os.path.splitext(os.path.basename(raw_path))[0]
        path_img = sitemap + '_' + basename.split('-')[1]
        return path_img

    def __getitem__(self, index):
        # select random image from list of images for that sample
        nb_images = self.get(index, 'numims')
        # im_idx = torch.randperm(nb_images)[0]
        im_idx = 0
        index_img = self.get(index, 'impos')[im_idx]
        path_img = self.format_path_img(self.get(index_img, 'imnames'))
        image_data = self.get(index_img, 'ims')
        # image_data = image_data.convert('RGB')

        #transform
        image_tensor = self.transform(image_data)
        pil_image = TF.to_pil_image(image_tensor)
        if self.do_normalize:
            image_tensor = self.normalize(image_tensor)

        # item = {'data': image_tensor, 'index': index_img, 'pil_image': pil_image}
        item = {'data': image_tensor, 'index': index_img, 'pil_image': pil_image, 'path': path_img}
        # item = {'data': image_data, 'index': index_img, 'path': path_img}

        return item


class Recipes(DatasetLMDB):
    def __init__(self, dir_data, split, path_tokenized_recipes, path_nutrs, path_ingrs, num_ingrs, max_instrs_len, max_ingrs_len, max_instrs, max_ingrs, nutr_names, nutr_scale, path_layer1 = None, path_description = None, path_mask_embed = None, **kwargs):
        super(Recipes, self).__init__(dir_data, split)

        self.path_tokenized_recipes = path_tokenized_recipes
        with open(self.path_tokenized_recipes, 'rb') as f:
            self.tokenized_recipes = json.load(f)

        self.path_nutrs = path_nutrs
        if self.path_nutrs:
            print('load nutr')
            with open(self.path_nutrs) as f:
                self.nutrs = json.load(f)

        self.path_ingrs = path_ingrs
        if self.path_ingrs:
            print('load ingr')
            with open(self.path_ingrs) as f:
                self.ingr_clss = json.load(f)
        self.path_layer1 = path_layer1
        if self.path_layer1:
            print('load layer1')
            with open(self.path_layer1, 'r') as f:
                layer1_ = json.load(f)
            if isinstance(layer1_,list):
                self.layer1 = {data['id']: data for data in layer1_}
            else:
                self.layer1 = layer1_

        self.path_description = path_description
        if self.path_description:
            print('load description')
            with open(self.path_description, 'rb') as f:
                self.descriptions = pickle.load(f)

        self.path_mask_embed = path_mask_embed
        if self.path_mask_embed:
            print('load mask_embed')
            with open(self.path_mask_embed, 'rb') as f:
                self.mask_embed = pickle.load(f)

        self.num_ingrs = num_ingrs
        self.nutr_names = nutr_names

        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True, lock=False)

        self.max_instrs_len = max_instrs_len
        self.max_ingrs_len = max_ingrs_len
        self.max_instrs = max_instrs
        self.max_ingrs = max_ingrs
        self.context_length = 77

        self.nutr_scale = nutr_scale

    def __getitem__(self, index):
        item = {}
        item['index'] = index
        item['ids'] = self.get(index, 'ids')
        recipe = self.tokenized_recipes[item['ids']]

        #get titles
        item['title'] = torch.LongTensor(recipe['title'])

        # get ingrs
        tokenized_ingrs = recipe['ingredients'][:self.max_ingrs]
        tokenized_ingrs = [l[:self.max_ingrs_len] for l in tokenized_ingrs]

        # for all texts (title + ingredients + instructions)
        ingrs_whole_text = torch.cat([torch.LongTensor(l) for l in tokenized_ingrs])

        max_len = max([len(l) for l in tokenized_ingrs])
        tokenized_ingrs = [l + (max_len - len(l))*[0] for l in tokenized_ingrs]
        item['ingrs'] = torch.LongTensor(tokenized_ingrs)

        # get instrs
        tokenized_instrs = recipe['instructions'][:self.max_instrs]

        tokenized_instrs = [l[:self.max_instrs_len] for l in tokenized_instrs]

        # for all texts (title + ingredients + instructions)
        instrs_whole_text = torch.cat([torch.LongTensor(l) for l in tokenized_instrs])

        max_len = max([len(l) for l in tokenized_instrs])
        tokenized_instrs = [l + (max_len - len(l))*[0] for l in tokenized_instrs]
        item['instrs'] = torch.LongTensor(tokenized_instrs)


        #get nutrs
        if self.path_nutrs:
            item['nutr'] = torch.tensor([self.nutrs[item['ids']][nutr_name] / self.nutr_scale[nutr_name] for nutr_name in self.nutr_names])

        #get ingrs
        if self.path_ingrs:
            item['ingr_clss'] = F.one_hot(torch.tensor(self.ingr_clss[item['ids']]),num_classes=self.num_ingrs).sum(0).bool().float()

        if self.path_layer1:
            item['raw'] = self.layer1[item['ids']]

        if self.path_description:
            item['description'] = self.descriptions[item['ids']]

        if self.path_mask_embed:
            # print(item['ids'])
            if item['ids'] in self.mask_embed.keys():
                # print('key founded')
                item['mask_embed'] = self.mask_embed[item['ids']]
                # print(item['mask_embed'])
            else:
                print(f"key not founded: {item['ids']}")
                item['mask_embed'] = torch.randn(512,dtype=torch.float32)

        #all texts title + ingredients + instructions
        item['texts'] = torch.cat([item['title'],ingrs_whole_text,instrs_whole_text])[:self.context_length]
        return item

class FoodData(DatasetLMDB):
    def __init__(self, dir_data, split, path_tokenized_recipes, path_nutrs, path_ingrs, num_ingrs, max_instrs_len, max_ingrs_len, max_instrs, max_ingrs, nutr_names, nutr_scale, **kwargs):
        super(FoodData, self).__init__(dir_data, split)
        self.images_dataset = Images(dir_data, split, **kwargs)
        self.recipes_dataset = Recipes(dir_data, split, path_tokenized_recipes, path_nutrs, path_ingrs, num_ingrs, max_instrs_len, max_ingrs_len, max_instrs, max_ingrs, nutr_names, nutr_scale, **kwargs)

    def __getitem__(self, index):
        item = {}
        item['index'] = index
        item['recipe'] = self.recipes_dataset[index]
        item['image'] = self.images_dataset[index]
        return item


def make_data_loader(dir_data, split, path_tokenized_recipes, path_nutrs, path_ingrs, num_ingrs, nutr_names, nutr_scale, batch_size=100, num_workers=4, shuffle=True, max_instrs_len=20, max_ingrs_len=15, max_instrs=20, max_ingrs=15, **kwargs):
    dataset = FoodData(dir_data, split, path_tokenized_recipes, path_nutrs, path_ingrs, num_ingrs, max_instrs_len, max_ingrs_len, max_instrs, max_ingrs, nutr_names, nutr_scale, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=Compose([
            ListDictsToDictLists(),
            PadTensors(value=0),
            StackTensors()
        ]),
    )
