import atexit
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from .loss import cos_dist

class Evaluator(nn.Module):

    def __init__(self, out_dir, retrieval_dir):
        super(Evaluator, self).__init__()

        self.nb_matchs_saved = 0
        self.identifiers = {'image': [], 'recipe': []}
        self.save_dir = out_dir
        os.makedirs(self.save_dir,exist_ok=True)
        self.identifiers['ids'] = []
        self.identifiers['img_path'] = []
        self.im2recipe_saved = False
        self.retrieval_dir = os.path.join(retrieval_dir,self.save_dir)
        os.makedirs(self.retrieval_dir,exist_ok=True)
        atexit.register(self.reinitialize)

    def create_file_path(self, identifier):
        file_path = os.path.join(self.retrieval_dir, f'{identifier}.pth')
        return file_path

    def save_activation(self, identifier, activation):
        file_path = self.create_file_path(identifier)
        torch.save(activation, file_path)

    def load_activation(self, identifier):
        file_path = self.create_file_path(identifier)
        return torch.load(file_path)

    def delete_activation(self, identifier):
        file_path = self.create_file_path(identifier)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def reinitialize(self):
        for identifier_image in self.identifiers['image']:
            self.delete_activation(identifier_image)
        for identifier_recipe in self.identifiers['recipe']:
            self.delete_activation(identifier_recipe)
        for identifier_recipe in self.identifiers['ids']:
            self.delete_activation(identifier_recipe)
        for identifier_recipe in self.identifiers['img_path']:
            self.delete_activation(identifier_recipe)
        self.identifiers = {'image': [], 'recipe': [], 'ids': [], 'img_path': []}
        self.nb_matchs_saved = 0

    def forward(self, out_img, out_recipe, batch):
        out = {}
        batch_size = len(out_img)
        for i in range(batch_size):
            # save img embeddings
            identifier = 'img_{}'.format(batch['image']['index'][i])
            self.save_activation(identifier, out_img[i].detach().cpu())
            self.identifiers['image'].append(identifier)

            # save recipe embeddings
            identifier = 'rcp_{}'.format(batch['recipe']['index'][i])
            self.save_activation(identifier, out_recipe[i].detach().cpu())
            self.identifiers['recipe'].append(identifier)

            self.nb_matchs_saved += 1
            self.identifiers['ids'].append(batch['recipe']['ids'][i])
            self.identifiers['img_path'].append(batch['image']['path'][i])

        return out

    def calculate_similarity(self):
        nb_identifiers_image = self.nb_matchs_saved
        nb_identifiers_recipe = self.nb_matchs_saved

        distances = np.zeros((nb_identifiers_image, nb_identifiers_recipe), dtype=float)

        # load
        im_matrix = torch.zeros(nb_identifiers_image, self.load_activation(self.identifiers['image'][0]).size(0))
        rc_matrix = torch.zeros(nb_identifiers_recipe, self.load_activation(self.identifiers['recipe'][0]).size(0))

        ids = []
        img_path = []

        for index in range(self.nb_matchs_saved):
            identifier_image = self.identifiers['image'][index]
            activation_image = self.load_activation(identifier_image)
            im_matrix[index] = activation_image

            identifier_recipe = self.identifiers['recipe'][index]
            activation_recipe = self.load_activation(identifier_recipe)
            rc_matrix[index] = activation_recipe

            ids.append(self.identifiers['ids'][index])
            img_path.append(self.identifiers['img_path'][index])

        distances = cos_dist(im_matrix, rc_matrix)

        im2recipe = np.argsort(distances.numpy(), axis=1)
        recipe2im = np.argsort(distances.numpy(), axis=0)
        with open(os.path.join(self.save_dir, 'saved_ids'), 'wb') as fp:
            pickle.dump(ids, fp)
        with open(os.path.join(self.save_dir, 'img_path'), 'wb') as fp:
            pickle.dump(img_path, fp)

        np.save(os.path.join(self.save_dir,'im2recipe'), im2recipe)
        np.save(os.path.join(self.save_dir, 'recipe2im'), recipe2im)
        np.save(os.path.join(self.save_dir, 'distances'), distances)


        # for addtional evaluation
        r2rdistances = cos_dist(rc_matrix, rc_matrix)
        i2idistances = cos_dist(im_matrix, im_matrix)
        r2r = np.argsort(r2rdistances.numpy(), axis=1)
        i2i = np.argsort(i2idistances.numpy(), axis=1)
        np.save(os.path.join(self.save_dir,'recipe2recipe'), r2r)
        np.save(os.path.join(self.save_dir, 'im2im'), i2i)
        np.save(os.path.join(self.save_dir, 'r2rdistances'), r2rdistances)
        np.save(os.path.join(self.save_dir, 'i2idistances'), i2idistances)

        self.im2recipe_saved = True
