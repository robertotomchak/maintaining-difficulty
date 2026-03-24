"""
    Defines the Datasets classes used in this project
"""

import os
import glob
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime

import pandas as pd
import math
import copy
import time

from run import euclidean_distance


# Dataset of images that will be used for triplets
# Used for training and evaluation
class TripletDataset(Dataset):
    def __init__(self, data_path, transform, imgs_per_id, ids_per_batch, neg_porcent=0, mode="train"):
        self.data_path = data_path
        self.transform = transform
        self.imgs_per_id = imgs_per_id
        self.ids_per_batch = ids_per_batch
        self.batch_size = int((1 + neg_porcent) * imgs_per_id * ids_per_batch)
        self.neg_quantity = self.batch_size -  imgs_per_id * ids_per_batch
        
        self.name = []                    # actual name of id
        self.imgs_from_id = []            # images of each id
        self.possible_idxs = []           # have at least imgs_per_id images
        self.negatives_imgs = []          # images from ids that have less than imgs_per_id images
        i = 0
        for name_id in os.listdir(data_path):
            imgs = os.listdir(os.path.join(data_path, name_id))
            self.name.append(name_id)
            self.imgs_from_id.append(imgs)
            if len(imgs) >= imgs_per_id:
                self.possible_idxs.append(i)
            else:
                self.negatives_imgs.extend([(img, i) for img in imgs])
            i += 1
        if mode == "test":
            self.shuffle(seed=42)
        else:
            self.shuffle()


    def __len__(self):
        return len(self.shuffled_imgs)
    
    def __getitem__(self, index):
        img, label = self.shuffled_imgs[index]
        full_path = os.path.join(self.data_path, self.name[label], img)
        return self.img_to_tensor(full_path), label
    
    # must be called at the start of each epoch
    # shuffles the data and prepares for batches
    # seed should only be set for valuation
    def shuffle(self, seed=None):
        if seed is None:
            random_inst = random
        else:
            random_inst = random.Random(seed)
        
        # shuffle identities
        random_inst.shuffle(self.possible_idxs)
        random_inst.shuffle(self.negatives_imgs)

        # shuffle the images of each identity
        for i, imgs in enumerate(self.imgs_from_id):
            random_inst.shuffle(self.imgs_from_id[i])

        # images shuffled, ready to be sampled
        self.shuffled_imgs = []

        # pointers to iterate though the images of each identity
        id_pointers = [0 for _ in range(len(self.name))]

        # ids that have at least imgs_per_id left
        available_ids = self.possible_idxs.copy()

        # ids that have less than imgs_per_id left
        negative_imgs = self.negatives_imgs.copy()
        negatives_pointer = 0

        # each iteration is one batch
        while len(available_ids) >= self.ids_per_batch and negatives_pointer+self.neg_quantity <= len(negative_imgs):
            selected_ids = random_inst.sample(available_ids, self.ids_per_batch)

            for id in selected_ids:
                id_ptr = id_pointers[id]
                for img in self.imgs_from_id[id][id_ptr: id_ptr + self.imgs_per_id]:
                    self.shuffled_imgs.append((img, id))
                id_pointers[id] += self.imgs_per_id

            # adding the negatives
            self.shuffled_imgs.extend(negative_imgs[negatives_pointer:negatives_pointer+self.neg_quantity])
            negatives_pointer += self.neg_quantity

            # update available ids and negative images
            new_available_ids = []
            for id in available_ids:
                if id_pointers[id] + self.imgs_per_id <= len(self.imgs_from_id[id]):
                    new_available_ids.append(id)
                else:
                    negative_imgs.extend([(img, id) for img in self.imgs_from_id[id]])
            available_ids = new_available_ids


    
    def img_to_tensor(self, img_path):
        with Image.open(img_path) as img:
            tensor = self.transform(img)
        return tensor



# Loader for pairs
# Used for testing
class PairDataset(Dataset):
    def __init__(self, pairs_path, transform, mode="test"):
        self.transform = transform
        df = pd.read_csv(pairs_path)
        self.pairs = []
        for i, row in df.iterrows():
            self.pairs.append(list(row))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        pair = self.pairs[index]
        image1 = self.transform(Image.open(pair[0]))
        image2 = self.transform(Image.open(pair[1]))
        same = torch.tensor(pair[2])

        return (image1, image2, same)
    

# Loader for just generating embeddings for each individual image
# Used for testing
class IndividualDataset(Dataset):
    def __init__(self, idx_path, transform):
        self.transform = transform
        df = pd.read_csv(idx_path)
        self.imgs = []
        # path, index, id
        for i, row in df.iterrows():
            self.imgs.append(list(row))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        data = self.imgs[index]
        image = self.transform(Image.open(data[0]))
        index = torch.tensor(data[1])
        id = torch.tensor(data[2])

        return (image, index, id)
    
