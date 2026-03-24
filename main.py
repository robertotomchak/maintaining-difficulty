"""
    Main
    Executes experiment based on config.py
"""

import os
import sys
import shutil
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

from Models import NN2
from Datasets import *
from run import *
from report import *
from metrics import analyse
from notifier_bot import write_heartbeat

from get_config import *
from paths import *

EXPERIMENTS_TO_RUN = get_experiments(json_ext=True)

def execute_experiment(data_dict, dataloaders, PATH_RESULTS, heartbeat_data):
    device = data_dict["device"]
    model = data_dict["model"]
    optimizer = data_dict["optimizer"]
    scheduler = data_dict["scheduler"]
    train_loss = data_dict["train_loss"]
    eval_loss = data_dict["eval_loss"]
    epochs = data_dict["epochs"]
    patience = data_dict["patience"]
    hard_filter = data_dict["hard_filter"]

    # training model
    run(device, model, optimizer, scheduler, train_loss, eval_loss, epochs, dataloaders, patience, PATH_RESULTS, 
        hard_filter, heartbeat_data)


def main():
    default_path = os.path.join(PATH_EXPERIMENTS, "default.json")
    print(EXPERIMENTS_TO_RUN)
    for exp, num in EXPERIMENTS_TO_RUN:
        for i in range(num):
            if num == 1:
                idx = ""
            else:
                idx = str(i)
            data_dict, json_content = get_config(os.path.join(PATH_EXPERIMENTS, exp), default_path)
            path_results = exp.removesuffix(".json") + idx
            prepare_env(path_results, json_content)

            # transforms
            mean, std = get_mean_std(os.path.join(data_dict["data_path"], "train"))
            if "mnist" in data_dict["data_path"]:
                trans = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                    trans = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])

            # datasets
            datasets = {
                "train": TripletDataset(os.path.join(data_dict["data_path"], "train"), trans, data_dict["imgs_per_id"],
                                        data_dict["ids_per_batch"], data_dict["neg_porcent"]),
                "eval": TripletDataset(os.path.join(data_dict["data_path"], "eval"), trans, data_dict["imgs_per_id"], 
                                    data_dict["eval_batch"] // data_dict["imgs_per_id"], data_dict["neg_porcent"], 
                                    mode="test"),
            }

            # dataloaders
            dataloaders = {
                "train": DataLoader(datasets["train"], batch_size=data_dict["train_batch"], 
                                    num_workers=data_dict["num_workers"], shuffle=False),
                "eval": DataLoader(datasets["eval"], batch_size=data_dict["eval_batch"], 
                                num_workers=data_dict["num_workers"], shuffle=False),
            }

            # run
            print(f"EXPERIMENT: {path_results}")
            heartbeat_data = {"exp": exp.removesuffix(".json"), "execution": str(i)}
            execute_experiment(data_dict, dataloaders, os.path.join(PATH_RESULTS, path_results), heartbeat_data)
            analyse(os.path.join(PATH_RESULTS, path_results), heartbeat_data)
    write_heartbeat({"exp": "DONE"})


if __name__ == "__main__":
    main()
