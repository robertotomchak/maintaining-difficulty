"""
    Runs an given model, both for training and testing
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
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime

from Datasets import *
from Loss import *
from report import create_hist_dist
from paths import *
from notifier_bot import write_heartbeat

import pandas as pd
import math
import copy
import time

def euclidean_distance(x, y):
    return (torch.sqrt(torch.sum((x - y) ** 2, dim=1, keepdim=True)))


def save_dists(pos_dists, neg_dists, path):
    pos_dists = pos_dists.detach().cpu().numpy()
    neg_dists = neg_dists.detach().cpu().numpy()
    same = [1] * len(pos_dists) + [0] * len(neg_dists)
    pd.DataFrame({"distances": np.append(pos_dists, neg_dists), "same": same}).to_csv(path, index=False)


def save_model(model, path):
    torch.save(model, path)


def save_triplets_data(triplets_info, results_path):
    # turning tensor into numpy array
    for k, v in triplets_info.items():
        triplets_info[k] = v.detach().cpu().numpy()
    pd.DataFrame(triplets_info).to_csv(results_path, index=False)


def getFilteredTriplets(embeddings, labels, alpha, dist_fn, filter="none"):
    device = embeddings.device
    batchSize = embeddings.size(0)
    
    dists = distance_matrix_batched(embeddings, dist_fn, batch_size=64)
    
    labels = labels.view(-1, 1)
    label_equal = (labels == labels.T)
    mask_anchor_positive = label_equal ^ torch.eye(batchSize, dtype=torch.bool, device=labels.device)  # mesma classe, exceto diagonal
    mask_anchor_negative = ~label_equal  # classes diferentes
    
    triplets = []

    # Para cada âncora, vamos usar broadcasting para calcular tudo de uma vez
    for anchor_idx in range(batchSize):
        d_pos_all = dists[anchor_idx][mask_anchor_positive[anchor_idx]]  # distâncias para todos positivos
        pos_idxs = torch.arange(batchSize, device=device)[mask_anchor_positive[anchor_idx]]

        d_neg_all = dists[anchor_idx][mask_anchor_negative[anchor_idx]]  # distâncias para todos negativos
        neg_idxs = torch.arange(batchSize, device=device)[mask_anchor_negative[anchor_idx]]

        if d_pos_all.numel() == 0 or d_neg_all.numel() == 0:
            continue

        # Condições semi-hard para todos positivos de uma vez
        # d_pos_all: [P], d_neg_all: [N]
        # queremos comparar cada d_pos com todos d_neg (P x N)
        d_pos_all = d_pos_all.unsqueeze(1)  # (P,1)
        d_neg_all = d_neg_all.unsqueeze(0)  # (1,N)

        if filter == "none":
            mask = torch.ones((d_pos_all.size(0), d_neg_all.size(1)), device=device, dtype=torch.bool)
        elif filter == "semihard":
            mask = (d_neg_all > d_pos_all) & (d_neg_all < d_pos_all + alpha)
        elif filter == "hard":
            mask = (d_neg_all < d_pos_all + alpha)
        elif filter == "superhard":
            mask = (d_neg_all < d_pos_all) & (d_neg_all < d_pos_all + alpha)

        # Para cada positivo, pega os negativos semi-hard
        for pos_i, pos_idx in enumerate(pos_idxs):
            semi_hard_negatives = neg_idxs[mask[pos_i]]
            if semi_hard_negatives.numel() == 0:
                continue

            for neg_idx in semi_hard_negatives:
                triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))

    return triplets, dists


# trains one epoch
def train(dataloader, model, loss_fn, optimizer, device, hard_filter):
    total_loss = 0
    total_triplets = 0
    hard_triplets = 0
    model.train()
    i = 1
    is_optimized = isinstance(loss_fn, OptimizedTripletMarginLoss)

    # saves the distances between the triplets
    triplets_info = {
        "batch": [],
        "d_ap": [],
        "d_an": [],
        "d_pn": []
    }

    for images, labels in dataloader:
        print(f"\tBATCH (TRAIN): {i}/{len(dataloader)} ({round(100*i/len(dataloader), 2)})%", end="\r")
        images = images.to(device)
        labels = labels.to(device)
        embeddings = model(images)

        triplets, dists = getFilteredTriplets(embeddings, labels, loss_fn.margin, loss_fn.distance_function, filter=hard_filter)
        total_triplets += num_triplets(labels)
        hard_triplets += len(triplets)
        if len(triplets) == 0:
            continue

        a_idx, p_idx, n_idx = zip(*triplets)
        anchors = embeddings[torch.tensor(a_idx)]
        positives = embeddings[torch.tensor(p_idx)]
        negatives = embeddings[torch.tensor(n_idx)]

        if is_optimized:
            if loss_fn.reduction == "mean":
                total_num_triplets = num_triplets(labels)
            else:
                total_num_triplets = 0
            loss = loss_fn(anchors, positives, negatives, num_total_triplets=total_num_triplets)
        else:
            # calculate dists...
            loss = loss_fn(anchors, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # saving data
        triplets_info["batch"].append(torch.full((len(triplets),), i))
        triplets_info["d_ap"].append(dists[torch.tensor(a_idx), torch.tensor(p_idx)])
        triplets_info["d_an"].append(dists[torch.tensor(a_idx), torch.tensor(n_idx)])
        triplets_info["d_pn"].append(dists[torch.tensor(p_idx), torch.tensor(n_idx)])

        i += 1

    # turning python list into tensor
    for k, v in triplets_info.items():
        triplets_info[k] = torch.cat(v, axis=0)

    return total_loss / len(dataloader), triplets_info, (total_triplets - hard_triplets) / total_triplets


# evaluation at the end of epoch
def eval(dataloader, model, loss_fn, device):
    is_optimized = isinstance(loss_fn, OptimizedTripletMarginLoss)
    if is_optimized:
        filter = "hard"
    else:
        filter = "none"
    model.eval()
    total_loss = 0
    i = 1

    # saves the distances between the triplets
    triplets_info = {
        "batch": [],
        "d_ap": [],
        "d_an": [],
        "d_pn": []
    }

    for images, labels in dataloader:
        print(f"\tBATCH (EVAL): {i}/{len(dataloader)} ({round(100*i/len(dataloader), 2)})%", end="\r")
        images = images.to(device)
        labels = labels.to(device)
        embeddings = model(images)

        triplets, dists = getFilteredTriplets(embeddings, labels, loss_fn.margin, loss_fn.distance_function, filter)
        if len(triplets) == 0:
            continue

        a_idx, p_idx, n_idx = zip(*triplets)
        anchors = embeddings[torch.tensor(a_idx)]
        positives = embeddings[torch.tensor(p_idx)]
        negatives = embeddings[torch.tensor(n_idx)]

        if is_optimized:
            if loss_fn.reduction == "mean":
                total_num_triplets = num_triplets(labels)
            else:
                total_num_triplets = 0
            loss = loss_fn(anchors, positives, negatives, num_total_triplets=total_num_triplets)
        else:
            # calculate dists...
            loss = loss_fn(anchors, positives, negatives)
        total_loss += loss.item()

        # saving data
        triplets_info["batch"].append(torch.full((len(triplets),), i))
        triplets_info["d_ap"].append(dists[torch.tensor(a_idx), torch.tensor(p_idx)])
        triplets_info["d_an"].append(dists[torch.tensor(a_idx), torch.tensor(n_idx)])
        triplets_info["d_pn"].append(dists[torch.tensor(p_idx), torch.tensor(n_idx)])

        i += 1

    # turning python list into tensor
    for k, v in triplets_info.items():
        triplets_info[k] = torch.cat(v, axis=0)

    return total_loss / len(dataloader), triplets_info


def run(device, model, optimizer, scheduler, train_loss_fn, eval_loss_fn, epochs, dataloader, patience, 
        results_path, hard_filter, heartbeat_data):
    columns = ["epoch", "train_loss", "eval_loss", "time", "margin"]
    rows = []

    exp_name = f"{heartbeat_data['exp']} ({heartbeat_data['execution']})"

    # train
    best_params = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = -1
    patience_count = 0
    for j in range(1, epochs+1):
        print("-"*30)
        print(f"EPOCH {j}/{epochs} ({exp_name})")
        heartbeat_data["epoch"] = f"{j} / {epochs} (train)"
        write_heartbeat(heartbeat_data)

        start = time.time()

        filter = hard_filter.next()
        if patience_count > patience:
            break

        # shuffle training data, to create different triplets
        dataloader["train"].dataset.shuffle()
        train_loss, train_triplets, easy_percent = train(dataloader["train"], model, train_loss_fn, optimizer, 
                                            device, filter)
        with torch.no_grad():
            # eval
            eval_loss, eval_triplets = eval(dataloader["eval"], model, eval_loss_fn, device)
        # saving best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_params = copy.deepcopy(model.state_dict())
            best_epoch = j
            patience_count = 0
        else:
            patience_count += 1

        if scheduler:
            scheduler.step()
        if isinstance(train_loss_fn, AdaptiveTripletMarginLoss):
            train_loss_fn.step(easy_percent)
        end = time.time()

        save_triplets_data(train_triplets, os.path.join(results_path, "train", f"epoch{j}.csv"))
        save_triplets_data(eval_triplets, os.path.join(results_path, "eval", f"epoch{j}.csv"))
        save_model(model.state_dict(), os.path.join(results_path, "train", f"model{j}.pt"))
        # print some stats
        print(f"\tTRAIN LOSS = {train_loss}")
        print(f"\tEVAL LOSS = {eval_loss}")
        if scheduler:
            print(f"\tLEARNING RATE = {scheduler.get_last_lr()[0]}")
        print(f"\tMARGIN: {train_loss_fn.margin}")
        print(f"\tEASY PERCENT: {round(100*easy_percent)}%")
        print(f"\tFILTER: {filter}")
        print(f"\tTIME = {time.time() - start} sec")
        expected_time_left = int((epochs - j) * (end - start))
        hours = expected_time_left // 3600
        minutes = (expected_time_left % 3600) // 60
        print(f"\tEXPECTED TIME LEFT:{hours}h{minutes}min")
        print(f"\tBEST EPOCH = {best_epoch} (loss = {best_loss})")
        print("-"*30)

        rows.append([j, train_loss, eval_loss, end - start, train_loss_fn.margin])

    # save general results
    pd.DataFrame(rows, columns=columns).to_csv(os.path.join(results_path, "loss.csv"), index=False)
    torch.save(best_params, os.path.join(results_path, "best_model.pth"))
    torch.save(model.state_dict(), os.path.join(results_path, "last_model.pth"))
    return best_params

    
def get_mean_std(data_path):
    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        # images shape: (batch_size, channels, height, width)
        batch_samples = images.size(0)  # batch size (may be smaller at end)
        images = images.view(batch_samples, images.size(1), -1)  # flatten H and W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std
