"""
    Defines the loss functions and related scheduling
"""

import os
import glob
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime

from math import e, log

# calculates the total number of triplets that can be generated from the given labels
def num_triplets(labels):
    batch_size = len(labels)
    _, counts = torch.unique(labels, return_counts=True)
    triplets_per_anchor = counts * (counts - 1) * (batch_size - counts)
    return triplets_per_anchor.sum().item()

def distance_matrix_batched(embeds, dist_fn, batch_size=256):
    N, D = embeds.shape
    device = embeds.device

    dist_matrix = torch.empty(N, N, device=device)

    for i in range(0, N, batch_size):
        x = embeds[i:i+batch_size]
        B = x.size(0)

        x_exp = x[:, None, :].expand(B, N, D)
        y_exp = embeds[None, :, :].expand(B, N, D)

        x_flat = x_exp.reshape(-1, D)
        y_flat = y_exp.reshape(-1, D)

        dists = dist_fn(x_flat, y_flat)
        dist_matrix[i:i+batch_size] = dists.view(B, N)

    return dist_matrix

# warmup + scheduler linear
class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_lr, initial_lr, last_lr, warmup_epochs, normal_epochs, last_epoch=-1):
        self.warmup_rate = (initial_lr - warmup_lr) / warmup_epochs
        self.normal_rate = (last_lr - initial_lr) / normal_epochs
        self.end_warmup = warmup_epochs
        self.lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.lr
        if self.last_epoch < self.end_warmup:
            self.lr += self.warmup_rate
        else:
            self.lr += self.normal_rate
        return [lr for _ in self.optimizer.param_groups]
            

# possible schedulers for the margin
def create_exp_margin(initial_margin, last_margin, epochs):
    ratio = e ** (log(last_margin / initial_margin) / (epochs-1))
    def next_margin(margin):
        return ratio * margin
    return next_margin

def create_linear_margin(initial_margin, last_margin, epochs):
    ratio = (last_margin - initial_margin) / epochs
    def next_margin(margin):
        return ratio + margin
    return next_margin

# defines the optimized margin loss
class OptimizedTripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean", distance_function=None):
        super(OptimizedTripletMarginLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        if distance_function is None:
            self.dist = nn.PairwiseDistance()
        else:
            self.dist = distance_function

    def forward(self, anchor, positive, negative, num_total_triplets=0):
        pos_dist = self.dist(anchor, positive)
        neg_dist = self.dist(anchor, negative)

        loss = pos_dist - neg_dist
        loss_sum = loss.sum() + self.margin * pos_dist.size(0)
        if self.reduction == "mean":
            return loss_sum / num_total_triplets
        return loss_sum


# defines the margin loss, with increase
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, swap=False, next_margin=None, reduction="mean"):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.next_margin = next_margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        if self.swap:
            second_neg_dist = torch.norm(positive - negative, p=2, dim=1)
            neg_dist = torch.max(neg_dist, second_neg_dist)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
    
    def step(self):
        if self.next_margin:
            self.margin = self.next_margin(self.margin)


# defines the margin loss, with adaptive increase
class AdaptiveTripletMarginLoss(nn.Module):
    def __init__(self, margin=0.0, threshold=0.95, increase=0.05, swap=False, reduction="mean", distance_function=None):
        super(AdaptiveTripletMarginLoss, self).__init__()
        self.margin = margin
        self.threshold = threshold
        self.increase = increase
        self.swap = swap
        self.reduction = reduction
        if distance_function is None:
            self.dist = nn.PairwiseDistance()
        else:
            self.dist = distance_function

    def forward(self, anchor, positive, negative):
        pos_dist = self.dist(anchor, positive)
        neg_dist = self.dist(anchor, negative)

        if self.swap:
            second_neg_dist = self.dist(positive, negative)
            neg_dist = torch.max(neg_dist, second_neg_dist)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
    
    def step(self, easy_percent):
        if easy_percent >= self.threshold:
            self.margin += self.increase



# defines a scheduler for the filtering of triplets
class TripletFiltering():
    def __init__(self, filter=None, epochs=100):
        self.i = 0
        if filter is None:
            self.filters = ["none"] * epochs
        elif "scheduler" in filter:
            change_epochs = [int(x) for x in filter.split(" ")[1:]]
            semihard_range = change_epochs[0] - 1
            hard_range = change_epochs[1] - change_epochs[0]
            superhard_range = epochs - change_epochs[1] + 1
            self.filters = ["semihard"] * semihard_range + ["hard"] * hard_range + ["superhard"] * superhard_range
        else:
            self.filters = [filter] * epochs

    def next(self):
        filter = self.filters[self.i]
        self.i += 1
        return filter
