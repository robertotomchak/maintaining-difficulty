"""
    Gets the experiment variables from a json file
"""
import json
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from math import log, e

from Models import *
from Loss import *
from paths import *


def get_model(model_name, output_shape, pre_trained):
    if model_name == "mobilenet":
        return MobileNetV3SmallBackbone(output_shape, pre_trained=pre_trained)
    if model_name == "efficientnet":
        return KaggleEfficientNetB0Backbone(output_shape, pre_trained=pre_trained)
    if model_name == "nn2":
        return NN2(output_shape)
    raise NotImplementedError(f"{model_name} is not a valid model")

def get_distance(distance):
    if distance == "euclidean":
        return nn.PairwiseDistance()
    if distance == "manhattan":
        return nn.PairwiseDistance(p=1)
    if distance == "cosine":
        return lambda a, b: -nn.functional.cosine_similarity(a, b)
    if distance == "max":
        return nn.PairwiseDistance(p=float("inf"))
    if distance == "l2squared":
        return lambda a, b: torch.norm(a - b, p=2) ** 2 
    return NotImplementedError(f"{distance} is not a valid distance function")

def get_loss(loss_name, margin, last_margin, epochs, reduction, distance):
    dist = get_distance(distance)
    if last_margin < 0:
        last_margin = margin
    if "triplet" in loss_name:
        if "optimized" in loss_name:
            train_loss = OptimizedTripletMarginLoss(margin=margin, reduction=reduction, distance_function=dist)
            eval_loss = OptimizedTripletMarginLoss(margin=1.0, reduction="mean", distance_function=dist)
        elif "adaptive" in loss_name:
            train_loss = AdaptiveTripletMarginLoss(margin=margin, reduction=reduction, distance_function=dist)
            eval_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, reduction="mean", distance_function=dist)
        else:
            train_loss = nn.TripletMarginWithDistanceLoss(margin=margin, reduction=reduction, distance_function=dist)
            eval_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, reduction="mean", distance_function=dist)
        return train_loss, eval_loss
    raise NotImplementedError(f"{loss_name} is not a valid loss")

def get_optimizer(optimizer_name, learning_rate, model):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer used in the Efficient Net paper
    if optimizer_name == "efficientnet":
        return optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, momentum=0.9, weight_decay=1e-5)
    raise NotImplementedError(f"{optimizer_name} is not a valid optimizer")

def get_scheduler(scheduler_name, optimizer, initial_lr, epochs):
    if scheduler_name == "none":
        return None
    if scheduler_name == "exponential":
        # we will use last lr = 0.01 * initial_lr
        gamma = e ** (log(0.01) / (epochs-1))
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # optimizer used in the Efficient Net paper
    if scheduler_name == "efficientnet":
        # approx. of lr decay 0.97 after 2.4 epochs
        gamma = 0.97 ** (1 / 2.4)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # using warmup, with some default parameters
    if scheduler_name == "warmup":
        # warmup = 10% of epochs
        warmup_epochs = int(0.1 * epochs)
        normal_epochs = epochs - warmup_epochs
        # warmup_lr = 0.01 * initial_lr
        warmup_lr = 0.01 * initial_lr
        # last_lr = 0.01 * initial_lr
        last_lr = 0.01 * initial_lr
        return LinearWarmupScheduler(optimizer, warmup_lr, initial_lr, last_lr, warmup_epochs, normal_epochs)
    raise NotImplementedError(f"{scheduler_name} is not a valid scheduler")


# gets the experiment variables from a json file
# if uses a default path, uses this as the dict, and change/add the keys/values from json_path
# returns both the python dict and the json content
def get_config(json_path, default_path=None):
    if default_path:
        with open(default_path, "r") as file:
            data_dict = json.load(file)
    else:
        data_dict = dict()
    with open(json_path, "r") as file:
        data_dict.update(json.load(file))

        json_content = data_dict.copy()

    # get the python object for the necessary parts
    data_dict["model"] = get_model(data_dict["model"], data_dict["output_shape"], data_dict["pre_trained"]).to(data_dict["device"])
    train_loss, eval_loss = get_loss(data_dict["loss"], data_dict["margin"], data_dict["last_margin"], data_dict["epochs"], data_dict["reduction"], data_dict["distance"])
    data_dict["train_loss"] = train_loss
    data_dict["eval_loss"] = eval_loss
    data_dict["optimizer"] = get_optimizer(data_dict["optimizer"], data_dict["learning_rate"], data_dict["model"])
    data_dict["scheduler"] = get_scheduler(data_dict["scheduler"], data_dict["optimizer"], data_dict["learning_rate"], data_dict["epochs"])
    data_dict["hard_filter"] = TripletFiltering(data_dict["hard_filter"], data_dict["epochs"])

    return data_dict, json_content


# prepares the environment for the experiments
def prepare_env(exp_name, json_content):
    results_full_path = os.path.join(PATH_RESULTS, exp_name)
    # if the results path already exists, rewrites it
    if os.path.exists(results_full_path):
        shutil.rmtree(results_full_path)
    os.makedirs(results_full_path)
    # adds configs of experiment
    with open(os.path.join(results_full_path, "config.json"), "w") as file:
        json.dump(json_content, file, indent=4)
    # creates directories for the distance results and batch sizes
    os.makedirs(os.path.join(results_full_path, "train"))
    os.makedirs(os.path.join(results_full_path, "eval"))
    os.makedirs(os.path.join(results_full_path, "test"))
    os.makedirs(os.path.join(results_full_path, "triplets"))


# returns experiments to run, and how many times each should be executed
# returns with .json if specified
def get_experiments(path=PATH_RUN_EXPERIMENTS, json_ext=False):
    with open(path, "r") as file:
        experiments = file.read().splitlines()
    experiments_executions = []
    for exp in experiments:
        exp = exp.split(" ")
        if len(exp) == 1:
            experiments_executions.append((exp[0], 1))
        else:
            name = exp[0]
            if exp[1].isnumeric():
                num = int(exp[1])
            else:
                num = 1
            experiments_executions.append((name, num))
    if json_ext:
        experiments_executions = [(name+".json", num) for name, num in experiments_executions]
    return experiments_executions
