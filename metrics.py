"""
    Calculates metrics and creates graphics
"""

import json
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, adjusted_rand_score
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from get_config import *
from Datasets import *
from paths import *
from run import *
from notifier_bot import write_heartbeat

DEVICE = "cuda:1"


def euclidean_distance(x, y):
    return (torch.sqrt(torch.sum((x - y) ** 2, dim=1, keepdim=True)))

def dist_max_matrix(x, y):
    return (x[:, None, :] - y[None, :, :]).abs().amax(dim=-1)


def pos_neg_dists(embeds, labels):
    dists = torch.cdist(embeds, embeds, p=2)
    
    labels = labels.view(-1, 1)
    label_equal = (labels == labels.T)
    mask_same = label_equal ^ torch.eye(len(labels), dtype=torch.bool, device=labels.device)  # ignores itself
    mask_diff = ~label_equal

    pos = dists[mask_same]
    neg = dists[mask_diff]

    return pos.mean().item(), neg.mean().item()


def get_pairs_dists(embeds, df_pairs, dist_fn):
    idx1 = list(df_pairs["image1"])
    idx2 = list(df_pairs["image2"])
    same = df_pairs["same"]

    embeds1 = embeds[idx1]
    embeds2 = embeds[idx2]
    dists = dist_fn(embeds1, embeds2).detach().cpu().numpy().reshape(-1)
    return dists, same


# creates a histogram of the distance and saves to path
def create_hist_dist(dist_same, dist_diff, title, path):
    bins = np.linspace(0, 2, 20)

    # Plot stacked histogram
    counts, bins, patches = plt.hist(
        [dist_same, dist_diff],
        bins=bins,
        stacked=True,
        label=["Same", "Diff"],
        color=["grey", "white"],       # base fill
        edgecolor="grey"
    )

    # Add hatching to the second class (Class B)
    for patch in patches[1]:
        patch.set_hatch("///")   # hatch style

    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(path)
    plt.close()


# returns number of total, easy, semihard and superhard triplets
def count_types_triplets(df, margin, swap=False):
    if swap:
        df["d_n"] = np.minimum(df["d_an"], df["d_pn"])
    else:
        df["d_n"] = df["d_an"]
    mask_hard = df["d_ap"] - df["d_n"] + margin > 0

    hard = df[mask_hard]
    mask_semihard = hard["d_n"] > hard["d_ap"]

    count_semihard = mask_semihard.sum()

    return len(df), len(df) - mask_hard.sum(), count_semihard, mask_hard.sum() - count_semihard


def calc_best_threshold(fpr, tpr, thresholds):
    dists = fpr ** 2 +(tpr - 1) ** 2
    return thresholds[dists.argmin()]


def load_model(model_path, data_dict):
    model_params = torch.load(model_path, weights_only=True)
    model = get_model(data_dict["model"], data_dict["output_shape"], False)
    model.load_state_dict(model_params)
    return model


def get_test_embeds(model, dataloader, device):
    model.to(device)
    model.eval()
    all_embeds = []
    all_ids = []
    i = 0
    for images, idxs, ids in dataloader:
        print(f"\tBATCH (TEST): {i}/{len(dataloader)} ({round(100*i/len(dataloader), 2)})%", end="\r")
        i += 1
        images = images.to(device)

        embeds = model(images)
        all_embeds.append(embeds.detach().cpu())
        all_ids.append(ids)
    all_embeds = torch.cat(all_embeds, axis=0)
    all_ids = torch.cat(all_ids, axis=0)
    return {"embeds": all_embeds, "labels": all_ids}

def recall_k(embeds, labels, k=1, dist_type="max"):
    # filtering only labels that appear more than once
    unique, counts = torch.unique(labels, return_counts=True)
    useful_labels = unique[counts > 1]
    mask = torch.isin(labels, useful_labels)
    embeds = embeds[mask]
    labels = labels[mask]

    if dist_type == "max":
        dists = dist_max_matrix(embeds, embeds)
    else:
        raise Exception("YOU DID NOT IMPLEMENT FOR OTHER FUNCTIONS")
    # exclude itself
    dists.fill_diagonal_(float("inf"))

    knn = dists.topk(k, largest=False).indices
    # hists[i] = 1 if knn contains same label
    hits = (labels[knn] == labels[:, None]).any(dim=1)

    return hits.float().mean().item()


def get_test_dataloader(data_dict):
    data_path = os.path.join(PATH_DATASETS, data_dict["data_path"])
    # transforms
    mean, std = get_mean_std(os.path.join(data_path, "train"))
    if "mnist" in data_path:
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
    dataset = IndividualDataset(os.path.join(data_path, "test_index.csv"), trans)
    dataloader = DataLoader(dataset, batch_size=data_dict["test_batch"], 
                                    num_workers=data_dict["num_workers"], shuffle=False)
    return dataloader


# computes and saves most useful metrics
def analyse(path_results, heartbeat_data):
    # gets parameters from config file
    with open(os.path.join(path_results, "config.json"), "r") as file:
        config = json.load(file)
    swap = "swap" in config["loss"]
    dist_fn = get_distance(config["distance"])

    # pair datasets for test metrics
    df_test_pairs = pd.read_csv(os.path.join(PATH_DATASETS, config["data_path"], "test_pairs.csv"))

    # loss data
    df_loss = pd.read_csv(os.path.join(path_results, "loss.csv"))
    margins = list(df_loss["margin"])

    # triplets data
    triplets = {
        "epoch": [],
        "total": [],
        "easy": [],
        "semihard": [],
        "superhard": [],
        "hard": []
    }

    # test dataloader
    dataloader = get_test_dataloader(config)
    # for each epoch
    for i in range(1, len(df_loss)+1):
        heartbeat_data["epoch"] = f"{i} / {len(df_loss)} (test)"
        write_heartbeat(heartbeat_data)

        print(f"\t{i}/{len(df_loss)} ({round(100*i/len(df_loss), 2)}%)", end="\r")
        # TRAIN DATA
        train_df = pd.read_csv(os.path.join(path_results, "train", f"epoch{i}.csv"))
        # train triplets
        total, easy, semihard, superhard = count_types_triplets(train_df, margins[i-1], swap)
        triplets["epoch"].append(i)
        triplets["total"].append(total)
        triplets["easy"].append(easy)
        triplets["semihard"].append(semihard)
        triplets["superhard"].append(superhard)
        triplets["hard"].append(semihard + superhard)
    
    # best epoch
    best_epoch_data = {}
    best = df_loss.iloc[df_loss["eval_loss"].argmin()].name
    # EVAL
    eval_df = pd.read_csv(os.path.join(path_results, "eval", f"epoch{best}.csv"))
    # calcule eval auc and best threshold
    eval_dists = np.concatenate((eval_df["d_ap"], eval_df["d_an"]))
    eval_same = np.concatenate((np.ones(len(eval_df)), np.zeros(len(eval_df))))
    fpr, tpr, thresholds = roc_curve(eval_same, eval_dists, pos_label=0)
    best_epoch_data["val_auc"] = auc(fpr, tpr)
    threshold = calc_best_threshold(fpr, tpr, thresholds)
    best_epoch_data["threshold"] = threshold

    # TEST
    model = load_model(os.path.join(path_results, "train", f"model{best}.pt"), config)
    embeds_info = get_test_embeds(model, dataloader, DEVICE)
    test_dists, test_same = get_pairs_dists(embeds_info["embeds"], df_test_pairs, dist_fn)
    # test metrics
    fpr, tpr, _ = roc_curve(test_same, test_dists, pos_label=0)
    best_epoch_data["test_auc"] = auc(fpr, tpr)
    pred = test_dists <= threshold
    acc = accuracy_score(test_same, pred)
    best_epoch_data["test_acc"] = acc
    #best_epoch_data["recall@5"] = recall_k(embeds_info["embeds"], embeds_info["labels"], k=5)

    # save data
    with open(os.path.join(path_results, "best_epoch.json"), "w") as file:
        json.dump(best_epoch_data, file, indent=4)
    pd.DataFrame(triplets).to_csv(os.path.join(path_results, "triplets.csv"), index=False)
    print()



def main():
    for res in get_experiments():
        name, times = res
        for i in range(times):
            path = name + str(i)
            print(f"Creating results for {path}")
            path = os.path.join(PATH_RESULTS, path)
            heartbeat_data = {"exp": name, "execution": str(i)}
            analyse(path, heartbeat_data)


if __name__ == "__main__":
    main()
    
