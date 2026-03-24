"""
    Writes reports of experiments
"""

import json
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# creates a histogram of the distance and saves to path
def create_hist_dist(distances, y_true, title, path):
    dist_same = distances[y_true == True]
    dist_diff = distances[y_true == False]

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


# configurations of experiment
# saves as a json file
def config_report(info, path):
    path = os.path.join(path, "config.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


# data collected during training (as a pd dataframe)
# saves as a csv file
def train_report(data, path):
    path = os.path.join(path, "train.csv")
    data.to_csv(path, index=False)


# data collected during testings (as a pd dataframe)
# saves a csv file with raw data, and a small report (json) with main metrics and threshold info
# returns a dict with main metrics
def test_report(data, threshold, path):
    csv_path = os.path.join(path, "test.csv")
    roc_path = os.path.join(path, "roc.png")
    stats_path = os.path.join(path, "test_stats.json")
    hist_path = os.path.join(path, "distances", "test.png")


    data.to_csv(csv_path, index=False)

    p_same = data[data["true"] == 1]
    p_diff = data[data["true"] == 0]

    # using given threshold
    acc = len(data[data["true"] == data["predict"]]) / len(data)
    val = len(p_same[p_same["predict"] == 1]) / len(p_same)
    far = len(p_diff[p_diff["predict"] == 1]) / len(p_diff)

    # using roc curve info
    fpr, tpr, _ = roc_curve(data["true"], -data["distance"])
    roc_auc = auc(fpr, tpr)

    target_fpr = 1e-3
    idx = np.where(fpr <= target_fpr)[0][-1]
    tpr_at_target = tpr[idx]

    # plotting roc curve
    plt.figure()
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")  # diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig(roc_path)
    plt.close()

    # plotting histogram of distances
    create_hist_dist(data["distance"], data["true"], "Distances (Test)", hist_path)

    stats = {
        "acc": acc,
        "val": val,
        "far": far,
        "threshold": threshold,
        "auc": roc_auc,
        "val@10-3far": tpr_at_target
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    return stats
