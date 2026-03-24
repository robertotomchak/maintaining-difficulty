"""
    Code to get the Labelled Faces in the Wild (LFW) Dataset
    You must have Kaggle API credentials
"""


import os
import shutil
import random
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DELETE_ORIGINAL = False
SPLIT = (0.4, 0.1, 0.5)  # (train, val, test)
DOWNLOAD_PATH = "/home/rsmt23/.cache/kagglehub/datasets/wenewone/cub2002011/versions/7"
DATA_PATH = "cub-200-2011"
IMAGES_PATH = os.path.join(DOWNLOAD_PATH, "CUB_200_2011", "images")


# splits values based on keys and split porcent (of first group)
def split(data, percents):
    groups = sorted(data.keys())
    train = []
    eval = []
    test = []
    temp = []  # stores until can do split
    for k in groups:
        temp.extend(data[k])
        # if can split in a way that gives at least one value to each group
        if min(percents) * len(temp) > 0:
            random.shuffle(temp)
            split1 = int(percents[0] * len(temp))
            split2 = int(percents[1] * len(temp))
            train.extend(temp[:split1])
            eval.extend(temp[split1:split1+split2])
            test.extend(temp[split1+split2:])
            temp = []
        # if can't split, just stores and goes on
    # if there's something remaining, store it aniways
    random.shuffle(temp)
    split1 = int(percents[0] * len(temp))
    split2 = int(percents[1] * len(temp))
    train.extend(temp[:split1])
    eval.extend(temp[split1:split1+split2])
    test.extend(temp[split1+split2:])
    return train, eval, test


# creates directory with data for each dataset
def create_dataset(path, people):
    full_path = os.path.join(DATA_PATH, path)
    for person in people:
        original_path = os.path.join(IMAGES_PATH, person)
        target_path = os.path.join(full_path, person)
        shutil.copytree(original_path, target_path)


def main():
    data = dict()   
    for category in os.listdir(IMAGES_PATH):
        n = len(os.listdir(os.path.join(IMAGES_PATH, category)))
        if n in data:
            data[n].append(category)
        else:
            data[n] = [category]
    train, eval, test = split(data, SPLIT)

    # creating the data directory
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    os.makedirs(DATA_PATH)
    create_dataset("train", train)
    create_dataset("eval", eval)
    create_dataset("test", test)

    if DELETE_ORIGINAL:
        shutil.rmtree(DOWNLOAD_PATH)


if __name__ == "__main__":
    main()





