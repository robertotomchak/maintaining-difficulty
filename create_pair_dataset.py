"""
    Creates a csv with pair from given dataset
    Used to create distance histograms that are consistent betwen epochs and experiments
"""


import os
import random
import pandas as pd

SIZE = 1000
SAME_PROB = 0.5

DATA_PATH = "cub-200-2011/test"
IDX_PATH = "cub-200-2011/test_index.csv"
TARGET_PATH = "cub-200-2011/test_pairs.csv"

columns = ["image1", "image2", "same"]
rows = []

n_same = int(SAME_PROB * SIZE)
n_diff = SIZE - n_same

df_idx = pd.read_csv(IDX_PATH)
idxs = list(df_idx["index"])
imgs = list(df_idx["path"])
idx_from_img = {k: v for k, v in zip(imgs, idxs)}

multi_img_classes = [x for x in os.listdir(DATA_PATH) if len(os.listdir(os.path.join(DATA_PATH, x))) > 1]

# avoid getting duplicate pairs
previous_pairs = set()

while len(previous_pairs) < n_same:
    class_same = random.choice(multi_img_classes)
    # person must have two or more images
    img1, img2 = random.sample(os.listdir(os.path.join(DATA_PATH, class_same)), 2)
    img1 = os.path.join(DATA_PATH, class_same, img1)
    img2 = os.path.join(DATA_PATH, class_same, img2)
    if (img1, img2) not in previous_pairs and (img2, img1) not in previous_pairs:
        rows.append([idx_from_img[img1], idx_from_img[img2], 1])
        previous_pairs.add((img1, img2))

previous_pairs = set()
while len(previous_pairs) < n_diff:
    class1, class2 = random.sample(os.listdir(DATA_PATH), 2)
    img1 = random.choice(os.listdir(os.path.join(DATA_PATH, class1)))
    img2 = random.choice(os.listdir(os.path.join(DATA_PATH, class2)))
    img1 = os.path.join(DATA_PATH, class1, img1)
    img2 = os.path.join(DATA_PATH, class2, img2)
    if (img1, img2) not in previous_pairs and (img2, img1) not in previous_pairs:
        rows.append([idx_from_img[img1], idx_from_img[img2], 0])
        previous_pairs.add((img1, img2))

# not actually necessary to shuffle
random.shuffle(rows)

pd.DataFrame(rows, columns=columns).to_csv(TARGET_PATH, index=False)