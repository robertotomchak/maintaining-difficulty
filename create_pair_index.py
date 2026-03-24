"""
    Creates a csv with pair from given dataset
    Used to create distance histograms that are consistent betwen epochs and experiments
"""


import os
import random
import pandas as pd

DATA_PATH = "cub-200-2011/test"
TARGET_PATH = "cub-200-2011/test_index.csv"

columns = ["path", "index", "id"]
rows = []

id_idx = 0
for id in os.listdir(os.path.join(DATA_PATH)): 
    for obs in os.listdir(os.path.join(DATA_PATH, id)):
        full_path = os.path.join(DATA_PATH, id, obs)
        rows.append([full_path, len(rows), id_idx])
    id_idx += 1


pd.DataFrame(rows, columns=columns).to_csv(TARGET_PATH, index=False)