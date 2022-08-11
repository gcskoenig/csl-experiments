"""Create a dictionary with all targets:
- Sample targets for synthetic datasets
- add targets for semi-synthetic datasets"""
import random
import pandas as pd
from utils import create_folder
import os
import pickle

# set random seed
random.seed(1902)


def main():
    # initiate targets dictionary (asia and hepar have fixed targets)
    target_dict = {"asia": 'dysp', "hepar": "Cirrhosis"}
    target_df = pd.DataFrame()
    target_df["asia"] = ["dysp"]
    target_df["hepar"] = ["Cirrhosis"]
    # datasets for which to sample targets
    path_of_the_directory = "data/"
    ext = ".csv"
    datasets = []
    for files in os.listdir(path_of_the_directory):
        if files.endswith(ext):
            datasets.append(files[:len(files)-4])
        else:
            continue

    # sample targets and append dictionary
    for data in datasets:
        if data in ["asia", "hepar"]:
            pass
        else:
            df = pd.read_csv(f"data/{data}.csv")
            col_names = df.columns
            target = random.choice(col_names)
            target_dict[data] = target
            target_df[data] = [target]

    create_folder("data/temp/")
    # save dictionary
    with open('data/temp/targets.pkl', 'wb') as f:
        pickle.dump(target_dict, f)


if __name__ == "__main__":
    main()
