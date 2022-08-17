"""Create a dictionary with all targets:
- Sample targets for synthetic datasets
- add targets for semi-synthetic datasets"""
import random
import pandas as pd
from utils import create_folder
import os
import pickle
from utils import convert_amat

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
            # check if degree of target node > 0 (if not, resample target)
            amat = pd.read_csv(f"data/true_amat/{data}.csv")
            if data in ["alarm", "sachs"]:
                amat = convert_amat(amat)
            else:
                amat = convert_amat(amat, col_names=True)
            degree = 0
            while degree < 1:
                target = random.choice(col_names)
                degree = amat[target].sum() + amat.loc[target, :].sum()
                print("degree of", {data}, degree)
            target_dict[data] = target
            target_df[data] = [target]
    create_folder("data/temp/")
    # save dictionary
    with open('data/temp/targets.pkl', 'wb') as f:
        pickle.dump(target_dict, f)


if __name__ == "__main__":
    main()
