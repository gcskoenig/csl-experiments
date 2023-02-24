"""Create a dictionary with all targets:
- Sample targets for synthetic datasets
- add targets for semi-synthetic datasets

Sample target was performed on different machines for different datasets, if you want to replicate the study with the
exact same targets, execute python sample_targets.py, if you want to randomly sample targets, use the argument
--sample True
"""
import random
import pandas as pd
from utils import create_folder
import os
import pickle
from utils import convert_amat
import argparse


# set random seed
random.seed(1902)


parser = argparse.ArgumentParser(description="Complete file for model fitting and SAGE estimation")

parser.add_argument(
    "-s",
    "--sample",
    default=False,
    type=bool,
    help="Whether to sample data or create predefined target dictionary")

arguments = parser.parse_args()


def main(args):
    # initiate targets dictionary (asia and hepar have fixed targets)
    target_dict = {"asia": 'dysp', "hepar": "Cirrhosis"}
    target_df = pd.DataFrame()
    target_df["asia"] = ["dysp"]
    target_df["hepar"] = ["Cirrhosis"]
    # datasets for which to sample targets
    path_of_the_directory = "data/"
    ext = ".csv"
    datasets = []
    # if args.sample is True other targets than used in the study may be sampled
    if args.sample:
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
    else:
        target_dict = {"asia": 'dysp', "hepar": "Cirrhosis", "dag_s_0.2": "8", "dag_s_0.3": "1", "dag_s_0.4": "1",
                       "dag_sm_0.1": "17", "dag_sm_0.15": "2", "dag_sm_0.2": "16", "dag_m_0.04": "4",
                       "dag_m_0.06": "32", "dag_m_0.08": "2", "sachs": "Erk", "alarm": "BP",
                       "dag_l_0.02": "4", "dag_l_0.03": "66", "dag_l_0.04": "66"}

    create_folder("data/temp/")
    # save dictionary
    with open('data/temp/targets.pkl', 'wb') as f:
        pickle.dump(target_dict, f)


if __name__ == "__main__":
    main(arguments)

