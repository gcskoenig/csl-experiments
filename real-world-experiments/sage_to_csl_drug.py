"""Convert results from SAGE Estimation to csl-sage"""

import pandas as pd
import networkx as nx
import argparse
from utils import convert_amat, create_folder
import regex as re

parser = argparse.ArgumentParser(description="SAGE to CSL SAGE")

parser.add_argument(
    "-a",
    "--alg",
    type=str,
    default="tabu",
    help="hc or tabu")

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    help="")

arguments = parser.parse_args()


def main(args):
    orderings = pd.read_csv(f"real-world-experiments/results/sage/order_sage_runs5.csv", index_col="ordering")
    values = pd.read_csv(f"real-world-experiments/results/sage/sage_scores_runs5.csv")
    values_copy = values
    adj_mat = pd.read_csv(f"real-world-experiments/results/{args.alg}/drug_consumption.csv")
    adj_mat = convert_amat(adj_mat)
    print(adj_mat)
    target = 'Nicotine'
    # create graph
    g = nx.DiGraph(adj_mat)

    # initiate vector for the estimated value for the summand when a d-separation is found
    diffs = []

    for i in range(int(len(orderings)/args.runs)):
        # if foi is d-sep from target given features before foi in current ordering set value to zero
        for j in range(args.runs):
            try:
                ordering = list(orderings.loc[i][orderings.loc[i]["sample"] == j].loc[i])[1]
            except:
                ordering = "nan"
            if str(ordering) == 'nan':
                pass
            else:
                ordering = re.sub("\n", "", ordering)
                ordering = list(ordering.split(" "))
                ordering[0] = ordering[0][1:]
                ordering[-1] = ordering[-1][:-1]
                for k in range(len(ordering)):
                    ordering[k] = ordering[k][1:-1]
                for m in range(len(ordering)):
                    J = ordering[m]
                    if m == 0:
                        d_sep = nx.d_separated(g, {J}, {target}, set())
                    else:
                        C = set(ordering[0:m])
                        d_sep = nx.d_separated(g, {J}, {target}, C)

                    if d_sep:
                        idx = i * args.runs + j
                        diffs.append(values.loc[idx, J])
                        values.loc[idx, J] = 0
                    else:
                        pass

    values.to_csv("real-world-experiments/results/sage/csl_sage_scores_runs5.csv", index=False)
    values_o = values.groupby('ordering').mean().drop("sample", axis=1).reset_index().drop("i", axis=1)
    values_o.to_csv("real-world-experiments/results/sage/csl_sage_o_runs5.csv", index=False)
    values_r = values.groupby('sample').mean().drop("ordering", axis=1)
    values_r["sample"] = [0, 1, 2, 3, 4]
    values_r.rename(columns={'i': 'sample'}, inplace=True)

    values_r.to_csv("real-world-experiments/results/sage/csl_sage_r_runs5.csv", index=False)

    values = values.drop("ordering", axis=1)
    values = values.drop("sample", axis=1)
    values = values.drop("i", axis=1)

    means_stds = pd.DataFrame()
    means_stds["means"] = values.mean()
    means_stds["std"] = values.std()
    means_stds = means_stds.reset_index()
    means_stds.columns = ['feature', 'mean', 'std']
    means_stds.to_csv("real-world-experiments/results/sage/csl_sage_runs5.csv", index=False)

    differences = pd.DataFrame()
    differences["all"] = diffs
    differences.to_csv("real-world-experiments/results/sage/differences.csv", index=False)
    print(values_copy)
    values_copy = values_copy.loc[:, ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore',
                              'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']]

    no_deltas = 12*2590 - values_copy.isna().sum().sum()
    skipped = len(differences)
    share = skipped/no_deltas

    print(12*2590, no_deltas, skipped, share)

if __name__ == "__main__":
    main(arguments)
