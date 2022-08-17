"""Convert results from SAGE Estimation to csl-sage"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import argparse


parser = argparse.ArgumentParser(description="Complete file for model fitting and SAGE estimation")

parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Dataset from ~/data/ folder; string w/o suffix")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="lm",
    help="linear model ('lm') or random forest regression ('rf'); default: 'lm'")

arguments = parser.parse_args()


def main(args):
    orderings = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv")
    values = pd.read_csv(f"results/{args.data}/sage_o_{args.data}_{args.model}.csv")
    if 'ordering' in values.columns:
        values = values.drop(['ordering'], axis=1)
    adj_mat = pd.read_csv("scripts/csl-experiments/data/true_amat/dag_s_0.22222.csv")

    # dag_s amat to zeros and ones
    col_names_str = []
    for k in range(len(adj_mat.columns)):
        col_names_str.append(str(k+1))
    adj_mat.columns = col_names_str
    mapping_rf = {False: 0, True: 1}
    col_names = adj_mat.columns
    for j in col_names:
        adj_mat[j] = adj_mat.replace({j: mapping_rf})[j]
    # modify adjacency matrix for use in networkx package
    adj_mat = adj_mat.set_axis(col_names, axis=0)

    target = "10"

    # create graph
    g = nx.DiGraph(adj_mat)
    predictors = adj_mat.columns.drop(target)

    # initiate vector for the estimated value for the summand when a d-separation is found
    diffs = []

    for i in range(len(orderings)):
        # TODO if it is the first ordering to go through, set value to zero if dsep found, else set value to value
        # from ordering before, also safe the difference b/w current value and value from ordering before
        # make 'ordering string' a list of numerics
        ordering = orderings["ordering"][i]     # this is a string
        ordering = filter(str.isdigit, ordering)
        ordering = " ".join(ordering)
        ordering = ordering.split()     # this is a list of strings
        # if list of integers required, uncomment following two lines
        # ordering = map(int, ordering)
        # ordering = list(ordering)
        for j in range(len(ordering)):
            column_index = int(ordering[j]) - 1
            if j == 0:
                J = set(ordering[0:j])
                C = set(ordering[j:])
                d_sep = nx.d_separated(g, J, {target},  C)
                if d_sep:
                    print("yes")
                    if i == 0:
                        diffs.append(values.iloc[i, column_index])
                        values.iloc[i, column_index] = 0
                    else:
                        diffs.append(values.iloc[i, column_index]-values.iloc[i-1, column_index])
                        values.iloc[i, column_index] = values.iloc[i-1, column_index]   # TODO (cl): set to zero
                else:
                    print("no")

            else:
                J = set(ordering[0:j])
                C = set(ordering[j:])
                d_sep = nx.d_separated(g, J, {target}, C)

                if d_sep:
                    print("yes")
                    if i == 0:
                        diffs.append(values.iloc[i, column_index])
                        values.iloc[i, column_index] = 0
                    else:
                        diffs.append(values.iloc[i, column_index]-values.iloc[i-1, column_index])
                        values.iloc[i, column_index] = values.iloc[i-1, column_index]   # TODO (cl): set to zero
                else:
                    print("no")

    # TODO plot the differences between you know what

    values.to_csv("scripts/csl-experiments/new_results/results/continuous/dag_s/csl_sage_o_dag_s_0.22222_lm.csv")


if __name__ == "__main__":
    main()
