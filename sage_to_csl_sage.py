"""Convert results from SAGE Estimation to csl-sage"""

import pandas as pd
import networkx as nx
import argparse
from utils import convert_amat, create_folder


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
    adj_mat = pd.read_csv(f"scripts/csl-experiments/data/true_amat/{args.data}.csv")

    # modify adjacency matrix for use in networkx package
    adj_mat = convert_amat(adj_mat)

    target = "10"

    # create graph
    g = nx.DiGraph(adj_mat)
    predictors = adj_mat.columns.drop(target)

    # initiate vector for the estimated value for the summand when a d-separation is found
    diffs = []

    for i in range(len(orderings)):
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

    values.to_csv(f"results/{args.data}/csl_sage_o_{args.data}_{args.model}.csv")


if __name__ == "__main__":
    main(arguments)
