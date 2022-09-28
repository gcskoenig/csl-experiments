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
    orderings = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv", index_col="ordering")
    values = pd.read_csv(f"results/{args.data}/sage_scores_{args.data}_{args.model}.csv")
    adj_mat = pd.read_csv(f"data/true_amat/{args.data}.csv")
    adj_mat = convert_amat(adj_mat, col_names=True)
    target = str(pd.read_csv(f"results/{args.data}/model_details_{args.data}_{args.model}.csv")["target"][0])
    # create graph
    g = nx.DiGraph(adj_mat)
    # initiate vector for the estimated value for the summand when a d-separation is found
    diffs = []

    for i in range(int(len(orderings)/5)):
        # if foi is d-sep from target given features before foi in current ordering set value to zero
        for j in range(5):
            ordering = list(orderings.loc[i][orderings.loc[i]["sample"] == j].loc[i])[1]
            if str(ordering) == 'nan':
                pass
            else:
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
                        idx = i * 5 + j
                        diffs.append(values.loc[idx, J])
                        values.loc[idx, J] = 0
                    else:
                        pass

    values.to_csv(f"results/{args.data}/csl_sage_scores_{args.data}_{args.model}.csv", index=False)
    values_o = values.groupby('ordering').mean().drop("sample", axis=1).reset_index().drop("i", axis=1)
    values_o.to_csv(f"results/{args.data}/csl_sage_o_{args.data}_{args.model}.csv", index=False)
    values_r = values.groupby('sample').mean().drop("ordering", axis=1)
    values_r["sample"] = [0, 1, 2, 3, 4]
    values_r.rename(columns={'i': 'sample'}, inplace=True)

    values_r.to_csv(f"results/{args.data}/csl_sage_r_{args.data}_{args.model}.csv", index=False)

    values = values.drop("ordering", axis=1)
    values = values.drop("sample", axis=1)
    values = values.drop("i", axis=1)

    means_stds = pd.DataFrame()
    means_stds["means"] = values.mean()
    means_stds["std"] = values.std()
    means_stds = means_stds.reset_index()
    means_stds.columns = ['feature', 'mean', 'std']
    means_stds.to_csv(f"results/{args.data}/csl_sage_{args.data}_{args.model}.csv", index=False)

    differences = pd.DataFrame()
    differences["diffs"] = diffs
    differences.to_csv(f"results/{args.data}/differences.csv", index=False)


if __name__ == "__main__":
    main(arguments)