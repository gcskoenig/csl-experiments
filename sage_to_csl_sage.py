"""Convert results from SAGE Estimation to d-sage"""

import pandas as pd
import networkx as nx
import argparse
from utils import convert_amat, create_folder
import regex as re

parser = argparse.ArgumentParser(description="")

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

parser.add_argument(
    "-a",
    "--alg",
    type=str,
    default="tabu",
    help="hc or tabu")

parser.add_argument(
    "-o",
    "--obs",
    type=int,
    default=10000,
    help="1000 to 1000000")

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    help="")

arguments = parser.parse_args()


def main(args):
    orderings = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv", index_col="ordering")
    values = pd.read_csv(f"results/{args.data}/sage_scores_{args.data}_{args.model}.csv")
    adj_mat = pd.read_csv(f"bnlearn/results/{args.alg}/{args.data}_{args.obs}_obs.csv")
    adj_mat = convert_amat(adj_mat, col_names=True)

    adj_mat_true = pd.read_csv(f"data/true_amat/{args.data}.csv")
    adj_mat_true = convert_amat(adj_mat_true, col_names=True)
    target = str(pd.read_csv(f"results/{args.data}/model_details_{args.data}_{args.model}.csv")["target"][0])
    # create graph
    g = nx.DiGraph(adj_mat)
    g_true = nx.DiGraph(adj_mat_true)

    # initiate vector for the estimated value for the summand when a d-separation is found
    diffs = []
    diffs_false_positives = []

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
                        d_sep_true = nx.d_separated(g_true, {J}, {target}, set())
                    else:
                        C = set(ordering[0:m])
                        d_sep = nx.d_separated(g, {J}, {target}, C)
                        d_sep_true = nx.d_separated(g_true, {J}, {target}, C)

                    if d_sep and d_sep_true:
                        idx = i * args.runs + j
                        diffs.append(values.loc[idx, J])
                        values.loc[idx, J] = 0
                    elif d_sep and not d_sep_true:
                        idx = i * args.runs + j
                        diffs.append(values.loc[idx, J])
                        diffs_false_positives.append(values.loc[idx, J])
                        values.loc[idx, J] = 0
                    else:
                        pass

    values.to_csv(f"results/{args.data}/csl_sage_scores_{args.data}_{args.model}.csv", index=False)
    values_o = values.groupby('ordering').mean().drop("sample", axis=1).reset_index().drop("i", axis=1)
    values_o.to_csv(f"results/{args.data}/csl_sage_o_{args.data}_{args.model}.csv", index=False)
    values_r = values.groupby('sample').mean().drop("ordering", axis=1)
    sample_list = [i for i in range(len(values_r))]
    values_r["sample"] = sample_list
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
    differences["all"] = diffs
    differences.to_csv(f"results/{args.data}/differences_{args.model}.csv", index=False)

    diffs_fp = pd.DataFrame()
    diffs_fp["false pos"] = diffs_false_positives
    diffs_fp.to_csv(f"results/{args.data}/diffs_fp_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
