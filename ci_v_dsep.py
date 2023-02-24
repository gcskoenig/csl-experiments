"""CI v dsep"""

import pandas as pd
import networkx as nx
import argparse
from utils import convert_amat
from tqdm import tqdm
import regex as re
import networkx as nx
from pingouin import partial_corr
from utils import convert_amat, powerset
import pandas as pd
import time
import argparse
from utils import create_folder
import pickle

parser = argparse.ArgumentParser(description="CI v dsep")

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
    "-rs",
    "--seed",
    type=int,
    default=1902,
    help="Numpy random seed; default: 1902",
)

parser.add_argument(
    "-a",
    "--alg",
    type=str,
    default='tabu',
    help="bnlearn alg",
)

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    default=5,
    help="no of runs used",
)

parser.add_argument(
    "-o",
    "--order",
    type=int,
    default=100,
    help="orderings",
)

arguments = parser.parse_args()


def main(args):

    # for saving purposes
    create_folder(f"results/ci_v_dsep")
    create_folder(f"results/ci_v_dsep/{args.data}")
    savepath = f"results/ci_v_dsep/{args.data}"

    # the train data
    df = pd.read_csv(f"data/{args.data}.csv")

    #
    if args.order is None:
        orderings = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv", index_col="ordering")
    else:
        orderings = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv", index_col="ordering")[0:args.order]

    adj_mat = pd.read_csv(f"bnlearn/results/{args.alg}/{args.data}_10000_obs.csv")
    adj_mat = convert_amat(adj_mat, col_names=True)

    adj_mat_true = pd.read_csv(f"data/true_amat/{args.data}.csv")
    adj_mat_true = convert_amat(adj_mat_true, col_names=True)
    target = str(pd.read_csv(f"results/{args.data}/model_details_{args.data}_{args.model}.csv")["target"][0])
    # create graph
    g = nx.DiGraph(adj_mat)
    g_true = nx.DiGraph(adj_mat_true)

    # list of nodes (strings)
    predictors = list(g_true.nodes)
    # sort list to get consistent results across different graphs learned on same features (when using mc)
    predictors.sort()
    # remove the target from list of predictors
    predictors.remove(target)
    n = len(predictors)

    # initiate vector for the estimated value for the summand when a d-separation is found
    dsep_est = []
    dsep_true = []
    cis = []

    time_dsep = 0
    time_cis = 0
    for i in tqdm(range(int(len(orderings)/args.runs))):
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
                        dsep_start = time.time()
                        dsep_est.append(nx.d_separated(g, {J}, {target}, set()))
                        time_dsep += time.time() - dsep_start
                        dsep_true.append(nx.d_separated(g_true, {J}, {target}, set()))
                        cis_start = time.time()
                        cis.append(0 in partial_corr(df, str(J), str(target), [])["CI95%"]["pearson"])
                        time_cis += time.time() - cis_start
                    else:
                        C = set(ordering[0:m])
                        dsep_start = time.time()
                        dsep_est.append(nx.d_separated(g, {J}, {target}, C))
                        time_dsep += time.time() - dsep_start
                        dsep_true.append(nx.d_separated(g_true, {J}, {target}, C))
                        cis_start = time.time()
                        cis.append(0 in partial_corr(df, str(J), str(target), list(C))["CI95%"]["pearson"])
                        time_cis += time.time() - cis_start

    count_dsep = 0
    count_ci = 0

    for i in range(len(dsep_est)):
        if dsep_est[i] == dsep_true[i]:
            count_dsep += 1
        else:
            pass
    for i in range(len(cis)):
        if cis[i] == dsep_true[i]:
            count_ci += 1
        else:
            pass
    share_dsep = count_dsep/len(dsep_est)
    share_cis = count_ci/len(cis)

    ci_v_dsep = pd.DataFrame(columns=['time dsep', 'time ci', 'acc dsep', 'acc ci', 'total'])
    ci_v_dsep.loc[0] = [time_dsep, time_cis, share_dsep, share_cis, len(dsep_est)]

    if args.order is None:
        ci_v_dsep.to_csv(f'{savepath}/ci_v_dsep.csv')
    else:
        ci_v_dsep.to_csv(f'{savepath}/ci_v_dsep_{args.order}_orderings.csv')


if __name__ == "__main__":
    main(arguments)
