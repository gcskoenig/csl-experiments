"""
SAGE Evaluation for continuous data for experiments in 'Causal Structure Learning for Efficient SAGE Estimation'

Command line args:
    --data csv-file in folder ~/data/ (string without suffix)
    --model choice between linear model ('lm') and random forest regression ('rf')
    --size slice dataset to df[0:size] (int)
    --runs nr_runs in explainer.sage()
    --orderings nr_orderings in explainer.sage()
    --thresh threshold for convergence detection
"""
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rfi.explainers import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
import time
import argparse
from utils import create_folder
import pickle
import networkx as nx
from utils import convert_amat
import regex as re


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

parser.add_argument(
    "-n",
    "--size",
    type=int,
    default=10000,
    help="Custom sample size to slice df to, default: 20000",
)

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    default=5,
    help="Number of runs for each SAGE estimation; default: 5",
)

parser.add_argument(
    "-no",
    "--no_order",
    type=int,
    default=10,
    help="Orderings to evaluate",
)


parser.add_argument(
    "-s",
    "--split",
    type=float,
    default=0.2,
    help="Train test split; default: 0.2 (test set size)",
)

parser.add_argument(
    "-rs",
    "--seed",
    type=int,
    default=1902,
    help="Numpy random seed; default: 1902",
)


arguments = parser.parse_args()

# seed
np.random.seed(arguments.seed)
random.seed(arguments.seed)


def main(args):
    # create results folder
    create_folder("results/")
    create_folder(f"results/{args.data}")
    savepath = f"results/{args.data}"

    # df to store some metadata TODO (cl) Make this a dataframe for saved runtime
    col_names_meta = ["dsep pos", "dsep neg", "ai_via pos", "ai_via neg"]
    metadata = pd.DataFrame(columns=col_names_meta)

    # import and prepare data
    df = pd.read_csv(f"data/{args.data}.csv")
    if args.size is not None:
        df = df[0:args.size]
    col_names = df.columns.to_list()
    target = str(pd.read_csv(f"results/{args.data}/model_details_{args.data}_{args.model}.csv")["target"][0])
    col_names.remove(target)
    X = df[col_names]
    y = df[target]

    # split data for train and test purpose
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.split, random_state=args.seed
    )

    # fit model
    if args.model == "lm":
        # fit model
        model = LinearRegression()
    if args.model == "rf":
        # fit model
        model = RandomForestRegressor(n_estimators=100)     # TODO (cl) command line argument?

    model.fit(X_train, y_train)
    # model evaluation
    y_pred = model.predict(X_test)

    # model prediction linear model
    def model_predict(x):
        return model.predict(x)

    # set up sampler and decorrelator
    sampler = GaussianSampler(X_train)
    decorrelator = NaiveGaussianDecorrelator(X_train)

    # features of interest
    fsoi = X_train.columns
    # SAGE explainer
    wrk = Explainer(model_predict, fsoi, X_train, loss=mean_squared_error, sampler=sampler,
                    decorrelator=decorrelator)

    # load everything required for the runtime benchmark
    orderings_sage = pd.read_csv(f"results/{args.data}/order_sage_{args.data}_{args.model}.csv", index_col="ordering")
    # adj_mat = pd.read_csv(f'bnlearn/results/tabu/{args.data}.csv')
    adj_mat = pd.read_csv(f'data/true_amat/{args.data}.csv')
    adj_mat = convert_amat(adj_mat, col_names=True)
    g = nx.DiGraph(adj_mat)
    orderings_no = list(range(int(len(orderings_sage)/5)))
    orderings_random = np.random.permutation(orderings_no)
    # init times:
    time_dsep_test_pos = 0
    time_dsep_test_neg = 0
    time_ai_via_pos = 0
    time_ai_via_neg = 0
    orderings_evaluated = 0
    for i in orderings_random:
        j = random.choice(range(0, 5))
        ordering = list(orderings_sage.loc[i][orderings_sage.loc[i]["sample"] == j].loc[i])[1]
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
                    C = set()
                    print(C, J, target)
                    start_dsep = time.time()
                    d_sep = nx.d_separated(g, {J}, {target}, C)
                    time_d_sep = time.time() - start_dsep
                else:
                    C = set(ordering[0:m])
                    print(C, J, target)
                    start_dsep = time.time()
                    d_sep = nx.d_separated(g, {J}, {target}, C)
                    time_d_sep = time.time() - start_dsep

                K = set(ordering) - C
                K = K - {J}
                if d_sep:
                    time_dsep_test_pos += time_d_sep
                    time_ai_via = time.time()
                    ex_d_sage = wrk.ai_via(list({J}), list(C), list(K), X_test, y_test, nr_runs=args.runs)
                    time_ai_via_pos += time.time() - time_ai_via
                else:
                    time_dsep_test_neg += time_d_sep
                    time_ai_via = time.time()
                    ex_d_sage = wrk.ai_via(list({J}), list(C), list(K), X_test, y_test, nr_runs=args.runs)
                    time_ai_via_neg += time.time() - time_ai_via
        orderings_evaluated += 1
        if orderings_evaluated > args.no_order:
            break

    times = [time_dsep_test_pos, time_dsep_test_neg, time_ai_via_pos, time_ai_via_neg]
    metadata.loc[0] = times
    metadata.to_csv(f"{savepath}/ai_via_{args.data}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
