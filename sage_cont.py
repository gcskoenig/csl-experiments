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
    "-o",
    "--orderings",
    type=int,
    default=10000,
    help="Number of orderings in SAGE algorithm; default : 10000",
)

parser.add_argument(
    "-t",
    "--thresh",
    type=float,
    default=0.01,
    help="Threshold for convergence detection; default: 0.01",
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

parser.add_argument(
    "-sas",
    "--sage_seed",
    type=int,
    default=None,
    help="Numpy random seed; default: 1902",
)

parser.add_argument(
    "-e",
    "--extra",
    type=int,
    default=100,
    help="Extra orderings after convergence has been detected, if detection on; default: 100",
)

parser.add_argument(
    "-te",
    "--test",
    type=bool,
    default=False,
    help="test results",
)

parser.add_argument(
    "-run",
    "--run",
    type=int,
    default=None,
    help="test results",
)


arguments = parser.parse_args()

# seed
np.random.seed(arguments.seed)


def main(args):
    # create results folder
    create_folder("results/")
    create_folder(f"results/{args.data}")
    if args.test:
        create_folder(f"results/{args.data}/test/")
        savepath = f"results/{args.data}/test"
    else:
        savepath = f"results/{args.data}"

    # df to store some metadata TODO (cl) do we need to store any other data?
    col_names_meta = ["data", "model", "runtime", "sample size"]
    metadata = pd.DataFrame(columns=col_names_meta)

    # import and prepare data
    df = pd.read_csv(f"data/{args.data}.csv")
    if args.size is not None:
        df = df[0:args.size]
    col_names = df.columns.to_list()
    with open('data/temp/targets.pkl', 'rb') as f:
        target_dict = pickle.load(f)
    target = target_dict[args.data]
    col_names.remove(target)
    X = df[col_names]
    y = df[target]

    # split data for train and test purpose
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.split, random_state=args.seed
    )

    # capture model performance
    col_names_model = ["data", "model", "target", "mse", "r2"]
    model_details = pd.DataFrame(columns=col_names_model)

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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # fill df with info about model
    model_details.loc[len(model_details)] = [args.data, args.model, target, mse, r2]
    if args.test:
        model_details.to_csv(
            f"{savepath}/model_details_{args.data}_{args.model}_{args.run}.csv", index=False
        )
    else:
        model_details.to_csv(
            f"{savepath}/model_details_{args.data}_{args.model}.csv", index=False
        )

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

    # NO partial order
    partial_order = [tuple(X_train.columns)]

    np.random.seed(arguments.sage_seed)

    # track time with time module
    start_time = time.time()
    ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_orderings=args.orderings,
                                         nr_runs=args.runs, detect_convergence=True, thresh=args.thresh,
                                         extra_orderings=args.extra)
    time_sage = time.time() - start_time

    if args.test:

        # save  orderings
        orderings_sage.to_csv(f'{savepath}/order_sage_{args.data}_{args.model}_{args.run}.csv')

        # save SAGE values for every ordering split by runs
        sage_values_scores = ex_d_sage.scores
        sage_values_scores.to_csv(f"{savepath}/sage_scores_{args.data}_{args.model}_{args.run}.csv")

        # save SAGE values for every ordering not split by runs
        sage_values_ordering = ex_d_sage.scores.mean(level=0)
        sage_values_ordering.to_csv(f"{savepath}/sage_o_{args.data}_{args.model}_{args.run}.csv")

        # fi_values for the runs
        ex_d_sage.fi_vals().to_csv(f"{savepath}/sage_r_{args.data}_{args.model}_{args.run}.csv")

        # fi_mean values across runs + stds
        ex_d_sage.fi_means_stds().to_csv(f"{savepath}/sage_{args.data}_{args.model}_{args.run}.csv")

        content = [args.data, args.model, time_sage, args.size]
        # fill evaluation table with current run
        metadata.loc[len(metadata)] = content
        metadata.to_csv(f"{savepath}/metadata_{args.data}_{args.model}_{args.run}.csv", index=False)

    else:

        # save  orderings
        orderings_sage.to_csv(f'{savepath}/order_sage_{args.data}_{args.model}.csv')

        # save SAGE values for every ordering split by runs
        sage_values_scores = ex_d_sage.scores
        sage_values_scores.to_csv(f"{savepath}/sage_scores_{args.data}_{args.model}.csv")

        # save SAGE values for every ordering not split by runs
        sage_values_ordering = ex_d_sage.scores.mean(level=0)
        sage_values_ordering.to_csv(f"{savepath}/sage_o_{args.data}_{args.model}.csv")

        # fi_values for the runs
        ex_d_sage.fi_vals().to_csv(f"{savepath}/sage_r_{args.data}_{args.model}.csv")

        # fi_mean values across runs + stds
        ex_d_sage.fi_means_stds().to_csv(f"{savepath}/sage_{args.data}_{args.model}.csv")

        content = [args.data, args.model, time_sage, args.size]
        # fill evaluation table with current run
        metadata.loc[len(metadata)] = content
        metadata.to_csv(f"{savepath}/metadata_{args.data}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
