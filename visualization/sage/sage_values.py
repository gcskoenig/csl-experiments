"""Plot SAGE values and their differences to SAGE_CG and SAGE_CG_CD"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import argparse
from utils import fi_hbarplot, hbar_text_position, coord_height_to_pixels


parser = argparse.ArgumentParser(description="SAGE values")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="lm",
    help="lm or rf")

parser.add_argument(
    "-d",
    "--degree",
    type=int,
    default=2,
    help="lm or rf")

parser.add_argument(
    "-t",
    "--top",
    type=int,
    default=5,
    help="top values")


args = parser.parse_args()


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# data, model
datasets1 = [f"dag_s_{args.degree/10}", f"dag_m_{args.degree/50}"]  # must be s und m
datasets2 = [f"dag_sm_{args.degree/20}", f"dag_l_{args.degree/100}"]    # must be sm and l
datasets_list = [datasets1, datasets2]
model = args.model


#ax[0, 2].set_title('SAGE')
#ax[0, 3].set_title(r'SAGE - $d$-SAGE')

for k in range(2):
    # initiate plot, fill it in for loop
    fig, ax = plt.subplots(2, 2, figsize=(5, 7))
    fig.tight_layout(pad=2.1)
    ax[0, 0].set_title('SAGE')
    ax[0, 1].set_title(r'SAGE - $d$-SAGE')
    datasets = datasets_list[k]
    for i in range(2):
        # load data
        data = datasets[i]
        sage = pd.read_csv(f"results/{data}/sage_{data}_{model}.csv")
        dsage = pd.read_csv(f"results/{data}/csl_sage_{data}_{model}.csv")

        try:
            sage_mean = sage["mean"]
            sage_mean = abs(sage_mean)
            sage_ordered = sage_mean.sort_values(ascending=False)
            sage_five = sage_ordered.iloc[0:args.top]
            indices = sage_five.index
            labels = []

            for a in range(args.top):
                labels.append(int(indices[a]+2))
        except:
            sage_mean = sage["mean"]
            sage_mean = abs(sage_mean)
            sage_ordered = sage_mean.sort_values(ascending=False)
            sage_five = sage_ordered.iloc[0:9]
            indices = sage_five.index
            labels = []

            for a in range(9):
                labels.append(int(indices[a]+2))

        sage_r = pd.read_csv(f"results/{data}/sage_r_{data}_{model}.csv")
        dsage_r = pd.read_csv(f"results/{data}/csl_sage_r_{data}_{model}.csv")
        sage_r = sage_r.drop(columns="sample")
        dsage_r = dsage_r.drop(columns=["sample", "sample.1"])

        sage = sage[sage["feature"].isin(labels)]
        dsage = dsage[dsage["feature"].isin(labels)]

        labels_str = []
        for a in range(len(indices)):
            labels_str.append(str(indices[a]+2))

        # differences between sage and sage_cg per run
        diff = sage_r - dsage_r
        # make df of differences w corresponding stds
        diff_sage_dsage = pd.DataFrame(diff.mean(), columns=['mean'])
        diff_sage_dsage['std'] = diff.std()
        diff_sage_dsage['feature'] = diff_sage_dsage.index
        diff_sage_dsage = diff_sage_dsage[diff_sage_dsage["feature"].isin(labels_str)]


        # plots
        if i == 0:
            fi_hbarplot(sage, diff_sage_dsage, ax=ax[0, 0], std=False)
            fi_hbarplot(diff_sage_dsage, diff_sage_dsage, ax=ax[0, 1])

        if i == 1:
            fi_hbarplot(sage, diff_sage_dsage, ax=ax[1, 0], std=False)
            fi_hbarplot(diff_sage_dsage, diff_sage_dsage, ax=ax[1, 1])

        if i == 2:
            fi_hbarplot(sage, diff_sage_dsage, ax=ax[1, 0], std=False)
            fi_hbarplot(diff_sage_dsage, diff_sage_dsage, ax=ax[1, 1])

        if i == 3:
            fi_hbarplot(sage, diff_sage_dsage, ax=ax[1, 2], std=False)
            fi_hbarplot(diff_sage_dsage, diff_sage_dsage, ax=ax[1, 3])


    if k == 0:
        ax[0, 0].set_ylabel('DAG$_{s}$', fontsize=14)
        #ax[0, 2].set_ylabel('DAG$_{sm}$')
        ax[1, 0].set_ylabel('DAG$_{m}$', fontsize=14)
        #ax[1, 2].set_ylabel('DAG$_{l}$')
        ax[1, 0].set_xticks([0.0, 0.2, 0.4])
        ax[1, 0].set_xticklabels([0.0, 0.2, 0.4])
    if k == 1:
        ax[0, 0].set_ylabel('DAG$_{sm}$', fontsize=14)
        #ax[0, 2].set_ylabel('DAG$_{sm}$')
        ax[1, 0].set_ylabel('DAG$_{l}$', fontsize=14)
        #ax[1, 2].set_ylabel('DAG$_{l}$')

    ax[0, 1].yaxis.set_visible(False)
    #ax[0, 3].yaxis.set_visible(False)
    ax[1, 1].yaxis.set_visible(False)
    #ax[1, 3].yaxis.set_visible(False)


    plt.subplots_adjust(hspace=0.3, wspace=0)

    plt.savefig(f"plots/sage/sage_values_{args.model}_{args.degree}_{k+1}.png", dpi=600, transparent=True)
    plt.clf()
