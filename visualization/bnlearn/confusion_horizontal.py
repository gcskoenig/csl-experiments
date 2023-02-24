"""Confusion matrices of d-separation inference by causal structure learning algorithms"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import sys
import os
import inspect
import argparse
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import create_folder

create_folder("plots/")
create_folder("plots/bnlearn/")
create_folder("plots/bnlearn/confusion")

# type 1 font
matplotlib.rcParams['text.usetex'] = True
plt.style.use('tableau-colorblind10')

parser = argparse.ArgumentParser(description="confusion matrices")

parser.add_argument(
    "-a",
    "--alg",
    type=str,
    default="tabu",
    help="hc or tabu")

parser.add_argument(
    "-n",
    "--size",
    type=int,
    default=10000,
    help="1000, 10000, 1+e05, 1+e06",
)

parser.add_argument(
    "-t",
    "--latex",
    type=bool,
    default=True,
    help="latex font or not",
)

arguments = parser.parse_args()

dataframe = pd.read_csv("bnlearn/results/graph_evaluation.csv")


def conf(df, data, alg="tabu", size=10000, latex=True):

    if latex:
        # set plt font to standard Latex fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    df = df[df['Method'] == f"{alg}"]

    # sample size
    n = size

    # get rows for the graphs learned with n=10,000
    df = df[df['Graph'] == f"{data}_{n}_obs"]

    # create vectors with true and predicted labels for every graph
    # true labels
    true_labels = []
    for i in range(int(df['TP'].iloc[0]) + int(df['FN'].iloc[0])):
        true_labels.append(1)
    for j in range(int(df['TN'].iloc[0]) + int(df['FP'].iloc[0])):
        true_labels.append(0)

    # predictions
    pred_labels = []
    for i in range(int(df['TP'].iloc[0])):
        pred_labels.append(1)
    for j in range(int(df['FN'].iloc[0])):
        pred_labels.append(0)
    for k in range(int(df['TN'].iloc[0])):
        pred_labels.append(0)
    for m in range(int(df['FP'].iloc[0])):
        pred_labels.append(1)

    return confusion_matrix(true_labels, pred_labels, labels=[0, 1], normalize='all')


def main(args):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(3, 4, figsize=(6, 4))

    disp_dag_s_02 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_s_0.2", alg=args.alg,
                                           size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                            r"$\perp_{\mathcal{G}}$"])

    disp_dag_s_03 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_s_0.3", alg=args.alg,
                                           size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                            r"$\perp_{\mathcal{G}}$"])

    disp_dag_s_04 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_s_0.4", alg=args.alg,
                                           size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                            r"$\perp_{\mathcal{G}}$"])

    disp_dag_sm_01 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_sm_0.1", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_sm_015 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_sm_0.15", alg=args.alg,
                                             size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                              r"$\perp_{\mathcal{G}}$"])

    disp_dag_sm_02 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_sm_0.2", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_m_004 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_m_0.04", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_m_006 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_m_0.06", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_m_008 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_m_0.08", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_l_002 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_l_0.02", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_l_003 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_l_0.03", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    disp_dag_l_004 = ConfusionMatrixDisplay(confusion_matrix=conf(dataframe, data="dag_l_0.04", alg=args.alg,
                                            size=args.size), display_labels=[r"$\not\perp_{\mathcal{G}}$",
                                                                             r"$\perp_{\mathcal{G}}$"])

    # avg degree 2
    disp_dag_s_02.plot(ax=ax[0, 0], cmap="Blues")
    disp_dag_s_02.ax_.set_title(r"DAG$_s$")
    disp_dag_s_02.im_.colorbar.remove()
    disp_dag_s_02.ax_.set_xlabel('')
    disp_dag_s_02.ax_.set_ylabel('')
    ax[0, 0].xaxis.set_visible(False)

    disp_dag_sm_01.plot(ax=ax[0, 1], cmap="Blues")
    disp_dag_sm_01.ax_.set_title(r"DAG$_{sm}$")
    disp_dag_sm_01.im_.colorbar.remove()
    disp_dag_sm_01.ax_.set_xlabel('')
    disp_dag_sm_01.ax_.set_ylabel('')
    ax[0, 1].xaxis.set_visible(False)
    ax[0, 1].yaxis.set_visible(False)

    disp_dag_m_004.plot(ax=ax[0, 2], cmap="Blues")
    disp_dag_m_004.ax_.set_title(r"DAG$_m$")
    disp_dag_m_004.im_.colorbar.remove()
    disp_dag_m_004.ax_.set_xlabel('')
    disp_dag_m_004.ax_.set_ylabel('')
    ax[0, 2].xaxis.set_visible(False)
    ax[0, 2].yaxis.set_visible(False)

    disp_dag_l_002.plot(ax=ax[0, 3], cmap="Blues")
    disp_dag_l_002.ax_.set_title(r"DAG$_l$")
    disp_dag_l_002.im_.colorbar.remove()
    disp_dag_l_002.ax_.set_xlabel('')
    disp_dag_l_002.ax_.set_ylabel('')
    ax[0, 3].xaxis.set_visible(False)
    ax[0, 3].yaxis.set_visible(False)

    # avg degree 3
    disp_dag_s_03.plot(ax=ax[1, 0], cmap="Greens")
    disp_dag_s_03.im_.colorbar.remove()
    disp_dag_s_03.ax_.set_xlabel('')
    disp_dag_s_03.ax_.set_ylabel('')
    ax[1, 0].xaxis.set_visible(False)

    disp_dag_sm_015.plot(ax=ax[1, 1], cmap="Greens")
    disp_dag_sm_015.im_.colorbar.remove()
    disp_dag_sm_015.ax_.set_xlabel('')
    disp_dag_sm_015.ax_.set_ylabel('')
    ax[1, 1].xaxis.set_visible(False)
    ax[1, 1].yaxis.set_visible(False)

    disp_dag_m_006.plot(ax=ax[1, 2], cmap="Greens")
    disp_dag_m_006.im_.colorbar.remove()
    disp_dag_m_006.ax_.set_xlabel('')
    disp_dag_m_006.ax_.set_ylabel('')
    ax[1, 2].xaxis.set_visible(False)
    ax[1, 2].yaxis.set_visible(False)

    disp_dag_l_003.plot(ax=ax[1, 3], cmap="Greens")
    disp_dag_l_003.im_.colorbar.remove()
    disp_dag_l_003.ax_.set_xlabel('')
    disp_dag_l_003.ax_.set_ylabel('')
    ax[1, 3].xaxis.set_visible(False)
    ax[1, 3].yaxis.set_visible(False)

    # avg degree 4
    disp_dag_s_04.plot(ax=ax[2, 0], cmap="Oranges")
    disp_dag_s_04.im_.colorbar.remove()
    disp_dag_s_04.ax_.set_xlabel('')
    disp_dag_s_04.ax_.set_ylabel('')

    disp_dag_sm_02.plot(ax=ax[2, 1], cmap="Oranges")
    disp_dag_sm_02.im_.colorbar.remove()
    disp_dag_sm_02.ax_.set_xlabel('')
    disp_dag_sm_02.ax_.set_ylabel('')
    ax[2, 1].yaxis.set_visible(False)

    disp_dag_m_008.plot(ax=ax[2, 2], cmap="Oranges")
    disp_dag_m_008.im_.colorbar.remove()
    disp_dag_m_008.ax_.set_xlabel('')
    disp_dag_m_008.ax_.set_ylabel('')
    ax[2, 2].yaxis.set_visible(False)

    disp_dag_l_004.plot(ax=ax[2, 3], cmap="Oranges")
    disp_dag_l_004.im_.colorbar.remove()
    disp_dag_l_004.ax_.set_xlabel('')
    disp_dag_l_004.ax_.set_ylabel('')
    ax[2, 3].yaxis.set_visible(False)

    # custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color='b', lw=2, label='2'),
                       Line2D([0], [0], color='g', lw=2, label='3'),
                       Line2D([0], [0], color='orange', lw=2, label='4')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.53, -0.0), title="Average Degree",
               fancybox=True, shadow=True, ncol=3, fontsize=8)

    fig.text(0.05, 0.5, 'True label', va='center', rotation='vertical')

    fig.text(0.53, 0.12, 'Predicted label', ha='center')
    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.22)

    plt.savefig(f"plots/bnlearn/confusion/confusion_cont_{args.alg}_{args.size}_tight.png", dpi=300, transparent=True)
    plt.clf()


if __name__ == "__main__":
    main(arguments)
