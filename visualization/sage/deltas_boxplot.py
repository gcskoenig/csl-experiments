"""Plot Confidence Intervals for the differences of the deltas between SAGE
and SAGE_CG if the is conditional independence and corresponding boxplot

One confidence interval per experiment, confidence over nr_orderings"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


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
    help="")

args = parser.parse_args()


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.style.use('ggplot')

# data, model

diffs1 = pd.read_csv(f"results/dag_s_{args.degree/10}/differences_{args.model}.csv")
diffs2 = pd.read_csv(f"results/dag_sm_{args.degree/20}/differences_{args.model}.csv")
diffs3 = pd.read_csv(f"results/dag_m_{args.degree/50}/differences_{args.model}.csv")
diffs4 = pd.read_csv(f"results/dag_l_{args.degree/100}/differences_{args.model}.csv")

list_diffs = [diffs1, diffs2, diffs3, diffs4]

df = pd.DataFrame()
df["DAG$_s$"] = diffs1["all"]
df["DAG$_{sm}$"] = diffs2["all"]
df["DAG$_m$"] = diffs3["all"]
df["DAG$_l$"] = diffs4["all"]

# Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(len(df.columns), 1, figsize=(10, 5))

for i, ax in enumerate(axs.flat):
    ax.boxplot(list_diffs[i], flierprops=red_circle, vert=False)
    ax.set_title(df.columns[i], fontsize=22, fontweight='bold')
    ax.tick_params(axis='x', labelsize=22)
    ax.set_yticklabels([r"$\Delta_{j|S}$"], fontsize=22)
    ax.text(0.9, 0.8, fr"n$_\Delta$={len(list_diffs[i])}", fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #ax.set_xticklabels(r"$\Delta$", fontsize=10)
plt.tight_layout()
#plt.text(0.04, 4.3, f"n={len(diffs1)}", fontsize=16)
#plt.text(0.04, 3.3, f"n={len(diffs2)}", fontsize=16)
#plt.text(0.04, 2.3, f"n={len(diffs3)}", fontsize=16)
#plt.text(0.04, 1.3, f"n={len(diffs4)}", fontsize=16)


#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
# df.plot(kind="box")

# plt.xticks([0, 1, 2, 3], ["$\Delta$"])
plt.savefig(f"plots/sage/deltas_{args.model}_{args.degree}.png", dpi=400, transparent=False)
plt.clf()
