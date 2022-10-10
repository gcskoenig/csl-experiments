"""Plot Confidence Intervals for the differences of the deltas between SAGE
and SAGE_CG if the is conditional independence and corresponding boxplot

One confidence interval per experiment, confidence over nr_orderings"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.style.use('ggplot')

# data, model

diffs1 = pd.read_csv("results/dag_s_0.2/differences_lm.csv")
diffs2 = pd.read_csv("results/dag_sm_0.1/differences_lm.csv")
diffs3 = pd.read_csv("results/dag_m_0.04/differences_lm.csv")
diffs4 = pd.read_csv("results/dag_l_0.02/differences_lm.csv")

df = pd.DataFrame()
df["DAG$_s$"] = diffs1["all"]
df["DAG$_{sm}$"] = diffs2["all"]
df["DAG$_m$"] = diffs3["all"]
df["DAG$_l$"] = diffs4["all"]

# Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(len(df.columns), 1, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:, i], flierprops=red_circle, vert=False)
    ax.set_title(df.columns[i], fontsize=22, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.set_yticklabels([r"$\Delta_{sage}$"], fontsize=22)
    #ax.set_xticklabels(r"$\Delta$", fontsize=10)
plt.tight_layout()


#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
# df.plot(kind="box")

# plt.xticks([0, 1, 2, 3], ["$\Delta$"])
plt.savefig(f"plots/sage/deltas_bp_def.png", dpi=400, transparent=False)
plt.clf()
