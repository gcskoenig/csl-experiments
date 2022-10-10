"Visualization of runtime data"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import create_folder


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


create_folder("plots/sage/")

# dag s
times1 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
times2 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
times3 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])

# TODO import factor from estimated runtime
factors1 = 0.5
factors2 = 0.5
factors3 = 0.5

# dag sm
timesm1 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timesm2 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timesm3 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])

# TODO import factor from estimated runtime
factorsm1 = 0.5
factorsm2 = 0.5
factorsm3 = 0.5

# dag m
timem1 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timem2 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timem3 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])

# TODO import factor from estimated runtime
factorm1 = 0.5
factorm2 = 0.5
factorm3 = 0.5

# dag l
timel1 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timel2 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])
timel3 = float(pd.read_csv("results/dag_s_0.2/metadata_dag_s_0.2_lm.csv")["runtime"])

# TODO import factor from estimated runtime
factorl1 = 0.5
factorl2 = 0.5
factorl3 = 0.5

#  x data (sample size)
x_ticks = [r"2", r"3", r"4"]
x1 = [0.9, 1.9, 2.9]
x2 = [1.1, 2.1, 3.1]

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
fig.tight_layout()

# DAG_s
axes[0, 0].set_title(r'DAG$_{s}$')
# axes[0].set_xlabel('Sample Size')
b1 = axes[0, 0].bar(x1, [times1, times2, times3], width=0.2, color='b', align='center')
b2 = axes[0, 0].bar(x2, [factors1*times1, factors2*times2, factors3*times3], width=0.2, color='g', align='center')
axes[0, 0].set_xticks([1, 2, 3])
axes[0, 0].set_xticklabels(x_ticks, fontsize=10)

# DAG_sm
axes[0, 1].set_title(r'DAG$_{sm}$')
# axes[0].set_xlabel('Sample Size')
axes[0, 1].bar(x1, [timesm1, timesm2, timesm3], width=0.2, color='b', align='center')
axes[0, 1].bar(x2, [factorsm1*timesm1, factorsm2*timesm2, factorsm3*timesm3], width=0.2, color='g', align='center')
axes[0, 1].set_xticks([1, 2, 3])
axes[0, 1].set_xticklabels(x_ticks, fontsize=10)

# DAG_m
axes[1, 0].set_title(r'DAG$_{m}$')
# axes[0].set_xlabel('Sample Size')
axes[1, 0].bar(x1, [timem1, timem2, timem3], width=0.2, color='b', align='center')
axes[1, 0].bar(x2, [factorm1*timem1, factorm2*timem2, factorm3*timem3], width=0.2, color='g', align='center')
axes[1, 0].set_xticks([1, 2, 3])
axes[1, 0].set_xticklabels(x_ticks, fontsize=10)

# DAG_l
axes[1, 1].set_title(r'DAG$_{l}$')
# axes[0].set_xlabel('Sample Size')
axes[1, 1].bar(x1, [timel1, timel2, timel3], width=0.2, color='b', align='center')
axes[1, 1].bar(x2, [factorl1*timel1, factorl2*timem2, factorl3*timel3], width=0.2, color='g', align='center')
axes[1, 1].set_xticks([1, 2, 3])
axes[1, 1].set_xticklabels(x_ticks, fontsize=10)


legend_labels = [r"SAGE", r"$d$-SAGE"]
fig.legend([b1, b2],     # The line objects
           labels=legend_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, 0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=2, fontsize=8
           )
plt.subplots_adjust(bottom=0.32)
fig.text(0.5, 0.18, 'Sample Size', ha='center')
plt.savefig("plots/sage/cont_runtime.png", dpi=400, transparent=True)
plt.clf()
