"Visualization of runtime data"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import create_folder

create_folder("plots/bnlearn/f1")


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)


parser = argparse.ArgumentParser(description="confusion matrices")

parser.add_argument(
    "-a",
    "--alg",
    type=str,
    default="tabu",
    help="hc or tabu")


parser.add_argument(
    "-t",
    "--latex",
    type=bool,
    default=True,
    help="latex font or not",
)

args = parser.parse_args()



# now for cont
cont = pd.read_csv(f"bnlearn/results/{args.alg}/runtime_data_cont.csv")

#  x data (sample size)
x_ticks = [r"1k", r"10k", r"100k", r"1000k"]
x1 = [0.8, 1.8, 2.8, 3.8]
x2 = [1.0, 2.0, 3.0, 4.0]
x3 = [1.2, 2.2, 3.2, 4.2]


# y data (runtime per graph)
y1_s2 = cont[cont['Graph'] == "dag_s_0.2"]["Runtime in s"]
y1_s3 = cont[cont['Graph'] == "dag_s_0.3"]["Runtime in s"]
y1_s4 = cont[cont['Graph'] == "dag_s_0.4"]["Runtime in s"]

y1_sm2 = cont[cont['Graph'] == "dag_sm_0.1"]["Runtime in s"]
y1_sm3 = cont[cont['Graph'] == "dag_sm_0.15"]["Runtime in s"]
y1_sm4 = cont[cont['Graph'] == "dag_sm_0.2"]["Runtime in s"]

y1_m2 = cont[cont['Graph'] == "dag_m_0.04"]["Runtime in s"]
y1_m3 = cont[cont['Graph'] == "dag_m_0.06"]["Runtime in s"]
y1_m4 = cont[cont['Graph'] == "dag_m_0.08"]["Runtime in s"]

y1_l2 = cont[cont['Graph'] == "dag_l_0.02"]["Runtime in s"]
y1_l3 = cont[cont['Graph'] == "dag_l_0.03"]["Runtime in s"]
y1_l4 = cont[cont['Graph'] == "dag_l_0.04"]["Runtime in s"]


# y data (f1 scores per graph)
df = pd.read_csv("bnlearn/results/graph_eval_new.csv")
y1 = df[df['Method'] == f"{args.alg}"]

f1_s2 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_s2.append(y1[y1["Graph"] == f"dag_s_0.2_{i}_obs"]["F1"])

f1_s3 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_s3.append(y1[y1["Graph"] == f"dag_s_0.3_{i}_obs"]["F1"])

f1_s4 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_s4.append(y1[y1["Graph"] == f"dag_s_0.4_{i}_obs"]["F1"])

f1_sm2 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_sm2.append(y1[y1["Graph"] == f"dag_sm_0.1_{i}_obs"]["F1"])

f1_sm3 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_sm3.append(y1[y1["Graph"] == f"dag_sm_0.15_{i}_obs"]["F1"])

f1_sm4 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_sm4.append(y1[y1["Graph"] == f"dag_sm_0.2_{i}_obs"]["F1"])

f1_m2 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_m2.append(y1[y1["Graph"] == f"dag_m_0.04_{i}_obs"]["F1"])

f1_m3 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_m3.append(y1[y1["Graph"] == f"dag_m_0.06_{i}_obs"]["F1"])

f1_m4 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_m4.append(y1[y1["Graph"] == f"dag_m_0.08_{i}_obs"]["F1"])

f1_l2 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_l2.append(y1[y1["Graph"] == f"dag_l_0.02_{i}_obs"]["F1"])

f1_l3 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_l3.append(y1[y1["Graph"] == f"dag_l_0.03_{i}_obs"]["F1"])

f1_l4 = []
for i in ["1000", "10000", "1e+05", "1e+06"]:
    f1_l4.append(y1[y1["Graph"] == f"dag_l_0.04_{i}_obs"]["F1"])

fig, axes = plt.subplots(2, 2, figsize=(6.5, 6))
#fig.tight_layout()

# DAG_s
axes[0, 0].set_title(r'DAG$_{s}$')
# axes[0].set_xlabel('Sample Size')
# axes[0, 0].set_ylabel('Runtime in s')

axes[0, 0].scatter(x2, f1_s2, color='b', s=0.4)
axes[0, 0].plot(x2, f1_s2, color='b', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 0].scatter(x2, f1_s3, color='g', s=0.4)
axes[0, 0].plot(x2, f1_s3, color='g', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 0].scatter(x2, f1_s4, color='r', s=0.4)
axes[0, 0].plot(x2, f1_s4, color='r', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 0].set_xticks([1, 2, 3, 4])
axes[0, 0].set_xticklabels(x_ticks, fontsize=10)
ax1 = axes[0, 0].twinx()
ax1.bar(x1, y1_s2, width=0.2, color='b', align='center', alpha=0.3)
ax1.bar(x2, y1_s3, width=0.2, color='g', align='center', alpha=0.3)
ax1.bar(x3, y1_s4, width=0.2, color='r', align='center', alpha=0.3)



# DAG_sm
axes[0, 1].set_title(r'DAG$_{sm}$')
# axes[0].set_xlabel('Sample Size')
# axes[0, 1].set_ylabel('Runtime in s')
axes[0, 1].scatter(x2, f1_sm2, color='b', s=0.4)
axes[0, 1].plot(x2, f1_sm2, color='b', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 1].scatter(x2, f1_sm3, color='g', s=0.4)
axes[0, 1].plot(x2, f1_sm3, color='g', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 1].scatter(x2, f1_sm4, color='r', s=0.4)
axes[0, 1].plot(x2, f1_sm4, color='r', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[0, 1].set_xticks([1, 2, 3, 4])
axes[0, 1].set_xticklabels(x_ticks, fontsize=10)
ax2 = axes[0, 1].twinx()
ax2.bar(x1, y1_sm2, width=0.2, color='b', align='center', alpha=0.3)
ax2.bar(x2, y1_sm3, width=0.2, color='g', align='center', alpha=0.3)
ax2.bar(x3, y1_sm4, width=0.2, color='r', align='center', alpha=0.3)

# DAG_sm
axes[1, 0].set_title(r'DAG$_{m}$')
# axes[0].set_xlabel('Sample Size')
#axes[0, 1].set_ylabel('Runtime in s')
axes[1, 0].scatter(x2, f1_m2, color='b', s=0.4)
axes[1, 0].plot(x2, f1_m2, color='b', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 0].scatter(x2, f1_m3, color='g', s=0.4)
axes[1, 0].plot(x2, f1_m3, color='g', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 0].scatter(x2, f1_m4, color='r', s=0.4)
axes[1, 0].plot(x2, f1_m4, color='r', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 0].set_xticks([1, 2, 3, 4])
axes[1, 0].set_xticklabels(x_ticks, fontsize=10)
ax3 = axes[1, 0].twinx()
ax3.bar(x1, y1_m2, width=0.2, color='b', align='center', alpha=0.3)
ax3.bar(x2, y1_m3, width=0.2, color='g', align='center', alpha=0.3)
ax3.bar(x3, y1_m4, width=0.2, color='r', align='center', alpha=0.3)

# DAG_l
axes[1, 1].set_title(r'DAG$_{l}$')
# axes[0].set_xlabel('Sample Size')
# axes[0, 1].set_ylabel('Runtime in s')
axes[1, 1].scatter(x2, f1_l2, color='b', s=0.4)
axes[1, 1].plot(x2, f1_l2, color='b', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 1].scatter(x2, f1_l3, color='g', s=0.4)
axes[1, 1].plot(x2, f1_l3, color='g', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 1].scatter(x2, f1_l4, color='r', s=0.4)
axes[1, 1].plot(x2, f1_l4, color='r', linestyle='-', markevery=x_ticks, linewidth=0.7)
axes[1, 1].set_ylim([0.3, 1])
axes[1, 1].set_xticks([1, 2, 3, 4])
axes[1, 1].set_xticklabels(x_ticks, fontsize=10)
ax4 = axes[1, 1].twinx()
ax4.bar(x1, y1_l2, width=0.2, color='b', align='center', alpha=0.3)
ax4.bar(x2, y1_l3, width=0.2, color='g', align='center', alpha=0.3)
ax4.bar(x3, y1_l4, width=0.2, color='r', align='center', alpha=0.3)


#legend_labels = [r"2", r"3",  r"4"]
#fig.legend([b1, b2, b3],     # The line objects
#           labels=legend_labels,   # The labels for each line
#           loc="lower center",   # Position of legend
#           bbox_to_anchor=(0.5, 0.0),
#           title="Average Degree",  # Title for the legend
#           fancybox=False, shadow=False, ncol=3, fontsize=8
#           )
plt.subplots_adjust(bottom=0.15)
fig.text(0.5, 0.05, 'Sample Size', ha='center')
fig.text(0.03, 0.5, 'F1 score', va='center', rotation='vertical')
fig.text(0.975, 0.5, 'Runtime in s', va='center', rotation='vertical')
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.3)

plt.savefig(f"plots/bnlearn/f1/cont_runtime_{args.alg}.png", dpi=400, transparent=True)
plt.clf()
