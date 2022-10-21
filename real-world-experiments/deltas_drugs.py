"""Plot Confidence Intervals for the differences of the deltas between SAGE
and SAGE_CG if the is conditional independence and corresponding boxplot

One confidence interval per experiment, confidence over nr_orderings"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.style.use('ggplot')

# data, model

diffs1 = pd.read_csv(f"real-world-experiments/results/sage/differences.csv")['all']

# Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.boxplot(diffs1, flierprops=red_circle, vert=False)
ax.set_title('Drug Consumption', fontsize=22, fontweight='bold')
ax.tick_params(axis='x', labelsize=22)
ax.set_yticklabels([r"$\Delta_{j|S}$"], fontsize=22)
ax.text(0.9, 0.8, fr"n$_\Delta$={len(diffs1)}", fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
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
plt.savefig(f"real-world-experiments/results/sage/deltas.png", dpi=400, transparent=False)
plt.clf()
