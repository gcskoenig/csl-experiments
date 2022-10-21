"""Plot SAGE values and their differences to SAGE_CG and SAGE_CG_CD"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import argparse
from utils_viz import fi_hbarplot, hbar_text_position, coord_height_to_pixels


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


# initiate plot, fill it in for loop
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
fig.tight_layout(pad=2.1)
ax[0].set_title('SAGE')
ax[1].set_title(r'SAGE - $d$-SAGE')
ax[2].set_title(r'$d$-SAGE')


# load data
sage = pd.read_csv(f"real-world-experiments/results/sage/sage_runs5.csv")
dsage = pd.read_csv(f"real-world-experiments/results/sage/csl_sage_runs5.csv")

#sage_mean = sage["mean"]
#sage_mean = abs(sage_mean)
#sage_ordered = sage_mean.sort_values(ascending=False)
#sage_top = sage_ordered.iloc[0:len(sage)]
#indices = sage_top.index
#labels = []
#for a in range(args.top):
#    labels.append(int(indices[a]+2))

sage_r = pd.read_csv(f"real-world-experiments/results/sage/sage_r_runs5.csv")
dsage_r = pd.read_csv(f"real-world-experiments/results/sage/csl_sage_r_runs5.csv")
sage_r = sage_r.drop(columns="sample")
dsage_r = dsage_r.drop(columns=["sample", "sample.1"])
#sage = sage[sage["feature"].isin(labels)]
#dsage = dsage[dsage["feature"].isin(labels)]
#labels_str = []
#for a in range(len(indices)):
#    labels_str.append(str(indices[a]+2))
# differences between sage and sage_cg per run
diff = sage_r - dsage_r
# make df of differences w corresponding stds
diff_sage_dsage = pd.DataFrame(diff.mean(), columns=['mean'])
diff_sage_dsage['std'] = diff.std()
diff_sage_dsage['feature'] = diff_sage_dsage.index
#diff_sage_dsage = diff_sage_dsage[diff_sage_dsage["feature"].isin(labels_str)]
# plots
fi_hbarplot(sage, diff_sage_dsage, ax=ax[0], std=False)
fi_hbarplot(diff_sage_dsage, diff_sage_dsage, ax=ax[1])
fi_hbarplot(dsage, diff_sage_dsage, ax=ax[2], std=False)
ax[2].set_xticks([0.0, 0.005, 0.01, 0.015])
ax[2].set_xticklabels([0.0, 0.005, 0.01, 0.015])
ax[1].yaxis.set_visible(False)
ax[2].yaxis.set_visible(False)

plt.subplots_adjust(hspace=0.3, wspace=0, left=0.2)
plt.savefig(f"real-world-experiments/results/sage/sage_values_plot_nosd.png", dpi=600, transparent=True)
plt.clf()
