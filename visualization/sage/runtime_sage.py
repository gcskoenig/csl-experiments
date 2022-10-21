"Visualization of runtime data"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import create_folder
import argparse

parser = argparse.ArgumentParser(description="Complete file for model fitting and SAGE estimation")


parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="lm",
    help="lm or rf",
)

parser.add_argument(
    "-t",
    "--latex",
    type=bool,
    default=True,
    help="latex font or not",
)

args = parser.parse_args()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


create_folder("plots/sage/")

# dag s

times1 = (float(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_lm.csv")["ai_via neg"]))

times2 = (float(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_lm.csv")["ai_via neg"]))

times3 = (float(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_lm.csv")["ai_via neg"]))


times1r = (float(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_rf.csv")["ai_via neg"]))

times2r = (float(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_rf.csv")["ai_via neg"]))

times3r = (float(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_rf.csv")["ai_via neg"]))


# TODO import factor from estimated runtime
factors1 = (sum(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_lm.csv").loc[0][3])/times1
factors2 = (sum(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_lm.csv").loc[0][0:2]) +
                pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_lm.csv").loc[0][3])/times2
factors3 = (sum(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_lm.csv").loc[0][3])/times3

factors1r = (sum(pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_s_0.2/ai_via_dag_s_0.2_rf.csv").loc[0][3])/times1r
factors2r = (sum(pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_s_0.3/ai_via_dag_s_0.3_rf.csv").loc[0][3])/times2r
factors3r = (sum(pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_s_0.4/ai_via_dag_s_0.4_rf.csv").loc[0][3])/times3r


# dag_s no_orderings as multiplicator
#lm
orders1 = pd.read_csv("results/dag_s_0.2/order_sage_dag_s_0.2_lm.csv")
orders1 = orders1[orders1["sample"] == 0]
orders1 = orders1.dropna()
orders1 = len(orders1)/100

orders2 = pd.read_csv("results/dag_s_0.3/order_sage_dag_s_0.3_lm.csv")
orders2 = orders2[orders2["sample"] == 0]
orders2 = orders2.dropna()
orders2 = len(orders2)/100

orders3 = pd.read_csv("results/dag_s_0.4/order_sage_dag_s_0.4_lm.csv")
orders3 = orders3[orders3["sample"] == 0]
orders3 = orders3.dropna()
orders3 = len(orders3)/100

#rf
orders1r = pd.read_csv("results/dag_s_0.2/order_sage_dag_s_0.2_rf.csv")
orders1r = orders1r[orders1r["sample"] == 0]
orders1r = orders1r.dropna()
orders1r = len(orders1r)/100

orders2r = pd.read_csv("results/dag_s_0.3/order_sage_dag_s_0.3_rf.csv")
orders2r = orders2r[orders2r["sample"] == 0]
orders2r = orders2r.dropna()
orders2r = len(orders2r)/100

orders3r = pd.read_csv("results/dag_s_0.4/order_sage_dag_s_0.4_rf.csv")
orders3r = orders3r[orders3r["sample"] == 0]
orders3r = orders3r.dropna()
orders3r = len(orders3r)/100

# dag sm
timesm1 = (float(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_lm.csv")["ai_via neg"]))

timesm2 = (float(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_lm.csv")["ai_via neg"]))

timesm3 = (float(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_lm.csv")["ai_via neg"]))

timesm1r = (float(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_rf.csv")["ai_via neg"]))

timesm2r = (float(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_rf.csv")["ai_via neg"]))

timesm3r = (float(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_rf.csv")["ai_via neg"]))


# TODO import factor from estimated runtime
factorsm1 = (sum(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_lm.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_lm.csv").loc[0][3])/timesm1
factorsm2 = (sum(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_lm.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_lm.csv").loc[0][3])/timesm2
factorsm3 = (sum(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_lm.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_lm.csv").loc[0][3])/timesm3

factorsm1r = (sum(pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_rf.csv").loc[0][0:2]) +
              pd.read_csv("results/dag_sm_0.1/ai_via_dag_sm_0.1_rf.csv").loc[0][3])/timesm1r
factorsm2r = (sum(pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_rf.csv").loc[0][0:2]) +
              pd.read_csv("results/dag_sm_0.15/ai_via_dag_sm_0.15_rf.csv").loc[0][3])/timesm2r
factorsm3r = (sum(pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_rf.csv").loc[0][0:2]) +
              pd.read_csv("results/dag_sm_0.2/ai_via_dag_sm_0.2_rf.csv").loc[0][3])/timesm3r

# dag_sm no_orderings as multiplicator
#lm
ordersm1 = pd.read_csv("results/dag_sm_0.1/order_sage_dag_sm_0.1_lm.csv")
ordersm1 = ordersm1[ordersm1["sample"] == 0]
ordersm1 = ordersm1.dropna()
ordersm1 = len(ordersm1)/100

ordersm2 = pd.read_csv("results/dag_sm_0.15/order_sage_dag_sm_0.15_lm.csv")
ordersm2 = ordersm2[ordersm2["sample"] == 0]
ordersm2 = ordersm2.dropna()
ordersm2 = len(ordersm2)/100

ordersm3 = pd.read_csv("results/dag_sm_0.2/order_sage_dag_sm_0.2_lm.csv")
ordersm3 = ordersm3[ordersm3["sample"] == 0]
ordersm3 = ordersm3.dropna()
ordersm3 = len(ordersm3)/100


#lm
ordersm1r = pd.read_csv("results/dag_sm_0.1/order_sage_dag_sm_0.1_rf.csv")
ordersm1r = ordersm1r[ordersm1r["sample"] == 0]
ordersm1r = ordersm1r.dropna()
ordersm1r = len(ordersm1r)/100

ordersm2r = pd.read_csv("results/dag_sm_0.15/order_sage_dag_sm_0.15_rf.csv")
ordersm2r = ordersm2r[ordersm2r["sample"] == 0]
ordersm2r = ordersm2r.dropna()
ordersm2r = len(ordersm2r)/100

ordersm3r = pd.read_csv("results/dag_sm_0.2/order_sage_dag_sm_0.2_rf.csv")
ordersm3r = ordersm3r[ordersm3r["sample"] == 0]
ordersm3r = ordersm3r.dropna()
ordersm3r = len(ordersm3r)/100



# dag m
timem1 = (float(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_lm.csv")["ai_via neg"]))

timem2 = (float(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_lm.csv")["ai_via neg"]))

timem3 = (float(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_lm.csv")["ai_via neg"]))

timem1r = (float(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_rf.csv")["ai_via neg"]))

timem2r = (float(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_rf.csv")["ai_via neg"]))

timem3r = (float(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_rf.csv")["ai_via neg"]))



# TODO import factor from estimated runtime
factorm1 = (sum(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_lm.csv").loc[0][3])/timem1
factorm2 = (sum(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_lm.csv").loc[0][3])/timem2
factorm3 = (sum(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_lm.csv").loc[0][3])/timem3
factorm1r = (sum(pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_m_0.04/ai_via_dag_m_0.04_rf.csv").loc[0][3])/timem1r
factorm2r = (sum(pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_m_0.06/ai_via_dag_m_0.06_rf.csv").loc[0][3])/timem2r
factorm3r = (sum(pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_m_0.08/ai_via_dag_m_0.08_rf.csv").loc[0][3])/timem3r

# dag_m no_orderings as multiplicator
#lm
orderm1 = pd.read_csv("results/dag_m_0.04/order_sage_dag_m_0.04_lm.csv")
orderm1 = orderm1[orderm1["sample"] == 0]
orderm1 = orderm1.dropna()
orderm1 = len(orderm1)/100

orderm2 = pd.read_csv("results/dag_m_0.06/order_sage_dag_m_0.06_lm.csv")
orderm2 = orderm2[orderm2["sample"] == 0]
orderm2 = orderm2.dropna()
orderm2 = len(orderm2)/100

orderm3 = pd.read_csv("results/dag_m_0.08/order_sage_dag_m_0.08_lm.csv")
orderm3 = orderm3[orderm3["sample"] == 0]
orderm3 = orderm3.dropna()
orderm3 = len(orderm3)/100

# rf
orderm1r = pd.read_csv("results/dag_m_0.04/order_sage_dag_m_0.04_rf.csv")
orderm1r = orderm1r[orderm1r["sample"] == 0]
orderm1r = orderm1r.dropna()
orderm1r = len(orderm1r)/100

orderm2r = pd.read_csv("results/dag_m_0.06/order_sage_dag_m_0.06_rf.csv")
orderm2r = orderm2r[orderm2r["sample"] == 0]
orderm2r = orderm2r.dropna()
orderm2r = len(orderm2r)/100

orderm3r = pd.read_csv("results/dag_m_0.08/order_sage_dag_m_0.08_rf.csv")
orderm3r = orderm3r[orderm3r["sample"] == 0]
orderm3r = orderm3r.dropna()
orderm3r = len(orderm3r)/100



# dag l
timel1 = (float(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_lm.csv")["ai_via neg"]))

timel2 = (float(pd.read_csv("results/dag_l_0.03/ai_via_dag_l_0.03_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.03/ai_via_dag_l_0.03_lm.csv")["ai_via neg"]))

timel3 = (float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_lm.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_lm.csv")["ai_via neg"]))

timel1r = (float(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_rf.csv")["ai_via neg"]))

# ToDO change
timel2r = (float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv")["ai_via neg"]))

timel3r = (float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv")["ai_via pos"]) +
          float(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv")["ai_via neg"]))


# TODO import factor from estimated runtime
factorl1 = (sum(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_lm.csv").loc[0][3])/timel1
factorl2 = (sum(pd.read_csv("results/dag_l_0.03/ai_via_dag_l_0.03_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_l_0.03/ai_via_dag_l_0.03_lm.csv").loc[0][3])/timel2
factorl3 = (sum(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_lm.csv").loc[0][0:2]) +
            pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_lm.csv").loc[0][3])/timel3

# TODO change
factorl1r = (sum(pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_l_0.02/ai_via_dag_l_0.02_rf.csv").loc[0][3])/timel1r
factorl2r = (sum(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv").loc[0][3])/timel2r
factorl3r = (sum(pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv").loc[0][0:2]) +
             pd.read_csv("results/dag_l_0.04/ai_via_dag_l_0.04_rf.csv").loc[0][3])/timel3r


orderl1 = pd.read_csv("results/dag_l_0.02/order_sage_dag_l_0.02_lm.csv")
orderl1 = orderl1[orderl1["sample"] == 0]
orderl1 = orderl1.dropna()
orderl1 = len(orderl1)/100

orderl2 = pd.read_csv("results/dag_l_0.03/order_sage_dag_l_0.03_lm.csv")
orderl2 = orderl2[orderl2["sample"] == 0]
orderl2 = orderl2.dropna()
orderl2 = len(orderl2)/100

orderl3 = pd.read_csv("results/dag_l_0.04/order_sage_dag_l_0.04_lm.csv")
orderl3 = orderl3[orderl3["sample"] == 0]
orderl3 = orderl3.dropna()
orderl3 = len(orderl3)/100


orderl1r = pd.read_csv("results/dag_l_0.02/order_sage_dag_l_0.02_rf.csv")
orderl1r = orderl1r[orderl1r["sample"] == 0]
orderl1r = orderl1r.dropna()
orderl1r = len(orderl1r)/100

# todo change
orderl2r = pd.read_csv("results/dag_l_0.04/order_sage_dag_l_0.04_rf.csv")
orderl2r = orderl2r[orderl2r["sample"] == 0]
orderl2r = orderl2r.dropna()
orderl2r = len(orderl2r)/100

orderl3r = pd.read_csv("results/dag_l_0.04/order_sage_dag_l_0.04_rf.csv")
orderl3r = orderl3r[orderl3r["sample"] == 0]
orderl3r = orderl3r.dropna()
orderl3r = len(orderl3r)/100


#  x data (sample size)
x_ticks = [r"2", r"3", r"4"]

x0 = [0.7, 1.7, 2.7]
x1 = [0.9, 1.9, 2.9]
x2 = [1.1, 2.1, 3.1]
x3 = [1.3, 2.3, 3.3]

fig, axes = plt.subplots(1, 4, figsize=(7, 2))
fig.tight_layout()

# DAG_s
axes[0].set_title(r'DAG$_{s}$')
# axes[0].set_xlabel('Sample Size')
b0 = axes[0].bar(x0, [times1*orders1, times2*orders2, times3*orders3], width=0.2, color='b', align='center')
b1 = axes[0].bar(x1, [factors1*times1*orders1, factors2*times2*orders2, factors3*times3*orders3], width=0.2, color='g', align='center')
b2 = axes[0].bar(x2, [times1r*orders1r, times2r*orders2r, times3r*orders3r], width=0.2, color='c', align='center')
b3 = axes[0].bar(x3, [factors1r*times1r*orders1r, factors2r*times2r*orders2r, factors3r*times3r*orders3r], width=0.2, color='m', align='center')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(x_ticks, fontsize=10)

# DAG_sm
axes[1].set_title(r'DAG$_{sm}$')
# axes[0].set_xlabel('Sample Size')
axes[1].bar(x0, [timesm1*ordersm1, timesm2*ordersm2, timesm3*ordersm3], width=0.2, color='b', align='center')
axes[1].bar(x1, [factorsm1*timesm1*ordersm1, factorsm2*timesm2*ordersm2, factorsm3*timesm3*ordersm3], width=0.2, color='g', align='center')
axes[1].bar(x2, [timesm1r*ordersm1r, timesm2r*ordersm2r, timesm3r*ordersm3r], width=0.2, color='c', align='center')
axes[1].bar(x3, [factorsm1r*timesm1r*ordersm1r, factorsm2r*timesm2r*ordersm2r, factorsm3r*timesm3r*ordersm3r], width=0.2, color='m', align='center')
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(x_ticks, fontsize=10)

# DAG_m
axes[2].set_title(r'DAG$_{m}$')
# axes[0].set_xlabel('Sample Size')
axes[2].bar(x0, [timem1*orderm1, timem2*orderm2, timem3*orderm3], width=0.2, color='b', align='center')
axes[2].bar(x1, [factorm1*timem1*orderm1, factorm2*timem2*orderm2, factorm3*timem3*orderm3], width=0.2, color='g', align='center')
axes[2].bar(x2, [timem1r*orderm1r, timem2r*orderm2r, timem3r*orderm3r], width=0.2, color='c', align='center')
axes[2].bar(x3, [factorm1r*timem1r*orderm1r, factorm2r*timem2r*orderm2r, factorm3r*timem3r*orderm3r], width=0.2, color='m', align='center')
axes[2].set_xticks([1, 2, 3])
axes[2].set_xticklabels(x_ticks, fontsize=10)

# DAG_l
axes[3].set_title(r'DAG$_{l}$')
# axes[0].set_xlabel('Sample Size')
axes[3].bar(x0, [timel1*orderl1, timel2*orderl2, timel3*orderl3], width=0.2, color='b', align='center')
axes[3].bar(x1, [factorl1*timel1*orderl1, factorl2*timem2*orderl2, factorl3*timel3*orderl3], width=0.2, color='g', align='center')
axes[3].bar(x2, [timel1r*orderl1r, timel2r*orderl2r, timel3r*orderl3r], width=0.2, color='c', align='center')
axes[3].bar(x3, [factorl1r*timel1r*orderl1r, factorl2r*timem2r*orderl2r, factorl3r*timel3r*orderl3r], width=0.2, color='m', align='center')
axes[3].set_xticks([1, 2, 3])
axes[3].set_xticklabels(x_ticks, fontsize=10)


legend_labels = [r"SAGE (lm)", r"$d$-SAGE (lm)", r"SAGE (rf)", r"$d$-SAGE (rf)"]
fig.legend([b0, b1, b2, b3],     # The line objects
           labels=legend_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, -0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=4, fontsize=8
           )
plt.subplots_adjust(bottom=0.4, left=0.08)
fig.text(0.52, 0.234, 'Average Degree', ha='center')
fig.text(0.0, 0.6, 'Runtime in s', va='center', rotation='vertical')
# fig.text(1.0, 0.6, 'Runtime in s', va='center', rotation='vertical')

# fig.subplots_adjust(hspace=0.25)

plt.savefig(f"plots/sage/cont_runtime.png", dpi=400, transparent=True)
plt.clf()
