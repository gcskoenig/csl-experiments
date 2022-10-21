"""File to evaluate estimated graphs used in Causal Structure Learning for SAGE Estimation w.r.t. inferred
d-separations by comparison to ground truth graph using exact and approx comparison functions from utils.py
"""
import networkx as nx
import pandas as pd
from utils import exact, approx, convert_amat, create_folder
import pickle

# file for comparison   # TODO More elegant way than try-except-statement?
try:
    graph_evaluation = pd.read_csv("bnlearn/results/graph_evaluation.csv")
except:
    col_names = ["Graph", "Target", "Method", "MC",
                 "True total", "False total", "D-sep share", "TP", "TN", "FP", "FN", "TP rate",
                 "TN rate", "FP rate", "FN rate", "Precision", "Recall", "F1"]
    graph_evaluation = pd.DataFrame(columns=col_names)

# to access the target nodes in graph loop
with open('data/temp/targets.pkl', 'rb') as f:
    targets = pickle.load(f)

# adapt for final experiments
#discrete_graphs = ["asia", "sachs", "alarm", "hepar"]
discrete_graphs = []
cont_graphs = ["dag_s_0.2", "dag_s_0.3", "dag_s_0.4", "dag_sm_0.1", "dag_sm_0.15", "dag_sm_0.2",
               "dag_m_0.04", "dag_m_0.06", "dag_m_0.08", "dag_l_0.02", "dag_l_0.03", "dag_l_0.04"]

sample_sizes = ["1000", "10000", "1e+05", "1e+06"]
discrete_algs = ["h2pc", "mmhc", "hc", "tabu"]
cont_algs = ["hc", "tabu"]

# compare graphs with less than 1M d-separations by evaluating exact inference
for graph in discrete_graphs:
    # ground truth graph
    true_amat = pd.read_csv(f"data/true_amat/{graph}.csv")
    true_amat = convert_amat(true_amat)
    g_true = nx.DiGraph(true_amat)
    for method in discrete_algs:
        for n in sample_sizes:
            est_amat = pd.read_csv(f"bnlearn/results/{method}/{graph}_{n}_obs.csv")
            est_amat = convert_amat(est_amat)
            g_est = nx.DiGraph(est_amat)
            if graph in ["asia", "sachs"]:
                tp, tn, fp, fn, d_separated_total, d_connected_total = exact(g_true, g_est, targets[f"{graph}"])
                mc = "n/a"
            elif graph in ["alarm", "hepar"]:
                mc = 1000000
                tp, tn, fp, fn, d_separated_total, d_connected_total = approx(g_true, g_est, targets[f"{graph}"],
                                                                              mc=mc, rand_state=42)
            else:
                print("graph is not defined properly")
                print(true_amat)
                break

            dsep_share = d_separated_total / (d_separated_total + d_connected_total)
            if d_separated_total == 0:
                TP_rate = 0
                FN_rate = 0
            else:
                TP_rate = tp / d_separated_total
                FN_rate = fn / d_separated_total
            if d_connected_total == 0:
                TN_rate = 0
                FP_rate = 0
            else:
                TN_rate = tn / d_connected_total
                FP_rate = fp / d_connected_total
            # F1 score
            precision = tp / (tp + fp)
            recall = TP_rate
            F1 = (2 * precision * recall) / (precision + recall)
            content = [f"{graph}_{n}_obs", targets[f"{graph}"], method, mc, d_separated_total, d_connected_total,
                       dsep_share, tp, tn, fp, fn, TP_rate, TN_rate, FP_rate, FN_rate, precision, recall, F1]
            graph_evaluation.loc[len(graph_evaluation)] = content

for graph in cont_graphs:
    # ground truth graph
    true_amat = pd.read_csv(f"data/true_amat/{graph}.csv")
    true_amat = convert_amat(true_amat, col_names=True)
    g_true = nx.DiGraph(true_amat)
    for method in cont_algs:
        for n in sample_sizes:
            est_amat = pd.read_csv(f"bnlearn/results/{method}/{graph}_{n}_obs.csv")
            est_amat = convert_amat(est_amat, col_names=True)
            g_est = nx.DiGraph(est_amat)
            if graph in ["dag_s_0.2", "dag_s_0.3", "dag_s_0.4"]:
                tp, tn, fp, fn, d_separated_total, d_connected_total = exact(g_true, g_est, targets[f"{graph}"])
                mc = "n/a"
            elif graph in ["dag_sm_0.1", "dag_sm_0.15", "dag_sm_0.2", "dag_m_0.04", "dag_m_0.06", "dag_m_0.08",
                           "dag_l_0.02", "dag_l_0.03", "dag_l_0.04"]:
                mc = 1000000
                tp, tn, fp, fn, d_separated_total, d_connected_total = approx(g_true, g_est, targets[f"{graph}"],
                                                                              mc=mc, rand_state=42)
            else:
                print("graph is not defined properly")
                print(true_amat)
                break
            dsep_share = d_separated_total / (d_separated_total + d_connected_total)
            if d_separated_total == 0:
                TP_rate = 0
                FN_rate = 0
            else:
                TP_rate = tp / d_separated_total
                FN_rate = fn / d_separated_total
            if d_connected_total == 0:
                TN_rate = 0
                FP_rate = 0
            else:
                TN_rate = tn / d_connected_total
                FP_rate = fp / d_connected_total
            # F1 score
            try:
                precision = tp / (tp + fp)
            except:
                precision = 0
            recall = TP_rate
            try:
                F1 = (2 * precision * recall) / (precision + recall)
            except:
                F1 = 0
            content = [f"{graph}_{n}_obs", targets[f"{graph}"], method, mc, d_separated_total, d_connected_total,
                       dsep_share, tp, tn, fp, fn, TP_rate, TN_rate, FP_rate, FN_rate, precision, recall, F1]
            graph_evaluation.loc[len(graph_evaluation)] = content

graph_evaluation.to_csv("bnlearn/results/graph_evaluation.csv", index=False)
