import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
import networkx as nx
import random
import scipy.special as sp
import numpy as np
import os
from scipy.stats import bernoulli, norm
import warnings

def powerset(items):
    """computes power set of iterable object items

    Args:
        items: iterables object, e.g. list, array

    Returns:
         power set of items
    """

    combo = []
    for r in range(len(items) + 1):
        # use a list to coerce an actual list from the combinations generator
        combo.append(list(combinations(items, r)))
    return combo


def d_separation(g, y, g2=None, mc=None, random_state=None):
    """Test d-separation of each single node and y given every possible conditioning set in graph g

    Args:
        g : nx.DiGraph
        y : target node with respect to which d-separation is tested
        g2: potential second graph to be tested, that must contain the same nodes as g (typically an estimate of g)
        mc : if int given, mc sampling a subset of d-separations to be tested, recommended for large graphs
        random_state : seed for random and np.random when mc is not None

    Returns:
        if mc is None:
         pandas.DataFrame of Bools for d-separation for every node except y
        if mc is not None:
         Array of Bools for d-separation; it cannot be traced back which d-separations were tested explicitly
    """
    # list of nodes (strings)
    predictors = list(g.nodes)
    # sort list to get consistent results across different graphs learned on same features (when using mc)
    predictors.sort()
    # remove the target from list of predictors
    predictors.remove(y)
    n = len(predictors)

    # number of possible d-separations between one feature and target, i.e. number of potential conditioning sets
    if mc is None:
        # number of potential d-separations per node
        no_d_seps = (2 ** (n-1))
        # warn user if number of d-separation tests is large
        if no_d_seps > 1000000:
            warnings.warn("Warning: No. of d-separation tests per node > 1M, can lead to large runtime")
        # initiate dataframe for all d-separations
        d_separations = pd.DataFrame(index=predictors, columns=range(no_d_seps))

    if mc is not None:
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            rng = np.random.default_rng(seed=random_state)
        else:
            rng = np.random.default_rng()
        # initiate vector to store True/False for d-separations (cannot track nodes and conditioning sets)
        d_seps_bool = []
        if g2 is not None:
            d_seps_bool_2 = []
        # get a vector of probabilities to draw the size of the conditioning set; note that for n-1 potential
        # deconfounders there are n different sizes of conditioning sets because of the empty set
        probs = []
        for jj in range(n):
            probs.append((sp.comb(n-1, jj)) / (2**(n-1)))
        k = 0
        while k < mc:
            # draw index for feature of interest
            ind = random.randint(0, n-1)
            # retrieve feature of interest
            node = predictors[ind]
            # list of all features but feature of interest
            deconfounders = list(g.nodes)
            deconfounders.remove(y)
            deconfounders.remove(node)
            # sample a conditioning set from deconfounders
            # draw a cardinality
            card = np.random.choice(np.arange(n), p=probs)
            if card == 0:
                # test d-separation with empty set as conditioning set
                cond_set = set()
                d_seps_bool.append(nx.d_separated(g, {node}, {y}, cond_set))
                if g2 is not None:
                    d_seps_bool_2.append(nx.d_separated(g2, {node}, {y}, cond_set))

            else:
                # draw as many as 'card' numbers from range(n-1) as indices for conditioning set
                indices = rng.choice(n-1, size=card, replace=False)
                cond_set = set()
                for ii in range(len(indices)):
                    # index for first
                    index = indices[ii]
                    cond_set.add(deconfounders[index])
                d_seps_bool.append(nx.d_separated(g, {node}, {y}, cond_set))
                if g2 is not None:
                    d_seps_bool_2.append(nx.d_separated(g2, {node}, {y}, cond_set))
            k += 1
        if g2 is not None:
            return d_seps_bool, d_seps_bool_2
        else:
            return d_seps_bool  # vector of booleans
    else:
        if g2 is not None:
            print("Note: g2 only for Monte Carlo inference, will not be regarded here")

        for i in range(n):
            # test d-separation w.r.t. target using all conditional sets possible
            # for current predictor at i-th position in predictors
            node = predictors[i]
            deconfounders = list(g.nodes)
            deconfounders.remove(y)
            deconfounders.remove(node)
            power_set = powerset(deconfounders)
            j = 0
            while j < no_d_seps:
                for k in range(len(power_set)):
                    # k == 0 refers to the empty set in power_set
                    if k == 0:
                        cond_set = set()
                        d_separations.iloc[i, j] = nx.d_separated(g, {node}, {y}, cond_set)
                        j += 1
                    else:
                        for jj in range(len(power_set[k])):
                            cond_set = {power_set[k][jj][0]}
                            for m in range(len(power_set[k][jj]) - 1):
                                cond_set.add(power_set[k][jj][m + 1])
                            d_separations.iloc[i, j] = nx.d_separated(
                                g, {node}, {y}, cond_set
                            )
                            j += 1
        return d_separations    # df with Boolean indicators of d-separation for every predictor


# compute the number of d-separation statements from n

def dsep_mb(n, mb):
    """computes a lower bound for the number of d-separation statements between a target node
    and every other node in the graph given the size of the target node's Markov blanket

    Args:
        n: number of nodes in graph
        mb: size of Markov blanket of target

    Returns:
        Lower bound for number of d-separations
    """

    # number of nodes other than y and Markov blanket: n-1-mb
    # number of d-separations for such nodes 2**(n-2-mb)
    return (n - 1 - mb) * 2 ** (n - 2 - mb)


def dsep_degree(n, max_degree, sink=False):
    """computes a lower bound for the number of d-separation statements between a target node
        and every other node in the graph given the max degree of a node

        Args:
            n : number of nodes in graph
            max_degree : maximal degree of a node in the graph (max. number of edges associated to a node
            sink : Bool, whether target is sink node or not
        Returns:
            Lower bound for number of d-separations
        """
    if sink is False:
        # maximal size of Markov blanket
        max_mb = max_degree + max_degree ** 2
        return dsep_mb(n, max_mb)
    else:
        max_mb = max_degree
        return dsep_mb(n, max_mb)


def potential_dseps(n):
    """For a graph of size n, return the maximal number of d-separations between each node and a potentially
    dedicated target node

    Args:
        n: number of nodes in graph

    Return:
        Number of potential d-separation statements (float)
    """
    return (n - 1) * (2 ** (n - 2))


def create_folder(directory):
    """Creates directory as specified in function argument if not already existing

    Args:
        directory: string specifying the directory
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory: " + directory)


def create_amat(n, p):
    """Create a random adjacency matrix of size n x n

    Args:
        n: number of nodes
        p: probability of existence of an edge for each pair of nodes

    Returns:
        Adjacency matrix as pd.DataFrame

    """
    # create col_names
    variables = []
    for i in range(n):
        variables.append(str(i+1))

    # create df for amat
    amat = pd.DataFrame(columns=variables, index=variables)

    for j in range(n):
        for k in range(n):
            amat.iloc[j, k] = bernoulli(p)


def exact(g_true, g_est, y):
    # the two dataframes of d-separations
    true_dseps = d_separation(g_true, y)
    est_dseps = d_separation(g_est, y)
    # now compare every entry
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(true_dseps.shape[0]):
        for j in range(true_dseps.shape[1]):
            if true_dseps.iloc[i, j] == est_dseps.iloc[i, j]:
                if true_dseps.iloc[i, j] is True:
                    tp += 1
                else:
                    tn += 1
            else:
                if true_dseps.iloc[i, j] is True:
                    fn += 1
                else:
                    fp += 1
    # total number of d-separations in true graph
    d_separated_total = tp + fn
    d_connected_total = tn + fp
    return tp, tn, fp, fn, d_separated_total, d_connected_total


def approx(g_true, g_est, y, mc=None, rand_state=None):
    true_dseps, est_dseps = d_separation(g_true, y, g2=g_est, mc=mc, random_state=rand_state)
    # now compare every entry
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(true_dseps)):
        if true_dseps[i] == est_dseps[i]:
            if true_dseps[i] is True:
                tp += 1
            else:
                tn += 1
        else:
            if true_dseps[i] is True:
                fn += 1
            else:
                fp += 1
    # total number of d-separation among tested nodes (make a node if d-separations were approximated via mc)
    d_separated_total = tp + fn
    d_connected_total = tn + fp
    return tp, tn, fp, fn, d_separated_total, d_connected_total


def convert_amat(df, col_names=False):
    """Convert adjacency matrix of Bools to 0-1 for use in networkx package
    Args:
        df: adjacency matrix
        col_names: toggle overriding of column names with strings starting from "1"
    """

    if col_names:
        col_names_str = []
        for k in range(len(df.columns)):
            col_names_str.append(str(k+1))
        df.columns = col_names_str

    mapping_rf = {False: 0, True: 1}
    col_names = df.columns
    for j in col_names:
        df[j] = df.replace({j: mapping_rf})[j]  # TODO (cl) [j] required in the end?

    # modify adjacency matrix for use in networkx package
    df = df.set_axis(df.columns, axis=0)
    # store modified adjacency matrix
    return df


# Functions from utils file of rfi.plots
def coord_height_to_pixels(ax, height):
    p1 = ax.transData.transform((0, height))
    p2 = ax.transData.transform((0, 0))

    pix_height = p1[1] - p2[1]
    return pix_height


def hbar_text_position(rect, x_pos=0.5, y_pos=0.5):
    rx, ry = rect.get_xy()
    width = rect.get_width()
    height = rect.get_height()

    tx = rx + (width * x_pos)
    ty = ry + (height * y_pos)
    return (tx, ty)


def fi_hbarplot(ex, df , textformat='{:5.2f}', ax=None, figsize=None, std=True, std_factor=1):
    """
    Function that plots the result of an RFI computation as a barplot
    Args:
        figsize:
        ax:
        textformat:
        ex: Explanation object
    """

    names = df.index
    rfis = ex['mean'].to_numpy()
    stds = ex['std'].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ixs = np.arange(rfis.shape[0] + 0.5, 0.5, -1)
    if std:
        #ax.barh(ixs, rfis, tick_label=names, xerr=std_factor*stds, capsize=5, color=['mistyrose',
        #                                                              'salmon', 'tomato',
        #                                                              'darksalmon', 'coral'])
        ax.barh(ixs, rfis, tick_label=names, xerr=std_factor*stds, capsize=5)
    else:
        #ax.barh(ixs, rfis, tick_label=names, capsize=5, color=['mistyrose',
        #                                                              'salmon', 'tomato',
        #                                                              'darksalmon', 'coral'])
        ax.barh(ixs, rfis, tick_label=names, capsize=5)


    # color = ['lightcoral',
    #          'moccasin', 'darkseagreen',
    #          'paleturquoise', 'lightsteelblue']
    return ax


def convergence_plot(data, scores = None, top=None, bottom=None, choose=None, latex_font=False, alpha=0.05, ax=None, figsize=None,
                     ci_bands="sd", std="run", legend=True, loc='upper right', time_axis=False, runtime=2000):
    """
    Function that plots the result of an RFI computation as a convergence plot based on the values per ordering
    Args:
        data: explanation object from package
        top: number of top values to be plotted (cannot be combined with bottom)
        bottom: number of bottom values to be plotted (cannot be combined with top)
        choose: [x,y], x int, y int, range of values to plot, x is (x+1)th largest, y is y-th largest value
        latex_font: Bool - toggle LaTeX font
        alpha: alpha for confidence bands
        ax:
        figsize:
        legend: addd legend to plot/axes
        loc: position of legend; default: 'upper right' # TODO (cl): custom location of legend
    """

    # TODO (cl) check correct syntax for version > 3.8
    # TODO (cl) for package uncomment next line, delete l33
    # data = data.scores.groupby(level='orderings').mean()

    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # drop column 'ordering' if it is present
    if 'ordering' in data.columns:
        data = data.drop(['ordering'], axis=1)

    # latex font TODO (cl) more generic? use all plt options?
    if latex_font:
        # for latex font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # get the sage values (mean across all orderings)
    sage_mean = data.mean()

    if (top is not None and bottom is not None) or (top is not None and choose is not None) \
            or (bottom is not None and choose is not None):
        raise ValueError("Arguments top, bottom or choose cannot be used together.")

    if top is not None:
        # absolute values to rank sage values and then retrieve the top values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)   # i.e descending -> top first
        sage_top = sage_ordered.iloc[0:top]
        # indices of top values
        indices = sage_top.index

    elif bottom is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=True)
        sage_bottom = sage_ordered.iloc[0:bottom]
        # indices of bottom values
        indices = sage_bottom.index

    elif choose is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)
        sage_choose = sage_ordered.iloc[choose[0]:choose[1]]
        # indices of bottom values
        indices = sage_choose.index

    else:
        indices = sage_mean.index

    # trim to relevant data
    data = data[indices]

    # running means up to current ordering (this is the line to be plotted)
    running_mean = pd.DataFrame(columns=data.columns)
    for j in range(2, len(data)):
        running_mean.loc[j] = data[0:j + 1].mean()
    running_mean = running_mean.reset_index(drop=True)

    # get convergence bands for sage from running means of the [nr_runs] runs if scores are passed
    if std == "run":
        if scores is None:
            raise ValueError("Scores df has to be passed")
        running_means = []
        for i in range(max(scores["sample"]) + 1):
            run_mean = pd.DataFrame(columns=scores.columns)
            scores_copy = scores[scores["sample"] == i]
            print("scores copy: ", len(scores_copy))
            # running means up to current ordering
            for j in range(2, len(scores_copy)):
                run_mean.loc[j] = scores_copy[0:j + 1].mean()
            run_mean = run_mean.reset_index(drop=True)
            running_means.append(run_mean)
            # running_means now has [nr_runs] df's of running means, for each

        # get sds from running_means
        std_sage = pd.DataFrame(columns=data.columns)
        # k is the number of orderings to go through
        print("len running means 0: ", len(running_means[0]))
        for k in range(len(running_means[0])):
            # get a row of stds for the first running mean, then append the df, then second etc
            sds = []
            for col_index in indices:
                current_means = []
                for run in range(len(running_means)):
                    current_means.append(running_means[run][col_index][k])
                sds.append(np.std(current_means))
            std_sage.loc[k] = sds

    # get the standard deviations after every ordering starting with the third
    if std == "ordering":
        std_sage = pd.DataFrame(columns=data.columns)
        for i in range(2, len(data)):
            # TODO rewrite and shorten
            diffs = data[0:i+1] - data[0:i+1].mean()
            # squared differences
            diffs2 = diffs*diffs
            # sum of squared diffs
            diffs2_sum = diffs2.sum()
            # sum of diffs
            diffs_sum = diffs.sum()
            # diffs_sum2 = (diffs_sum * diffs_sum)
            # diffs_sum2_n = (diffs_sum2/ii)
            variance = (diffs2_sum - ((diffs_sum * diffs_sum)/i)) / (i - 1)
            std = variance**0.5
            std_sage.loc[i-2] = std
            # std_sage is a (no_order - 2) x len(indices) matrix

    # make confidence bands
    lower = pd.DataFrame(columns=running_mean.columns)
    upper = pd.DataFrame(columns=running_mean.columns)

    for k in range(len(running_mean)):
        # NOTE: k+3 because first 3 rows were dropped before
        if ci_bands == "Gaussian":
            # TODO: use t-distribution (in general: Gaussian in the limit, because every summand is a RV -> are they indep? NO)
            # TODO: this is wrong anyways, but 'sd' is fine
            lower.loc[k] = running_mean.loc[k] + norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))
            upper.loc[k] = running_mean.loc[k] - norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))
        elif ci_bands == "sd":
            lower.loc[k] = running_mean.loc[k] + std_sage.loc[k]
            upper.loc[k] = running_mean.loc[k] - std_sage.loc[k]

    x_axis = []
    for ll in range(len(running_mean)):
        x_axis.append(ll)

    # fig = plt.figure(figsize=(5,5))
    ax.plot(x_axis, running_mean, linewidth=0.7)

    for n in lower.columns:
        ax.fill_between(x_axis, lower[n], upper[n], alpha=.1)

    labels = data.columns
    if legend:
        ax.legend(loc=loc, labels=labels)

    # TODO: times_axis does not work yet
    if time_axis:
        runtime = runtime
        times = []
        for i in range(len(running_mean)):
            times.append(i * (runtime / len(running_mean)))
        ax2 = ax.twinx()
        ax2.plot(x_axis, times, alpha=0.5)
        # ax2 = ax.twiny()
        # ax2.plot(range(0, time_axis, int(time_axis/20)), np.ones(time_axis))

    return ax, std_sage, indices

def convergence_plot_dsage(data, scores = None, top=None, bottom=None, choose=None, latex_font=False, alpha=0.05, ax=None, figsize=None,
                     ci_bands="sd", std="run", legend=True, loc='upper right', time_axis=False, runtime=2000, index_set=None):
    """
    Function that plots the result of an RFI computation as a convergence plot based on the values per ordering
    Args:
        data: explanation object from package
        top: number of top values to be plotted (cannot be combined with bottom)
        bottom: number of bottom values to be plotted (cannot be combined with top)
        choose: [x,y], x int, y int, range of values to plot, x is (x+1)th largest, y is y-th largest value
        latex_font: Bool - toggle LaTeX font
        alpha: alpha for confidence bands
        ax:
        figsize:
        legend: addd legend to plot/axes
        loc: position of legend; default: 'upper right' # TODO (cl): custom location of legend
    """

    # TODO (cl) check correct syntax for version > 3.8
    # TODO (cl) for package uncomment next line, delete l33
    # data = data.scores.groupby(level='orderings').mean()

    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # drop column 'ordering' if it is present
    if 'ordering' in data.columns:
        data = data.drop(['ordering'], axis=1)

    # latex font TODO (cl) more generic? use all plt options?
    if latex_font:
        # for latex font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # get the sage values (mean across all orderings)
    sage_mean = data.mean()

    if (top is not None and bottom is not None) or (top is not None and choose is not None) \
            or (bottom is not None and choose is not None):
        raise ValueError("Arguments top, bottom or choose cannot be used together.")

    if top is not None:
        # absolute values to rank sage values and then retrieve the top values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)   # i.e descending -> top first
        sage_top = sage_ordered.iloc[0:top]
        # indices of top values
        indices = sage_top.index

    elif bottom is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=True)
        sage_bottom = sage_ordered.iloc[0:bottom]
        # indices of bottom values
        indices = sage_bottom.index

    elif choose is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)
        sage_choose = sage_ordered.iloc[choose[0]:choose[1]]
        # indices of bottom values
        indices = sage_choose.index

    elif index_set is not None:
        indices = index_set

    else:
        indices = sage_mean.index

    # trim to relevant data
    data = data[indices]

    # running means up to current ordering (this is the line to be plotted)
    running_mean = pd.DataFrame(columns=data.columns)
    for j in range(2, len(data)):
        running_mean.loc[j] = data[0:j + 1].mean()
    running_mean = running_mean.reset_index(drop=True)

    # get convergence bands for sage from running means of the [nr_runs] runs if scores are passed
    if std == "run":
        if scores is None:
            raise ValueError("Scores df has to be passed")
        running_means = []
        for i in range(max(scores["sample"]) + 1):
            run_mean = pd.DataFrame(columns=scores.columns)
            scores_copy = scores[scores["sample"] == i]
            # running means up to current ordering
            for j in range(2, len(scores_copy)):
                run_mean.loc[j] = scores_copy[0:j + 1].mean()
            run_mean = run_mean.reset_index(drop=True)
            running_means.append(run_mean)
            # running_means now has [nr_runs] df's of running means, for each

        # get sds from running_means
        std_sage = pd.DataFrame(columns=data.columns)
        # k is the number of orderings to go through
        for k in range(len(running_means[0])):
            # get a row of stds for the first running mean, then append the df, then second etc
            sds = []
            for col_index in indices:
                current_means = []
                for run in range(len(running_means)):
                    current_means.append(running_means[run][col_index][k])
                sds.append(np.std(current_means))
            std_sage.loc[k] = sds

    # get the standard deviations after every ordering starting with the third
    if std == "ordering":
        std_sage = pd.DataFrame(columns=data.columns)
        for i in range(2, len(data)):
            # TODO rewrite and shorten
            diffs = data[0:i+1] - data[0:i+1].mean()
            # squared differences
            diffs2 = diffs*diffs
            # sum of squared diffs
            diffs2_sum = diffs2.sum()
            # sum of diffs
            diffs_sum = diffs.sum()
            # diffs_sum2 = (diffs_sum * diffs_sum)
            # diffs_sum2_n = (diffs_sum2/ii)
            variance = (diffs2_sum - ((diffs_sum * diffs_sum)/i)) / (i - 1)
            std = variance**0.5
            std_sage.loc[i-2] = std
            # std_sage is a (no_order - 2) x len(indices) matrix

    # make confidence bands
    lower = pd.DataFrame(columns=running_mean.columns)
    upper = pd.DataFrame(columns=running_mean.columns)
    for k in range(len(running_mean)):
        # NOTE: k+3 because first 3 rows were dropped before
        if ci_bands == "Gaussian":
            # TODO: use t-distribution (in general: Gaussian in the limit, because every summand is a RV -> are they indep? NO)
            # TODO: this is wrong anyways, but 'sd' is fine
            lower.loc[k] = running_mean.loc[k] + norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))
            upper.loc[k] = running_mean.loc[k] - norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))
        elif ci_bands == "sd":
            lower.loc[k] = running_mean.loc[k] + std_sage.loc[k]
            upper.loc[k] = running_mean.loc[k] - std_sage.loc[k]

    # if dsage, trim everything to correct length
    # make a dict with running_means for every var (different length of running means series)
    running_mean_dict = {}
    lower_dict = {}
    upper_dict = {}
    for kk in indices:
        # print(len([data[2:][kk] != 0]))
        # print(len(running_mean[kk]))
        running_mean_ = running_mean[kk]
        boolean_index = data[2:][kk] != 0
        boolean_index = boolean_index.reset_index(drop=True)
        running_mean_dict[kk] = running_mean_[boolean_index].reset_index(drop=True)
        lower_ = lower[kk]
        upper_ = upper[kk]
        lower_dict[kk] = lower_[boolean_index].reset_index(drop=True)
        upper_dict[kk] = upper_[boolean_index].reset_index(drop=True)

    x_axis = []

    for ll in range(len(running_mean)):
        x_axis.append(ll)

    first_index = indices[0]
    first_axis = [i for i in range(0, len(running_mean_dict[first_index] + 1))]
    ax.plot(first_axis, running_mean_dict[first_index], linewidth=0.7)
    for i in range(1, len(indices)):
        next_axis = [i for i in range(0, len(running_mean_dict[indices[i]] + 1))]
        ax.plot(next_axis, running_mean_dict[indices[i]], linewidth=0.7)

    # plt.xlim(len(x_axis))
    #for n in lower.columns:
    #    ax.fill_between(x_axis, lower[n], upper[n], alpha=.1)

    labels = data.columns
    if legend:
        ax.legend(loc=loc, labels=labels)

    ax.fill_between(first_axis, lower_dict[first_index], upper_dict[first_index], alpha=.1)
    for i in range(1, len(indices)):
        next_axis = [i for i in range(0, len(running_mean_dict[indices[i]] + 1))]
        ax.fill_between(next_axis, lower_dict[indices[i]], upper_dict[indices[i]], alpha=.1)

    # TODO: times_axis does not work yet
    if time_axis:
        runtime = runtime
        times = []
        for i in range(len(running_mean)):
            times.append(i * (runtime / len(running_mean)))
        ax2 = ax.twinx()
        ax2.plot(x_axis, times, alpha=0.5)
        # ax2 = ax.twiny()
        # ax2.plot(range(0, time_axis, int(time_axis/20)), np.ones(time_axis))

    return ax, std_sage