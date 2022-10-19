# dataset source https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#
import random

random.seed(123)


import pandas as pd
import numpy as np
import torch

np.random.seed(123)
torch.random.seed(123)

var_names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore',
             'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl',
             'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
             'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
df = pd.read_csv('drug_consumption.csv', header=None)
df.columns = var_names

inputs = var_names[1:13]
target = ['Nicotine']

X, y_cat = df.loc[:, inputs], df.loc[:, target]

y = y_cat.isin(['CL4', 'CL5', 'CL6'])
y = np.array(y).reshape(-1)

cat_vars = ['Gender', 'Education', 'Country', 'Ethnicity']

for cat_var in cat_vars:
    X.loc[:, cat_var] = X.loc[:, cat_var].astype('category').cat.codes

import seaborn as sns
import matplotlib.pyplot as plt

g = sns.PairGrid(X)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
plt.savefig('drug_consumption_features_distplot.pdf')
plt.close()

# train / test da with logreg

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
print(log_loss(y_test, logreg.predict_proba(X_test)))

def predict_proba(*args, **kwargs):
    pred = logreg.predict_proba(*args, **kwargs)
    return pred[:, 1]

print(log_loss(y_test, predict_proba(X_test)))


from rfi.samplers import SequentialSampler, GaussianSampler, UnivRFSampler
from rfi.explainers.explainer import Explainer

cat_sampler = UnivRFSampler(X_train)
cont_sampler = GaussianSampler(X_train)

sequential_sampler = SequentialSampler(X_train, categorical_fs=cat_vars, cat_sampler=cat_sampler,
                                       cont_sampler=cont_sampler)
wrk = Explainer(predict_proba, X_train.columns, X_train, sampler=sequential_sampler, loss=log_loss)

ex = wrk.dis_from_baselinefunc(X_train.columns, X_test, y_test)
ex.fi_means_stds()

ex2 = wrk.ais_via_contextfunc(X_train.columns, X_test, y_test)
ex2.fi_means_stds()

ex_sage = wrk.sage(X_test, y_test, [tuple(X_test.columns)], nr_runs=1, nr_resample_marginalize=10,
                   detect_convergence=True, thresh=0.01, nr_orderings=1500)
ex_sage.fi_means_stds()