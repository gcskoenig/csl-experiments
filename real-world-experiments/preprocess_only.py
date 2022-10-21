# dataset source https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#
import random

random.seed(123)


import pandas as pd
import numpy as np
import torch

np.random.seed(123)
torch.random.seed()

var_names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore',
             'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl',
             'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
             'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
df = pd.read_csv('real-world-experiments/drug_consumption.csv', header=None)
df.columns = var_names

df_cols = var_names[1:13]
df_cols.append('Nicotine')
df=df.loc[:, df_cols]
df.to_csv(f'real-world-experiments/drug_consumption_preprocessed.csv', index=False)
