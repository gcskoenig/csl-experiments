"""Dictionaries for mixed data to get lists of continuous and categorical data"""

import pandas as pd
import os
from utils import create_folder
import pickle


# load data
mixed_dict = {"dataset_cat": ["1", "2"], "dataset_cont": ["3", "4"]}
create_folder("data/temp/")
# save dictionary
with open('data/temp/mixed_dict.pkl', 'wb') as f:
    pickle.dump(mixed_dict, f)
