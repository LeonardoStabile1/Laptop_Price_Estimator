from classes import DataCleaner, GPUTierExtractor
from functions import *

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def full_pipeline(num_attribs, cat_attribs):
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('gpu_engineer', GPUTierExtractor(gpu_col='gpu_name')),
        ('features', ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs)
        ]))
    ])

def get_attribs(data):
    temp_pipe = Pipeline([
        ('cleaner', DataCleaner()),
        ('gpu_engineer', GPUTierExtractor(gpu_col='gpu_name'))
    ])
    
    temp_data = temp_pipe.fit_transform(data)
    
    num_attribs = temp_data.select_dtypes(include=['number']).columns.tolist()
    cat_attribs = temp_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return num_attribs, cat_attribs