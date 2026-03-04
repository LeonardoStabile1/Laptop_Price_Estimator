from classes import DataCleaner, GPUTierExtractor
from functions import *
#----------------------------------

import pandas as pd
import numpy as np

#-----------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def full_pipeline(data):
    base_pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('gpu_engineer', GPUTierExtractor(gpu_col='gpu_name'))
    ])

    num_attribs, cat_attribs = get_attribs(data, base_pipeline)

    col_transformer = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs)
    ])

    full_pipeline = Pipeline([
        ('base_prep', base_pipeline),
        ('features', col_transformer)
    ])

    return full_pipeline

def fit_transform_data(data, pipeline):
    return pipeline.fit_transform(data)

def get_attribs(data, pipeline):
    temp_data = fit_transform_data(data,pipeline)
    num_attribs = temp_data.select_dtypes(include=['number']).drop(columns=['Price'], errors='ignore').columns.tolist()
    cat_attribs = temp_data.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_attribs, cat_attribs
