from classes import *
from functions import *
from pipelines import *
from ml_algorithm import *
#----------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#-----------------------------------

from sklearn.metrics import mean_squared_error
from scipy import stats

DATASET_URL="dataset/Laptops.csv"

raw_data=import_data(DATASET_URL)

data, test_set = split_data(raw_data)
data = cut_price_outliers(data)

X_train = data.drop("Price", axis=1)
Y_train = log_target(data)

X_test = test_set.drop("Price", axis=1)
Y_test = log_target(test_set)

num_cols, cat_cols = get_attribs(X_train)

final_pipeline = full_pipeline(num_cols, cat_cols) 

dataset_prepared = final_pipeline.fit_transform(X_train)

model, best_parameters = train_RFR_random(dataset_prepared, Y_train)

rmse, interval = comparison(X_test,Y_test, final_pipeline, model)

save_results_txt(best_parameters, rmse, interval)

export_model(model=model, pipeline=final_pipeline)