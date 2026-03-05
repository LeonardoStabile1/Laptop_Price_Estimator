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
MODELS = ["RFR", "XGB"]

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

rmse_list = []
interval_list = []

best_rmse = float('inf')
best_trained_model = None
best_final_params = None
best_final_interval = None
best_model_name = ""

for model_name in MODELS:
    
    match model_name:
        case "RFR": #RandomForestRegression()
            current_model, current_params = train_RFR_random(dataset_prepared, Y_train)
            current_rmse, current_interval = comparison(X_test, Y_test, final_pipeline, current_model)
            rmse_list.append(current_rmse)
            interval_list.append(current_interval)
            
        case "XGB": #XGBoost()
            current_model, current_params = train_XGB_random(dataset_prepared, Y_train)
            current_rmse, current_interval = comparison(X_test, Y_test, final_pipeline, current_model)
            rmse_list.append(current_rmse)
            interval_list.append(current_interval)
            
    
    if current_rmse < best_rmse:
        best_rmse = current_rmse
        best_trained_model = current_model
        best_final_params = current_params
        best_final_interval = current_interval
        best_model_name = model_name

print(f"\nSeleção concluída! O melhor modelo foi {best_model_name} com RMSE de ${best_rmse:.2f}")


save_results_txt(best_final_params, best_rmse, best_final_interval, best_model_name)
export_model(model=best_trained_model, pipeline=final_pipeline)