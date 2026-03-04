from classes import *
from functions import *
from pipelines import full_pipeline
from model_train import *
#----------------------------------

import pandas as pd
import numpy as np

#-----------------------------------

from sklearn.metrics import mean_squared_error

DATASET_URL="dataset/Laptops.csv"

raw_data=import_data(DATASET_URL)

data, test_set = split_data(raw_data)
data = cut_price_outliers(data)
target = log_target(data)

print("Dados carregados e tratados. Iniciando Pipeline treino")

final_pipeline=full_pipeline(data)
dataset_prepared = final_pipeline.fit_transform(data)

model,best_parameters = train_RFR_random(dataset_prepared, target)

print("Best parameters are: " +best_parameters)