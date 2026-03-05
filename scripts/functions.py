import requests
import pandas as pd
import numpy as np
import os
from scipy import stats
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def import_data(dataset_url):
    dataset=pd.read_csv(dataset_url)
    return dataset
    

def split_data(dataset, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
    return train_set, test_set



def cut_price_outliers(df):
    if 'Price' not in df.columns:
        print(" 'Price' Column not include on the dataset.")
        return df
    new_df = df[df['Price'] < 150000].copy()
    return new_df


def price_to_usd(df):
    if not isinstance(df, pd.DataFrame):
        print("Error: Please pass a full DataFrame, not just a column.")
        return df

    if 'Price' not in df.columns:
        return df

    new_df = df.copy()
    
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    fallback = 83.12
    try:
        response = requests.get(url, timeout=5)
        rate = response.json()['rates']['INR']
    except Exception as e:
        print(f"Using fallback rate due to: {e}")
        rate = fallback
    new_df['Price'] = (new_df['Price'] / rate).round(2)
    
    return new_df

def clean_target(dataset):
    if 'Price' not in dataset.columns:
        print(" 'Price' Column not include on the dataset.")
        return dataset
    dataset['Price'] = price_to_usd(dataset)['Price']
    return dataset

def target_extraction(dataset):
    if 'Price' not in dataset.columns:
        print(" 'Price' Column not include on the dataset.")
        return dataset
    return clean_target(dataset)['Price']

def log_target(target):
    extracted_target = target_extraction(target)
    return np.log(extracted_target)


def comparison(x_test,y_test, pipeline, model, confidence = 0.95):
    X_test_prepared = pipeline.transform(x_test) 
    final_predictions = model.predict(X_test_prepared)
    predictions_usd = np.exp(final_predictions)
    Y_test_usd = np.exp(y_test)
    final_mse_usd = mean_squared_error(Y_test_usd, predictions_usd)
    final_rmse_usd = np.sqrt(final_mse_usd)
    squared_errors_usd = (predictions_usd - Y_test_usd) ** 2
    mean_sq_err_usd = squared_errors_usd.mean()
    standard_error_usd = stats.sem(squared_errors_usd)
    interval_usd = np.sqrt(stats.t.interval(
        confidence, 
        len(squared_errors_usd) - 1, 
        loc=mean_sq_err_usd, 
        scale=standard_error_usd
    ))
    return final_rmse_usd, interval_usd


def save_results_txt(best_params, rmse, confidence_interval, model_name, file_name="models/results/model_report.txt"):
    """
    Saves model metrics and hyperparameters to a text file.
    Automatically creates the destination folder if it doesn't exist.
    """
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"TRAINING REPORT - {timestamp}\n")
        f.write(f"Model: ${model_name} with RandomizedSearchCV\n")
        f.write("="*50 + "\n")
        f.write("BEST HYPERPARAMETERS FOUND:\n")

        for param, value in best_params.items():
            f.write(f" - {param}: {value}\n")
        
        f.write("\nTEST SET PERFORMANCE:\n")
        f.write(f" - Final RMSE: ${rmse:.2f}\n")
        f.write(f" - 95% Confidence Interval: ${confidence_interval[0]:.2f} to ${confidence_interval[1]:.2f}\n")
        f.write("="*50 + "\n")
        
    print(f"Report successfully saved to: {file_name}")

def export_model(model, pipeline, file_name="models/final_model_prod.pkl"):
    """
    Exports the model and the pipeline as a single dictionary object.
    Automatically creates the destination folder if it doesn't exist.
    """
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data_to_save = {
        "model": model,
        "full_pipeline": pipeline
    }
    
    joblib.dump(data_to_save, file_name, compress=3)
    print(f"Model and Pipeline exported to: {file_name}")