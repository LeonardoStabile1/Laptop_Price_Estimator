import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def cut_price_outliers(df):
    if 'Price' not in df.columns:
        print(" 'Price' Column not include on the dataset.")
        return df
    new_df = df[df['Price'] < 150000].copy()
    return new_df


def target_extraction(dataset):
    return clean_target(dataset)

def log_target(target):
    extracted_target = target_extraction(target)
    return np.log(extracted_target)

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
    return price_to_usd(dataset)['Price']


def import_data(dataset_url):
    dataset=pd.read_csv(dataset_url)
    return dataset
    

def split_data(dataset, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
    return train_set, test_set


