import pandas as pd
import joblib
import os
import sys
import numpy as np


sys.path.append(os.path.abspath("scripts"))

def predict_new_data(csv_path, model_path="models/final_model_prod.pkl"):
    """
    Loads the saved model and pipeline, processes a new dataset, 
    and returns the predictions.
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    print(f"Loading model from {model_path}...")
    saved_artifacts = joblib.load(model_path)
    
    model = saved_artifacts["model"]
    full_pipeline = saved_artifacts["full_pipeline"]
    
    print(f"Reading new data from {csv_path}...")
    new_data = pd.read_csv(csv_path)

    print("Preprocessing data...")
    X_processed = full_pipeline.transform(new_data)
    
    print("Generating predictions...")
    predictions = model.predict(X_processed)
    
    new_data['predicted_price'] = np.exp(predictions)
    
    return new_data

if __name__ == "__main__":
    new_laptops_path = "dataset/synthetic_laptops.csv"
    results = predict_new_data(new_laptops_path)
    print(results['predicted_price'].head(10))