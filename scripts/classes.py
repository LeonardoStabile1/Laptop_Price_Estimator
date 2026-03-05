import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, clean_data=True, op_sys_col='OpSys'):
        self.clean_data = clean_data
        self.op_sys_col = op_sys_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.clean_data:
            return X
        X_copy = X.copy()

        if self.op_sys_col in X_copy.columns:
            X_copy[self.op_sys_col] = X_copy[self.op_sys_col].fillna("No Operation System")

        if ('Memory') in X_copy.columns:
            X_copy = X_copy.drop(columns=['Memory'])

        if ('indx') in X_copy.columns:
            X_copy = X_copy.drop(columns=['indx'])

        return X_copy
    
class GPUTierExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, gpu_col='gpu_name'):
        self.gpu_col = gpu_col

    def fit(self, X, y=None):
        return self

    def _categorize_gpu(self, gpu_string):
        gpu_string = str(gpu_string).upper()
        if 'HD GRAPHICS' in gpu_string or 'UHD' in gpu_string or 'IRIS' in gpu_string:
            return 'Integrated_Graphics'
        elif 'GTX' in gpu_string or 'RTX' in gpu_string:
            return 'Nvidia_Premium'
        elif 'QUADRO' in gpu_string or 'FIREPRO' in gpu_string:
            return 'Workstation_GPU'
        elif 'RADEON' in gpu_string or 'RX ' in gpu_string:
            return 'AMD_Radeon'
        elif 'GEFORCE' in gpu_string or 'NVIDIA' in gpu_string:
            return 'Nvidia_Standard'
        else:
            return 'Other'

    def transform(self, X):
        X_copy = X.copy()
        
        if self.gpu_col in X_copy.columns:
            X_copy['gpu_tier'] = X_copy[self.gpu_col].apply(self._categorize_gpu)
            X_copy = X_copy.drop(columns=[self.gpu_col])
            
        return X_copy

