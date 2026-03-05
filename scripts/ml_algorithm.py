from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint


def train_RFR_random(dataset_prepared, target):
    param_dist = {
        'n_estimators': randint(100, 500),        
        'max_features': ['sqrt', 'log2', None],   
        'max_depth': [None, 10, 20, 30, 40, 50],  
        'min_samples_split': randint(2, 11),      
        'min_samples_leaf': randint(1, 5),        
        'bootstrap': [True, False]               
    }

    new_forest_reg = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=new_forest_reg,
        param_distributions=param_dist,
        n_iter=100,                                
        cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        random_state=42,                          
        n_jobs=-1                                
    )


    print("Starting the hyperparameter fitting. This can take some time...")
    random_search.fit(dataset_prepared, target)
    print("Finished!")

    return random_search.best_estimator_, random_search.best_params_

from xgboost import XGBRegressor
from scipy.stats import randint, uniform

def train_XGB_random(dataset_prepared, target):
    param_dist = {
        'n_estimators': randint(100, 500),        
        'learning_rate': [0.01, 0.05, 0.1, 0.2],   
        'max_depth': randint(3, 12),              
        'min_child_weight': randint(1, 10),      
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }


    xgb_reg = XGBRegressor(random_state=42, objective='reg:squarederror')

    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=100,                                
        cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        random_state=42,                           
        n_jobs=-1                                  
    )

    print("Starting the hyperparameter fitting. This can take some time...")
    random_search.fit(dataset_prepared, target)
    print("Finished!")

    return random_search.best_estimator_, random_search.best_params_