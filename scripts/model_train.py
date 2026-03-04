from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint

def train_RFR(dataset_prepared, target):

    param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
    ]

    new_forest_reg= RandomForestRegressor()

    grid_search = GridSearchCV(new_forest_reg, param_grid,cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(dataset_prepared,target)

    return grid_search.best_estimator_, grid_search.best_params_

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