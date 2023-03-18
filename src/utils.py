'''
This file will contain all the common functionalities that we can use for the entire project
'''

import os
import sys
import numpy as np
import pandas as pd
import dill #used to dump python objects to file
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from src.logger import logging

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            model_key=list(models.keys())[i]
            para=params[list(models.keys())[i]]

            # Search for best hyperparameters
            grid = GridSearchCV(estimator=model, param_grid=para, cv=3)           
            grid.fit(X_train,y_train)

            logging.info(f"{model_key}")
            # Get the results
            logging.info(grid.best_score_)
            logging.info(grid.best_estimator_)
            logging.info(grid.best_params_)
            
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train,y_train)  #Train Model
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score= r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e,sys)