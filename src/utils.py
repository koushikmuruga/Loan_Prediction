import pandas as py
import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from src.logger import logging

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        #dir_path = os.path.dirname(path)

        #os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(path):
    with open(path,"rb") as file_obj:
        return pickle.load(file_obj)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:

        report={}

        for m in range(len(list(models))):
            model=list(models.values())[m]
            para=param[list(models.keys())[m]]

            gridSearch = GridSearchCV(model,para,cv=5)
            gridSearch.fit(x_train,y_train)

            model.set_params(**gridSearch.best_params_)
            model.fit(x_train,y_train)

            train_pred = model.predict(x_train)

            test_pred = model.predict(x_test)

            train_model_score = accuracy_score(y_train, train_pred)

            test_model_score = accuracy_score(y_test, test_pred)

            logging.info(f"Model: {model}, Train Score:{train_model_score}, Test Score: {test_model_score}")

            report[list(models.keys())[m]] = test_model_score

            return report
        
    except Exception as e:
        raise CustomException(e,sys)
