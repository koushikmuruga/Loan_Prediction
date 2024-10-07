import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestion:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logging.info('Entered Data Ingestion')

        try:
            train_path=os.path.join('Data_Folder','train.csv')
            test_path=os.path.join('Data_Folder','test.csv')
            raw_path=os.path.join('Data_Folder','raw.csv')

            os.makedirs('Data_Folder',exist_ok=True)

            df=pd.read_csv('notebooks\stud.csv')
            df.to_csv(raw_path,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Train_test split done')

            train_set.to_csv(train_path,index=False,header=True)
            test_set.to_csv(test_path,index=False,header=True)
            logging.info('Data Ingestion Completed')

            return train_path,test_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    dt_obj=DataTransformation()
    train_arr,test_arr=dt_obj.initiate_data_transformation(train_path,test_path)

    model_obj=ModelTrainer()
    accuracy=model_obj.initiate_model_trainer(train_arr,test_arr)