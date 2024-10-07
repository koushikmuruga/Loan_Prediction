import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,FunctionTransformer
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path=os.path.join('Data_Folder','preprocessor.pkl')      #to save the preprocessor object

    def get_data_transformer_object(self):  #Function for data transformation
        
        try:
            numerical_columns = ['Total_Income', 'LoanAmount','Loan_Amount_Term']
            categorical_columns_nominal = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
            categorical_columns_ordnial=['Credit_History']

            logging.info(f"Categorical columns: {categorical_columns_nominal}")
            logging.info(f"Categorical columns: {categorical_columns_ordnial}")
            logging.info(f"Numerical columns: {numerical_columns}")


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            num_pipeline_1= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("gaussian transform",FunctionTransformer(np.log1p, validate=True))
                ]
            )
            cat_pipeline_nominal=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline_ordinal=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",LabelEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,['LoanAmount','Loan_Amount_Term']),
                ("num_pipeline_1",num_pipeline_1,['Total_Income'])
                ("cat_pipelines_1",cat_pipeline_nominal,categorical_columns_nominal),
                ("cat_pipelines_2",cat_pipeline_ordinal,categorical_columns_ordnial)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            train_df['Total_Income']=train_df['ApplicantIncome']+train_df['CoapplicantIncome']
            test_df['Total_Income']=test_df['ApplicantIncome']+test_df['CoapplicantIncome']

            target_column_name="Loan_Status"
            columns_to_drop=['ApplicantIncome','CoapplicantIncome','Loan_ID',target_column_name]

            input_feature_train_df=train_df.drop(columns=columns_to_drop,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=columns_to_drop,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_df=target_feature_train_df.map({'Y':1,'N':0})
            target_feature_test_df=target_feature_test_df.map({'Y':1,'N':0})
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (train_arr,test_arr)
        
        except Exception as e:
            raise CustomException(e,sys)
