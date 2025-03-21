import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifacts_entity import DataTransformationArtifact,DataValidationArtifact
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.utils.main_utils.utils import save_object,save_numpy_array_data



class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:    
            raise NetworkSecurityException(e,sys)     


    def get_data_transformer_object(cls)->Pipeline:
        logging.info("Method of transfromation class")
        try:
            imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor:Pipeline=Pipeline([("imputer",imputer)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("INITIATE DATA TRANSFORMATION")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)



            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)

            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)


            preprocessor=self.get_data_transformer_object()
            preprocessor_obj=preprocessor.fit(input_feature_train_df)

            
            transformed_input_train_features=preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_features=preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[transformed_input_train_features,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test_features,np.array(target_feature_test_df)]
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)  
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_obj)
            save_object("final_models/preprocessor.pkl",preprocessor_obj)


            data_transformation_artifact=DataTransformationArtifact(transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,transformed_object_file_path=self.data_transformation_config.transformed_object_file_path)

            return data_transformation_artifact

            



        except Exception as e:
            raise NetworkSecurityException(e,sys)


