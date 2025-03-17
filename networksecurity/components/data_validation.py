from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.artifacts_entity import DataIngestionArtifact
from networksecurity.entity.artifacts_entity import DataValidationArtifact
from networksecurity.logging import logger
import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from networksecurity.utils.main_utils.utils import read_yaml_file
import pandas as pd
import numpy as np
import os
import sys
from networksecurity.utils.main_utils.utils import write_yaml_file


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:    
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:    
            raise NetworkSecurityException(e,sys)   



    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys) 
        
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        
        try:
            expected_numerical_columns = {field["name"] for field in self._schema_config.get("columns", []) if field.get("type") == "integer"}

            actual_columns = set(dataframe.columns)

            logging.info(f"Expected numerical columns: {expected_numerical_columns}")
            logging.info(f"Actual columns: {actual_columns}")

            missing_numerical_columns = expected_numerical_columns - actual_columns
            if missing_numerical_columns:
                logging.warning(f" Missing numerical columns: {missing_numerical_columns}")
                return False 
            else:
                logging.info(" All numerical columns exist in the DataFrame.")
                return True

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_sample_dist=ks_2samp(d1,d2)
                if threshold<=is_sample_dist.pvalue:
                    is_found=False

                else:
                    is_found=True
                    status=False
                report.update({column:{"p-value":is_sample_dist.pvalue,"is_same_distribution":is_found}})
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(drift_report_file_path,content=report)
            
                
          
          
        except Exception as e:
          raise NetworkSecurityException(e,sys)
            


        

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
       
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.tested_file_path

            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            #validate number of columns
            status=self.validate_number_of_columns(train_dataframe)
            if not status:
                error_message=f"Number of columns doesn't match expected"

            status=self.validate_number_of_columns(test_dataframe)
            if not status:
                error_message=f"Number of columns doesn't match expected"

            #validate numerical columns
            status=self.validate_numerical_columns(train_dataframe)
            if not status:
                error_message=f"Numerical columns doesn't match expected"

            status=self.validate_numerical_columns(test_dataframe)
            if not status:
                error_message=f"Numerical columns doesn't match expected"

            #lets check drift status
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,index=False,header=True)
            
            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.tested_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
                )

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
            
            

          
            

           
            
                
          
