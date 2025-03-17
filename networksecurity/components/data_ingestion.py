from networksecurity.entity.config_entity import DataIngestionConfig
from  networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.entity.artifacts_entity import DataIngestionArtifact
import os
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from typing import List
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")





class DataIngestion:
    def __init__(self,data_ingetsion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingetsion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        


    
    def export_collection_as_dataframe(self):
        database_name=self.data_ingestion_config.database_name
        collection_name=self.data_ingestion_config.collection_name
        mongo_client=pymongo.MongoClient(MONGO_DB_URL)
        collection=mongo_client[database_name][collection_name]
        df=pd.DataFrame(list(collection.find()))
        if "_id" in df.columns.to_list():
              df=df.drop(columns=["_id"],axis=1)


        df.replace({np.nan:None},inplace=True)
        return df
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
         
        try:
              feature_store_file_path=self.data_ingestion_config.feature_store_file_path
              dir_path=os.path.dirname(feature_store_file_path)
              os.makedirs(dir_path,exist_ok=True)
              dataframe.to_csv(feature_store_file_path,index=False,header=True)
              return dataframe
        except Exception as e:
              raise NetworkSecurityException(e,sys)
        
    def train_test_split(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)    
            logging.info(f"train test split done")
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"exporting train data to feature store")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            logging.info(f"exporting test data to feature store")
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"train test split done")

            
        
    

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

        


    def initiate_data_ingestion(self):
            
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.train_test_split(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            tested_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:      
             raise NetworkSecurityException(e,sys)
        
            
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()