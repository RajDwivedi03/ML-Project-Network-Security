from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
import sys
from networksecurity.entity.config_entity import TrainingPipelineConfig


if __name__=="__main__":
    try:
        TrainingPipelineConfig=TrainingPipelineConfig()
        DataIngestionConfig=DataIngestionConfig(TrainingPipelineConfig)
        DataIngestion=DataIngestion(DataIngestionConfig)
        logging.info("initiate data ingestion")
        dataingestionartifact=DataIngestion.initiate_data_ingestion()
        print(dataingestionartifact)

     
    except Exception as e:
        raise NetworkSecurityException(e,sys)