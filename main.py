from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_training import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
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

        DataValidationConfig=DataValidationConfig(TrainingPipelineConfig)
        DataValidation=DataValidation(DataValidationConfig,dataingestionartifact)
        logging.info("initiate data validation")
        datavalidationartifact=DataValidation.initiate_data_validation()
        print(datavalidationartifact)
        datatransformationconfig=DataTransformationConfig(TrainingPipelineConfig)
        DataTransformation=DataTransformation(datatransformationconfig,datavalidationartifact)
        logging.info("initiate data transformation")
        datatransformationartifact=DataTransformation.initiate_data_transformation()
        print(datatransformationartifact)
        ModelTrainingConfig=ModelTrainerConfig(TrainingPipelineConfig)
        ModelTrainer=ModelTrainer(ModelTrainingConfig,datatransformationartifact)
        logging.info("initiate model training")
        modeltrainerartifact=ModelTrainer.initiate_model_trainer()
        print(modeltrainerartifact)




     
    except Exception as e:
        raise NetworkSecurityException(e,sys)