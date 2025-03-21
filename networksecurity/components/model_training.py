import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifacts_entity import DataTransformationArtifact,ModelTrainerArtifact,DataValidationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import *
from networksecurity.entity.artifacts_entity import ClassifiactionMetricArtifact
from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.ML_Utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.ML_Utils.model.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier ,
    AdaBoostClassifier         
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import mlflow
import dagshub
dagshub.init(repo_owner='RajDwivedi03', repo_name='ML-Project-Network-Security', mlflow=True)


import sys

class ModelTrainer:    
    def __init__(self,model_trainer_config:ModelTrainerConfig,
    data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def track_mlflow(self,best_model,Classsificationmetric):
        with mlflow.start_run():
            f1_score=Classsificationmetric.f1_score
            precision_score=Classsificationmetric.precision_score
            recall_score=Classsificationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
           
        

    def train_model(self,x_train,y_train,x_test,y_test):
       models={
           "Logistic Regression":LogisticRegression(verbose=1),
           "Random Forest":RandomForestClassifier(verbose=1),
           "Gradient Boosting":GradientBoostingClassifier(verbose=1),
           "AdaBoost":AdaBoostClassifier(),
           "Decision Tree":DecisionTreeClassifier(),
           "KNeighborsClassifier":KNeighborsClassifier(),
           "SVC":SVC()

       }
       params={
           

                "Logistic Regression":{
                    
                },
                "Decision Tree": {
                    'criterion':['gini','entropy','log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsClassifier":{},
                "SVC":{}
                
            }
       model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

    
       best_model_score = max(sorted(model_report.values()))
       best_model_name = list(model_report.keys())[
           list(model_report.values()).index(best_model_score)

       ]
       best_model = models[best_model_name]
       y_train_pred = best_model.predict(x_train)

       classification_train_metric=get_classification_metric(y_true=y_train,y_pred=y_train_pred)
       
       y_test_pred = best_model.predict(x_test)
       classification_test_metric=get_classification_metric(y_true=y_test,y_pred=y_test_pred)

       #MLFLOW
       self.track_mlflow(best_model=best_model,Classsificationmetric=classification_train_metric)




       preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
       model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
       os.makedirs(model_dir_path,exist_ok=True)
       Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
       save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_Model)
       save_object("final_models/model.pkl",best_model)

       model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
       train_metric_artifact=classification_train_metric,test_metric_artifact=classification_test_metric)
       return model_trainer_artifact,best_model_name


       if best_model_score<0.6:
           raise Exception("No best model found")
    def initiate_model_trainer(self,)->ModelTrainerArtifact:
        try:
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)


            x_train,y_train,x_test,y_test=(
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:,:-1],
                    test_arr[:,-1]
                )
            model=self.train_model(x_train,y_train,x_test,y_test)
            return model
        except Exception as e:      
             raise NetworkSecurityException(e,sys)