from networksecurity.entity.artifacts_entity import ClassifiactionMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.metrics import f1_score,precision_score,recall_score
import sys



def get_classification_metric(y_true,y_pred)->ClassifiactionMetricArtifact:
    try:
        model_f1=f1_score(y_true=y_true,y_pred=y_pred)
        model_precision=precision_score(y_true=y_true,y_pred=y_pred)
        model_recall=recall_score(y_true=y_true,y_pred=y_pred)
        classification_metric=ClassifiactionMetricArtifact(f1_score=model_f1,precision_score=model_precision,recall_score=model_recall)
        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e,sys)