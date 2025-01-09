import os
import glob
import pandas as pd
import numpy as np
import sys
import click
from dotenv import load_dotenv
import joblib
import requests

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, KFold, cross_val_score

from src.logger import logging
from loguru import logger
from src.Exception import AppException
from src.entity.artifact_entity import ModelTrainingArtifact
from src.entity.config_entity import ModelTrainingConfig, ModelTrainingPipelineConfig
from src.entity.config_entity import FeatureEngineeringConfig
from src.components.component import Component
from src.utils.main_utils import get_models, train_and_save, compute_accuracy_metrics, tracking_details_init

import comet_ml
from comet_ml import API, Experiment
from comet_ml.query import Metric
import mlflow

load_dotenv()

class ModelTraining(Component):
    def __init__(self, model_training_config: ModelTrainingConfig = ModelTrainingConfig(),
                 feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        
        self.model_training_config = model_training_config
        self.feature_engineering_config = feature_engineering_config
        
    
    def load_data(self) -> tuple:
        try:
            logger.info("Loading feature engineered data(features.parquet and target.csv)")
            transformed_features_file_path = os.path.join(self.feature_engineering_config.feature_engineering_dir)
            X_df = pd.read_parquet(os.path.join(transformed_features_file_path, "features.parquet"))
            y_df = pd.read_csv(os.path.join(transformed_features_file_path, "target.csv"))  
            logger.info("Processed data loaded and converted to NumPy arrays successfully")
            return X_df, y_df
        except Exception as e:
            raise AppException(e, sys)
    
    def split_data(self, X, y) -> tuple:
        logger.info("Splitting data into training, validation, and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
        logger.info("Data split completed successfully")
        return X_train, y_train, X_test, y_test, X_valid, y_valid

    def train_model(self, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> str:
        models = get_models()
        logger.info("Training model phase")
        experiment = tracking_details_init()
        # go through each model in the dict, train, valid and test and keep track of the results, model, metrics in comet_ml
        for model_name, model in models.items():
            experiment.add_tag(f"Model: {model_name}")
            with experiment.train():
                logger.info(f"Training {model_name}")
                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_valid_scaled = sc.transform(X_valid)
                # model_pipeline = make_pipeline(StandardScaler(), model)
                y_pred, model_path = train_and_save(X_train_scaled, X_valid_scaled, y_train, model_name, model)
                experiment.log_model(f"{model_name}", model_path)
                acc, f1, precision, recall, roc_auc = compute_accuracy_metrics(y_pred, y_valid)

                # Log metrics for this model
                experiment.log_metric("Accuracy", acc)
                experiment.log_metric("F1 Score", f1)
                experiment.log_metric("Precision", precision)
                experiment.log_metric("Recall", recall)
                experiment.log_metric("ROC", roc_auc)

        exp_key = experiment.get_key()
        logger.info("models trained and evaluated successfully")
        return exp_key

    def save_traing_data(self, X_train: pd.DataFrame, X_valid: pd.DataFrame,
                         y_train: pd.Series, y_valid: pd.Series, filename: str= 'training_data')-> None:
        
        logger.info("Saving training data to parquet format")
        X_train.to_parquet(os.path.join(os.getenv("DATA"), filename, "X_train.parquet"))
        X_valid.to_parquet(os.path.join(filename, "X_valid.parquet"))
        y_train.to_csv(os.path.join(filename, "y_train.csv"))
        y_valid.to_csv(os.path.join(filename, "y_valid.csv"))
        logger.debug("Training data saved successfully")



