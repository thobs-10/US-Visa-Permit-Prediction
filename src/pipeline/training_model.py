import os
import sys

from src.logger import logging
from src.Exception import AppException

from src.components.model_training import ModelTraining

from src.entity.artifact_entity import ModelTrainingArtifact
from src.entity.config_entity import ModelTrainingConfig, ModelTrainingPipelineConfig

from src.entity.config_entity import FeatureEngineeringConfig

class ModelTrainingPipeline:
    def __init__(self, config: ModelTrainingConfig, artifact : ModelTrainingArtifact, featureEngineeringConfig: FeatureEngineeringConfig):
        self.config = config
        self.artifact = artifact
        self.feature_engineering_config = featureEngineeringConfig
        # self.model_training = ModelTraining(self.artifact)
    
    def run_model_training(self) -> ModelTrainingArtifact:
        try:
            logging.info("Starting model training pipeline")
            model_training = ModelTraining(self.config, self.artifact)
            X, y = model_training.load_data(self.feature_engineering_config)
            X_train, X_valid, X_test, y_train, y_valid, y_test = model_training.split_data(X, y)
            exp_key = model_training.train_model(X_train, X_valid, y_train, y_valid)
            hyperparameter_exp_key = model_training.hyperparameter_tuning(exp_key, X_train, y_train, X_valid, y_valid)
            model_training.evaluate_model(hyperparameter_exp_key, X_test, y_test)
            model_training.register_model(hyperparameter_exp_key)
            # model_training.model_registration()
        except Exception as e:
            raise AppException(e, sys)
