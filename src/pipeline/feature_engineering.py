import os
import sys

from src.logger import logging
from src.Exception import AppException
from src.components.feature_engineering import FeatureEngineering

from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact
from src.entity.artifact_entity import DataIngestionArtifact

class FeatureEngineeringPipeline:
    def __init__(self, config: FeatureEngineeringConfig, artifact : FeatureEngineeringArtifact, data_ingestion_artifact : DataIngestionArtifact):
        self.config = config
        self.artiffact = artifact
        self.data_ingestion_artifact = data_ingestion_artifact
    
    def run_feature_engineering(self)-> FeatureEngineeringArtifact:
        try:
            logging.info("Starting feature engineering pipeline")
            feature_engineering = FeatureEngineering(self.artiffact, self.config)
            processed_data = feature_engineering.load_cleaned_data(self.data_ingestion_artifact)
            data = feature_engineering.feature_extraction(processed_data)
            df, y = feature_engineering.feature_transformations(data)
            X = feature_engineering.feature_scaling(df)
            X_res, y_res = feature_engineering.resampling_dataset(X, y)
            feature_engineering.save_processed_data(X_res, y_res)
            logging.info("Feature engineering pipeline completed successfully")
        except Exception as e:
            raise AppException(e, sys)