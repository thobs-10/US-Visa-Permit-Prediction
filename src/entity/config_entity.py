import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataPipelineConfig:
    artifacts_dir: str

@dataclass 
class FeatureEngineeringPipelineConfig:
    feature_engineering_dir: str

@dataclass
class ModelTrainingPipelineConfig:
    model_training_dir: str

@dataclass
class DataIngestionConfig:
    raw_folder: str
    processed_folder: str

    def __init__(self, data_pipeline_config: DataPipelineConfig):
        self.raw_folder = os.path.join(
            data_pipeline_config.artifacts_dir, "raw_data"
        )
        self.processed_folder = os.path.join(
            data_pipeline_config.artifacts_dir, "processed_data"
        )


@dataclass
class FeatureEngineeringConfig:
    feature_engineering_dir: str
    transformed_data_dir: str
    transformed_features_file: str
    transformed_target_file: str

    def __init__(self, feature_engineering_pipeline_config: FeatureEngineeringPipelineConfig):
        self.feature_engineering_dir = os.path.join(
            feature_engineering_pipeline_config.feature_engineering_dir, "feature_engineered_data"
        )
        self.transformed_data_dir = os.path.join(self.feature_engineering_dir, "transformed_data")
        self.transformed_features_file = os.path.join(self.transformed_data_dir, "features.parquet")
        self.transformed_target_file = os.path.join(self.transformed_data_dir, "target.parquet")

@dataclass
class ModelTrainingConfig:
    feature_engineered_data_dir: str
    model_output_dir: str
    model_file: str

    def __init__(self, model_training_pipeline_config: ModelTrainingPipelineConfig, feature_engineering_config: FeatureEngineeringConfig):
        self.feature_engineered_data_dir = feature_engineering_config.feature_engineering_dir
        self.model_output_dir = os.path.join(
            model_training_pipeline_config.model_training_dir, "model_output"
        )
        self.model_file = os.path.join(self.model_output_dir, "trained_model.pkl")
