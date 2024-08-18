from src.pipeline.data_ingestion import DataIngestionPipeline
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig,DataPipelineConfig

from src.pipeline.feature_engineering import FeatureEngineeringPipeline
from src.entity.artifact_entity import FeatureEngineeringArtifact
from src.entity.config_entity import FeatureEngineeringConfig,FeatureEngineeringPipelineConfig

from src.pipeline.training_model import ModelTrainingPipeline
from src.entity.artifact_entity import ModelTrainingArtifact
from src.entity.config_entity import ModelTrainingConfig, ModelTrainingPipelineConfig

def run_data_ingestion_pipeline():
    # Define the base directory for artifacts
    base_artifact_dir = "data"
    # Create the DataPipelineConfig
    data_pipeline_config = DataPipelineConfig(artifacts_dir=base_artifact_dir)
    # Create the DataIngestionConfig
    data_ingestion_config = DataIngestionConfig(data_pipeline_config=data_pipeline_config) 
    # Create the DataIngestionArtifact
    data_ingestion_artifact = DataIngestionArtifact(
        raw_data_path=data_ingestion_config.raw_folder,
        processed_data_path=data_ingestion_config.processed_folder
    )
   # Create and run the DataIngestionPipeline
    data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config, artifact=data_ingestion_artifact)
    artifact = data_ingestion_pipeline.run_data_ingestion()

def run_feature_engineering_pipeline():
    # Define the base directory for artifacts
    base_artifact_dir = "data"
    # Create the FeatureEngineeringPipelineConfig
    feature_engineering_pipeline_config = FeatureEngineeringPipelineConfig(feature_engineering_dir=base_artifact_dir)
    # Create the FeatureEngineeringConfig
    feature_engineering_config = FeatureEngineeringConfig(
        feature_engineering_pipeline_config=feature_engineering_pipeline_config)
    # Create the FeatureEngineeringArtifact
    feature_engineering_artifact = FeatureEngineeringArtifact(
        feature_engineered_data_path=feature_engineering_config.feature_engineering_dir
    )
    base_artifact_dir = "data"
    # Create the DataPipelineConfig
    data_pipeline_config = DataPipelineConfig(artifacts_dir=base_artifact_dir)
    # Create the DataIngestionConfig
    data_ingestion_config = DataIngestionConfig(data_pipeline_config=data_pipeline_config) 
    # Create the DataIngestionArtifact
    data_ingestion_artifact = DataIngestionArtifact(
        raw_data_path=data_ingestion_config.raw_folder,
        processed_data_path=data_ingestion_config.processed_folder
    )
    # Create and run the FeatureEngineeringPipeline
    feature_engineering_pipeline = FeatureEngineeringPipeline(
        config=feature_engineering_config, artifact=feature_engineering_artifact,
        data_ingestion_artifact=data_ingestion_artifact  # Use the same artifact for data ingestion and feature engineering to save time and resources.
    )
    artifact = feature_engineering_pipeline.run_feature_engineering()

def run_training_model_pipeline():
    # Define the base directory for artifacts
    base_artifact_dir = "data"
    # Create the ModelTrainingPipelineConfig
    model_training_pipeline_config = ModelTrainingPipelineConfig(model_training_dir=base_artifact_dir)
    # Create the FeatureEngineeringPipelineConfig
    feature_engineering_pipeline_config = FeatureEngineeringPipelineConfig(feature_engineering_dir=base_artifact_dir)
    # Create the FeatureEngineeringConfig
    feature_engineering_config = FeatureEngineeringConfig(
        feature_engineering_pipeline_config=feature_engineering_pipeline_config)
    # Create the ModelTrainingConfig
    model_training_config = ModelTrainingConfig(
        model_training_pipeline_config=model_training_pipeline_config, 
        feature_engineering_config=feature_engineering_config)
    # Create the ModelTrainingArtifact
    model_training_artifact = ModelTrainingArtifact()
    # Create and run the ModelTrainingPipeline
    model_training_pipeline = ModelTrainingPipeline(model_training_config, model_training_artifact,feature_engineering_config)
    model_training_pipeline.run_model_training()

if __name__ == "__main__":
    # run_data_ingestion_pipeline()
    # run_feature_engineering_pipeline()
    run_training_model_pipeline()