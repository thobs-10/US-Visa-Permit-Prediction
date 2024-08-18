import os
import sys

from src.logger import logging
from src.Exception import AppException
from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.entity.config_entity import DataPipelineConfig

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig, artifact: DataIngestionArtifact):
        self.config = config
        self.artifact = artifact
        # self.data_ingestion = DataIngestion(self.artifact)
        
    def run_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion pipeline")
            data_ingestion = DataIngestion(self.artifact)
            raw_data = data_ingestion.load_data()
            cleaned_data = data_ingestion.data_cleaning(raw_data)
            data_ingestion.save_data(cleaned_data)
            logging.info("Data ingestion pipeline completed successfully")
            return self.artifact
        except Exception as e:
            raise AppException(e, sys)
        

# if __name__ == "__main__":
# # Define configurations
# data_ingestion_config = DataIngestionConfig(data_pipeline_config)
# data_pipeline_config = DataIngestionPipeline(artifacts_dir='path/to/artifacts')


# # Define artifacts
# data_ingestion_artifact = DataIngestionArtifact(
#     raw_data_path=data_ingestion_config.raw_folder,
#     processed_data_path=data_ingestion_config.processed_folder
# )

# # Create and start data ingestion pipeline
# data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config, artifact=data_ingestion_artifact)
# data_ingestion_pipeline.start_data_ingestion()


