from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str = "data\\raw_data\\Visadataset.csv"
    processed_data_path: str = "data\\processed_data"

@dataclass
class FeatureEngineeringArtifact:
    feature_engineered_data_path: str = "data\\feature_engineered_data"
    processed_data_path: str = "data\\process_data"

@dataclass
class ModelTrainingArtifact:
    model_path: str = "models\\model_output"
    feature_engineered_data_path: str = "data\\feature_engineered_data"
