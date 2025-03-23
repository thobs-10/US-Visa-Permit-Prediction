from src.pipeline.data_ingestion_pipeline import run_data_ingestion
from src.pipeline.feature_engineering_pipeline import run_feature_engineering
from src.pipeline.training_pipeline import run_model_training


def run_all_pipelines() -> None:
    run_data_ingestion()
    run_feature_engineering()
    run_model_training()


if __name__ == "__main__":
    run_all_pipelines()
