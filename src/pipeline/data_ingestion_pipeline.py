from loguru import logger
from zenml import pipeline

from src.components.data_ingestion import (
    handling_data_type,
    handling_duplicates,
    handling_null_values,
    load_data,
    save,
)


# class DataIngestionPipeline:
@pipeline
def run_data_ingestion() -> None:
    try:
        logger.info("Starting data ingestion pipeline")
        raw_data = load_data()
        cleaned_data = handling_null_values(raw_data)
        cleaned_data = handling_data_type(cleaned_data)
        cleaned_data = handling_duplicates(cleaned_data)
        save(cleaned_data)
        logger.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        raise e


if __name__ == "__main__":
    run_data_ingestion()
