from loguru import logger
from zenml import pipeline

from src.components.feature_engineering import (
    feature_extraction,
    feature_transformations,
    load_data,
    removing_outliers,
    save,
    save_to_feast_feature_store,
)


@pipeline
def run_feature_engineering() -> None:
    try:
        logger.info("Starting feature engineering pipeline")
        processed_data = load_data()
        df = feature_extraction(processed_data)
        df = removing_outliers(df)
        df, X, y = feature_transformations(df)
        save(X, y, df)
        save_to_feast_feature_store()
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        raise e


if __name__ == "__main__":
    run_feature_engineering()
