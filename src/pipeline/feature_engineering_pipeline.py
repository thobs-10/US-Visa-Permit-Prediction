from loguru import logger
from zenml import pipeline
from src.components.feature_engineering import (
    load_data,
    feature_extraction,
    removing_outliers,
    feature_transformations,
    save,
)


@pipeline
def run_feature_engineering() -> None:
    try:
        logger.info("Starting feature engineering pipeline")
        processed_data = load_data()
        df = feature_extraction(processed_data)
        df = removing_outliers(df)
        df, X, y = feature_transformations(df)
        save(X, y)
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        raise e


if __name__ == "__main__":
    run_feature_engineering()
