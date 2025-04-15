from loguru import logger

from src.components.data_ingestion import (
    handling_data_type,
    handling_duplicates,
    handling_null_values,
    load_raw_data,
    save_processed_data,
)
from src.components.feature_engineering import (
    feature_extraction,
    feature_transformations,
    load_processed_data,
    removing_outliers,
    save,
    save_to_feast_feature_store,
)
from src.components.model_registry import register_model
from src.components.model_training import (
    load_data,
    save_preprocessor,
    save_training_data,
    split_data,
    train_model,
)
from src.components.model_tuning import hyperparameter_tuning
from src.components.model_validation import model_evaluation, save_model_pipeline


def run_data_ingestion() -> None:
    try:
        logger.info("Starting data ingestion pipeline")
        raw_data = load_raw_data()
        cleaned_data = handling_null_values(raw_data)
        cleaned_data = handling_data_type(cleaned_data)
        cleaned_data = handling_duplicates(cleaned_data)
        save_processed_data(cleaned_data)
        logger.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        raise e


def run_feature_engineering() -> None:
    try:
        logger.info("Starting feature engineering pipeline")
        processed_data = load_processed_data()
        df = feature_extraction(processed_data)
        df = removing_outliers(df)
        df, X, y = feature_transformations(df)
        save(X, y, df)
        save_to_feast_feature_store()
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        raise e


def run_model_training() -> None:
    try:
        logger.info("Starting model training pipeline")
        X, y = load_data()
        X_train, y_train, X_test, y_test, X_valid, y_valid = split_data(X, y)
        column_transformer, chosen_model_path, best_model_name = train_model(X_train, X_valid, y_train, y_valid)
        save_preprocessor(column_transformer)
        save_training_data(X_train, X_valid, y_train, y_valid)
        best_model = hyperparameter_tuning(
            X_train,
            X_valid,
            y_train,
            y_valid,
            chosen_model_path,
            best_model_name,
            column_transformer,
        )
        model = model_evaluation(X_test, y_test, best_model, column_transformer)
        save_model_pipeline(model)
        register_model(model)
    except Exception as e:
        raise e
