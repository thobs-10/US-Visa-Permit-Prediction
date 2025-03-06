import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from loguru import logger
from src.entity.config_entity import ModelTrainingConfig
from src.entity.config_entity import FeatureEngineeringConfig
from src.components.component import Component
from src.utils.main_utils import (
    get_models,
    train_and_save,
    tracking_details_init,
    logging_metrics,
    feature_scaling,
)
from zenml import step
from feast import FeatureStore


load_dotenv()


@step(enable_cache=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feature engineered data from the feast feature store"""
    try:
        logger.info("Loading feature engineered data from feature store.")
        features_df = pd.read_parquet(
            os.path.join(
                FeatureEngineeringConfig.feature_engineering_dir, "features.parquet"
            )
        )
        target_df = pd.read_csv(
            os.path.join(FeatureEngineeringConfig.feature_engineering_dir, "target.csv")
        )
        fs = FeatureStore(
            repo_path="/Users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/US-Visa-Permit-Prediction/my_feature_store/feature_repo",
        )
        fs.get_historical_features(
            entity_df=features_df,
            features=fs.get_feature_service("passenger_features"),
        )
        logger.info("Processed data loaded and converted to NumPy arrays successfully")
        return features_df, target_df
    except Exception as e:
        raise e


# @step
def split_data(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    logger.info("Splitting data into training, validation, and test sets")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.dropna(inplace=True)

    target = y.iloc[:, 1]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, target, train_size=0.7, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=42
    )
    logger.info("Data split completed successfully")
    return X_train, y_train, X_test, y_test, X_valid, y_valid


def train_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> Tuple[str, ColumnTransformer]:
    models = get_models()
    logger.info("Training model phase")
    experiment = tracking_details_init()
    for model_name, model in models.items():
        experiment.add_tag(f"Model: {model_name}")
        with experiment.train():
            logger.info(f"Training {model_name}")
            column_transformer, X_train_scaled, X_valid_scaled = feature_scaling(
                X_train,
                X_valid,
            )
            y_pred, model_path = train_and_save(
                X_train_scaled, X_valid_scaled, y_train, model_name, model
            )
            experiment.log_model(f"{model_name}", model_path)
            logging_metrics(experiment, y_pred, y_valid)

    exp_key = experiment.get_key()
    logger.info("models trained and evaluated successfully")
    return exp_key, column_transformer


def save_preprocessor(
    preprocessor: ColumnTransformer, filename: str = "preprocessor.joblib"
) -> None:
    try:
        logger.info("Saving preprocessor..")
        os.makedirs(ModelTrainingConfig.model_artifact_dir, exist_ok=True)
        output_path = os.path.join(ModelTrainingConfig.model_artifact_dir, filename)
        joblib.dump(preprocessor, output_path)
    except Exception as e:
        raise e


def save_training_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    filename: str = "training_data",
) -> None:
    logger.info("Saving training data to parquet format")
    X_train.to_parquet(os.path.join(os.environ["DATA"], filename, "X_train.parquet"))
    X_valid.to_parquet(os.path.join(filename, "X_valid.parquet"))
    y_train.to_csv(os.path.join(filename, "y_train.csv"))
    y_valid.to_csv(os.path.join(filename, "y_valid.csv"))
    logger.debug("Training data saved successfully")
