import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
from joblib import Memory
from typing import Tuple, Dict, Any
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from loguru import logger
from src.entity.config_entity import ModelTrainingConfig
from src.entity.config_entity import FeatureEngineeringConfig
from src.utils.main_utils import (
    get_models,
    train_and_save,
    log_mlflow_metrics,
    feature_scaling,
    get_best_performing_model,
)
from zenml import step
from feast import FeatureStore
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
memory = Memory(location="cachedir", verbose=0)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


@step(enable_cache=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feature engineered data from the feast feature store"""
    try:
        logger.info("Loading feature engineered data from feature store.")
        features_df = pd.read_parquet(
            os.path.join(FeatureEngineeringConfig.feature_engineering_dir, "features.parquet")
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
@memory.cache
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
    X_train, X_temp, y_train, y_temp = train_test_split(X, target, train_size=0.7, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=42
    )
    logger.info("Data split completed successfully")
    return X_train, y_train, X_test, y_test, X_valid, y_valid


def train_model_parallel(
    X_train_scaled: np.ndarray[Any, Any] | spmatrix,
    X_valid_scaled: np.ndarray[Any, Any] | spmatrix,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> Dict[np.ndarray, str]:
    models = get_models()
    logger.info("Training model using parallelization of threads")
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                train_and_save,
                X_train_scaled,
                X_valid_scaled,
                y_train,
                y_valid,
                model_name,
                model,
            ): model_name
            for model_name, model in models.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                y_pred, artifact_path = future.result()
                results[model_name] = (y_pred, artifact_path)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

    return results


# @step
def train_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> Tuple[ColumnTransformer, str, str]:
    logger.info("Training model phase")
    mlflow.set_experiment("Model Training Phase")
    mlflow.set_experiment_tag("model-training", "v1.0.0")

    column_transformer, X_train_scaled, X_valid_scaled = feature_scaling(
        X_train,
        X_valid,
    )

    results = train_model_parallel(
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
    )
    model_performance_dict = {}
    for model_name, (y_pred, model_path) in results.items():
        logger.info(f"logging results for model: {model_name}: path: {model_path}")
        with mlflow.start_run():
            mlflow.set_tag("model-training:", model_name)
            signature = infer_signature(X_valid_scaled, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model_path,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name,
            )
            logger.info(f"y_pred: {y_pred} and type:{type(y_pred)}")
            metrics_dict = log_mlflow_metrics(np.array(y_pred), y_valid)
            mlflow.log_metrics(metrics=metrics_dict)
            model_performance_dict[model_name] = (metrics_dict["Accuracy"], model_path)
    logger.info("models trained and evaluated successfully")
    best_model, best_model_path = get_best_performing_model(model_performance_dict)
    chosen_model_path = os.path.join(model_path, best_model)
    return column_transformer, chosen_model_path, best_model


# @step
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


# @step
def save_training_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    filename: str = "training_data",
) -> None:
    logger.info("Saving training data to parquet format")
    os.makedirs(
        os.path.join(FeatureEngineeringConfig.feature_engineering_dir, filename),
        exist_ok=True,
    )
    X_train.to_parquet(
        os.path.join(
            FeatureEngineeringConfig.feature_engineering_dir,
            filename,
            "X_train.parquet",
        )
    )
    X_valid.to_parquet(
        os.path.join(
            FeatureEngineeringConfig.feature_engineering_dir,
            filename,
            "X_valid.parquet",
        )
    )
    logger.debug("Training data saved successfully")
    y_train.to_csv(
        os.path.join(
            FeatureEngineeringConfig.feature_engineering_dir,
            filename,
            "y_train.csv",
        )
    )
    y_valid.to_csv(
        os.path.join(
            FeatureEngineeringConfig.feature_engineering_dir,
            filename,
            "y_valid.csv",
        )
    )
    logger.debug("Training data saved successfully")
