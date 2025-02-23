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


load_dotenv()


class ModelTraining(Component):
    def __init__(
        self,
        model_training_config: ModelTrainingConfig = ModelTrainingConfig(),
        feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
    ):
        self.model_training_config = model_training_config
        self.feature_engineering_config = feature_engineering_config

    def load_data(self) -> tuple:
        try:
            logger.info(
                "Loading feature engineered data(features.parquet and target.csv)"
            )
            transformed_features_file_path = os.path.join(
                self.feature_engineering_config.feature_engineering_dir
            )
            X_df = pd.read_parquet(
                os.path.join(transformed_features_file_path, "features.parquet")
            )
            y_df = pd.read_csv(
                os.path.join(transformed_features_file_path, "target.csv")
            )
            logger.info(
                "Processed data loaded and converted to NumPy arrays successfully"
            )
            return X_df, y_df
        except Exception as e:
            raise e

    def split_data(self, X, y) -> tuple:
        logger.info("Splitting data into training, validation, and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=0.7, random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=1 / 3, random_state=42
        )
        logger.info("Data split completed successfully")
        return X_train, y_train, X_test, y_test, X_valid, y_valid

    def train_model(
        self,
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
        self, preprocessor: ColumnTransformer, filename: str = "preprocessor.joblib"
    ) -> None:
        try:
            logger.info("Saving preprocessor..")
            path = os.environ["ARTIFACTS_PATH"]
            os.makedirs(path, exist_ok=True)
            output_path = os.path.join(path, filename)
            joblib.dump(preprocessor, output_path)
        except Exception as e:
            raise e

    def save_training_data(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        filename: str = "training_data",
    ) -> None:
        logger.info("Saving training data to parquet format")
        X_train.to_parquet(
            os.path.join(os.environ["DATA"], filename, "X_train.parquet")
        )
        X_valid.to_parquet(os.path.join(filename, "X_valid.parquet"))
        y_train.to_csv(os.path.join(filename, "y_train.csv"))
        y_valid.to_csv(os.path.join(filename, "y_valid.csv"))
        logger.debug("Training data saved successfully")
