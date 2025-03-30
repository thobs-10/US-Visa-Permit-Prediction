import os
from datetime import datetime

import joblib
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from zenml import step

from src.entity.config_entity import ModelTuningConfig
from src.utils.main_utils import log_metrics_terminal


@step
def model_evaluation(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: LogisticRegression | RandomForestClassifier | DecisionTreeClassifier | GradientBoostingClassifier | KNeighborsClassifier | SVC | AdaBoostClassifier,
    column_transformer: ColumnTransformer,
    threshold: float = 0.50,
) -> Pipeline:
    logger.info(" Starting the model evaluation phase")
    model_pipeline = make_pipeline(column_transformer, model)
    logger.info("Model prediction phase")
    y_pred = model_pipeline.predict(X_test)
    y_test_numpy = y_test.to_numpy()
    metrics_dict = log_metrics_terminal(y_pred, y_test_numpy)  # type: ignore
    if metrics_dict["Accuracy"] and metrics_dict["F1 Score"] < threshold:
        logger.error("Model evaluation failed. F1 score is less than threshold.")
        raise ValueError("Model evaluation failed, accuracy is less than threshold.")
    logger.info("Completed model evaluation.")
    return model_pipeline


@step
def save_model_pipeline(model_pipeline: Pipeline) -> None:
    try:
        saving_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(ModelTuningConfig.model_artifact_dir, " model_pipeline")
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model_pipeline, os.path.join(model_path, f"model_pipeline_{saving_timestamp}.pkl"))
        logger.info(f"Saving model pipeline to {model_path}")
    except Exception as e:
        raise e
