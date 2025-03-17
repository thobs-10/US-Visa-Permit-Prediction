import os
import pandas as pd
from loguru import logger
import joblib
from typing import Union

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.entity.config_entity import ModelTuningConfig
from src.utils.main_utils import log_metrics_terminal


def model_evaluation(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame,
    model: Union[
        LogisticRegression,
        RandomForestClassifier,
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        SVC,
        AdaBoostClassifier,
    ],
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


def save_model_pipeline(model_pipeline: Pipeline) -> None:
    try:
        model_path = os.path.join(
            ModelTuningConfig.model_artifact_dir, " model_pipeline"
        )
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model_pipeline, os.path.join(model_path, "model_pipeline.pkl"))
        logger.info(f"Saving model pipeline to {model_path}")
    except Exception as e:
        raise e
