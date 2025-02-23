import os
import pandas as pd
import numpy as np
from loguru import logger
import joblib
from typing import Optional, Union

from comet_ml import Experiment
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
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
from src.utils.main_utils import feature_scaling
from src.entity.config_entity import ModelTuningConfig
from src.utils.main_utils import log_metrics_terminal


def evaluate_model(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
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
    experiment: Experiment,
    exp_key: Optional[str] = None,
) -> BaseEstimator:
    logger.info(" Starting the evaluation phase")
    column_transformer, X_valid_scaled, X_test_scaled = feature_scaling(
        X_val,
        X_test,
    )
    y_pred = model.predict(X_test_scaled)
    log_metrics_terminal(y_pred, y_test)
    logger.info("Completed model evaluation.")

    return model


def save_model_pipeline(
    model: BaseEstimator, column_transformer: ColumnTransformer
) -> None:
    try:
        model_pipeline = make_pipeline(column_transformer, model)
        model_path = os.path.join(
            ModelTuningConfig.model_artifact_dir, " model_pipeline"
        )
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Saving model pipeline to {model_path}")
        joblib.dump(model_pipeline, model_path)
    except Exception as e:
        raise e
