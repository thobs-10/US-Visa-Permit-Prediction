from typing import Any
import mlflow.artifacts
import pandas as pd
from loguru import logger
from src.utils.main_utils import (
    log_mlflow_metrics,
    load_local_model,
)
from src.entity.config_entity import randomcv_models
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
)
from sklearn.compose import ColumnTransformer
import multiprocessing as mp
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


tracking_uri = mlflow.get_tracking_uri()
mlflow_client = MlflowClient(tracking_uri=tracking_uri)


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    chosen_model_path: str,
    chosen_model_name: str,
    column_transformer: ColumnTransformer,
    max_evals: int = 5,
) -> Any:
    """create new experiment for hyperparameter tuning using mlflow, get model assets from previous experiments and
    perform hyperparameter tuning and log the model parameters"""

    mlflow.set_experiment("Hyperparameter Tuning Phase")
    mlflow.set_experiment_tag("model-tuning", "v1.0.0")

    model = load_local_model(f"{chosen_model_path}/{chosen_model_name}.pkl")
    if not model:
        raise ValueError("Failed to load the chosen model.")

    tuple_item = [item for item in randomcv_models if item[0] == chosen_model_name]
    if not tuple_item:
        raise ValueError("Could not find the model for the given model name.")
    model_name, chosen_model, search_space = tuple_item[0]
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    random_cv_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=2,
        cv=kf,
        verbose=2,
        n_jobs=mp.cpu_count(),
        random_state=42,
        scoring="accuracy",
    )
    X_train_scaled = column_transformer.fit_transform(X_train)  # type: ignore
    X_valid_scaled = column_transformer.transform(X_val)  # type: ignore
    random_cv_model.fit(X_train_scaled, y_train)
    best_model = random_cv_model.best_estimator_
    best_model.set_params(**random_cv_model.best_params_)
    y_pred = best_model.predict(X_valid_scaled)  # type: ignore
    signature = infer_signature(X_valid_scaled, y_pred)
    metrics_dict = log_mlflow_metrics(y_pred, y_val)
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=best_model,
            signature=signature,
            artifact_path="model",
            registered_model_name=chosen_model_name,
        )
        mlflow.log_params(random_cv_model.best_params_)
        mlflow.log_metrics(metrics_dict)
    logger.info(f"Completed hyperparameter tuning for model: {model_name}")

    return best_model
