import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from loguru import logger
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.entity.config_entity import ModelTrainingConfig

memory = Memory(location="cachedir", verbose=0)


def get_skewed_features(
    df: pd.DataFrame,
    features: list[str],
) -> list[str]:
    skewed_features = df[features].apply(lambda x: x.skew()).abs()
    transform_features = skewed_features[skewed_features > 1.0].index.tolist()
    return transform_features


def separate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x_values = df.drop("case_status", axis=1)
    y_values = df["case_status"]
    return x_values, y_values


def apply_power_transform(
    df: pd.DataFrame,
    x_values: pd.DataFrame,
    transform_features: list[str],
) -> pd.DataFrame:
    x_copy = x_values.copy()

    power_transformer = PowerTransformer(method="yeo-johnson")
    x_copy[transform_features] = power_transformer.fit_transform(x_values[transform_features])
    return x_copy


def encode_target(y_values: pd.Series) -> pd.Series:
    y_values = pd.Series(np.where(y_values == "Denied", 1, 0))
    return y_values


def get_latest_modified_file(
    processed_files: list[str],
    processed_folder: str,
) -> str:
    full_file_paths = [os.path.join(processed_folder, f) for f in processed_files]
    latest_file = max(full_file_paths, key=os.path.getmtime)
    return latest_file


def get_statistical_properties(
    column: str,
    df: pd.DataFrame,
) -> tuple[float, float, float]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    inter_quartile_range = q3 - q1
    return q1, q3, inter_quartile_range


def instantiate_encoders() -> tuple[StandardScaler, OneHotEncoder, OrdinalEncoder, Pipeline]:
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()
    ordinal_encoder = OrdinalEncoder()
    transform_pipe = Pipeline(steps=[("transformer", PowerTransformer(method="yeo-johnson"))])
    return numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe


def get_column_list(x_values: pd.DataFrame):
    num_features = list(x_values.select_dtypes(exclude="object").columns)
    or_columns = [
        "has_job_experience",
        "requires_job_training",
        "full_time_position",
        "education_of_employee",
    ]
    oh_columns = ["continent", "unit_of_wage", "region_of_employment"]
    transform_columns = ["no_of_employees", "company_age"]
    return num_features, or_columns, oh_columns, transform_columns


def get_column_transformer(x_values: pd.DataFrame):
    num_features, or_columns, oh_columns, transform_columns = get_column_list(x_values)
    numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe = instantiate_encoders()
    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, oh_columns),
            ("Ordinal_Encoder", ordinal_encoder, or_columns),
            ("Transformer", transform_pipe, transform_columns),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )
    return preprocessor


def get_models() -> dict:
    models = {
        "Random_Forest": RandomForestClassifier(),
        "Decision_Tree": DecisionTreeClassifier(),
        "Gradient_Boosting": GradientBoostingClassifier(),
        "Logistic_Regression": LogisticRegression(),
        "K-Neighbors_Classifier": KNeighborsClassifier(),
        "Support_Vector_Classifier": SVC(),
        "AdaBoost_Classifier": AdaBoostClassifier(),
    }
    return models


def train_and_save(
    x_train: np.ndarray[Any, Any] | spmatrix,
    x_valid: np.ndarray[Any, Any] | spmatrix,
    y_train: pd.Series,
    model_name: str,
    model: LogisticRegression | RandomForestClassifier | DecisionTreeClassifier | GradientBoostingClassifier | KNeighborsClassifier | SVC | AdaBoostClassifier,
) -> tuple[np.ndarray, str]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    os.makedirs(os.path.join(ModelTrainingConfig.model_artifact_dir, model_name), exist_ok=True)
    joblib.dump(
        model,
        os.path.join(ModelTrainingConfig.model_artifact_dir, model_name, f"{model_name}.pkl"),
    )
    return y_pred, ModelTrainingConfig.model_artifact_dir


def compute_accuracy_metrics(
    y_pred: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[float, float, float, float, float]:
    acc = float(accuracy_score(y_valid, y_pred))
    f1_score_metric = float(f1_score(y_valid, y_pred, average="weighted"))
    precision = float(precision_score(y_valid, y_pred, average="weighted"))
    recall = float(recall_score(y_valid, y_pred, average="weighted"))
    roc_auc = float(roc_auc_score(y_valid, y_pred, multi_class="ovr"))

    return acc, f1_score_metric, precision, recall, roc_auc


def log_mlflow_metrics(
    y_pred: np.ndarray,
    y_valid: pd.Series,
) -> dict[str, float]:
    # convert y_valid from pd.Series to numpy array
    y_valid_array = y_valid.to_numpy()
    acc, f1_score_metric, precision, recall, roc_auc = compute_accuracy_metrics(y_pred, y_valid_array)
    return {
        "Accuracy": acc,
        "F1 Score": f1_score_metric,
        "Precision": precision,
        "Recall": recall,
        "ROC AUC Score": roc_auc,
    }


def log_metrics_terminal(
    y_pred: np.ndarray,
    y_valid: np.ndarray,
) -> dict:
    acc, f1_score_metric, precision, recall, roc_auc = compute_accuracy_metrics(y_pred, y_valid)
    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 Score: {f1_score_metric}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"ROC AUC Score: {roc_auc}")
    return {
        "Accuracy": acc,
        "F1 Score": f1_score_metric,
        "Precision": precision,
        "Recall": recall,
        "ROC AUC Score": roc_auc,
    }


def get_chosen_model_from_search(
    model_assets: list[dict[str, Any]],
    randomcv_models_dict: dict[str, tuple[DecisionTreeClassifier, dict[str, Any]]],
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    for model_asset in model_assets:
        model_file_name = model_asset["fileName"]
        model_name = model_file_name.split(".")[0]
        if model_name in randomcv_models_dict:
            chosen_model, chosen_params = randomcv_models_dict[model_name]
            return chosen_model, chosen_params

    raise ValueError("Could not find the model for the given model name.")


@memory.cache
def feature_scaling(
    x_train: pd.DataFrame, x_valid: pd.DataFrame
) -> tuple[
    ColumnTransformer,
    np.ndarray[Any, Any] | spmatrix,
    np.ndarray[Any, Any] | spmatrix,
]:
    # now we need to exclude the last column from the feature scaling which is timestamp and has data type of datetime
    x_train_to_be_scaled = x_train.iloc[:, :-1]
    x_valid_to_be_scaled = x_valid.iloc[:, :-1]
    column_transformer = get_column_transformer(x_train_to_be_scaled)
    x_train_scaled = column_transformer.fit_transform(x_train_to_be_scaled)
    x_valid_scaled = column_transformer.transform(x_valid_to_be_scaled)
    logger.info("Feature scaling completed successfully")
    return column_transformer, x_train_scaled, x_valid_scaled


def get_best_performing_model(
    all_model_dict: dict[str, tuple[float, str]],
) -> tuple[str, str]:
    """Arrange models by accuracy and return the best-performing one."""
    if not all_model_dict:
        raise ValueError("all_model_dict is nonexistent")
    best_model = max(all_model_dict, key=lambda x: all_model_dict[x][0])
    best_model_path = all_model_dict[best_model][1]
    return best_model, best_model_path


def load_local_model(model_path: str) -> Any:
    """Load a local model from the given path."""
    with open(model_path, "rb") as file:
        return joblib.load(file)
