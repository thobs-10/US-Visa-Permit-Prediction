import os
import pickle
import yaml
from typing import List, Tuple, Union, Any, Dict
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib
from loguru import logger
import comet_ml
from comet_ml import API, Experiment, APIExperiment
from comet_ml.query import Metric


def read_yaml_file(filepath: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.

    Parameters:
    filepath (str): The path to the YAML file to be read.

    Returns:
    dict: The content of the YAML file as a dictionary.

    Raises:
    AppException: If an error occurs while reading the file or parsing the YAML content.
    """
    try:
        with open(filepath, "r") as file:
            logger.info("Reading YAML file")
            return yaml.safe_load(file)
    except Exception as e:
        raise e


def write_yaml_file(
    filepath: str,
    data: dict,
    replace: bool,
) -> None:
    try:
        if replace:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Removing old YAML filename {filepath}")

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w") as file:
                logger.info(f"Writing new YAML file to {filepath}")
                yaml.dump(data, file, default_flow_style=False)
    except Exception as e:
        raise e


def get_skewed_features(
    df: pd.DataFrame,
    features: List[str],
) -> List[str]:
    skewed_features = df[features].apply(lambda x: x.skew()).abs()
    transform_features = skewed_features[skewed_features > 1.0].index.tolist()
    return transform_features


def separate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop("case_status", axis=1)
    y = df["case_status"]
    return X, y


def apply_power_transform(
    df: pd.DataFrame,
    X: pd.DataFrame,
    transform_features: List[str],
) -> pd.DataFrame:
    X_copy = X.copy()
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method="yeo-johnson")
    X_copy[transform_features] = pt.fit_transform(X[transform_features])
    return X_copy


def encode_target(y: pd.Series) -> pd.Series:
    y = pd.Series(np.where(y == "Denied", 1, 0))
    return y


def get_latest_modified_file(
    processed_files: List[str],
    processed_folder: str,
) -> str:
    full_file_paths = [os.path.join(processed_folder, f) for f in processed_files]
    latest_file = max(full_file_paths, key=os.path.getmtime)
    return latest_file


def get_statistical_properties(
    column: str,
    df: pd.DataFrame,
) -> Tuple[float, float, float]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR


def instantiate_encoders() -> Tuple[
    StandardScaler, OneHotEncoder, OrdinalEncoder, Pipeline
]:
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()
    ordinal_encoder = OrdinalEncoder()
    transform_pipe = Pipeline(
        steps=[("transformer", PowerTransformer(method="yeo-johnson"))]
    )
    return numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe


def get_column_list(X: pd.DataFrame):
    num_features = list(X.select_dtypes(exclude="object").columns)
    or_columns = [
        "has_job_experience",
        "requires_job_training",
        "full_time_position",
        "education_of_employee",
    ]
    oh_columns = ["continent", "unit_of_wage", "region_of_employment"]
    transform_columns = ["no_of_employees", "company_age"]
    return num_features, or_columns, oh_columns, transform_columns


def get_column_transformer(X: pd.DataFrame):
    num_features, or_columns, oh_columns, transform_columns = get_column_list(X)
    numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe = (
        instantiate_encoders()
    )
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
    X_train: np.ndarray[Any, Any] | spmatrix,
    X_valid: np.ndarray[Any, Any] | spmatrix,
    y_train: np.ndarray,
    model_name: str,
    model: Union[
        LogisticRegression,
        RandomForestClassifier,
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        SVC,
        AdaBoostClassifier,
    ],
) -> Tuple[np.ndarray, str]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    model_path = f"src/models/{model_name}.pkl"
    joblib.dump(model, model_path)
    return y_pred, model_path


def compute_accuracy_metrics(
    y_pred: np.ndarray,
    y_valid: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    acc = float(accuracy_score(y_valid, y_pred))
    f1 = float(f1_score(y_valid, y_pred, average="weighted"))
    precision = float(precision_score(y_valid, y_pred, average="weighted"))
    recall = float(recall_score(y_valid, y_pred, average="weighted"))
    roc_auc = float(roc_auc_score(y_valid, y_pred, multi_class="ovr"))

    return acc, f1, precision, recall, roc_auc


def tracking_details_init() -> Experiment:
    API_key = os.environ["COMET_API_KEY"]
    proj_name = os.environ["COMET_PROJECT_NAME"]
    comet_ml.init()
    experiment = comet_ml.Experiment(
        api_key=API_key,
        project_name=proj_name,
        workspace=os.environ["MLOPS_WORKSPACE_NAME"],
    )
    return experiment


def logging_metrics(
    experiment: Experiment,
    y_pred: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    acc, f1, precision, recall, roc_auc = compute_accuracy_metrics(y_pred, y_valid)
    experiment.log_metric("Accuracy", acc)
    experiment.log_metric("F1 Score", f1)
    experiment.log_metric("Precision", precision)
    experiment.log_metric("Recall", recall)
    experiment.log_metric("ROC", roc_auc)


def log_metrics_terminal(
    y_pred: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    acc, f1, precision, recall, roc_auc = compute_accuracy_metrics(y_pred, y_valid)
    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"ROC AUC Score: {roc_auc}")


def get_tracking_api_experiment(exp_key: str) -> APIExperiment:
    api = API(api_key=os.environ["API_KEY"])
    api_experiment = api.get_experiment_by_key(exp_key)
    if api_experiment is None:
        logger.error(f"Experiment with key {exp_key} not found.")
        raise ValueError("Experiment with key {exp_key} not found")
    return api_experiment


def get_matching_experiments() -> Union[list[Any] | list[APIExperiment | None], None]:
    query_condition = (Metric("train_Accuracy") > 0.75) & (
        Metric("train_Accuracy") < 0.9
    )
    WORKSPACE_NAME = os.environ["WORKSPACE_NAME"]
    PROJECT_NAME = os.environ["PROJECT_NAME"]
    api = API(api_key=os.environ["API_KEY"])
    matching_api_experiments = api.query(WORKSPACE_NAME, PROJECT_NAME, query_condition)
    return matching_api_experiments


def get_chosen_model_from_search(
    model_assets: List[Dict[str, Any]],
    randomcv_models_dict: dict[str, tuple[DecisionTreeClassifier, dict[str, Any]]],
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    for model_asset in model_assets:
        model_file_name = model_asset["fileName"]
        model_name = model_file_name.split(".")[0]
        if model_name in randomcv_models_dict:
            chosen_model, chosen_params = randomcv_models_dict[model_name]
            return chosen_model, chosen_params

    raise ValueError("Could not find the model for the given model name.")


def feature_scaling(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> tuple[
    ColumnTransformer,
    np.ndarray[Any, Any] | spmatrix,
    np.ndarray[Any, Any] | spmatrix,
]:
    column_transformer = get_column_transformer(X_train)
    X_train_scaled = column_transformer.fit_transform(X_train)
    X_valid_scaled = column_transformer.transform(X_valid)
    logger.info("Feature scaling completed successfully")
    return column_transformer, X_train_scaled, X_valid_scaled


def saving_model_object_zip_file(
    best_model: BaseEstimator, experiment: Experiment
) -> None:
    model_filename = "best_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)

    experiment.log_model("Best Model", model_filename)
    logger.info("Model object saved as a zip file")
    os.remove(model_filename)
