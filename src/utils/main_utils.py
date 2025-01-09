import os.path
from pathlib import Path
import sys
import yaml
import base64
from typing import List,Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
import joblib
from src.logger import logging
from src.Exception import AppException
import comet_ml
from comet_ml import API, Experiment

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
        with open(filepath, 'r') as file:
            logging.info("Reading YAML file")
            return yaml.safe_load(file)
    except Exception as e:
        raise AppException(e, sys)
    
def write_yaml_file(filepath: str, data: dict, replace: bool) -> None:
    try:
        if replace:
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Removing old YAML filename {filepath}")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as file:
                logging.info(f"Writing new YAML file to {filepath}")
                yaml.dump(data, file, default_flow_style = False)
    except Exception as e:
        raise AppException(e, sys)
    
def get_skewed_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    skewed_features = df[features].apply(lambda x: x.skew()).abs()
    transform_features = skewed_features[skewed_features > 1.0].index.tolist()
    return transform_features

def separate_data(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop('case_status', axis=1)
    y = df['case_status']
    return X, y

def apply_power_transform(df:pd.DataFrame, X: pd.DataFrame ,transform_features: List[str]) -> pd.DataFrame:
    X_copy = X.copy()
    pt = PowerTransformer(method='yeo-johnson')
    X_copy[transform_features] = pt.fit_transform(X[transform_features])
    df[transform_features] = X_copy[transform_features]
    return df

def encode_target(y: pd.Series) -> pd.Series:
    y = np.where(y == 'Denied', 1, 0)
    return y

def get_latest_modified_file(processed_files: List[str], processed_folder: str) -> str:
    full_file_paths = [os.path.join(processed_folder, f) for f in processed_files]
    latest_file = max(full_file_paths, key=os.path.getmtime)
    return latest_file

def get_statistical_properties(column: str, df: pd.DataFrame)-> Tuple[int, int, int]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

def instantiate_encoders() -> Tuple[StandardScaler, OneHotEncoder, OrdinalEncoder, Pipeline]:
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()
    ordinal_encoder = OrdinalEncoder()
    transform_pipe = Pipeline(steps=[
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])
    return numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe


def get_column_list(X:pd.DataFrame):
    num_features = list(X.select_dtypes(exclude="object").columns)
    or_columns = ['has_job_experience','requires_job_training','full_time_position','education_of_employee']
    oh_columns = ['continent','unit_of_wage','region_of_employment']
    transform_columns= ['no_of_employees','company_age']
    return num_features, or_columns, oh_columns, transform_columns


def get_column_transformer(X:pd.DataFrame):
    num_features, or_columns, oh_columns, transform_columns = get_column_list(X)
    numeric_transformer, oh_transformer, ordinal_encoder, transform_pipe = instantiate_encoders()
    preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, oh_columns),
                ("Ordinal_Encoder", ordinal_encoder, or_columns),
                ("Transformer", transform_pipe, transform_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ]
        )
    return preprocessor

def get_models()-> dict:
    models = {
                "Random_Forest": RandomForestClassifier(),
                "Decision_Tree": DecisionTreeClassifier(),
                "Gradient_Boosting": GradientBoostingClassifier(),
                "Logistic_Regression": LogisticRegression(),
                "K-Neighbors_Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting_Classifier": CatBoostClassifier(verbose=False),
                "Support_Vector_Classifier": SVC(),
                "AdaBoost_Classifier": AdaBoostClassifier()

    }
    return models

def train_and_save(X_train: pd.DataFrame, X_valid: pd.DataFrame, 
                   y_train: pd.Series, model_name: str,
                    model: Union[BaseEstimator, Any])-> Tuple[pd.Series, str]:

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    model_path = f"src\models\{model_name}.pkl"
    joblib.dump(model, model_path)
    return y_pred, model_path

def compute_accuracy_metrics(y_pred: pd.Series, 
                             y_valid: pd.Series)-> Tuple[float, float, float, float, float]:
    
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted')
    precision = precision_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_valid, y_pred, multi_class='ovr')

    return acc, f1, precision, recall, roc_auc

def tracking_details_init() -> Experiment:
        API_key = os.getenv("COMET_API_KEY")
        proj_name = os.getenv("COMET_PROJECT_NAME")
        comet_ml.init()
        experiment = comet_ml.Experiment(
            api_key=API_key,
            project_name=proj_name,
            workspace=os.getenv("MLOPS_WORKSPACE_NAME"),
        )
        return experiment