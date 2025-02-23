import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(root_dir, ".env"))


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.environ["RAW_DATA_FILE"]


@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = os.environ["PROCESSED_DATA_FILE"]


@dataclass
class FeatureEngineeringConfig:
    feature_engineering_dir: str = os.environ["FEATURE_ENGINEERED_DATA_FILE"]


@dataclass
class ModelTrainingConfig:
    model_artifact_dir: str = os.environ["ARTIFACTS_PATH"]


@dataclass
class ModelTuningConfig:
    model_artifact_dir: str = os.environ["TUNING_ARTIFACTS_PATH"]


search_space = {
    # 'Random_Forest': {
    #     'n_estimators': [100, 200, 300, 400],
    #     'max_depth': [10, 20, 30, 40, 50],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 3, 4, 5],
    #     'criterion': ['gini', 'entropy'],
    # },
    "Decision_Tree": {
        "max_depth": [10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "criterion": ["gini", "entropy"],
    },
    # 'Gradient_Boosting': {
    #     'n_estimators': [100, 200, 300, 400],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    #     'max_depth': [10, 20, 30, 40, 50],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 3, 4, 5],
    # },
    # 'Logistic_Regression': {
    #     'C': [0.01, 0.1, 1, 10],
    #     'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
    # },
    # 'K-Neighbors_Classifier': {
    #     'n_neighbors': list(range(1, 31)),
    #     'weights': ['uniform', 'distance'],
    #     'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    # },
    # 'XGBClassifier': {
    #     'n_estimators': [100, 200, 300, 400],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    #     'max_depth': list(range(10, 51)),
    #     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # },
    # 'CatBoosting_Classifier': {
    #     'iterations': [100, 200, 300, 400],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    #     'depth': list(range(4, 11)),
    #     'l2_leaf_reg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # },
    # 'Support_Vector_Classifier': {
    #     'C': [0.1, 0.5, 1, 5, 10],
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # },
    # 'AdaBoost_Classifier': {
    #     'n_estimators': [50, 100, 200, 300],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    # }
}

# Models list for Hyperparameter tuning
randomcv_models = [
    # ('XGBClassifier', XGBClassifier(), search_space['XGBClassifier']),
    # ('Random_Forest', RandomForestClassifier(), search_space['Random_Forest']),
    # ('K-Neighbors_Classifier', KNeighborsClassifier(), search_space['K-Neighbors_Classifier']),
    ("Decision_Tree", DecisionTreeClassifier(), search_space["Decision_Tree"]),
    # ('Gradient_Boosting', GradientBoostingClassifier(), search_space['Gradient_Boosting']),
    # ('Logistic_Regression', LogisticRegression(), search_space['Logistic_Regression']),
    # ('Support_Vector_Classifier', SVC(), search_space['Support_Vector_Classifier']),
    # ('AdaBoost_Classifier', AdaBoostClassifier(),search_space['AdaBoost_Classifier']),
    # ('CatBoosting_Classifier', CatBoostClassifier(verbose=False), search_space['CatBoosting_Classifier'])
]
