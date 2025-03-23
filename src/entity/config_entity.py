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
from scipy.stats import uniform, norm, loguniform, randint

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
    "Random_Forest": {
        "n_estimators": randint(100, 500),  # Random integer between 100 and 500
        "max_depth": randint(10, 50),  # Random integer between 10 and 50
        "min_samples_split": randint(2, 10),  # Random integer between 2 and 10
        "min_samples_leaf": randint(1, 5),  # Random integer between 1 and 5
        "criterion": ["gini", "entropy"],  # Categorical (no distribution)
    },
    "Decision_Tree": {
        "max_depth": randint(10, 50),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "criterion": ["gini", "entropy"],
    },
    "Gradient_Boosting": {
        "n_estimators": randint(100, 500),
        "learning_rate": loguniform(1e-3, 1e-1),  # Log-uniform distribution between 0.001 and 0.1
        "max_depth": randint(10, 50),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
    },
    "Logistic_Regression": {
        "C": loguniform(1e-2, 1e2),  # Log-uniform distribution between 0.01 and 100
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
    },
    "K-Neighbors_Classifier": {
        "n_neighbors": randint(1, 31),  # Random integer between 1 and 30
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
    },
    "XGBClassifier": {
        "n_estimators": randint(100, 500),
        "learning_rate": loguniform(1e-3, 1e-1),
        "max_depth": randint(10, 50),
        "subsample": uniform(0.5, 0.5),  # Uniform distribution between 0.5 and 1.0
        "colsample_bytree": uniform(0.5, 0.5),
    },
    "CatBoosting_Classifier": {
        "iterations": randint(100, 500),
        "learning_rate": loguniform(1e-3, 1e-1),
        "depth": randint(4, 11),
        "l2_leaf_reg": randint(1, 11),
    },
    "Support_Vector_Classifier": {
        "C": loguniform(1e-1, 1e1),  # Log-uniform distribution between 0.1 and 10
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    "AdaBoost_Classifier": {
        "n_estimators": randint(50, 300),
        "learning_rate": loguniform(1e-3, 1e0),  # Log-uniform distribution between 0.001 and 1.0
    },
}

# Models list for Hyperparameter tuning
randomcv_models = [
    ("Random_Forest", RandomForestClassifier(), search_space["Random_Forest"]),
    ("K-Neighbors_Classifier", KNeighborsClassifier(), search_space["K-Neighbors_Classifier"]),
    ("Decision_Tree", DecisionTreeClassifier(), search_space["Decision_Tree"]),
    ("Gradient_Boosting", GradientBoostingClassifier(), search_space["Gradient_Boosting"]),
    ("Logistic_Regression", LogisticRegression(), search_space["Logistic_Regression"]),
    ("Support_Vector_Classifier", SVC(), search_space["Support_Vector_Classifier"]),
    ("AdaBoost_Classifier", AdaBoostClassifier(), search_space["AdaBoost_Classifier"]),
]
