import os
import glob
import pandas as pd
import numpy as np
import sys
import click
from dotenv import load_dotenv
import joblib
import requests

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, KFold, cross_val_score

from src.logger import logging
from src.Exception import AppException
from src.entity.artifact_entity import ModelTrainingArtifact
from src.entity.config_entity import ModelTrainingConfig, ModelTrainingPipelineConfig
from src.entity.config_entity import FeatureEngineeringConfig
from src.components.component import Component
from src.utils.main_utils import get_models

import comet_ml
from comet_ml import API
from comet_ml.query import Metric
import mlflow

load_dotenv()

class ModelTraining(Component):
    def __init__(self, model_training_config: ModelTrainingConfig = ModelTrainingConfig(),
                 feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        
        self.model_training_config = model_training_config
        self.feature_engineering_config = feature_engineering_config
        
    
    def load_data(self) -> tuple:
        try:
            logging.info("Loading feature engineered data(features.parquet and target.csv)")
            transformed_features_file_path = os.path.join(self.feature_engineering_config.feature_engineering_dir)
            X_df = pd.read_parquet(os.path.join(transformed_features_file_path, "features.parquet"))
            y_df = pd.read_csv(os.path.join(transformed_features_file_path, "target.csv"))  
            logging.info("Processed data loaded and converted to NumPy arrays successfully")
            return X_df, y_df
        except Exception as e:
            raise AppException(e, sys)
    
    def split_data(self, X, y) -> tuple:
        logging.info("Splitting data into training, validation, and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
        logging.info("Data split completed successfully")
        return X_train, y_train, X_test, y_test, X_valid, y_valid
     
    
    def tracking_details_init(self):
        API_key = os.getenv("COMET_API_KEY")
        proj_name = os.getenv("COMET_PROJECT_NAME")
        comet_ml.init()
        experiment = comet_ml.Experiment(
            api_key=API_key,
            project_name=proj_name,
            workspace=os.getenv("MLOPS_WORKSPACE_NAME"),
        )
        return experiment

    def train_model(self, X_train, X_valid, y_train, y_valid) -> str:
        models = get_models()
        try:
            logging.info("Training model phase")
            experiment = self.tracking_details_init()
            # go through each model in the dict, train, valid and test and keep track of the results, model, metrics in comet_ml
            for model_name, model in self.models.items():
                experiment.add_tag(f"Model: {model_name}")
                # logging.info(f"Training {model_name}")
                with experiment.train():
                    logging.info(f"Training {model_name}")
                    # model_pipeline = make_pipeline(StandardScaler(), model)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_valid)
                    model_path = f"src\models\{model_name}.pkl"
                    joblib.dump(model, model_path)
                    experiment.log_model(f"{model_name}", model_path)
                    # experiment.log_model(name = f"{model_name}",file_or_folder= "model_pipeline.bin")
                    
                    acc = accuracy_score(y_valid, y_pred)
                    f1 = f1_score(y_valid, y_pred, average='weighted')
                    precision = precision_score(y_valid, y_pred, average='weighted')
                    recall = recall_score(y_valid, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_valid, y_pred, multi_class='ovr')

                    # Log metrics for this model
                    experiment.log_metric("Accuracy", acc)
                    experiment.log_metric("F1 Score", f1)
                    experiment.log_metric("Precision", precision)
                    experiment.log_metric("Recall", recall)
                    experiment.log_metric("ROC", roc_auc)
            exp_key = experiment.get_key()
            logging.info(f"models trained and evaluated successfully")
            return exp_key
        except Exception as e:
            raise AppException(e, sys)
        

        
    def hyperparameter_tuning(self, exp_key:str,  X_train, Y_train,X_val, Y_val) -> str:
        
        try:
            logging.info("starting hyperparameter tuning")
            api = API(api_key=os.getenv('API_KEY'))
            api_experiment = api.get_experiment_by_key(exp_key)
            # get all the models with accuracy greater than 70% but less than 90% accuracy
            query_condition = (Metric("train_Accuracy") > 0.75) & (Metric("train_Accuracy") < 0.9)
            accuracy_threshold = 0.75
        
            # fetch from old experiment
            WORKSPACE_NAME = 'thobela'
            PROJECT_NAME = 'mlops-project'
            # matching_api_experiments = api.query(WORKSPACE_NAME, PROJECT_NAME, query_condition)
            # get 5 best runs and re-run
            search_space ={
                    # 'Random_Forest': {
                    #     'n_estimators': [100, 200, 300, 400],
                    #     'max_depth': [10, 20, 30, 40, 50],
                    #     'min_samples_split': [2, 5, 10],
                    #     'min_samples_leaf': [1, 2, 3, 4, 5],
                    #     'criterion': ['gini', 'entropy'],
                    # },
                    'Decision_Tree': {
                        'max_depth': [10, 20, 30, 40, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 3, 4, 5],
                        'criterion': ['gini', 'entropy'],
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
                ('Decision_Tree', DecisionTreeClassifier(), search_space['Decision_Tree']),
                # ('Gradient_Boosting', GradientBoostingClassifier(), search_space['Gradient_Boosting']),
                # ('Logistic_Regression', LogisticRegression(), search_space['Logistic_Regression']),
                # ('Support_Vector_Classifier', SVC(), search_space['Support_Vector_Classifier']),
                # ('AdaBoost_Classifier', AdaBoostClassifier(),search_space['AdaBoost_Classifier']),
                # ('CatBoosting_Classifier', CatBoostClassifier(verbose=False), search_space['CatBoosting_Classifier'])
            ]
            # Convert the list to a dictionary
            randomcv_models_dict = {name: (model, params) for name, model, params in randomcv_models}
            sc = StandardScaler()

            comet_ml.init()
            experiment = comet_ml.Experiment(
                api_key=os.getenv('API_KEY'),
                project_name=os.getenv('PROJECT_NAME'),
                workspace=os.getenv('WORKSPACE'),
            )
            for model_name, model  in self.models.items():
                assets = api_experiment.get_model_asset_list(model_name=model_name)
                if len(assets) > 0:
                    # Filter out the models
                    model_assets = [asset for asset in assets if asset['fileName'].endswith('.pkl')]
                    model_param = {}
                    # Download the models
                    for model_asset in model_assets:
                        model_file_name = model_asset['fileName']
                        model_url = model_asset['link']
                        # get the first part of the model before pkl
                        model_name = model_file_name.split('.')[0]
                        # clf_name = model_name.split('_')[0]
                        chosen_model, chosen_params = randomcv_models_dict[model_name]
                        # start a new experiment for cross validation
                        with experiment.train():
                            # Perform hyperparameter tuning using RandomizedSearchCV
                            logging.info(f"Performing hyperparameter tuning for model: {model_name}")
                        
                            # random_cv_model = GridSearchCV(estimator=chosen_model,
                            #                         param_grid=chosen_params,
                            #                         cv=2,
                            #                         verbose=2, 
                            #                         n_jobs=-1)
                            random_cv_model = RandomizedSearchCV(
                                                    estimator=chosen_model,
                                                    param_distributions=chosen_params,
                                                    n_iter=3,
                                                    cv=3,
                                                    verbose=2,
                                                    n_jobs=-1)
                            # kf = KFold(n_splits=5)
                            # scores = cross_val_score(chosen_model, X_train, Y_train, cv=kf)
                            # print(f"K-Fold CV Scores: {scores}")
                            random_cv_model.fit(X_train, Y_train)
                            model_param[model_name] = random_cv_model.best_params_
                            # log the model parameters and model itself
                            experiment.log_parameters(model_param[model_name])
                            tuned_model_path = "src/models/tuned_models/"
                            # model_pipeline = make_pipeline(StandardScaler(), random_cv_model)
                            os.makedirs(tuned_model_path, exist_ok=True)
                            model_path = os.path.join(tuned_model_path, f"{model_name}.pkl")
                            joblib.dump(random_cv_model, model_path)
                            experiment.log_model(model_name, model_path)

                            # get metrics then log the metrics
                            logging.info("logging metrics")
                            # X_val_scaled = sc.transform(X_val)
                            y_pred = random_cv_model.predict(X_val)
                            acc = accuracy_score(Y_val, y_pred)
                            f1 = f1_score(Y_val, y_pred, average='weighted')
                            precision = precision_score(Y_val, y_pred, average='weighted')
                            recall = recall_score(Y_val, y_pred, average='weighted')
                            roc_auc = roc_auc_score(Y_val, y_pred, multi_class='ovr')

                            # Log metrics for this model
                            experiment.log_metric("Accuracy", acc)
                            experiment.log_metric("f1", f1)
                            experiment.log_metric("Precision", precision)
                            experiment.log_metric("Recall", recall)
                            experiment.log_metric("ROC", roc_auc)
                            # experiment.log_artifact(model_file_name, f"./{model_file_name}")
                            logging.info(f"Completed hyperparameter tuning for model: {model_name}")
            hyper_exp_key = experiment.get_key()
            return hyper_exp_key
        except Exception as e:
            raise AppException(e, sys)
    
    def evaluate_model(self, exp_key, X_test, y_test)-> str:
        """
        Evaluates a model based on the experiment key and test data.

        Retrieves the experiment from Comet.ml using the provided experiment key,
        and then downloads the best model associated with the experiment.
        The model is then evaluated using the provided test data, and various metrics
        such as accuracy, f1 score, precision, recall, and ROC are calculated.
        These metrics are logged to the experiment in Comet.ml.

        Parameters:
            exp_key (str): The key of the experiment to retrieve from Comet.ml.
            X_test (array-like): The test data to use for model evaluation.
            y_test (array-like): The true labels for the test data.

        Returns:
            tuple: A tuple containing the accuracy, f1 score, precision, recall, and ROC of the model.
        """
        try:
            logging.info(" Starting the evaluation phase")
            # Initialize Comet.ml API
            api = API(api_key=os.getenv('API_KEY'))
        
            api_experiment = api.get_experiment_by_key(exp_key)
            # Get the models with accuracy greater than 90%
            query_condition = (Metric("train_Accuracy") > 0.70)
            matching_api_experiments = api.query(api_experiment.workspace, api_experiment.project_name, query_condition)
            
            if len(matching_api_experiments) > 0:
                # Get the best experiment
                best_experiment = matching_api_experiments[0]
                # Get the tracked model associated with the best experiment
                model_name = best_experiment.get_model_names()
                assets = best_experiment.get_model_asset_list(model_name=model_name[0])
                # Filter out the models
                model_assets = [asset for asset in assets if asset['fileName'].endswith('.pkl')]
                # Download the best model
                if len(model_assets) > 0:
                    model_asset = model_assets[0]
                    model_file_name = model_asset['fileName']
                    model_url = model_asset['curlDownload']
                    # Load the downloaded model
                    os.makedirs('src/models/best_model', exist_ok=True)
                    output_path = f'src/models/best_model/'
                    best_experiment.download_model(model_name[0], output_path= output_path)
                    file_extension="*.pkl"
                    files = glob.glob(os.path.join(output_path, file_extension))
                    latest_file = max(files, key=os.path.getmtime)
                    model = joblib.load(latest_file)
                    # Evaluate the model
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)  
                    f1 = f1_score(y_test, y_pred)  
                    precision = precision_score(y_test, y_pred)  
                    recall = recall_score(y_test, y_pred) 
                    roc_auc = roc_auc_score(y_test, y_pred)  
                    print('-------Evaluation summary---------')
                    print(f"Accuracy: {acc}\nf1 score: {f1}\nPrecision: {precision}\nRecall: {recall}\nRoc: {roc_auc}")
                    # Log metrics for this model
                    best_experiment.log_metric("Accuracy", acc)
                    best_experiment.log_metric("f1", f1)
                    best_experiment.log_metric("Precision", precision)
                    best_experiment.log_metric("Recall", recall)
                    best_experiment.log_metric("ROC", roc_auc)
                    # experiment.log_artifact(model_file_name, f"./{model_file_name}")  # log model artifact to the experiment in Comet.ml
                    logging.info(f"Completed model evaluation.")
                    
                else:
                    print("No models found with accuracy greater than 90%.")
            else:
                print("No experiments found with accuracy greater than 90%.")
            return exp_key
        except Exception as e:
            raise AppException(e, sys)
    
    def register_model(self, experiment_key):
        try:
            logging.info("Starting the model registration phase")
            api = API(api_key=os.getenv('API_KEY'))
            experiment = api.get_experiment_by_key(experiment_key)
            accuracy_condition = Metric("train_Accuracy") > 0.70
            matching_experiments = api.query(
                experiment.workspace, experiment.project_name, accuracy_condition)
            if matching_experiments:
                best_experiment = matching_experiments[0]
                model_name = best_experiment.get_model_names()[0]
                assets = best_experiment.get_model_asset_list(model_name=model_name)
                feedback = best_experiment.register_model(model_name=model_name)
                if feedback:
                    logging.info("Successfully registered the model")
        except Exception as e:
            raise AppException(e, sys)





