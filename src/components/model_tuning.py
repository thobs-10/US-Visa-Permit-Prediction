import os
import sys
import comet_ml
from comet_ml import API
from comet_ml.experiment import Experiment
from comet_ml.query import Metric
from loguru import logger
from src.components.component import Component
from comet_ml import API, Experiment

class HyperparameterTuning(Component):
    def hyperparameter_tuning(self, exp_key:str,  X_train, Y_train,X_val, Y_val) -> str:
        logger.info("starting hyperparameter tuning")
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
                        logger.info(f"Performing hyperparameter tuning for model: {model_name}")
                    
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
                        logger.info("logger metrics")
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
                        logger.info(f"Completed hyperparameter tuning for model: {model_name}")
        hyper_exp_key = experiment.get_key()
        return hyper_exp_key
