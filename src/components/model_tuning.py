import os
import pandas as pd
from typing import Optional, Tuple
import comet_ml
from loguru import logger
from src.utils.main_utils import (
    get_tracking_api_experiment,
    get_chosen_model_from_search,
    logging_metrics,
    feature_scaling,
    get_models,
    saving_model_object_zip_file,
)
from src.entity.config_entity import randomcv_models
from comet_ml import Experiment
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.base import BaseEstimator
import multiprocessing as mp


class HyperparameterTuning:
    def __init__(
        self,
    ):
        self.model_param = {}

    def hyperparameter_tuning(
        self, exp_key: str, X_train, Y_train, X_val, Y_val
    ) -> Tuple[str, Experiment]:
        logger.info("starting hyperparameter tuning")
        api_experiment = get_tracking_api_experiment(exp_key)
        randomcv_models_dict = {
            name: (model, params) for name, model, params in randomcv_models
        }
        comet_ml.init()
        experiment = comet_ml.Experiment(
            api_key=os.environ["API_KEY"],
            project_name=os.environ["PROJECT_NAME"],
            workspace=os.environ["WORKSPACE"],
        )
        for model_name, model in get_models.items():
            assets = api_experiment.get_model_asset_list(model_name=model_name)
            if len(assets) > 0:
                model_assets = [
                    asset for asset in assets if asset["fileName"].endswith(".pkl")
                ]
                chosen_model, chosen_params = get_chosen_model_from_search(
                    model_assets, randomcv_models_dict
                )
                with experiment.train():
                    logger.info(
                        f"Performing hyperparameter tuning for model: {model_name}"
                    )
                    random_cv_model = RandomizedSearchCV(
                        estimator=chosen_model,
                        param_distributions=chosen_params,
                        n_iter=10,
                        cv=5,
                        verbose=2,
                        n_jobs=-1,
                        random_state=42,
                    )
                    column_transformer, X_train_scaled, X_valid_scaled = (
                        feature_scaling(
                            X_train,
                            X_val,
                        )
                    )
                    random_cv_model.fit(X_train_scaled, Y_train)
                    y_pred = random_cv_model.predict(X_val)
                    logging_metrics(experiment, y_pred, Y_val)
                    logger.info(
                        f"Completed hyperparameter tuning for model: {model_name}"
                    )
        hyper_exp_key = experiment.get_key()
        return hyper_exp_key, experiment

    def peform_cross_validation(
        self,
        model: RandomizedSearchCV,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        experiment: Experiment,
        experiment_key: Optional[str] = None,
        threshold: float = 0.8,
    ) -> BaseEstimator:
        logger.info("Performing cross validation")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_model = model.best_estimator_
        scores = cross_val_score(best_model, x_val, y_val, cv=kf, n_jobs=mp.cpu_count())
        mean_score = scores.mean()
        if mean_score < threshold:
            raise ValueError("Cross validation score is less than threshold")
        experiment.log_metric("Cross Validation Score", mean_score)
        experiment.log_parameters(model.best_params_)
        # experiment.log_model("Best Model", best_model)
        saving_model_object_zip_file(best_model, experiment)
        logger.info(f"Cross validation score: {mean_score}")
        return best_model
