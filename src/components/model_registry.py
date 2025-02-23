import sys
from loguru import logger

from src.Exception import AppException
from src.utils.main_utils import get_matching_experiments


def register_model(experiment_key: str):
    try:
        logger.info("Starting the model registration phase")
        matching_experiments = get_matching_experiments()
        if matching_experiments:
            best_experiment = matching_experiments[0]
            model_name = best_experiment.get_model_names()[0]
            feedback = best_experiment.register_model(model_name=model_name)
            if feedback:
                logger.info("Successfully registered the model")
    except Exception as e:
        raise AppException(e, sys)
