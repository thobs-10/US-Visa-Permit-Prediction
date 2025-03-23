from loguru import logger
from src.components.model_training import (
    load_data,
    split_data,
    train_model,
    save_preprocessor,
    save_training_data,
)
from src.components.model_tuning import hyperparameter_tuning
from src.components.model_validation import model_evaluation, save_model_pipeline
from src.components.model_registry import register_model

from zenml import pipeline


@pipeline
def run_model_training() -> None:
    try:
        logger.info("Starting model training pipeline")
        X, y = load_data()
        X_train, y_train, X_test, y_test, X_valid, y_valid = split_data(X, y)
        column_transformer, chosen_model_path, best_model_name = train_model(
            X_train, X_valid, y_train, y_valid
        )
        save_preprocessor(column_transformer)
        save_training_data(X_train, X_valid, y_train, y_valid)
        best_model = hyperparameter_tuning(
            X_train,
            X_valid,
            y_train,
            y_valid,
            chosen_model_path,
            best_model_name,
            column_transformer,
        )
        model = model_evaluation(X_test, y_test, X_valid, best_model, column_transformer)
        save_model_pipeline(model)
        register_model(model)
    except Exception as e:
        raise e


if __name__ == "__main__":
    run_model_training()
