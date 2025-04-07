import mlflow
from loguru import logger
from mlflow.client import MlflowClient
from mlflow.entities import ViewType
from sklearn.pipeline import Pipeline
from zenml import step

tracking_uri = mlflow.get_tracking_uri()
client = MlflowClient(tracking_uri=tracking_uri)


@step
def register_model(model_pipeline: Pipeline) -> None:
    """
    Register the best model from the hyperparameter tuning experiment in MLflow.

    Parameters:
    best_model (BaseEstimator): The best model to be registered.

    Raises:
    ValueError: If the hyperparameter tuning experiment does not exist.
    """
    tuning_experiment = client.get_experiment_by_name("Hyperparameter Tuning Phase")
    if not tuning_experiment:
        raise ValueError("Hyperparameter tuning experiment does not exist")
    top_n = 1  # Define the variable top_n
    runs = client.search_runs(
        experiment_ids=tuning_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.Accuracy ASC"],
    )
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = model_pipeline.__class__.__name__
    mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Model '{model_name}' registered successfully")
