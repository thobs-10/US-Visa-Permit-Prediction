# Variables
FEAST_DIR := my_feature_store/feature_store
MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 8080

.PHONY: all install feast mlflow pipelines precommit

start-dev:
        install feast mlflow pipelines precommit

install:
        @echo "Installing the package..."
        pip install .

feast:
        @echo "Starting Feast UI..."
        @$(MAKE) -C $(FEAST_DIR) ui
        @cd ../..

mlflow:
        @echo "Starting MLflow server..."
        @mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) &

pipelines:
        @echo "Running data ingestion, feature engineering, and model training pipelines..."
        @python run_pipelines.py

precommit:
        @echo "Running pre-commit checks..."
        @SKIP=no-commit-to-branch pre-commit run --all-files

# Wait for mlflow and feast to start
feast mlflow:
        @sleep 5
