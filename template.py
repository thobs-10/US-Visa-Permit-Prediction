import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "src"

list_of_files = [
    "github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    "data/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    f"{project_name}/models/.gitkeep",
    "notebooks/eda.ipynb",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/preprocessing.py",
    f"{project_name}/components/feature_engineering.py",
    f"{project_name}/components/model_training.py",
    f"{project_name}/components/model_tuning.py",
    f"{project_name}/components/model_validation.py",
    f"{project_name}/components/model_registry.py",
    f"{project_name}/components/model_deployment.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/Exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/pipeline/data_ingestion.py",
    f"{project_name}/pipeline/feature_engineering.py",
    f"{project_name}/pipeline/training_model.py",
    f"{project_name}/pipeline/model_registy.py",
    f"{project_name}/pipeline/web_service_deployment.py",
    f"{project_name}/tests/.gitkeep",
    "README.md",
    "requirements.txt",
    "setup.py",
    "LICENSE",
    ".gitignore",
    ".env",
    "Dockerfile",
    "Makefile",
]

for filepath in list_of_files:
    file_path = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for filename: {filename}")
    if (not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            logging.info(f"Created file: {filename}")
    else:
        logging.info(f"File {filename} already exists.")
