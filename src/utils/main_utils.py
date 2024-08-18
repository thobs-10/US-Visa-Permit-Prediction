import os.path
import sys
import yaml
import base64

from src.logger import logging
from src.Exception import AppException

def read_yaml_file(filepath: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.

    Parameters:
    filepath (str): The path to the YAML file to be read.

    Returns:
    dict: The content of the YAML file as a dictionary.

    Raises:
    AppException: If an error occurs while reading the file or parsing the YAML content.
    """
    try:
        with open(filepath, 'r') as file:
            logging.info("Reading YAML file")
            return yaml.safe_load(file)
    except Exception as e:
        raise AppException(e, sys)
    
def write_yaml_file(filepath: str, data: dict, replace: bool) -> None:
    try:
        if replace:
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Removing old YAML filename {filepath}")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as file:
                logging.info(f"Writing new YAML file to {filepath}")
                yaml.dump(data, file, default_flow_style = False)
    except Exception as e:
        raise AppException(e, sys)