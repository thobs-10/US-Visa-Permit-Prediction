import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(root_dir, '.env'))


dataclass
class DataIngestionConfig:
    raw_data_path: str = os.getenv('RAW_DATA_FILE')

@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = os.getenv('PROCESSED_PATH_FILE')

