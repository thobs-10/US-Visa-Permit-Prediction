import logging
import os
from datetime import datetime
from from_root import from_root

PROJECT_ROOT = 'C:/Users/Thobs/Desktop/Portfolio/Projects/Data-Science-Projects/mlops-zoompcamp-project-2024'
LOG_FILE =  f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(PROJECT_ROOT, 'log', LOG_FILE)

os.makedirs(log_path, exist_ok=True)

# full log file path
log_file_path = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )