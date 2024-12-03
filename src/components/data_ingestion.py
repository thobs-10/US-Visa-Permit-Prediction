import os
import sys
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime

# from src.logger import logging
from loguru import logger
from src.Exception import AppException
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AppException(e, sys)
    
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading data from raw folder")
            filepath = os.path.join(self.data_ingestion_config.raw_data_path)
            data = pd.read_csv(filepath)
            logger.debug(f"Loaded {filepath} from raw folder")
            return data
        except Exception as e:
            raise AppException(e, sys)
    
    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Performing data cleaning")
            # Implement data cleaning logic here
            # features with nan value
            features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
            for feature in features_with_na:
                if (np.round(df[feature].isnull().mean()*100,5)) > 25:
                    df.drop(feature, axis=1, inplace=True)
                    logging.info(f"Dropped {feature} due to high missing value percentage")
                elif (np.round(df[feature].isnull().mean()*100,5)) > 50:
                    df[feature].fillna(df[feature].median(), inplace=True)
                    logging.info(f"Filled {feature} with median")
            
            # handling data types
            logging.info("Correcting data types")
            for column in df.columns:
                if pd.api.types.is_object_dtype(df[column]):
                    df[column] = df[column].astype(str)
                elif pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif pd.api.types.is_datetime64_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                else:
                    logging.warning(f"Unsupported data type for column: {column}")
            # handling duplicates
            logging.info("Handling duplicates")
            df.drop_duplicates(inplace=True)
            logging.info("Duplicates handled")
            # remove outliers from the dataset
            logging.info("Removing outliers")
            # Assuming df is a DataFrame and 'numeric_column' is the column to detect outliers
            # remove the outliers from each numeric column
            numeric_columns = df.select_dtypes(include=['number']).columns
            for column in numeric_columns:
                # Calculate Q1, Q3, and IQR for the column
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
                outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
                
                # Remove outliers from the DataFrame
                df = df[~outliers]
            logging.info("Outliers removed")
            return df
        except Exception as e:
            raise AppException(e, sys)
    
    def save_data(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Saving data to processed folder")
            os.makedirs(self.data_ingestion_artifact.processed_data_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_file_path = os.path.join(
                self.data_ingestion_artifact.processed_data_path, 
                f'processed_data_{timestamp}.parquet'
            )            
            df.to_parquet(processed_file_path, index=False)
        except Exception as e:
            raise AppException(e, sys)
    
    
        