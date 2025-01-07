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
from src.components.component import Component

class DataIngestion(Component):
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config.raw_data_path
   
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading data from raw folder")
            filepath = os.path.join(self.data_ingestion_config.raw_data_path)
            data = pd.read_csv(filepath)
            logger.debug(f"Loaded {filepath} from raw folder")
            return data
        except Exception as e:
            raise AppException(e, sys)
    
    def handling_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Performing data cleaning")
        features_with_null_values=[features for features in df.columns if df[features].isnull().sum()>=1]
        for feature in features_with_null_values:
            if df[feature].dtype == 'categorical':
                df[feature] = df[feature].fillna(df[feature].mode()[0], inplace=True)
            else:
                df[feature] = df[feature].fillna(df[feature].mean())
        logger.debug("Handled missing values successfully")
        return df
        
    def handling_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Correcting data types")
        for column in df.columns:
            if pd.api.types.is_object_dtype(df[column]):
                df[column] = df[column].astype(str)
            elif pd.api.types.is_numeric_dtype(df[column]):
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif pd.api.types.is_datetime64_dtype(df[column]):
                df[column] = pd.to_datetime(df[column], errors='coerce')
            else:
                logger.warning(f"Unsupported data type for column: {column}")
        logger.debug("Data types corrected successfully")
        return df
    
    def handling_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling duplicates")
        duplicated = df[df.duplicated(keep=False)]
        if duplicated.shape[0] > 0:
            df = df.drop_duplicates(inplace= True)
        logger.debug("Duplicates handled")
        return df
    
    def save(self, data: pd.DataFrame, filename: str='processed_data.parquet') -> None:
        try:
            logger.info("Saving data to processed folder")
            filepath = os.path.join(self.data_ingestion_config.processed_data_path, filename)
            data.to_parquet(filepath)
            logger.debug(f"Saved {filepath} to processed folder")
        except Exception as e:
            raise AppException(e, sys)
    
        