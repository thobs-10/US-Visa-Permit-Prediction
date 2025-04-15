import os
from datetime import datetime

import pandas as pd
from loguru import logger
from zenml import step

from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig


@step(enable_cache=False)
def load_raw_data() -> pd.DataFrame:
    try:
        logger.info("Loading data from raw folder")
        filepath = os.path.join(DataIngestionConfig.raw_data_path)
        data = pd.read_csv(filepath)
        logger.debug(f"Loaded {filepath} from raw folder")
        return data
    except Exception as e:
        raise e


@step
def handling_null_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Performing data cleaning")
    features_with_null_values = [features for features in df.columns if df[features].isnull().sum() >= 1]
    for feature in features_with_null_values:
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = df[feature].fillna(df[feature].mean())
        else:
            df[feature] = df[feature].fillna(df[feature].mode()[0])
    logger.debug("Handled missing values successfully")
    return df


@step
def handling_data_type(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Correcting data types")
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].astype(str)
        elif pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif pd.api.types.is_datetime64_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], errors="coerce")
        else:
            raise TypeError(f"Unsupported data type for column: {column}")
    logger.debug("Data types corrected successfully")
    return df


@step
def handling_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling duplicates")
    df = df.drop_duplicates()
    logger.debug("Duplicates handled successfully")
    return df


@step
def save_processed_data(
    data: pd.DataFrame,
    filename: str = "processed_data",
) -> None:
    try:
        logger.info("Saving data to processed folder")
        filepath = os.path.join(DataPreprocessingConfig.processed_data_path)
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.parquet"
        full_filepath = os.path.join(filepath, full_filename)
        data.to_parquet(full_filepath)
        logger.debug(f"Saved {full_filepath} to processed folder")
    except Exception as e:
        raise e
