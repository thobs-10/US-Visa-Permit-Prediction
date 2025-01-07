import os
import sys
import zipfile
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE

from src.logger import logging
from loguru import logger
from src.Exception import AppException
from src.entity.artifact_entity import FeatureEngineeringArtifact
from src.entity.config_entity import FeatureEngineeringConfig
from src.pipeline.data_ingestion import DataIngestionArtifact
from src.entity.config_entity import DataPreprocessingConfig
from src.components.component import Component
from src.utils.main_utils import (get_skewed_features, separate_data, apply_power_transform, 
                                  encode_target, get_latest_modified_file, get_statistical_properties,
                                  instantiate_encoders, get_column_transformer)

class FeatureEngineering(Component):
    def __init__(self, feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
                 data_preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig()):
        
        self.feature_engineering_config = feature_engineering_config
        self.data_preprocessing_config = data_preprocessing_config
    
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading cleaned data from processed folder")
            processed_folder =self.data_preprocessing_config.processed_data_path
            files = [f for f in os.listdir(processed_folder) if os.path.isfile(os.path.join(processed_folder, f))]
            processed_files = [f for f in files if f.startswith('processed_data_')]
            if not processed_files:
                raise FileNotFoundError("No processed data files found in the directory.")

            latest_file = get_latest_modified_file(processed_files, processed_folder)
            data = pd.read_parquet(latest_file)
            logger.info(f"Loaded cleaned data from {processed_folder}")
            return data
        except Exception as e:
            raise AppException(e, sys)
    
    def feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Performing feature extraction")
        todays_date = date.today()
        current_year= todays_date.year
        df['company_age'] = current_year-df['yr_of_estab']
        df.drop('yr_of_estab', inplace=True, axis=1)
        logger.info('successfully extracted  new feature(s)')
        return df
    
    def removing_outliers(df : pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing outliers")
        numeric_columns = df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            Q1, Q3, IQR = get_statistical_properties(column)
            outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
            df = df[~outliers]
        logger.info("Outliers removed")
        return df


    def feature_transformations(self,df: pd.DataFrame) -> tuple:
    
        logger.info("Performing feature transformations")
        continuous_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        transform_features = get_skewed_features(df, continuous_features)
        if len(transform_features) > 0:
            logger.info(f"Features to be transformed: {transform_features}")
            X, y = separate_data(df)
            y = encode_target(y)
            df = apply_power_transform(df, X, transform_features)
            logger.info("Feature transformations completed successfully")
            return df, y
        else:
            logger.info("No features to be transformed")
            X, y = separate_data(df)
            y = encode_target(y)
            return df, X, y
        
    def resampling_dataset(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Resampling dataset")
        smote = SMOTE(sampling_strategy='minority')
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("Resampling completed successfully")
        return X_res, y_res
    
    def save_preprocessor(self, preprocessor: ColumnTransformer, filename: str = "preprocessor.joblib") -> None:
        try:
            logger.info("Saving preprocessor..")
            path = os.getenv("ARTIFACTS_PATH")
            os.makedirs(path, exist_ok=True)
            output_path = os.path.join(path, filename)
            joblib.dump(preprocessor, output_path)
        except Exception as e:
            raise AppException(e, sys)

    def save(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            logger.info("Saving feature engineered data")
            fullpath = self.feature_engineering_config.feature_engineering_dir
            os.makedirs(fullpath, exist_ok=True)
            feature_filename = os.path.join(fullpath, "features.parquet")
            target_filename = os.path.join(fullpath, "target.csv")
            X.to_parquet(feature_filename)
            y.to_csv(target_filename)
            logger.info("Processed data saved successfully") 
        except Exception as e:
            raise AppException(e, sys)
        

    # def save_processed_data(self, X: pd.DataFrame, y: pd.Series):
    #     try:
    #         logger.info("Saving feature engineered data")
    #         fullpath = self.feature_engineering_config.feature_engineering_dir
    #         os.makedirs(fullpath, exist_ok=True)
    #         X.to_parquet(fullpath)
    #         y.to_csv(fullpath)
    #         logger.info("Processed data saved successfully")
    #     except Exception as e:
    #         raise AppException(e, sys)

