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
from src.Exception import AppException
from src.entity.artifact_entity import FeatureEngineeringArtifact
from src.entity.config_entity import FeatureEngineeringConfig
from src.pipeline.data_ingestion import DataIngestionArtifact

class FeatureEngineering:
    def __init__(self, feature_engineering_artifact: FeatureEngineeringArtifact, feature_engineering_config: FeatureEngineeringConfig):
        try:
            self.feature_engineering_artifact = feature_engineering_artifact
            self.feature_engineering_config = feature_engineering_config
        except Exception as e:
            raise AppException(e, sys)
    
    def load_cleaned_data(self, data_ingestion_artifact: DataIngestionArtifact) -> pd.DataFrame:
        try:
            logging.info("Loading cleaned data from processed folder")
            processed_folder = data_ingestion_artifact.processed_data_path
            # List all files in the directory
            files = [f for f in os.listdir(processed_folder) if os.path.isfile(os.path.join(processed_folder, f))]
            # Filter files with 'processed_data_' prefix and sort them by modification time
            processed_files = [f for f in files if f.startswith('processed_data_')]
            if not processed_files:
                raise FileNotFoundError("No processed data files found in the directory.")
            # Get full file paths and sort them by modification time
            full_file_paths = [os.path.join(processed_folder, f) for f in processed_files]
            latest_file = max(full_file_paths, key=os.path.getmtime)
            data = pd.read_parquet(latest_file)
            logging.info(f"Loaded cleaned data from {processed_folder}")
            return data
        except Exception as e:
            raise AppException(e, sys)
    
    def feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Performing feature extraction")
            # creating the date object of today's date
            todays_date = date.today()
            current_year= todays_date.year
            df['company_age'] = current_year-df['yr_of_estab']
            df.drop('yr_of_estab', inplace=True, axis=1)
            logging.info('successfully extracted  new feature(s)')
            return df
        except Exception as e:
            raise AppException(e, sys)
    
    def feature_transformations(self,df: pd.DataFrame) -> tuple:
        try:
            logging.info("Performing feature transformations")
            # Initialize PowerTransformer
            pt = PowerTransformer(method='yeo-johnson')
            # Identify continuous features
            continuous_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Check skewness and identify features with skewness > 2.0
            skewed_features = df[continuous_features].apply(lambda x: x.skew()).abs()
            transform_features = skewed_features[skewed_features > 1.0].index.tolist()
            if len(transform_features) > 0:
                logging.info(f"Features to be transformed: {transform_features}")
                # Prepare data
                X = df.drop('case_status', axis=1)
                y = df['case_status']
                # Encode target variable
                y = np.where(y == 'Denied', 1, 0)
                # Apply PowerTransformer to the identified skewed features
                X_copy = X.copy()
                X_copy[transform_features] = pt.fit_transform(X[transform_features])
                # Update the original DataFrame with the transformed features
                df[transform_features] = X_copy[transform_features]
                logging.info("Feature transformations completed successfully")
                return df, y
            else:
                logging.info("No features to be transformed")
                X = df.drop('case_status', axis=1)
                y = df['case_status']
                # Encode target variable
                y = np.where(y == 'Denied', 1, 0)
                return df, y
        except Exception as e:
            raise AppException(e, sys)
        
    def feature_scaling(self, df: pd.DataFrame):
        try:
            logging.info("Starting feature scaling")
            X = df.drop('case_status', axis=1)
            num_features = list(X.select_dtypes(exclude="object").columns)
            # Create Column Transformer with 3 types of transformers
            or_columns = ['has_job_experience','requires_job_training','full_time_position','education_of_employee']
            oh_columns = ['continent','unit_of_wage','region_of_employment']
            transform_columns= ['no_of_employees','company_age']
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )
            X = preprocessor.fit_transform(X)
            os.makedirs('src/models/best_model', exist_ok=True)
            path = f'src/models/best_model/'
            # Save the preprocessor to a file
            output_path = os.path.join(path, "preprocessor.pkl")
            # Save the preprocessor to a file for future use in model deployment and prediction stages.
            # Note: This file should be included in the model deployment package along with the trained model.
            # This way, the preprocessor can be used to transform new data before making predictions.
            # Note: This is just an example, the actual path and filename should be set according to your project structure.
            # Also, you may want to consider using a secure method to save the preprocessor, such as using environment variables or secure key storage.
            # Example: os.environ.get('PREPROCESSOR_PATH', 'path_to_your_file')
            # Example: output_path = os.path.join(os.environ.get('MODEL_DEPLOYMENT_PATH', 'path_to_your
            joblib.dump(preprocessor, output_path)
            logging.info("Feature scaling completed successfully")
            return X
        except Exception as e:
            raise AppException(e, sys)
    def resampling_dataset(self, X, y):
        try:
            logging.info("Resampling dataset")
            smote = SMOTE(sampling_strategy='minority')
            X_res, y_res = smote.fit_resample(X, y)
            logging.info("Resampling completed successfully")
            return X_res, y_res
        except Exception as e:
            raise AppException(e, sys)
    
    def save_processed_data(self, X: np.ndarray, y: np.ndarray):
        try:
            logging.info("Saving feature engineered data")
            # lets save without the use of configuration for now
            X_df = pd.DataFrame(X)
            y_df = pd.Series(y, name='case_status')
            # Ensure the artifact path exists
            os.makedirs(self.feature_engineering_config.transformed_data_dir, exist_ok=True)
            transformed_features_file_path = os.path.join(self.feature_engineering_config.transformed_features_file)
            transformed_target_file_path = os.path.join( self.feature_engineering_config.transformed_target_file)

            X_df.to_parquet(transformed_features_file_path, index=False)
            # pd.DataFrame(y, columns=['case_status']).to_parquet(transformed_target_file_path, index=False)
            y_df.to_frame().to_parquet(transformed_target_file_path, index=False)
            logging.info("Processed data saved successfully")
        except Exception as e:
            raise AppException(e, sys)

