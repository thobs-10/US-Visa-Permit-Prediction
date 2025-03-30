import os
from datetime import date, datetime

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    FileSource,
    ValueType,
)
from loguru import logger
from zenml import step

from src.entity.config_entity import DataPreprocessingConfig, FeatureEngineeringConfig
from src.utils.main_utils import (
    apply_power_transform,
    encode_target,
    get_latest_modified_file,
    get_skewed_features,
    get_statistical_properties,
    separate_data,
)


@step(enable_cache=False)
def load_data() -> pd.DataFrame:
    try:
        logger.info("Loading cleaned data from processed folder")
        processed_folder = DataPreprocessingConfig.processed_data_path
        if not os.path.exists(processed_folder):
            logger.warning("Cannot find processed folder")
            return pd.DataFrame()
        files = [f for f in os.listdir(processed_folder) if os.path.isfile(os.path.join(processed_folder, f))]
        processed_files = [f for f in files if f.startswith("processed_data_")]
        if not processed_files:
            logger.warning("No processed data files found in the directory.")
            return pd.DataFrame()

        latest_file = get_latest_modified_file(processed_files, processed_folder)
        data = pd.read_parquet(latest_file)
        logger.info(f"Loaded cleaned data from {processed_folder}")
        return data
    except Exception as e:
        raise e


@step
def feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Performing feature extraction")
    if df.empty:
        logger.error("The input DataFrame is empty.")
        raise ValueError("The input DataFrame is empty.")

    if "yr_of_estab" not in df.columns:
        logger.error("'yr_of_estab' column not found in the DataFrame.")
        raise KeyError("'yr_of_estab' column not found in the DataFrame.")

    todays_date = date.today()
    current_year = todays_date.year
    try:
        df["yr_of_estab"] = pd.to_numeric(df["yr_of_estab"], errors="raise")
    except ValueError as e:
        logger.exception(f"Invalid data in 'yr_of_estab' column: {e}")

    df["company_age"] = current_year - df["yr_of_estab"]
    df.drop("yr_of_estab", inplace=True, axis=1)
    logger.info("Successfully extracted new feature(s)")
    return df


@step
def removing_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing outliers")
    numeric_columns = df.select_dtypes(include=["number"]).columns
    outlier_mask = pd.Series(data=False, index=df.index)
    for column in numeric_columns:
        Q1, Q3, IQR = get_statistical_properties(column, df)
        column_outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outlier_mask = outlier_mask | column_outliers
    df = df[~outlier_mask]
    logger.info("Outliers removed")
    return df


@step
def feature_transformations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    logger.info("Performing feature transformations")
    continuous_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    transform_features = get_skewed_features(df, continuous_features)
    if len(transform_features) > 0:
        logger.info(f"Features to be transformed: {transform_features}")
        X, y = separate_data(df)
        y = encode_target(y)
        X = apply_power_transform(df, X, transform_features)
        logger.info("Feature transformations completed successfully")
        return df, X, y

    logger.info("No features to be transformed")
    X, y = separate_data(df)
    y = encode_target(y)
    return df, X, y


@step
def save(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> None:
    try:
        logger.info("Saving feature engineered data")
        fullpath = FeatureEngineeringConfig.feature_engineering_dir
        if not os.path.exists(fullpath):
            logger.info(f"Creating directory: {fullpath}")
            os.makedirs(fullpath, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {fullpath}")

        saving_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_filename = os.path.join(fullpath, f"features_{saving_timestamp}.parquet")
        target_filename = os.path.join(fullpath, f"target_{saving_timestamp}.csv")
        X["event_timestamp"] = datetime.now()
        X.to_parquet(feature_filename)
        y.to_csv(target_filename)
        df["event_timestamp"] = datetime.now()
        df.to_parquet(os.path.join(fullpath, f"full_data_{saving_timestamp}.parquet"))
        logger.info("Processed data saved successfully")
    except Exception as e:
        raise e


@step
def save_to_feast_feature_store() -> None:
    try:
        logger.info("Saving data to Feast feature store")
        passenger = Entity(name="case_id", value_type=ValueType.STRING)
        feature_view = FeatureView(
            name="passenger_features",
            entities=[Entity(name="case_id", value_type=ValueType.STRING)],
            ttl=None,
            source=FileSource(
                path="/Users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/US-Visa-Permit-Prediction/data/feature_store/features.parquet",
                event_timestamp_column="event_timestamp",
            ),
        )
        fs = FeatureStore(
            repo_path="/Users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/US-Visa-Permit-Prediction/my_feature_store/feature_repo",
        )
        passenger_feature_service = FeatureService(
            name="passenger_features",
            features=[feature_view],
            tags={},
        )
        fs.apply([passenger, feature_view, passenger_feature_service])
        fs.materialize_incremental(end_date=datetime.now())
        logger.info("Data saved to Feast feature store")
    except Exception as e:
        raise e
