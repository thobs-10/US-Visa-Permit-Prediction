import unittest
import os
import pandas as pd
import numpy as np

from src.components.feature_engineering import FeatureEngineering

from src.entity.config_entity import FeatureEngineeringConfig, FeatureEngineeringPipelineConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, DataIngestionArtifact
from src.pipeline.data_ingestion import DataIngestionPipeline

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.feature_engineering_config = FeatureEngineeringConfig(
            feature_engineering_pipeline_config=FeatureEngineeringPipelineConfig(
                feature_engineering_dir="tests/data/feature_engineered_data"
            )
        )
        self.feature_engineering_artifact = FeatureEngineeringArtifact(
            feature_engineered_data_path=self.feature_engineering_config.feature_engineering_dir
        )
        self.feature_engineering = FeatureEngineering(
            feature_engineering_artifact=self.feature_engineering_artifact,
            feature_engineering_config=self.feature_engineering_config
        )

    def test_load_cleaned_data(self):
        data_ingestion_artifact = DataIngestionArtifact(
            raw_data_path="tests/data/raw_data.csv",
            processed_data_path="tests/data/processed_data.parquet"
        )
        processed_data = self.feature_engineering.load_cleaned_data(
            data_ingestion_artifact=data_ingestion_artifact
        )
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.shape[0], 10)
        self.assertEqual(processed_data.shape[1], 8)

    def test_feature_extraction(self):
        processed_data = pd.read_parquet(
            os.path.join(self.feature_engineering_config.transformed_data_dir, "processed_data.parquet")
        )
        data = self.feature_engineering.feature_extraction(processed_data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 14)

    def test_feature_transformations(self):
        data = pd.read_parquet(os.path.join(self.feature_engineering_config.transformed_data_dir, "data.parquet"))
        df, y = self.feature_engineering.feature_transformations(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(df.shape[0], 10)
        self.assertEqual(df.shape[1], 14)
        self.assertEqual(y.shape[0], 10)

    def test_feature_scaling(self):
        df = pd.read_parquet(os.path.join(self.feature_engineering_config.transformed_data_dir, "df.parquet"))
        X = self.feature_engineering.feature_scaling(df)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(X.shape[1], 14)

    def test_resampling_dataset(self):
        X = np.load(os.path.join(self.feature_engineering_config.transformed_data_dir, "X.npy"))
        y = pd.read_parquet(os.path.join(self.feature_engineering_config.transformed_data_dir, "y.parquet"))
        X_res, y_res = self.feature_engineering.resampling_dataset(X, y)
        self.assertIsInstance(X_res, np.ndarray)
        self.assertIsInstance(y_res, pd.Series)
        self.assertEqual(X_res.shape[0], 10)
        self.assertEqual(X_res.shape[1], 14)
        self.assertEqual(y_res.shape[0], 10)

    def test_save_processed_data(self):
        X = np.load(os.path.join(self.feature_engineering_config.transformed_data_dir, "X_resampled.npy"))
        y = pd.read_parquet(os.path.join(self.feature_engineering_config.transformed_data_dir, "y_resampled.parquet"))
        self.feature_engineering.save_processed_data(X, y)
        self.assertTrue(os.path.exists(os.path.join(self.feature_engineering_config.transformed_data_dir, "X_resampled.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.feature_engineering_config.transformed_data_dir, "y_resampled.parquet")))

if __name__ == '__main__':
    unittest.main()