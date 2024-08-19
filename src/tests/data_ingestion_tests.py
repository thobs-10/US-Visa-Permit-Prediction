import unittest
import os
import pandas as pd
from src.components.data_ingestion import DataIngestion

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        self.data_ingestion = DataIngestion()

    def test_load_data(self):
        data = self.data_ingestion.load_data()
        self.assertIsInstance(data, pd.DataFrame)

    def test_data_cleaning(self):
        data = self.data_ingestion.load_data()
        cleaned_data = self.data_ingestion.data_cleaning(data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(data.shape, cleaned_data.shape)

    def test_save_data(self):
        data = self.data_ingestion.load_data()
        cleaned_data = self.data_ingestion.data_cleaning(data)
        self.data_ingestion.save_data(cleaned_data)
        self.assertTrue(os.path.exists(self.data_ingestion.data_ingestion_artifact.processed_data_path))

if __name__ == '__main__':
    unittest.main()