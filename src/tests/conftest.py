import pandas as pd
import pytest


@pytest.fixture
def mock_raw_data_path(monkeypatch):
    # Mock the environment variable for raw data path
    monkeypatch.setenv("PROCESSED_DATA_FILE", "data/raw_data/EasyVisa.csv")


@pytest.fixture
def sample_data():
    # Sample DataFrame to be returned by pd.read_csv
    return pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
        }
    )
