from unittest.mock import patch

import pandas as pd
import pytest
from zenml.steps.base_step import BaseStep


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


@pytest.fixture(autouse=True)
def mock_zenml_runtime():
    """Completely disable ZenML step execution during tests."""

    def mock_step_call(self, *args, **kwargs):
        return self.entrypoint(*args, **kwargs)

    with patch.object(BaseStep, "__call__", new=mock_step_call), patch("zenml.steps.step_decorator.step", lambda x: x):  # Bypass @step decorator
        yield
