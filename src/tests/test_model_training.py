import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np
from src.components.model_training import (
    load_data,
    split_data,
    train_model,
    save_preprocessor,
)
from src.entity.config_entity import ModelTrainingConfig
from sklearn.compose import ColumnTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_split_data_success():
    """Test successful splitting of data."""
    X = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = pd.DataFrame(
        {
            "Unnamed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "series": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    result = split_data(X, y)
    assert len(result) == 6
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.Series)
    assert isinstance(result[2], pd.DataFrame)
    assert isinstance(result[3], pd.Series)
    assert isinstance(result[4], pd.DataFrame)
    assert isinstance(result[5], pd.Series)


def test_split_data_with_nan_values():
    """Test splitting of data with NaN values."""
    X = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = pd.DataFrame(
        {
            "Unnamed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "series": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    X.iloc[0, 0] = np.nan
    y.iloc[0, 0] = np.nan
    result = split_data(X, y)
    assert len(result) == 6
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.Series)
    assert isinstance(result[2], pd.DataFrame)
    assert isinstance(result[3], pd.Series)
    assert isinstance(result[4], pd.DataFrame)
    assert isinstance(result[5], pd.Series)


def test_split_data_with_infinite_values():
    """Test splitting of data with infinite values."""
    X = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = pd.DataFrame(
        {
            "Unnamed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "series": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    X.iloc[0, 0] = np.inf
    y.iloc[0, 0] = np.inf
    result = split_data(X, y)
    assert len(result) == 6
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.Series)
    assert isinstance(result[2], pd.DataFrame)
    assert isinstance(result[3], pd.Series)
    assert isinstance(result[4], pd.DataFrame)
    assert isinstance(result[5], pd.Series)


def test_train_model_success():
    """Test successful training of model."""
    X_train = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    X_valid = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_valid = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    mock_column_transformer = MagicMock(spec=ColumnTransformer)
    mock_X_train_scaled = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    mock_X_valid_scaled = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    mock_results = {
        "model_1": (np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), "model_1_path"),
        "model_2": (np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), "model_2_path"),
    }

    mock_mlflow = MagicMock()
    mock_executor = MagicMock(spec=ThreadPoolExecutor)
    mock_future = MagicMock()
    mock_future.result.return_value = (np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), "model_1_path")

    with (
        patch(
            "src.components.model_training.feature_scaling",
            return_value=(
                mock_column_transformer,
                mock_X_train_scaled,
                mock_X_valid_scaled,
            ),
        ),
        patch(
            "src.components.model_training.train_model_parallel",
            return_value=mock_results,
        ),
        patch("src.components.model_training.mlflow", mock_mlflow),
        patch(
            "src.components.model_training.ThreadPoolExecutor",
            return_value=mock_executor,
        ),
        patch(
            "src.components.model_training.as_completed",
            return_value=[mock_future],
        ),
    ):
        result = train_model(X_train, X_valid, y_train, y_valid)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == mock_column_transformer
        assert isinstance(result[1], str)
        assert isinstance(result[2], str)

        mock_mlflow.set_experiment.assert_called_once_with("Model Training Phase")
        mock_mlflow.set_experiment_tag.assert_called_once_with("model-training", "v1.0.0")
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_metrics.assert_called()


def test_save_preprocessor_successfully():
    """Test successful saving of preprocessor."""
    mock_column_transformer = MagicMock(spec=ColumnTransformer)
    model_artifact_dir = "src/models/artifacts/"
    with patch("joblib.dump") as mock_joblib_dump:
        with patch(
            "src.entity.config_entity.ModelTrainingConfig.model_artifact_dir",
            model_artifact_dir,
        ):
            output_path = os.path.join(model_artifact_dir, "preprocessor.joblib")
            results = save_preprocessor(mock_column_transformer)
            mock_joblib_dump.assert_called_once_with(mock_column_transformer, output_path)
            assert results is None


def test_save_preprocessor_with_exception():
    """Test saving of preprocessor with an exception."""
    mock_column_transformer = MagicMock(spec=ColumnTransformer)
    with patch("joblib.dump") as mock_joblib_dump:
        mock_joblib_dump.side_effect = Exception("An error occurred")
        with pytest.raises(Exception):
            save_preprocessor(mock_column_transformer)
            mock_joblib_dump.assert_called_once()
