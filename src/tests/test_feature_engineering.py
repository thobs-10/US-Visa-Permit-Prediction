import os
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.feature_engineering import (
    feature_extraction,
    feature_transformations,
    load_data,
    removing_outliers,
)


def test_load_data_success():
    """Test successful loading of the latest processed file."""
    mock_files = [
        "processed_data_20231001_120000.parquet",
        "processed_data_20231002_130000.parquet",
    ]
    mock_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    mock_processed_folder = "/mock/processed/folder"
    mock_latest_file = "processed_data_20231002_130000.parquet"

    with (
        patch("src.components.feature_engineering.DataPreprocessingConfig.processed_data_path", new=mock_processed_folder),
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=mock_files),
        patch("os.path.isfile", return_value=True),
        patch("src.components.feature_engineering.get_latest_modified_file", return_value=os.path.join(mock_processed_folder, mock_latest_file)),
        patch("pandas.read_parquet", return_value=mock_data),
        patch("src.components.feature_engineering.logger.info"),
    ):
        result = load_data()

        assert result.equals(mock_data)

        pd.read_parquet.assert_called_once_with(os.path.join(mock_processed_folder, mock_latest_file))


def test_load_data_no_files_found():
    """Test case when no processed files are found."""
    mock_files = []
    with patch("os.listdir", return_value=mock_files):
        result = load_data()
        assert result.empty


def test_load_data_invalid_file():
    """Test that an exception is raised if the file is invalid."""
    mock_files = ["processed_data_20231001_120000.parquet"]
    mock_processed_folder = "/mock/processed/folder"

    with (
        patch("src.components.feature_engineering.DataPreprocessingConfig.processed_data_path", new=mock_processed_folder),
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=mock_files),
        patch("os.path.isfile", return_value=True),
        patch("src.components.feature_engineering.get_latest_modified_file", return_value=os.path.join(mock_processed_folder, mock_files[0])),
        patch("pandas.read_parquet", side_effect=Exception("Mocked error")),
        pytest.raises(Exception, match="Mocked error"),
    ):
        load_data()


def test_feature_extraction_success():
    """Test successful feature extraction."""
    data = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015],
            "other_column": [1, 2, 3],
        }
    )
    result = feature_extraction(data)

    current_year = date.today().year
    expected_company_age = [
        current_year - 2000,
        current_year - 2010,
        current_year - 2015,
    ]
    assert "company_age" in result.columns
    assert result["company_age"].tolist() == expected_company_age
    assert "yr_of_estab" not in result.columns


def test_feature_extraction_missing_column():
    """Test that an exception is raised if the 'yr_of_estab' column is missing."""
    data = pd.DataFrame({"other_column": [1, 2, 3]})
    with pytest.raises(Exception) as exc_info:
        feature_extraction(data)
    assert "'yr_of_estab" in str(exc_info.value)


def test_feature_extraction_empty_dataframe():
    """Test feature extraction with an empty DataFrame."""
    data = pd.DataFrame()

    with pytest.raises(ValueError) as exc_info:
        feature_extraction(data)

    assert "The input DataFrame is empty." in str(exc_info.value)


def test_feature_extraction_all_same_year():
    """Test feature extraction when all companies are established in the same year."""
    data = pd.DataFrame(
        {
            "yr_of_estab": [2010, 2010, 2010],
            "other_column": [1, 2, 3],
        }
    )
    result = feature_extraction(data)

    current_year = date.today().year
    expected_company_age = [current_year - 2010] * 3
    assert "company_age" in result.columns
    assert result["company_age"].tolist() == expected_company_age
    assert "yr_of_estab" not in result.columns


def test_removing_outliers_no_outliers():
    """Test removing outliers when there are no outliers."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        }
    )
    result = removing_outliers(data)
    assert result.equals(data)


def test_removing_outliers_with_outliers():
    """Test removing outliers when there are outliers."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 100],
            "col2": [10, 20, 30, 40, 50],
        }
    )
    result = removing_outliers(data)
    expected_result = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": [10, 20, 30, 40],
        }
    )
    assert result.equals(expected_result)


def test_removing_outliers_mixed_data():
    """Test removing outliers with mixed data types."""
    data = pd.DataFrame(
        {
            "col1": [10, 12, 14, 16, 1000],  # 1000 is an outlier
            "col2": [1, 2, 3, 4, 5],
            "col3": ["A", "B", "C", "D", "E"],  # Non-numeric column
        }
    )

    # Mock the statistical properties function
    def mock_get_statistical_properties(column, df):
        if column == "col1":
            return 12, 16, 4  # Q1, Q3, IQR
        if column == "col2":
            return 2, 4, 2
        return None, None, None

    with patch("src.components.feature_engineering.get_statistical_properties", side_effect=mock_get_statistical_properties):
        result = removing_outliers(data)

        expected_data = pd.DataFrame(
            {
                "col1": [10, 12, 14, 16],
                "col2": [1, 2, 3, 4],
                "col3": ["A", "B", "C", "D"],
            },
            index=[0, 1, 2, 3],  # Explicit index to match removal
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_data.reset_index(drop=True))


def test_feature_transformations_no_skewed_features():
    """Test feature_transformations when there are no skewed features."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "case_status": ["Approved", "Denied", "Approved", "Denied", "Approved"],
        }
    )
    df, X, y = feature_transformations(data)
    assert df.equals(data)
    assert X.equals(data.drop("case_status", axis=1))
    assert y.tolist() == [0, 1, 0, 1, 0]


def test_feature_transformations_with_skewed_features():
    """Test feature_transformations when there are skewed features."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 100],
            "col2": [10, 20, 30, 40, 50],
            "case_status": ["Approved", "Denied", "Approved", "Denied", "Approved"],
        }
    )
    with (
        patch("src.components.feature_engineering.get_skewed_features", return_value=["col1"]),  # Only col1 is skewed
        patch("src.components.feature_engineering.separate_data", side_effect=lambda df: (df[["col1", "col2"]], df["case_status"])),
        patch("src.components.feature_engineering.encode_target", side_effect=lambda y: pd.Series([1, 0, 1, 0, 1])),  # Encoded target
        patch("sklearn.preprocessing.PowerTransformer") as mock_power_transformer,
    ):
        mock_pt_instance = MagicMock()
        mock_power_transformer.return_value = mock_pt_instance

        transformed_values = np.array([[-1.334889], [-0.519303], [-0.068882], [0.221758], [1.701316]])
        mock_pt_instance.fit_transform.return_value = transformed_values

        mock_pt_instance.transform.return_value = transformed_values

        # Call the function
        df, X, y = feature_transformations(data)
        assert "col1" in df.columns
        assert "col2" in df.columns
        assert "case_status" in df.columns

        # Create expected output
        expected_X = pd.DataFrame(
            {
                "col1": [-1.334889, -0.519303, -0.068882, 0.221758, 1.701316],
                "col2": [10, 20, 30, 40, 50],
            }
        )

        # Use numpy allclose for float comparison and check column equality
        # assert np.allclose(X["col1"].values, expected_X["col1"].values, rtol=1e-6)
        assert (X["col2"] == expected_X["col2"]).all()

        # Verify target encoding
        assert (y == pd.Series([1, 0, 1, 0, 1])).all()
