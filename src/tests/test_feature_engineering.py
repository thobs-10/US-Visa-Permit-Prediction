import pandas as pd
import numpy as np
import pytest
from src.components.feature_engineering import (
    load_data,
    feature_extraction,
    removing_outliers,
    feature_transformations,
)
from unittest.mock import patch, MagicMock
from datetime import date


def test_load_data_success():
    """Test successful loading of the latest processed file."""
    mock_files = [
        "processed_data_20231001_120000.parquet",
        "processed_data_20231002_130000.parquet",
    ]
    # mock_latest_file = "processed_data_20231002_130000.parquet"
    mock_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    with (
        patch("os.listdir", return_value=mock_files),
        patch("pandas.read_parquet", return_value=mock_data),
    ):
        result = load_data()
        assert result.equals(mock_data)


def test_load_data_no_files_found():
    """Test case when no processed files are found."""
    mock_files = []
    with patch("os.listdir", return_value=mock_files):
        result = load_data()
        assert result.empty


def test_load_data_invalid_file():
    """Test that an exception is raised if the file is invalid."""
    mock_files = ["processed_data_20231001_120000.parquet"]
    with (
        patch("os.listdir", return_value=mock_files),
        patch("pandas.read_parquet", side_effect=Exception("Mocked error")),
        pytest.raises(Exception) as exc_info,
    ):
        load_data()
    assert "Mocked error" in str(exc_info.value)


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
    # Create an empty DataFrame
    data = pd.DataFrame()

    # Verify that a ValueError is raised
    with pytest.raises(ValueError) as exc_info:
        feature_extraction(data)

    # Verify the error message
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
    expected_data = pd.DataFrame(
        {
            "col1": [10, 12, 14, 16],
            "col2": [1, 2, 3, 4],
            "col3": ["A", "B", "C", "D"],
        }
    )
    result = removing_outliers(data)
    assert result.equals(expected_data)


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
    with patch("sklearn.preprocessing.PowerTransformer") as mock_power_transformer:
        mock_pt_instance = MagicMock()
        mock_power_transformer.return_value = mock_pt_instance

        mock_pt_instance.fit_transform.return_value = np.array(
            [[-1.334889], [-0.519303], [-0.068882], [0.221758], [1.701316]]
        )
        df, X, y = feature_transformations(data)

        assert "col1" in df.columns
        expected_X = pd.DataFrame(
            {
                "col1": [-1.334889, -0.519303, -0.068882, 0.221758, 1.701316],
                "col2": [10, 20, 30, 40, 50],
            }
        )
        assert X.equals(expected_X)
        assert y.tolist() == [0, 1, 0, 1, 0]
        mock_pt_instance.fit_transform.assert_called_once()
