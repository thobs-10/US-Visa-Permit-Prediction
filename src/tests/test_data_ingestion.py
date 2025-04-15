import numpy as np
import pandas as pd
import pytest

from src.components.data_ingestion import (
    handling_data_type,
    handling_duplicates,
    handling_null_values,
    load_raw_data,
)


def test_load_data_success(mock_raw_data_path, sample_data, mocker):
    """Testing load_data successfully."""
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=sample_data)
    data = load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw_data/EasyVisa.csv")
    assert not data.empty
    assert list(data.columns) == ["col1", "col2"]
    assert len(data) == 3


def test_load_raw_data_exception(mock_raw_data_path, mocker):
    """Testing load_raw_data with exception."""
    mock_read_csv = mocker.patch("pandas.read_csv", side_effect=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw_data/EasyVisa.csv")


def test_load_raw_data_empty_file(mock_raw_data_path, mocker):
    """Testing load_raw_data with an empty file."""
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    data = load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw_data/EasyVisa.csv")
    assert data.empty
    assert not list(data.columns)
    assert len(data) == 0


def test_handling_null_values_no_nulls():
    """Testing handling_null_values with no nulls."""
    data = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": ["A", "B", "C"]},
    )
    cleaned_data = handling_null_values(data)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data.equals(data)


def test_handling_null_values_numerical_nulls():
    """Testing handling_null_values with some nulls."""
    data = pd.DataFrame(
        {"col1": [1, 2, None], "col2": ["A", None, "C"]},
    )
    cleaned_data = handling_null_values(data)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data["col1"].iloc[2] == 1.5
    assert cleaned_data["col2"].iloc[1] == "A"


def test_handling_null_values_category_nulls():
    """Testing handling_null_values with some nulls."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": pd.Series(["A", None, "A"], dtype="category"),
        }
    )
    cleaned_data = handling_null_values(data)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data["col2"].iloc[1] == "A"


def test_handling_data_type_object():
    """Testing handling_data_type with object dtype."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
        }
    )
    data["col2"] = data["col2"].astype("object")
    cleaned_data = handling_data_type(data)
    assert cleaned_data["col1"].dtype == np.int64
    assert pd.api.types.is_object_dtype(cleaned_data["col2"])


def test_handling_data_type_numeric():
    """Testing handling_data_type with numeric dtype."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.5, 2.5, 3.5],
        }
    )
    cleaned_data = handling_data_type(data)
    assert pd.api.types.is_numeric_dtype(cleaned_data["col1"])
    assert pd.api.types.is_numeric_dtype(cleaned_data["col2"])


def test_handling_data_type_datetime():
    """Testing handling_data_type with datetime dtype."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["2022-01-01", "2022-01-02", "2022-01-03"],
        }
    )
    data["col2"] = pd.to_datetime(data["col2"])
    cleaned_data = handling_data_type(data)
    assert pd.api.types.is_datetime64_dtype(cleaned_data["col2"])


def test_handling_data_type_unsupported():
    """Testing handling_data_type with unsupported dtype."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [True, False, True],
        }
    )
    data["col2"] = data["col2"].astype("bool")
    with pytest.raises(TypeError) as exception:
        handling_data_type(data)
    assert "Unsupported data type for column: col2" in str(exception.value)


def test_handling_duplicates_present():
    """Testing handling_duplicates with duplicates."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 1],
            "col2": ["A", "B", "C", "A"],
        }
    )
    cleaned_data = handling_duplicates(data)
    assert cleaned_data.duplicated().sum() == 0
    assert cleaned_data.shape[0] == 3


def test_handling_duplicates_none():
    """Testing handling_duplicates with no duplicates."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
        }
    )
    cleaned_data = handling_duplicates(data)
    assert cleaned_data.duplicated().sum() == 0
    assert cleaned_data.shape[0] == 3


def test_handling_duplicates_all():
    """Testing handling_duplicates with all duplicates."""
    data = pd.DataFrame(
        {
            "col1": [1, 1, 1, 1],
            "col2": ["A", "A", "A", "A"],
        }
    )
    cleaned_data = handling_duplicates(data)
    assert cleaned_data.duplicated().sum() == 0
    assert cleaned_data.shape[0] == 1
