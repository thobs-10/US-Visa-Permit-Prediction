from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.components.model_tuning import hyperparameter_tuning


def test_hyperparameter_tuning_success():
    """Test successful hyperparameter tuning."""
    X_train = pd.DataFrame(
        {
            "yr_of_estab": [2000, 2010, 2015, 2020, 2005, 1995, 2018, 2008, 2012, 2003],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    X_val = pd.DataFrame(
        {
            "yr_of_estab": [2010, 2015, 2020],
            "other_column": [2, 3, 4],
        }
    )
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_val = pd.Series([0, 1, 0])
    chosen_model_path = "model_path"
    chosen_model_name = "Random_Forest"
    mock_column_transformer = Mock(spec=ColumnTransformer)
    mock_column_transformer.fit_transform.return_value = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
        ],
    )
    mock_column_transformer.transform.return_value = np.array(
        [[1, 2], [3, 4], [5, 6]],
    )

    class MockEstimator:
        def __init__(self, param1=1, param2=2):
            self.param1 = param1
            self.param2 = param2

        def get_params(self, deep=True):
            return {"param1": self.param1, "param2": self.param2}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def score(self, X, y):
            return 0.9

        def __eq__(self, other):
            if not isinstance(other, MockEstimator):
                return False
            return self.param1 == other.param1 and self.param2 == other.param2

    # Mock the model to behave like a scikit-learn estimator
    mock_model = MockEstimator()

    # Mock the best model
    mock_best_model = MockEstimator(param1=1, param2=2)
    # Mock the RandomizedSearchCV
    mock_random_cv_model = Mock(spec=RandomizedSearchCV)
    mock_random_cv_model.best_estimator_ = mock_best_model
    mock_random_cv_model.best_params_ = {"param1": 1, "param2": 2}
    mock_random_cv_model.fit.return_value = None

    mock_mlflow = Mock()

    mock_load_local_model = Mock(return_value=mock_model)

    mock_randomcv_models = [
        (
            chosen_model_name,
            mock_model,
            {"param1": [1, 2, 3], "param2": [4, 5, 6]},
        ),
    ]

    with (
        patch("src.components.model_training.mlflow", mock_mlflow),
        patch(
            "src.components.model_tuning.load_local_model",
            mock_load_local_model,
        ),
        patch(
            "src.entity.config_entity.randomcv_models",
            mock_randomcv_models,
        ),
        patch(
            "sklearn.model_selection.RandomizedSearchCV",
            return_value=mock_random_cv_model,
        ),
        patch(
            "sklearn.model_selection.KFold",
            return_value=MagicMock(spec=KFold),
        ),
        patch(
            "src.utils.main_utils.log_mlflow_metrics",
            return_value={"Accuracy": 0.9, "Precision": 0.8, "Recall": 0.7},
        ),
        patch(
            "src.components.model_training.infer_signature",
            return_value="mock_signature",
        ),
    ):
        # Call the function
        result = hyperparameter_tuning(
            X_train,
            X_val,
            y_train,
            y_val,
            chosen_model_path,
            chosen_model_name,
            mock_column_transformer,
            max_evals=5,
        )

        # Assertions
        assert result.param1 == mock_best_model.param1
        assert result.param2 == mock_best_model.param2

        assert isinstance(result, MockEstimator)
