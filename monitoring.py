import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from loguru import logger
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.entity.config_entity import FeatureEngineeringConfig


@dataclass
class MonitorConfig:
    PROMETHEUS_GATEWAY_URL: str = "localhost:9091"
    PROMETHEUS_JOB_NAME: str = "ml_monitoring"
    REFERENCE_DATA_PATH: str = os.path.join(FeatureEngineeringConfig.feature_engineering_dir, "full_data.parquet")
    NUMERICAL_FEATURES: list[str] = ["feature1", "feature2"]
    DRIFT_THRESHOLD: float = 0.5
    MISSING_VALUES_THRESHOLD: float = 0.1
    TARGET_COLUMN: str = "target"


registry = CollectorRegistry()
data_drift_score_gauge = Gauge("data_drift_score", "Evidently Data Drift Score", registry=registry)
missing_values_score_gauge = Gauge("missing_values_score", "Missing Values Score", registry=registry)


class Monitor:
    def __init__(self):
        self.reference_data_path: str = MonitorConfig.REFERENCE_DATA_PATH
        self.numerical_features: list[str] = MonitorConfig.NUMERICAL_FEATURES
        self.prediction_column: str = MonitorConfig.TARGET_COLUMN
        # self.current_prediction_data: pd.DataFrame = pd.DataFrame()
        self.prediction_df: pd.DataFrame = pd.DataFrame()

    def log_prediction(self, features: pd.DataFrame, prediction: float) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Features must be a pandas DataFrame")
        if not isinstance(prediction, float | int):
            raise TypeError("Prediction must be a float or int")
        features["prediction"] = prediction
        self.prediction_df = pd.concat([self.prediction_df, features], ignore_index=True)

    def _construct_reference_data(self) -> pd.DataFrame:
        try:
            features_df = pd.read_parquet(self.reference_data_path)
            sample_data = features_df.sample(10)
            return sample_data
        except FileNotFoundError as e:
            raise e

    def _construct_column_mapping(self) -> ColumnMapping:
        column_mapping = ColumnMapping(
            numerical_features=self.numerical_features,
            prediction=self.prediction_column,
        )
        return column_mapping

    def _generate_report(self) -> Report:
        report = Report(
            metrics=[
                ColumnDriftMetric(
                    column_name=self.prediction_column,
                ),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
            ]
        )
        return report

    @classmethod
    def process_data_drift_score_results(cls, drift_metric_result: dict[str, Any]) -> None:
        current_drift_score = drift_metric_result["result"]["drift_score"]
        data_drift_score_gauge.set(current_drift_score)
        logger.info(f"Data Drift Score: {current_drift_score}")
        if current_drift_score > MonitorConfig.DRIFT_THRESHOLD:
            logger.warning(f"HIGH DATA DRIFT DETECTED! Score: {current_drift_score} > Threshold: {MonitorConfig.DRIFT_THRESHOLD}")

    @classmethod
    def process_missing_values_score_results(cls, missing_values_metric_result: dict[str, Any]) -> None:
        current_missing_values_share = missing_values_metric_result["result"]["current"]["share_of_missing_values"]
        missing_values_score_gauge.set(current_missing_values_share)
        logger.info(f"Missing Values Share: {current_missing_values_share}")
        if current_missing_values_share > MonitorConfig.MISSING_VALUES_THRESHOLD:
            logger.warning(f"HIGH MISSING VALUES DETECTED! Share: {current_missing_values_share} > Threshold: {MonitorConfig.MISSING_VALUES_THRESHOLD}")

    def _evaluate_drift(self) -> None:
        if self._construct_reference_data().shape[0] == 0:
            logger.error("Reference data is not loaded. Cannot evaluate drift.")
            return

        if self.prediction_df.shape[0] == 0:
            logger.warning("No current prediction data logged yet. Skipping drift evaluation.")
            return

        report = self._generate_report()
        column_mapping = self._construct_column_mapping()

        try:
            report.run(
                reference_data=self._construct_reference_data(),
                current_data=self.prediction_df,
                column_mapping=column_mapping,
            )
            logger.info("Evidently report generated successfully.")
        except RuntimeError as e:
            logger.error(f"Error running Evidently report: {e}")
            raise e

        report_dict = report.as_dict()

        drift_metric_result = next((item for item in report_dict["metrics"] if item["metric_id"] == "DatasetDriftMetric"), None)
        if drift_metric_result and "result" in drift_metric_result and "drift_score" in drift_metric_result["result"]:
            self.process_data_drift_score_results(drift_metric_result)
        else:
            logger.warning("Could not find 'DatasetDriftMetric' result in Evidently report.")

        missing_values_metric_result = next((item for item in report_dict["metrics"] if item["metric_id"] == "DatasetMissingValuesMetric"), None)
        if (
            missing_values_metric_result
            and "result" in missing_values_metric_result
            and "current" in missing_values_metric_result["result"]
            and "share_of_missing_values" in missing_values_metric_result["result"]["current"]
        ):
            self.process_missing_values_score_results(missing_values_metric_result)
        else:
            logger.warning("Could not find 'DatasetMissingValuesMetric' result in Evidently report.")

        # Push to Prometheus
        try:
            push_to_gateway(MonitorConfig.PROMETHEUS_GATEWAY_URL, job=MonitorConfig.PROMETHEUS_JOB_NAME, registry=registry)
            logger.info(f"Successfully pushed metrics to Prometheus Gateway: {MonitorConfig.PROMETHEUS_GATEWAY_URL}")
        except RuntimeError as e:
            logger.error(f"Failed to push metrics to Prometheus Gateway at {MonitorConfig.PROMETHEUS_GATEWAY_URL}: {e}")

    def _trigger_retraining(self) -> None:
        raise NotImplementedError("Retraining logic is not implemented yet. This should be handled by the orchestrator.")

    def reset_current_data(self) -> None:
        self.prediction_df = pd.DataFrame()
        logger.info("Current prediction data has been reset.")

    def __call__(self, features: pd.DataFrame, prediction: float | int) -> None:
        self.log_prediction(features, prediction)
        logger.info("Prediction logged. Consider calling .evaluate_drift() periodically.")


ml_monitoring = Monitor()
