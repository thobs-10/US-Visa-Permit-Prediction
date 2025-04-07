import time
from datetime import datetime

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report


def load_logged_data(log_file) -> pd.DataFrame:
    data = []

    with open(log_file, mode="rb", encoding="utf-8") as file:
        for line in file:
            if "Model prediction" in line:
                prediction = int(line.split(":")[-1].strip())
                data.append({"case_status": prediction})
        file.close()
    return pd.DataFrame(data)


def create_dashboard(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    column_mapping = ColumnMapping()

    column_mapping.target = "case_status"
    column_mapping.prediction = "prediction"
    reports = [Report(metrics=[DataDriftPreset()]), Report(metrics=[TargetDriftPreset()]), Report(metrics=[DataQualityPreset()])]
    log_file = "dashboard_errors.log"

    for report in reports:
        try:
            if isinstance(report, Report) and report.metrics == [TargetDriftPreset()]:
                report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
                file_name = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                report.save_html(file_name)
                print(f"Dashboard saved as {file_name}")
            else:
                report.run(reference_data=reference_data, current_data=current_data)
                file_name = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                report.save_html(file_name)
                print(f"Dashboard saved as {file_name}")
            time.sleep(1)
        except FileNotFoundError as e:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Error generating report: {e}\n")
            print(f"Error generating report: {e}")
            f.close()


if __name__ == "__main__":
    reference_dataset = pd.read_parquet("data/feature_store/y_valid.parquet")
    current_dataset = load_logged_data("predictions.log")
    create_dashboard(reference_dataset, current_dataset)
