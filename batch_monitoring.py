import pickle
import pandas as pd
import os
from datetime import datetime
import time 

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset, classification_performance
from evidently import ColumnMapping

from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import NoTargetPerformanceTestPreset
from evidently.test_preset import DataQualityTestPreset
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataDriftTestPreset
from evidently.test_preset import BinaryClassificationTestPreset

from datetime import datetime

def load_logged_data(log_file) ->pd.DataFrame:
    data = []
    with open(log_file, 'r') as file:
        for line in file:
            if "Model prediction" in line:
                prediction = int(line.split(':')[-1].strip())
                data.append({'case_status': prediction})
    return pd.DataFrame(data)

def create_dashboard(reference_data, current_data):

    column_mapping = ColumnMapping()

    column_mapping.target = 'case_status'
    column_mapping.prediction = 'prediction'
    reports = [
        Report(metrics=[DataDriftPreset()]),
        Report(metrics=[TargetDriftPreset()]),
        Report(metrics=[DataQualityPreset()])
    ]
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
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"Error generating report: {e}\n")
            print(f"Error generating report: {e}")
   
if __name__ == '__main__':
    reference_data = pd.read_parquet('data/feature_store/y_valid.parquet')
    current_data = load_logged_data('predictions.log')
    # Load sample data
    # reference_data = pd.DataFrame({'target': [1, 0, 1, 0, 1]})
    # current_data = pd.DataFrame({'prediction': [1, 0, 1, 0, 1]})
    create_dashboard(reference_data, current_data)