import os
import json
import shutil
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from great_expectations.data_context import DataContext
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import PandasDataset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Paths
REFERENCE_DATA_PATH = "data/processed/merged/reference_data.csv"
CURRENT_DATA_PATH = "data/processed/merged/current_data.csv"
REPORTS_DIR = "reports"

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"⚠️ Missing file: {file_path}")
        return None

def validate_data(data):
    """Validate merged data using Great Expectations."""
    print(f"🔹 Running validation for merged dataset")

    context = DataContext()
    suite_name = "mbajk_merged_suite"

    # Create Expectation Suite if it does not exist
    try:
        suite = context.get_expectation_suite(suite_name)
    except:
        suite = ExpectationSuite(suite_name)
        context.add_expectation_suite(expectation_suite=suite)

    # Define expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "timestamp"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "available_bikes"}
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "available_bikes", "min_value": 0, "max_value": 50}
        )
    )

    # Create a PandasDataset
    dataset = PandasDataset(data)

    # Validate data
    results = dataset.validate(expectation_suite=suite, only_return_failures=False)

    # Convert results to JSON serializable format
    results_serialized = results.to_json_dict()

    # Save validation report
    validation_report_path = os.path.join(REPORTS_DIR, "validation_report_merged.json")
    with open(validation_report_path, "w") as f:
        json.dump(results_serialized, f, indent=4)

    if not results.success:
        print(f"❌ Validation failed for merged dataset! Aborting pipeline...")
        exit(1)
    else:
        print(f"✅ Validation successful for merged dataset!")

def test_data_drift(reference_data, current_data):
    """Generate Evidently data drift report and perform KS test."""
    print(f"🔹 Running data drift test for merged dataset")

    # Run Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save drift report
    drift_report_path = os.path.join(REPORTS_DIR, "data_drift_report_merged.html")
    report.save_html(drift_report_path)

    drift_results = report.as_dict()
    if drift_results["metrics"][0]["result"]["dataset_drift"]:
        print(f"❌ Data drift detected for merged dataset! Aborting pipeline...")
        exit(1)
    else:
        print(f"✅ No data drift detected for merged dataset. Proceeding...")

    # Run Kolmogorov-Smirnov test for each numeric column
    print(f"🔹 Running Kolmogorov-Smirnov test for feature distributions")
    numeric_columns = reference_data.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        ks_stat, ks_p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
        
        if ks_p_value < 0.05:
            print(f"❌ Kolmogorov-Smirnov test failed for column: {col} (p-value={ks_p_value:.5f})")
            exit(1)
        else:
            print(f"✅ KS test passed for column: {col} (p-value={ks_p_value:.5f})")

def main():
    """Validates and tests the merged dataset."""
    reference_data = load_data(REFERENCE_DATA_PATH)
    current_data = load_data(CURRENT_DATA_PATH)

    if reference_data is None or current_data is None:
        print("⚠️ Missing reference or current data! Aborting validation.")
        exit(1)

    validate_data(current_data)
    test_data_drift(reference_data, current_data)

    # Update reference data with new current data
    print(f"📌 Updating reference dataset for merged data...")
    shutil.copy(CURRENT_DATA_PATH, REFERENCE_DATA_PATH)

    print("🚀 Validation and testing complete for merged dataset!")

if __name__ == "__main__":
    main()