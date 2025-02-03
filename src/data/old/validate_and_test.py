import os
import json
import shutil
import pandas as pd
from scipy.stats import ks_2samp
from great_expectations.data_context import DataContext
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import PandasDataset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Paths
REFERENCE_DATA_DIR = "data/processed/reference_data"
CURRENT_DATA_DIR = "data/processed/current_data"
REPORTS_DIR = "reports"

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_station_data(station_name, base_path):
    """Load a specific station's CSV file into a DataFrame."""
    file_path = os.path.join(base_path, f"{station_name}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["station_name"] = station_name  # Track which station the data belongs to
        return df
    else:
        return None

station_names = [f.replace(".csv", "") for f in os.listdir(REFERENCE_DATA_DIR) if f.endswith(".csv")]

def validate_data(data, station_name):
    """Validate data for a specific station using Great Expectations."""
    print(f"🔹 Running validation for station: {station_name}")

    context = DataContext()
    suite_name = f"mbajk_suite_{station_name}"

    try:
        suite = context.get_expectation_suite(suite_name)
    except:
        suite = ExpectationSuite(suite_name)
        context.add_expectation_suite(expectation_suite=suite)

    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "timestamp"}))
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "available_bikes"}))
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_values_to_be_between", kwargs={"column": "available_bikes", "min_value": 0, "max_value": 50}))

    dataset = PandasDataset(data)
    results = dataset.validate(expectation_suite=suite, only_return_failures=False)
    results_serialized = results.to_json_dict()

    validation_report_path = os.path.join(REPORTS_DIR, f"validation_report_{station_name}.json")
    with open(validation_report_path, "w") as f:
        json.dump(results_serialized, f, indent=4)

    if not results.success:
        print(f"❌ Validation failed for station {station_name}!")
        exit(1)
    else:
        print(f"✅ Validation successful for station {station_name}!")

def test_data_drift(reference_data, current_data, station_name):
    """Generate Evidently data drift report for a specific station."""
    print(f"🔹 Running data drift test for station: {station_name}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_report_path = os.path.join(REPORTS_DIR, f"data_drift_report_{station_name}.html")
    report.save_html(drift_report_path)

    drift_results = report.as_dict()
    if drift_results["metrics"][0]["result"]["dataset_drift"]:
        print(f"❌ Data drift detected for station {station_name}! Aborting pipeline...")
        exit(1)
    else:
        print(f"✅ No data drift detected for station {station_name}. Proceeding...")

def kolmogorov_smirnov_test(reference_data, current_data, station_name):
    """Izvede Kolmogorov-Smirnov test za vse numerične stolpce"""
    print(f"🔹 Running Kolmogorov-Smirnov test for station: {station_name}")

    numeric_columns = reference_data.select_dtypes(include=['number']).columns

    for column in numeric_columns:
        stat, p_value = ks_2samp(reference_data[column].dropna(), current_data[column].dropna())

        print(f"📊 KS test for {column}: p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(f"❌ Kolmogorov-Smirnov test FAILED for {column}! Possible data drift detected.")
            exit(1)

    print(f"✅ Kolmogorov-Smirnov test PASSED for station {station_name}.")

def validate_and_test_station(station_name):
    """Validates and tests data for a specific station."""
    print(f"🚦 Processing station: {station_name}")

    reference_data = load_station_data(station_name, REFERENCE_DATA_DIR)
    current_data = load_station_data(station_name, CURRENT_DATA_DIR)

    if reference_data is None or current_data is None:
        print(f"⚠️ Missing data for station {station_name}, skipping...")
        return

    validate_data(current_data, station_name)
    test_data_drift(reference_data, current_data, station_name)
    kolmogorov_smirnov_test(reference_data, current_data, station_name)

    print(f"📌 Updating reference data for station {station_name}...")
    shutil.copy(
        os.path.join(CURRENT_DATA_DIR, f"{station_name}.csv"),
        os.path.join(REFERENCE_DATA_DIR, f"{station_name}.csv"),
    )

for station_name in station_names:
    validate_and_test_station(station_name)

print("🚀 Validation and testing complete for all stations!")