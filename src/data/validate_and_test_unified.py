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
import argparse

def load_data(file_path):
    """Naloži CSV podatke v pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["timestamp"])
    else:
        print(f"⚠️ Missing file: {file_path}")
        return None

def validate_data(data, suite_name, station_name):
    """Validacija podatkov s Great Expectations."""
    print(f"🔹 Running validation for station: {station_name}")

    context = DataContext()
    
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

    if not results.success:
        print(f"❌ Validation failed for {station_name}! Aborting pipeline...")
        exit(1)
    else:
        print(f"✅ Validation successful for {station_name}!")

def test_data_drift(reference_data, current_data, station_name):
    """Izvede Evidently test za odkrivanje data drift-a."""
    print(f"🔹 Running data drift test for station: {station_name}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_results = report.as_dict()
    if drift_results["metrics"][0]["result"]["dataset_drift"]:
        print(f"❌ Data drift detected for station {station_name}! Aborting pipeline...")
        exit(1)
    else:
        print(f"✅ No data drift detected for station {station_name}. Proceeding...")

def kolmogorov_smirnov_test(reference_data, current_data, station_name):
    """Kolmogorov-Smirnov test za odkrivanje sprememb v distribuciji podatkov."""
    print(f"🔹 Running Kolmogorov-Smirnov test for station: {station_name}")

    numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        ks_stat, ks_p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())

        if ks_p_value < 0.05:  # ✅ Popravljeno (prej: p_value)
            print(f"❌ KS test failed for column: {col} (p-value={ks_p_value:.5f}) in station {station_name}")
            exit(1)
        else:
            print(f"✅ KS test passed for column: {col} (p-value={ks_p_value:.5f}) in station {station_name}")

def main():
    parser = argparse.ArgumentParser(description="Validate and test data drift.")
    parser.add_argument("--reference", required=True, help="Path to reference dataset.")
    parser.add_argument("--current", required=True, help="Path to current dataset.")
    parser.add_argument("--suite_name", required=True, help="Name for validation suite.")

    args = parser.parse_args()

    # Če je argument mapa, iteriraj čez vse postaje
    if os.path.isdir(args.reference) and os.path.isdir(args.current):
        stations = [f.replace(".csv", "") for f in os.listdir(args.reference) if f.endswith(".csv")]

        for station in stations:
            reference_path = os.path.join(args.reference, f"{station}.csv")
            current_path = os.path.join(args.current, f"{station}.csv")

            reference_data = load_data(reference_path)
            current_data = load_data(current_path)

            if reference_data is None or current_data is None:
                print(f"⚠️ Missing reference or current data for station {station}. Skipping...")
                continue

            validate_data(current_data, args.suite_name, station)
            test_data_drift(reference_data, current_data, station)
            kolmogorov_smirnov_test(reference_data, current_data, station)

            # Posodobi referenčne podatke
            print(f"📌 Updating reference dataset for station {station}...")
            shutil.copy(current_path, reference_path)

        print("🚀 Validation and testing complete for all stations!")

    else:
        # Obdelava za združene podatke (merged dataset)
        reference_data = load_data(args.reference)
        current_data = load_data(args.current)

        if reference_data is None or current_data is None:
            print("⚠️ Missing reference or current data! Aborting validation.")
            exit(1)

        validate_data(current_data, args.suite_name, "merged_dataset")
        test_data_drift(reference_data, current_data, "merged_dataset")
        kolmogorov_smirnov_test(reference_data, current_data, "merged_dataset")

        # Posodobi referenčne podatke
        print(f"📌 Updating reference dataset for merged dataset...")
        shutil.copy(args.current, args.reference)

        print("🚀 Validation and testing complete for merged dataset!")

if __name__ == "__main__":
    main()