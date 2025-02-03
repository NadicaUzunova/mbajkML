import os
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from dotenv import load_dotenv
import argparse
import json

# Nastavitev MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow_client = MlflowClient()

# Poti do testnih podatkov
CONFIG_PATH = "src/utils/config.json"

with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

def get_latest_model(model_name):
    """
    Pridobi zadnjo verzijo modela iz MLflow Model Registry, ki je v fazi 'None'.
    """
    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=["None"])
        if versions:
            return versions[0].version
    except Exception as e:
        print(f"⚠️ Napaka pri pridobivanju zadnje verzije modela: {e}")
        return None
    return None

def get_production_model(model_name):
    """
    Pridobi trenutno produkcijsko verzijo modela.
    """
    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].version
    except Exception as e:
        print(f"⚠️ Napaka pri pridobivanju produkcijskega modela: {e}")
        return None
    return None

def evaluate_model(model_uri, test_data):
    """
    Izvede evalvacijo modela na testnih podatkih in vrne metrike.
    """
    try:
        model = mlflow.sklearn.load_model(model_uri)  # Naloži model iz MLflow
    except Exception:
        print("❌ Napaka pri nalaganju modela!")
        return None, None, None
    
    features = config["features"]
    target = config["target"]
    
    X_test = test_data[features].values
    y_test = test_data[target].values

    predictions = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    
    return mae, mse, evs

def main():
    parser = argparse.ArgumentParser(description="Evaluate and register models dynamically.")
    parser.add_argument("--model_type", choices=["station", "merged"], required=True, help="Model type (station or merged).")
    args = parser.parse_args()

    model_name = "merged_model" if args.model_type == "merged" else "station_model"

    latest_version = get_latest_model(model_name)
    if not latest_version:
        print("❌ Ni nove verzije modela za evalvacijo.")
        return

    latest_model_uri = f"models:/{model_name}/{latest_version}"

    # Naloži testne podatke
    test_data_path = config["merged_test_data"] if args.model_type == "merged" else config["station_test_data"]
    test_data = pd.read_csv(test_data_path)

    # Evaluacija novega modela
    latest_mae, latest_mse, latest_evs = evaluate_model(latest_model_uri, test_data)
    if latest_mae is None:
        return

    print(f"🔎 Novi model (verzija {latest_version}) - MAE: {latest_mae:.3f}, MSE: {latest_mse:.3f}, EVS: {latest_evs:.3f}")

    # Pridobitev trenutnega produkcijskega modela
    production_version = get_production_model(model_name)
    
    if not production_version:
        print("✅ Ni obstoječega produkcijskega modela. Novi model bo označen kot 'Production'.")
        mlflow_client.transition_model_version_stage(model_name, latest_version, stage="Production")
        return

    production_model_uri = f"models:/{model_name}/{production_version}"
    production_mae, production_mse, production_evs = evaluate_model(production_model_uri, test_data)
    
    if production_mae is None:
        return

    print(f"🔎 Produkcijski model (verzija {production_version}) - MAE: {production_mae:.3f}, MSE: {production_mse:.3f}, EVS: {production_evs:.3f}")

    # Primerjava modelov
    if latest_mse < production_mse and latest_evs > production_evs:
        print("🚀 Novi model je boljši. Posodabljam produkcijski model.")
        mlflow_client.transition_model_version_stage(model_name, latest_version, stage="Production")
        mlflow_client.transition_model_version_stage(model_name, production_version, stage="Archived")
    else:
        print("📉 Novi model ni boljši. Ostanemo pri trenutnem produkcijskem modelu.")
        mlflow_client.transition_model_version_stage(model_name, latest_version, stage="Archived")

if __name__ == "__main__":
    main()