import os
import json
import mlflow
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Nalaganje okolijskih spremenljivk
load_dotenv()

# MLflow konfiguracija
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# MongoDB konfiguracija
client = MongoClient("mongodb+srv://nadicauzunova:7H8mP7RhyTaYlpy7@mbajkml.q7lre.mongodb.net/?retryWrites=true&w=majority&appName=mbajkML")
db = client["mbajkML"]
collection = db["predictions"]

def evaluate_model(model_name):
    """Pridobi napovedi iz MongoDB, izračuna metrike in jih shrani v MLflow."""
    print(f"🔍 Pridobivam podatke za model: {model_name}")

    # Pridobi podatke, kjer `actual_value` obstaja
    query = {
        "model_name": model_name,
        "actual_value": {"$ne": None}  # Izključi napovedi brez prave vrednosti
    }
    data = list(collection.find(query))

    if not data:
        print(f"❌ Ni podatkov za evalvacijo modela '{model_name}'.")
        return

    # Pretvori podatke v DataFrame
    df = pd.DataFrame(data)

    # Izračunaj metrike
    y_true = df["actual_value"].values
    y_pred = df["predicted_value"].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"📊 MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # Zapiši metrike v MLflow v ločen eksperiment "Model Evaluation"
    mlflow.set_experiment("Model Evaluation")

    with mlflow.start_run(run_name=f"Eval_{model_name}_{datetime.utcnow().date()}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)

    print(f"✅ Metrike shranjene v MLflow.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions and log metrics to MLflow.")
    parser.add_argument("--model_type", choices=["station", "merged"], required=True, help="Choose station or merged model.")
    args = parser.parse_args()

    if args.model_type == "station":
        print("🔍 Evaluiram modele za vse postaje...")
        station_models = collection.distinct("model_name", {"model_name": {"$ne": "merged_model"}})
        for station in station_models:
            evaluate_model(station)
    else:
        evaluate_model("merged_model")

if __name__ == "__main__":
    main()