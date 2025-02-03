import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse

# Nalaganje konfiguracije
CONFIG_PATH = "src/utils/config.json"

with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

# Nastavitev MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow_client = MlflowClient()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

def train_model(train_data_path, model_name):
    """Treniranje modela s podanimi podatki in registracija v MLflow."""
    print(f"🚀 Training model: {model_name}")

    # Nalaganje podatkov
    df = pd.read_csv(train_data_path)

    # Definirane značilke
    features = config["features"]
    target = config["target"]

    if not all(f in df.columns for f in features + [target]):
        print(f"❌ Error: Missing required columns in {train_data_path}")
        return

    X = df[features].values
    y = df[target].values

    # Delitev na train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predprocesiranje podatkov
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Model MLP
    mlp = MLPRegressor(max_iter=500, random_state=42)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("MLPR", mlp)
    ])

    # Optimizacija hiperparametrov
    param_grid = {
        "MLPR__hidden_layer_sizes": [(32,), (16,)],
        "MLPR__learning_rate_init": [0.001, 0.01]
    }

    search = GridSearchCV(pipe, param_grid, cv=3, verbose=2, n_jobs=-1)

    with mlflow.start_run(run_name=f"Train_{model_name}"):
        search.fit(X_train, y_train)
        best_params = search.best_params_

        final_model = Pipeline([
            ("preprocess", preprocessor),
            ("MLPR", MLPRegressor(
                hidden_layer_sizes=best_params["MLPR__hidden_layer_sizes"], 
                learning_rate_init=best_params["MLPR__learning_rate_init"], 
                max_iter=1000, random_state=42))
        ])
        final_model.fit(X_train, y_train)

        train_score = final_model.score(X_train, y_train)
        test_score = final_model.score(X_test, y_test)

        print(f"✅ Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")

        mlflow.log_param("best_hidden_layer_sizes", best_params["MLPR__hidden_layer_sizes"])
        mlflow.log_param("best_learning_rate", best_params["MLPR__learning_rate_init"])
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)

        model_uri = mlflow.sklearn.log_model(final_model, model_name)

        # ✅ Preveri, ali model že obstaja v MLflow Model Registry
        registered_models = [m.name for m in mlflow_client.search_registered_models()]
        if model_name not in registered_models:
            print(f"📌 Model '{model_name}' še ne obstaja v Model Registry. Registriram...")
            mlflow_client.create_registered_model(model_name)

        # ✅ Kreiraj novo verzijo modela (pusti v "None")
        model_version = mlflow_client.create_model_version(
            name=model_name,
            source=mlflow.get_artifact_uri(model_name),
            run_id=mlflow.active_run().info.run_id
        )

        print(f"📌 Model '{model_name}' verzije {model_version.version} je naučen in pripravljen za evalvacijo.")

def main():
    parser = argparse.ArgumentParser(description="Train models dynamically.")
    parser.add_argument("--model_type", choices=["station", "merged"], required=True, help="Type of model to train (station or merged).")
    args = parser.parse_args()

    if args.model_type == "station":
        # Trenira vsak model posamezne postaje
        for station in os.listdir(config["station_train_data"]):
            if station.endswith(".csv"):
                station_name = station.replace(".csv", "")
                train_model(os.path.join(config["station_train_data"], station), station_name)
    else:
        # Trenira združeni model
        train_model(config["merged_train_data"], "merged_model")

if __name__ == "__main__":
    main()