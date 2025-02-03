import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from dotenv import load_dotenv

# Nalaganje konfiguracije
CONFIG_PATH = "src/utils/config.json"
with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

# Nastavitev MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

def load_production_model(model_name):
    """Naloži najnovejši 'Production' model iz MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()
    models = client.get_latest_versions(model_name, stages=["Production"])

    if not models:
        print(f"❌ Ni modelov v 'Production' za {model_name}")
        return None

    model_uri = f"models:/{model_name}/{models[0].version}"
    print(f"✅ Nalagam model {model_name} (verzija {models[0].version})...")
    return mlflow.sklearn.load_model(model_uri)

def predict(model_name, input_data_path, output_path):
    """Izvede napovedi s produkcijskim modelom."""
    model = load_production_model(model_name)
    if model is None:
        return

    # Naloži podatke
    df = pd.read_csv(input_data_path)

    features = config["features"]
    if not all(f in df.columns for f in features):
        print(f"❌ Manjkajoče značilke v {input_data_path}")
        return

    X = df[features].values

    # Napovedi
    df["predicted_available_bikes"] = model.predict(X)

    # Shrani rezultate
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Napovedi shranjene v {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict using trained models.")
    parser.add_argument("--model_type", choices=["station", "merged"], required=True, help="Choose station or merged model.")
    parser.add_argument("--input_data", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save predictions.")

    args = parser.parse_args()

    if args.model_type == "station":
        station_name = os.path.basename(args.input_data).replace(".csv", "")
        predict(station_name, args.input_data, args.output)
    else:
        predict("merged_model", args.input_data, args.output)

if __name__ == "__main__":
    main()