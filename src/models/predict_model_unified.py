import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from dotenv import load_dotenv
from datetime import datetime
from pymongo import MongoClient

# Nalaganje konfiguracije
CONFIG_PATH = "src/utils/config.json"
with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

# Nalaganje okolijskih spremenljivk
load_dotenv()

# Nastavitev MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Vzpostavi povezavo z MongoDB
client = MongoClient("mongodb+srv://nadicauzunova:7H8mP7RhyTaYlpy7@mbajkml.q7lre.mongodb.net/?retryWrites=true&w=majority&appName=mbajkML")
db = client["mbajkML"]
collection = db["predictions"]

def save_prediction_to_mongo(input_data, predicted_values, actual_values, model_name):
    """Shrani vhodne podatke, napovedi, prave vrednosti in čas v MongoDB."""
    timestamp = datetime.now().isoformat()

    # Pridobi dejanska imena stolpcev
    feature_names = config["features"]
    
    # Oblikovanje podatkov za MongoDB
    documents = []
    for i in range(len(predicted_values)):
        doc = {
            "timestamp": timestamp,
            "model_name": model_name,
            "features": {feature_names[j]: float(input_data[i, j]) for j in range(len(feature_names))},
            "predicted_value": float(predicted_values[i]),
            "actual_value": float(actual_values[i]) if actual_values is not None else None
        }
        documents.append(doc)

    # Shrani v MongoDB
    collection.insert_many(documents)
    print(f"✅ Napovedi in dejanske vrednosti shranjene v MongoDB.")

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
    """Izvede napovedi s produkcijskim modelom in jih shrani v MongoDB."""
    model = load_production_model(model_name)
    if model is None:
        return

    # Naloži podatke
    df = pd.read_csv(input_data_path)

    features = config["features"]
    target = config.get("target", None)  # Prava vrednost (če obstaja)

    if not all(f in df.columns for f in features):
        print(f"❌ Manjkajoče značilke v {input_data_path}")
        return

    X = df[features].values
    actual_values = df[target].values if target and target in df.columns else None

    # Napovedi
    predictions = model.predict(X)

    # Shrani rezultate v MongoDB
    save_prediction_to_mongo(X, predictions, actual_values, model_name)

    print(f"✅ Napovedi in dejanske vrednosti shranjene v MongoDB.")

def main():
    parser = argparse.ArgumentParser(description="Predict using trained models.")
    parser.add_argument("--model_type", choices=["station", "merged"], required=True, help="Choose station or merged model.")
    parser.add_argument("--input_data", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save predictions.")  # Ni več nujno

    args = parser.parse_args()

    if args.model_type == "station":
        station_name = os.path.basename(args.input_data).replace(".csv", "")
        predict(station_name, args.input_data, args.output)
    else:
        predict("merged_model", args.input_data, args.output)

if __name__ == "__main__":
    main()