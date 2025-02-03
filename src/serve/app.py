import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, request, jsonify
import os
import csv
import datetime
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Nalaganje MLflow konfiguracije
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# MongoDB konfiguracija
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# Povezava z MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Konstante
window_size = 12
BIKE_API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'combined')

# 🔹 Dinamično pridobivanje modela iz MLflow Model Registry
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

# 🔹 Shrani napovedi v MongoDB
def save_prediction_to_mongo(input_data, predicted_values, model_name, actual_values=None):
    """Shrani vhodne podatke, napovedi in čas v MongoDB."""
    timestamp = datetime.datetime.now().isoformat()
    feature_names = ['position_lat', 'position_lng', 'available_bikes', 
                     'temperature', 'humidity', 'dew_point', 'apparent_temperature']
    
    # Oblikovanje podatkov za MongoDB
    documents = []
    for i in range(len(predicted_values)):
        doc = {
            "timestamp": timestamp,
            "model_name": model_name,
            "features": {feature_names[j]: float(input_data[i, j]) for j in range(len(feature_names))},
            "predicted_value": float(predicted_values[i])
        }
        if actual_values is not None:
            doc["actual_value"] = float(actual_values[i])  # Dodamo dejansko vrednost, če obstaja
        documents.append(doc)

    # Shrani v MongoDB
    collection.insert_many(documents)
    print(f"✅ Napovedi shranjene v MongoDB ({COLLECTION_NAME})")

# 🔹 Normalizacija imena postaje
def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

# 🔹 Napovedovanje prostih koles na postaji
@app.route('/mbajk/predict', methods=['POST'])
def process_time_series():
    """Processes time series data for prediction and saves it to MongoDB."""
    try:
        request_body = request.get_json()

        # Extract data and location
        data = request_body['data']
        location = request_body['location']

        # Pridobi model iz MLflow
        model = load_production_model(normalize_string(location))
        if model is None:
            return jsonify({"error": f"Model for location '{location}' not found."}), 404

        # Validate data length
        if len(data) != window_size:
            print(f"Invalid data length. Expected {window_size}, got {len(data)}")
            return jsonify({"error": f"Invalid array length. Expected {window_size} rows."}), 400

        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'position_lat', 'position_lng', 'available_bike_stands',
            'available_bikes', 'temperature', 'humidity', 'dew_point',
            'apparent_temperature', 'precipitation'
        ])

        # Pretvori timestamp v datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Pretvori numerične stolpce
        numeric_columns = [
            'position_lat', 'position_lng', 'available_bike_stands',
            'available_bikes', 'temperature', 'humidity', 'dew_point',
            'apparent_temperature', 'precipitation'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Izberi značilke za napovedovanje
        features = ['position_lat', 'position_lng', 'available_bikes', 
                    'temperature', 'humidity', 'dew_point', 'apparent_temperature']
        X_features = df[features]

        # Preveri, ali imamo dovolj podatkov
        if X_features.shape[0] < window_size:
            print(f"Not enough data rows. Expected at least {window_size}, got {X_features.shape[0]}")
            return jsonify({"error": f"Not enough data rows for prediction. Expected at least {window_size}, got {X_features.shape[0]}."}), 400

        # Pretvori podatke v numpy array
        multi_data_np = X_features.to_numpy()
        print(f"Data shape before reshaping: {multi_data_np.shape}")

        X = multi_data_np[-window_size:].reshape((1, window_size, multi_data_np.shape[1]))

        # Napoved
        predictions = model.predict(X).flatten()

        # Shrani napovedi v MongoDB (če obstaja 'available_bikes', se doda kot actual_value)
        actual_values = df['available_bikes'].values if 'available_bikes' in df.columns else None
        save_prediction_to_mongo(multi_data_np, predictions, normalize_string(location), actual_values)

        # Zaokrožene napovedi
        rounded_predictions = [round(pred) for pred in predictions]

        return jsonify({"predictions": rounded_predictions}), 200
    except Exception as e:
        print(f"Error processing time series: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)