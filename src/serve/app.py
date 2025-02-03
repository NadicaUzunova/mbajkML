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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Nalaganje MLflow konfiguracije
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

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

# 🔹 Normalizacija imena postaje
def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

# 🔹 Pridobivanje trenutnih podatkov o postajah
@app.route('/live-data', methods=['POST'])
def fetch_live_data():
    """Fetches live bike data for a specific bike stand location."""
    try:
        request_body = request.get_json()
        location = request_body['location']

        # Fetch live bike data from external API
        response = requests.get(BIKE_API_URL)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch live data from the API"}), response.status_code

        bike_data = response.json()

        # Search for the specific location in the fetched data
        for station in bike_data:
            if station['name'].lower() == location.lower():
                return jsonify({
                    "available_bike_stands": station["available_bike_stands"],
                    "available_bikes": station["available_bikes"],
                    "timestamp": station["last_update"]
                }), 200

        return jsonify({"error": f"Location '{location}' not found in live data."}), 404
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return jsonify({"error": str(e)}), 500

# 🔹 Pridobivanje zadnjih 12 vrstic podatkov
@app.route('/data', methods=['POST'])
def fetch_last_12_rows():
    """Fetches the last 12 rows of data for a specific location."""
    try:
        request_body = request.get_json()
        location = request_body['location']
        location = location.replace('\u010c', 'Č').replace('\u017d', 'Ž')

        file_name = f"{location}.csv"
        file_path = os.path.join(data_directory, file_name)

        # Read last 12 rows from CSV
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_name}' not found.")

        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            last_12_rows = rows[-12:]

        # Log the last row for debugging
        print("Last row fetched from CSV:", last_12_rows[-1])

        return jsonify({"data": last_12_rows}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔹 Napovedovanje prostih koles na postaji
@app.route('/mbajk/predict', methods=['POST'])
def process_time_series():
    """Processes time series data for prediction."""
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

        # Zaokrožene napovedi
        rounded_predictions = [round(pred) for pred in predictions]

        return jsonify({"predictions": rounded_predictions}), 200
    except Exception as e:
        print(f"Error processing time series: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)