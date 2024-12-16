import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
import os
import csv
import datetime
from urllib.request import urlopen
import json
from flask_cors import CORS
import requests
import unicodedata



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

window_size = 12

models = {}
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'combined')

# Load all models into memory with normalized filenames
model_files = [file for file in os.listdir(directory) if file.endswith('.keras')]

for model_file in model_files:
    models[model_file] = load_model(os.path.join(directory, model_file))


def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)


def fill_nan_data(df: pd.DataFrame):
    """Fills missing values with column means."""
    for col in df.columns:
        df[col] = df[col].fillna(value=df[col].mean())
    return df


# Endpoint to fetch live bike data
BIKE_API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

@app.route('/live-data', methods=['POST'])
def fetch_live_data():
    """Fetches live data for a specific bike stand location."""
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

@app.route('/mbajk/predict', methods=['POST'])
def process_time_series():
    """Processes time series data for prediction."""
    try:
        request_body = request.get_json()

        # Extract data and location
        data = request_body['data']
        location = request_body['location']

        # Normalize the location to match the model name
        model_name = f"{normalize_string(location)}.keras"
        print(f"Looking for model: {model_name} in directory: {directory}")

        # Find matching model
        if model_name not in models:
            print(f"Model '{model_name}' not found in loaded models. Existing models: {list(models.keys())}")
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

        # Handle concatenated timestamp issue
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Parse timestamps
            except Exception as e:
                print(f"Timestamp parsing error: {e}")
                df['timestamp'] = None

        # Convert numeric columns to appropriate types
        numeric_columns = [
            'position_lat', 'position_lng', 'available_bike_stands',
            'available_bikes', 'temperature', 'humidity', 'dew_point',
            'apparent_temperature', 'precipitation'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Fill missing values and select relevant features
        df = fill_nan_data(df)

        # Select only the 7 features used during training
        X_features = df[[
            'position_lat', 'position_lng', 'available_bikes',  # Exclude 'available_bike_stands'
            'temperature', 'humidity', 'dew_point',
            'apparent_temperature'
        ]]

        # Ensure enough rows for prediction
        if X_features.shape[0] < window_size:
            print(f"Not enough data rows. Expected at least {window_size}, got {X_features.shape[0]}")
            return jsonify({"error": f"Not enough data rows for prediction. Expected at least {window_size}, got {X_features.shape[0]}."}), 400

        # Convert DataFrame to numpy array and reshape for prediction
        multi_data_np = X_features.to_numpy()
        print(f"Data shape before reshaping: {multi_data_np.shape}")

        X = multi_data_np[-window_size:].reshape((1, window_size, multi_data_np.shape[1]))

        # Predict using the model
        model = models[model_name]
        predictions = model.predict(X).flatten()  # Flatten predictions for easy processing

        # Round predictions to full numbers
        rounded_predictions = [round(pred) for pred in predictions]

        return jsonify({"predictions": rounded_predictions}), 200
    except Exception as e:
        print(f"Error processing time series: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)