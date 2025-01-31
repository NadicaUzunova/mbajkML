import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import unicodedata

def create_model(input_shape):
    model = Sequential([
        LSTM(units=32, return_sequences=True, input_shape=input_shape),
        LSTM(units=32),
        Dense(16, activation='relu'),
        Dense(7)  # Output layer now predicts 7 values for the next 7 hours
    ])
    return model

def create_multi_dataset(dataset, look_back=12):
    X, y = [], []
    for i in range(look_back, len(dataset) - 6):  
        X.append(dataset[i - look_back:i, 1:])  
        y.append(dataset[i:i + 7, 0])  
    return np.array(X), np.array(y)

def normalize_string(input_str: str) -> str:
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

# Configuration
window_size = 12
directory = "./data/processed/train_test_from_current/train"
output_dir = "./models/models_mlflow"  # NOVA LOKACIJA
os.makedirs(output_dir, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/NadicaUzunova/mbajkML.mlflow")
mlflow.set_experiment("mBajk - LSTM Model Training")

csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    complete_df = pd.read_csv(file_path)

    required_columns = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature', 
                        'humidity', 'dew_point', 'apparent_temperature', 'precipitation']

    if not all(col in complete_df.columns for col in required_columns):
        print(f"Skipping {csv_file}: Missing required columns.")
        continue

    df = complete_df[required_columns]

    if len(df) < window_size + 6:
        print(f"Skipping {csv_file}: Not enough rows.")
        continue

    data = df.to_numpy()
    X, y = create_multi_dataset(data, window_size)

    if X.size == 0 or y.size == 0:
        print(f"Skipping {csv_file}: Invalid dataset.")
        continue

    input_shape = (X.shape[1], X.shape[2])
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # MLflow tracking
    with mlflow.start_run(run_name=normalize_string(csv_file)):
        history = model.fit(X, y, validation_split=0.2, epochs=10, verbose=1)

        # Log parameters
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", 10)

        # Log metrics
        mlflow.log_metric("train_loss", np.mean(history.history['loss']))
        mlflow.log_metric("val_loss", np.mean(history.history['val_loss']))

        # Save model
        model_name = normalize_string(os.path.splitext(csv_file)[0]) + ".keras"
        model_path = os.path.join(output_dir, model_name)
        model.save(model_path)
        mlflow.tensorflow.log_model(model, artifact_path="models")

    print(f"✅ Model saved: {model_name}")

print("🚀 Training complete for all stations!")