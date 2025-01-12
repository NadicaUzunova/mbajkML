import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import mlflow
import mlflow.tensorflow

def create_model(input_shape):
    """Create and return an LSTM model."""
    model = Sequential([
        LSTM(units=32, return_sequences=True, input_shape=input_shape),
        LSTM(units=32),
        Dense(16, activation='relu'),
        Dense(7)  # Predict 7-hour output
    ])
    return model

def create_multi_dataset(dataset, look_back=12):
    """Prepare input-output pairs for time-series data with 7-hour predictions."""
    X, y = [], []
    for i in range(look_back, len(dataset) - 6):
        X.append(dataset[i - look_back:i, 1:])  # Exclude the target column from input
        y.append(dataset[i:i + 7, 0])  # Predict 7 values for the target column
    return np.array(X), np.array(y)

def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing special Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

# Configuration
window_size = 12
train_dir = './data/processed/train'
test_dir = './data/processed/test'
output_dir = './models/final'
os.makedirs(output_dir, exist_ok=True)

csv_files = [file for file in os.listdir(train_dir) if file.endswith('.csv')]
test_csv_files = [file for file in os.listdir(test_dir) if file.endswith('.csv')]

# Combine train and test datasets
combined_dfs = []
for csv_file in csv_files + test_csv_files:
    file_path = os.path.join(train_dir if csv_file in csv_files else test_dir, csv_file)
    combined_dfs.append(pd.read_csv(file_path))

# Concatenate all datasets
complete_df = pd.concat(combined_dfs, ignore_index=True)

required_columns = [
    'available_bike_stands', 'position_lat', 'position_lng', 'temperature',
    'humidity', 'dew_point', 'apparent_temperature', 'precipitation'
]

if not all(col in complete_df.columns for col in required_columns):
    raise ValueError("Missing required columns in the dataset.")

df = complete_df[required_columns]
data = df.to_numpy()

if len(data) < window_size + 6:
    raise ValueError(f"Not enough rows in the combined dataset (minimum required: {window_size + 6}).")

X, y = create_multi_dataset(data, window_size)

if X.size == 0 or y.size == 0:
    raise ValueError("Failed to create valid datasets.")

input_shape = (X.shape[1], X.shape[2])

# Set MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/<USERNAME>/<REPO_NAME>.mlflow')

# Start MLflow run
with mlflow.start_run():
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Log parameters
    mlflow.log_param('window_size', window_size)
    mlflow.log_param('input_shape', input_shape)

    # Train the model
    history = model.fit(X, y, validation_split=0.2, epochs=10, verbose=1)

    # Save the model
    model_name = 'final_model.keras'
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)

    # Log the model in MLflow
    mlflow.tensorflow.log_model(model, artifact_path="models/final_model")

    # Log training metrics
    mlflow.log_metric('loss', np.mean(history.history['loss']))
    mlflow.log_metric('val_loss', np.mean(history.history['val_loss']))

    print(f"Final model and metrics have been logged to MLflow.")