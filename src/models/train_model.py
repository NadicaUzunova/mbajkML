import os
import pandas as pd
import numpy as np
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
    for i in range(look_back, len(dataset) - 6):  # Ensure we have enough rows for 7-hour predictions
        X.append(dataset[i - look_back:i, 1:])  # Exclude the target column from input
        y.append(dataset[i:i + 7, 0])  # Predict 7 values for the target column
    return np.array(X), np.array(y)


def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing special Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)


# Configuration
window_size = 12
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'train')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
os.makedirs(output_dir, exist_ok=True)

csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    complete_df = pd.read_csv(file_path)

    # Check if required columns exist
    required_columns = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature', 
                        'humidity', 'dew_point', 'apparent_temperature', 'precipitation']

    if not all(col in complete_df.columns for col in required_columns):
        print(f"Skipping {csv_file}: Missing required columns.")
        continue

    # Select relevant features
    df = complete_df[required_columns]

    # Check if the dataset has enough rows
    if len(df) < window_size + 6:  # Account for 7-hour predictions
        print(f"Skipping {csv_file}: Not enough rows (minimum required: {window_size + 6}).")
        continue

    # Convert to numpy array
    data = df.to_numpy()

    # Prepare datasets
    X, y = create_multi_dataset(data, window_size)

    # Check if X and y are valid
    if X.size == 0 or y.size == 0:
        print(f"Skipping {csv_file}: Failed to create valid datasets.")
        continue

    # Define input shape
    input_shape = (X.shape[1], X.shape[2])

    # Create and compile the model
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X, y, validation_split=0.2, epochs=10, verbose=1)

    # Save the model with normalized filename
    model_name = normalize_string(os.path.splitext(csv_file)[0]) + '.keras'
    model.save(os.path.join(output_dir, model_name))

    # Save training metrics
    train_metrics = pd.DataFrame({
        'loss': [np.mean(history.history['loss'])],
        'val_loss': [np.mean(history.history['val_loss'])]
    })
    train_metrics.to_csv(os.path.join(output_dir, f"{normalize_string(os.path.splitext(csv_file)[0])}_train_metrics.csv"), index=False)