import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing special Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)


def create_multi_dataset(dataset, look_back=12):
    """Prepare input-output pairs for time-series data with 7-hour predictions."""
    X, y = [], []
    for i in range(look_back, len(dataset) - 6):  # Ensure enough rows for 7-hour predictions
        X.append(dataset[i - look_back:i, 1:])  # Exclude the target column from input
        y.append(dataset[i:i + 7, 0])  # Target is a sequence of 7 values
    return np.array(X), np.array(y)


# Configuration
window_size = 12
test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'test')
models_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
reports_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports', 'test')
os.makedirs(reports_directory, exist_ok=True)

csv_files = [file for file in os.listdir(test_directory) if file.endswith('.csv')]

for csv_file in csv_files:
    test_file_path = os.path.join(test_directory, csv_file)

    # Load test data
    complete_df = pd.read_csv(test_file_path)

    # Select relevant features
    required_columns = [
        'available_bike_stands',
        'position_lat',
        'position_lng',
        'temperature',
        'humidity',
        'dew_point',
        'apparent_temperature',
        'precipitation',
    ]

    # Check if all required columns exist
    if not all(col in complete_df.columns for col in required_columns):
        print(f"Skipping {csv_file}: Missing required columns.")
        continue

    df = complete_df[required_columns]

    # Convert to numpy array
    data = df.to_numpy()

    # Check if the dataset has enough rows
    if len(data) < window_size + 6:  # Account for 7-hour predictions
        print(f"Skipping {csv_file}: Not enough rows (minimum required: {window_size + 6}).")
        continue

    # Prepare datasets
    X, y = create_multi_dataset(data, window_size)

    # Check if X and y are valid
    if X.size == 0 or y.size == 0:
        print(f"Skipping {csv_file}: Failed to create valid datasets.")
        continue

    # Normalize the filename to match model names
    normalized_name = normalize_string(os.path.splitext(csv_file)[0])
    model_name = f"{normalized_name}.keras"
    model_path = os.path.join(models_directory, model_name)

    if not os.path.exists(model_path):
        print(f"Model {model_name} does not exist. Skipping.")
        continue

    # Load the corresponding model
    model = load_model(model_path)

    # Predict
    predictions = model.predict(X)  # Predictions are 7-hour sequences
    predictions = np.round(predictions)  # Round to full numbers

    # Evaluate predictions
    mae = mean_absolute_error(y.flatten(), predictions.flatten())
    mse = mean_squared_error(y.flatten(), predictions.flatten())
    evs = explained_variance_score(y.flatten(), predictions.flatten())

    # Save evaluation metrics
    metrics = pd.DataFrame({
        'mae': [mae],
        'mse': [mse],
        'explained_variance_score': [evs]
    })

    metrics_file = os.path.join(reports_directory, f"{normalized_name}_metrics.csv")
    metrics.to_csv(metrics_file, index=False)

    print(f"Metrics for {csv_file} saved to {metrics_file}.")