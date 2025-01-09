import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import mlflow

def normalize_string(input_str: str) -> str:
    """Normalize a string by replacing special Slovenian characters with ASCII equivalents."""
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

def create_multi_dataset(dataset, look_back=12):
    """Prepare input-output pairs for time-series data with 7-hour predictions."""
    X, y = [], []
    for i in range(look_back, len(dataset) - 6):
        X.append(dataset[i - look_back:i, 1:])
        y.append(dataset[i:i + 7, 0])
    return np.array(X), np.array(y)

# Configuration
window_size = 12
test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'test')
models_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
reports_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports', 'test')
os.makedirs(reports_directory, exist_ok=True)

csv_files = [file for file in os.listdir(test_directory) if file.endswith('.csv')]

mlflow.set_tracking_uri("https://dagshub.com/YOUR_USERNAME/YOUR_PROJECT_NAME.mlflow")

for csv_file in csv_files:
    test_file_path = os.path.join(test_directory, csv_file)

    complete_df = pd.read_csv(test_file_path)

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

    if not all(col in complete_df.columns for col in required_columns):
        print(f"Skipping {csv_file}: Missing required columns.")
        continue

    df = complete_df[required_columns]
    data = df.to_numpy()

    if len(data) < window_size + 6:
        print(f"Skipping {csv_file}: Not enough rows (minimum required: {window_size + 6}).")
        continue

    X, y = create_multi_dataset(data, window_size)

    if X.size == 0 or y.size == 0:
        print(f"Skipping {csv_file}: Failed to create valid datasets.")
        continue

    normalized_name = normalize_string(os.path.splitext(csv_file)[0])
    model_name = f"{normalized_name}.keras"
    model_path = os.path.join(models_directory, model_name)

    if not os.path.exists(model_path):
        print(f"Model {model_name} does not exist. Skipping.")
        continue

    model = load_model(model_path)

    predictions = model.predict(X)
    predictions = np.round(predictions)

    mae = mean_absolute_error(y.flatten(), predictions.flatten())
    mse = mean_squared_error(y.flatten(), predictions.flatten())
    evs = explained_variance_score(y.flatten(), predictions.flatten())

    metrics = pd.DataFrame({
        'mae': [mae],
        'mse': [mse],
        'explained_variance_score': [evs]
    })

    metrics_file = os.path.join(reports_directory, f"{normalized_name}_metrics.csv")
    metrics.to_csv(metrics_file, index=False)

    with mlflow.start_run():
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("explained_variance_score", evs)

    print(f"Metrics for {csv_file} saved to {metrics_file}.")