import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

def normalize_string(input_str: str) -> str:
    replacements = str.maketrans({'š': 's', 'č': 'c', 'ž': 'z', 'Š': 'S', 'Č': 'C', 'Ž': 'Z'})
    return input_str.translate(replacements)

def create_multi_dataset(dataset, look_back=12):
    X, y = [], []
    for i in range(look_back, len(dataset) - 6):
        X.append(dataset[i - look_back:i, 1:])
        y.append(dataset[i:i + 7, 0])
    return np.array(X), np.array(y)

# Configuration
window_size = 12
test_directory = "./data/processed/train_test_from_current/test"
models_directory = "./models/models_mlflow"  # NOVA LOKACIJA
reports_directory = "./reports/test_mlflow"  # NOVA LOKACIJA
os.makedirs(reports_directory, exist_ok=True)

csv_files = [file for file in os.listdir(test_directory) if file.endswith('.csv')]

for csv_file in csv_files:
    test_file_path = os.path.join(test_directory, csv_file)
    complete_df = pd.read_csv(test_file_path)

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

    # MLflow tracking
    with mlflow.start_run(run_name=f"Eval_{normalized_name}"):
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("Explained Variance", evs)

    print(f"✅ Metrics logged for {csv_file}: MAE={mae}, MSE={mse}, EVS={evs}")

print("🚀 Evaluation complete for all stations!")