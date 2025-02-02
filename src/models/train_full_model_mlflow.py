import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tqdm import tqdm
from mlflow.tracking import MlflowClient
from requests.exceptions import RequestException
from dotenv import load_dotenv

load_dotenv()

# Paths
train_data_path = "./data/processed/train_test_merged/train.csv"
test_data_path = "./data/processed/train_test_merged/test.csv"
final_train_path = "./data/processed/train_test_merged/final_train.csv"
output_dir = "./models/models_mlflow_full"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load train & test data
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Merge train & test into final_train
full_df = pd.concat([train_df, test_df])
full_df.to_csv(final_train_path, index=False)
print(f"✅ Merged train & test datasets into {final_train_path}.")

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

client = MlflowClient(tracking_uri="https://dagshub.com/NadicaUzunova/mbajkML.mlflow")

# Retry connecting to MLflow (10x poskusov)
for attempt in range(10):
    try:
        mlflow.set_experiment("mBajk - Full LSTM Model Training")
        print("✅ MLflow connected successfully.")
        break
    except RequestException as e:
        print(f"⚠️ Napaka pri povezovanju z MLflow ({attempt+1}/10): {e}")
        time.sleep(5)  # Počakaj 5 sekund in poskusi ponovno
else:
    raise Exception("❌ Neuspešno povezovanje z MLflow po 10 poskusih.")

# Required columns
features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
            'humidity', 'dew_point', 'apparent_temperature', 'precipitation']
target = 'available_bikes'

full_df = full_df[["timestamp"] + features + [target]]

# Convert to numpy
data = full_df[features + [target]].values

# Split into input/output
X, y = [], []
look_back = 12  # 12-hour window

for i in range(look_back, len(data)):
    X.append(data[i - look_back:i, :-1])
    y.append(data[i, -1])

X, y = np.array(X), np.array(y)

# Model definition
input_shape = (X.shape[1], X.shape[2])
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=input_shape),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)  # Predicting available_bikes
])
model.compile(optimizer='adam', loss='mean_squared_error')

# MLflow tracking
with mlflow.start_run():
    epochs = 10
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        history = model.fit(X, y, validation_split=0.2, epochs=1, verbose=1)

        # Log parameters & metrics
        mlflow.log_param("window_size", look_back)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("train_loss", np.mean(history.history['loss']))
        mlflow.log_metric("val_loss", np.mean(history.history['val_loss']))

    # Save model
    model_path = os.path.join(output_dir, "full_lstm_model.keras")
    model.save(model_path)
    mlflow.tensorflow.log_model(model, artifact_path="models")

print(f"✅ Full model trained and saved to {output_dir}.")