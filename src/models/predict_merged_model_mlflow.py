import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Paths
test_data_path = "./data/processed/train_test_merged/test.csv"
model_path = "./models/models_mlflow_merged/merged_lstm_model.keras"

# Load test data
df = pd.read_csv(test_data_path)

# Required columns
features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
            'humidity', 'dew_point', 'apparent_temperature', 'precipitation']
target = 'available_bikes'

df = df[["timestamp"] + features + [target]]

# Convert to numpy
data = df[features + [target]].values

# Split into input/output
X, y = [], []
look_back = 12  # 12-hour window

for i in range(look_back, len(data)):
    X.append(data[i - look_back:i, :-1])
    y.append(data[i, -1])

X, y = np.array(X), np.array(y)

# Load trained model
model = load_model(model_path)

# Predict
predictions = model.predict(X)
predictions = np.round(predictions)

# Metrics
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
evs = explained_variance_score(y, predictions)

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("Explained Variance", evs)

print(f"✅ Predictions completed. Metrics - MAE: {mae}, MSE: {mse}, EVS: {evs}")