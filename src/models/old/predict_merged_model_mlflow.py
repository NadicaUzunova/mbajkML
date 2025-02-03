import os
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Naloži okoljske spremenljivke
load_dotenv()

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

# Konfiguracija poti
test_data_path = "./data/processed/train_test_merged/test.csv"
model_registry_name = "mBajk-Merged-Prediction-Model"

try:
    # ✅ Poskusi pridobiti model iz "Production"
    model_version = client.get_latest_versions(model_registry_name, stages=["Production"])[0].version
except IndexError:
    print(f"❌ Ni produkcijskega modela za združene podatke. Preskakujem.")
    exit()

# Naloži model iz MLflow
model_uri = f"models:/{model_registry_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

# Naloži testne podatke
df_test = pd.read_csv(test_data_path)

# Definiraj značilke
features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
            'humidity', 'dew_point', 'apparent_temperature', 'precipitation']

if not all(col in df_test.columns for col in features):
    print(f"❌ Manjkajoče značilke v združenih podatkih. Preskakujem.")
    exit()

X_test = df_test[features].values

# Napovedi
predictions = model.predict(X_test)
df_test["predicted_available_bikes"] = predictions

# Shrani rezultate
prediction_path = "./data/predictions/predictions_merged.csv"
df_test.to_csv(prediction_path, index=False)

print(f"✅ Napovedi združenega modela shranjene v {prediction_path}!")