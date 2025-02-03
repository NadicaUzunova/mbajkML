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
test_directory = "./data/processed/train_test_from_current/test"
model_registry_name = "mBajk-Prediction-Model"

# Pridobi seznam vseh modelov iz MLflow
models = client.search_registered_models()

for model in models:
    model_name = model.name  # Ime modela (postaje)

    try:
        # ✅ Poskusi pridobiti model iz "Production"
        model_version = client.get_latest_versions(model_name, stages=["Production"])[0].version
    except IndexError:
        print(f"❌ Ni produkcijskega modela za {model_name}. Preskakujem.")
        continue

    # Naloži model iz MLflow
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Poišči testni set za to postajo
    test_file_path = os.path.join(test_directory, f"{model_name}.csv")
    
    if not os.path.exists(test_file_path):
        print(f"⚠️ Testni set za {model_name} ne obstaja. Preskakujem.")
        continue

    df_test = pd.read_csv(test_file_path)

    # Definiraj vhodne značilke
    features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
                'humidity', 'dew_point', 'apparent_temperature', 'precipitation']
    
    if not all(col in df_test.columns for col in features):
        print(f"❌ Manjkajoče značilke v {model_name}. Preskakujem.")
        continue

    X_test = df_test[features].values

    # Napovedi
    predictions = model.predict(X_test)
    df_test["predicted_available_bikes"] = predictions

    # Shrani rezultate
    prediction_path = f"./data/predictions/{model_name}_predictions.csv"
    df_test.to_csv(prediction_path, index=False)

    print(f"✅ Napovedi shranjene za {model_name}: {prediction_path}")

print("🚀 Evaluacija končana za vse postaje!")