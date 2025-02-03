import os
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

load_dotenv()

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

client = MlflowClient()
models = client.search_registered_models()

for model in models:
    print(f"Model: {model.name}")
    for version in model.latest_versions:
        print(f"  Version: {version.version}, Stage: {version.current_stage}")