import dagshub
import mlflow
import os

dagshub.init(repo_owner='NadicaUzunova', repo_name='mbajkML', mlflow=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.login(
    username=os.getenv("MLFLOW_TRACKING_USERNAME"),
    password=os.getenv("MLFLOW_TRACKING_PASSWORD")
)

with mlflow.start_run():
  # Your training code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')