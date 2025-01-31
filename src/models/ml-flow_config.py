import dagshub
import mlflow

dagshub.init(repo_owner='NadicaUzunova', repo_name='mbajkML', mlflow=True)


with mlflow.start_run():
  # Your training code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')