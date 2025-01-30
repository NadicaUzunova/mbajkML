import dagshub
dagshub.init(repo_owner='nadicauzunova',
             repo_name='mbajkML',
             mlflow=True)

import mlflow

mlflow.set_tracking_uri("https://dagshub.com/NadicaUzunova/mbajkML.mlflow")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)