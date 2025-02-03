import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.tracking import MlflowClient  # ✅ Dodano za model registry

# Naloži okoljske spremenljivke
load_dotenv()

# Paths
train_data_path = "./data/processed/train_test_merged/train.csv"
output_dir = "./models/models_mlflow_merged"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Load train data
df = pd.read_csv(train_data_path)

# Required columns
features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
            'humidity', 'dew_point', 'apparent_temperature', 'precipitation']
target = 'available_bikes'

df = df[["timestamp"] + features + [target]]

# Convert to numpy
X = df[features].values
y = df[target].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline za obdelavo podatkov
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Zapolnitev manjkajočih vrednosti
    ("scaler", StandardScaler())  # Standardizacija
])

# Definicija MLP regresorja
mlp = MLPRegressor(max_iter=500, random_state=42)

# Celoten `Pipeline`
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("MLPR", mlp)
])

# Definicija iskalnega prostora za optimizacijo hiperparametrov
parameter_space = {
    "MLPR__hidden_layer_sizes": [(32,), (16,)],
    "MLPR__learning_rate_init": [0.001, 0.01]
}

# Optimizacija hiperparametrov z GridSearchCV
search = GridSearchCV(pipe, parameter_space, cv=3, verbose=2, n_jobs=-1)

with mlflow.start_run():
    # Iskanje najboljših hiperparametrov
    search.fit(X_train, y_train)
    
    # Pridobitev najboljših hiperparametrov
    best_params = search.best_params_
    best_score = search.best_score_
    print(f"Best parameters found: {best_params} (CV Score={best_score:.3f})")

    # Končno učenje modela z najboljšimi parametri
    final_model = Pipeline([
        ("preprocess", preprocessor),
        ("MLPR", MLPRegressor(hidden_layer_sizes=best_params["MLPR__hidden_layer_sizes"], 
                              learning_rate_init=best_params["MLPR__learning_rate_init"], 
                              max_iter=1000, random_state=42))
    ])
    final_model.fit(X_train, y_train)

    # Evaluacija modela
    train_score = final_model.score(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    print(f"Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")

    # Logiranje v MLflow
    mlflow.log_param("best_hidden_layer_sizes", best_params["MLPR__hidden_layer_sizes"])
    mlflow.log_param("best_learning_rate", best_params["MLPR__learning_rate_init"])

    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)

    # Shranjevanje modela
    model_name = "Merged_Model"
    mlflow.sklearn.log_model(final_model, model_name)

    # ✅ Premik modela v MLflow Model Registry
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{model_name}"
    
    # Register the model
    mlflow.register_model(model_uri, model_name)

    # Premik v "Production"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    print(f"✅ Model '{model_name}' prestavljen v 'Production'!")