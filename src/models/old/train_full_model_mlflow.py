import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Naloži okoljske spremenljivke
load_dotenv()

# Paths
train_data_path = "./data/processed/train_test_merged/train.csv"
output_dir = "./models/models_mlflow_merged"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow_client = MlflowClient()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Load train data
df = pd.read_csv(train_data_path)

# Required columns
features = ['available_bike_stands', 'position_lat', 'position_lng', 'temperature',
            'humidity', 'dew_point', 'apparent_temperature', 'precipitation']
target = 'available_bikes'

X = df[features].values
y = df[target].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline za obdelavo podatkov
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# MLP regresor
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

# Optimizacija hiperparametrov
search = GridSearchCV(pipe, parameter_space, cv=3, verbose=2, n_jobs=-1)

with mlflow.start_run(run_name="Train_Merged_Model") as run:
    search.fit(X_train, y_train)
    best_params = search.best_params_

    # Končno učenje modela
    final_model = Pipeline([
        ("preprocess", preprocessor),
        ("MLPR", MLPRegressor(hidden_layer_sizes=best_params["MLPR__hidden_layer_sizes"], 
                              learning_rate_init=best_params["MLPR__learning_rate_init"], 
                              max_iter=1000, random_state=42))
    ])
    final_model.fit(X_train, y_train)

    # Logiranje v MLflow
    model_name = "merged_model"
    mlflow.sklearn.log_model(final_model, model_name)

    print(f"✅ Merged model trained and saved to MLflow.")

    # **Preveri, ali model že obstaja v Model Registry**
    registered_models = [m.name for m in mlflow_client.search_registered_models()]

    if model_name not in registered_models:
        print(f"📌 Model '{model_name}' še ne obstaja v Model Registry. Registriram...")
        mlflow_client.create_registered_model(model_name)

    # **Pravilno ustvarjanje nove verzije modela**
    model_version = mlflow_client.create_model_version(
        name=model_name,
        source=mlflow.get_artifact_uri(model_name),
        run_id=run.info.run_id
    )
    
    # **Preveri in arhiviraj stare modele**
    current_production_models = mlflow_client.get_latest_versions(model_name, stages=["Production"])

    if current_production_models:
        for prod_model in current_production_models:
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=prod_model.version,
                stage="Archived"
            )
            print(f"📌 Stari model {prod_model.version} premaknjen v 'Archived'.")

    # **Postavi nov model v "Production"**
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )

    print(f"🚀 Model '{model_name}' verzije {model_version.version} je zdaj v 'Production'!")