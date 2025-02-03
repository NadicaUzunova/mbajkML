import pytest
import os
from pymongo import MongoClient
from src.serve.app import app as flask_app  # Flask aplikacija

# 📌 MongoDB konfiguracija
client = MongoClient("mongodb+srv://nadicauzunova:7H8mP7RhyTaYlpy7@mbajkml.q7lre.mongodb.net/?retryWrites=true&w=majority&appName=mbajkML")
db = client["mbajkML"]
collection = db["predictions"]

# 📌 Testni odjemalec Flask
@pytest.fixture
def client():
    """Nastavi Flask testni odjemalec."""
    with flask_app.test_client() as client:
        yield client

# 📌 Testiranje /live-data endpointa
def test_live_data_valid_location(client):
    """Testira /live-data endpoint z veljavno lokacijo."""
    response = client.post('/live-data', json={"location": "DVORANA TABOR"})
    assert response.status_code == 200
    data = response.get_json()
    assert "available_bike_stands" in data, "Manjka ključ 'available_bike_stands'"
    assert "available_bikes" in data, "Manjka ključ 'available_bikes'"
    assert "timestamp" in data, "Manjka ključ 'timestamp'"

def test_live_data_invalid_location(client):
    """Testira /live-data endpoint z neveljavno lokacijo."""
    response = client.post('/live-data', json={"location": "NEVELJAVNA LOKACIJA"})
    assert response.status_code == 404
    assert "error" in response.get_json()

# 📌 Testiranje napovedi na /mbajk/predict
def test_predict_valid_data(client):
    """Testira /mbajk/predict endpoint z veljavnimi podatki."""
    valid_data = {
        "location": "DVORANA TABOR",
        "data": [["2024-12-15 19:00:00", "46.549946", "15.635611", "5", "15", "8.0", "60", "6.0", "7.5", "0.0"]] * 12
    }

    response = client.post('/mbajk/predict', json=valid_data)
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data, "Manjka ključ 'predictions'"
    assert len(data["predictions"]) == 7, "Napovedi niso dolžine 7"

# 📌 Testiranje, ali so napovedi shranjene v MongoDB
def test_predictions_stored_in_mongo():
    """Preveri, ali so napovedi shranjene v MongoDB."""
    client = MongoClient("mongodb+srv://nadicauzunova:7H8mP7RhyTaYlpy7@mbajkml.q7lre.mongodb.net/?retryWrites=true&w=majority&appName=mbajkML")
    db = client["mbajkML"]
    collection = db["predictions"]

    # Poiščemo zadnji zapis
    last_prediction = collection.find_one(sort=[("_id", -1)])

    assert last_prediction is not None, "Ni shranjenih napovedi v MongoDB"
    assert "predicted_value" in last_prediction, "Manjka ključ 'predicted_value' v MongoDB zapisu"
    assert "features" in last_prediction, "Manjka ključ 'features' v MongoDB zapisu"

# 📌 Testiranje nepravilnih vhodnih podatkov
def test_predict_invalid_data(client):
    """Testira /mbajk/predict endpoint z neveljavnimi podatki."""
    invalid_data = {
        "location": "DVORANA TABOR",
        "data": [["2024-12-15 19:00:00", "46.549946", "15.635611", "5", "15", "8.0"]]  # Premalo značilk
    }

    response = client.post('/mbajk/predict', json=invalid_data)
    assert response.status_code == 400
    assert "error" in response.get_json()

# 📌 Testiranje /data endpointa
def test_data_endpoint_valid(client):
    """Testira /data endpoint z veljavno lokacijo."""
    response = client.post('/data', json={"location": "DVORANA TABOR"})
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data, "Manjka ključ 'data'"
    assert len(data["data"]) == 12, "Vrnjeno napačno število vrstic"

def test_data_endpoint_invalid(client):
    """Testira /data endpoint z neobstoječo lokacijo."""
    response = client.post('/data', json={"location": "NEVELJAVNA"})
    assert response.status_code == 404
    assert "error" in response.get_json()