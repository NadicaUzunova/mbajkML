import pytest
import json
from src.serve.app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_predict_endpoint_with_valid_data(client):
    data = {"location": "DVORANA TABOR",
            "latitude": 46.54994670178013,
            "longitude": 15.635611927857214,
            "data": [["2024-03-20 19:00:00", "9.0", "67.0", "0.0", "0.0", "20.0", "18.0"],
                     ["2024-03-20 20:00:00", "7.5", "77.0", "0.0", "0.0", "20.0", "20.0"],
                     ["2024-03-20 21:00:00", "8.3", "74.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-20 22:00:00", "8.0", "74.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-20 23:00:00", "7.4", "76.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 00:00:00", "19.85689270315523", "72.56875492576894",
                      "20.103236357274433", "0.15397251817059357", "21.878957915831663",
                      "12.469072212579226"],
                     ["2024-03-21 01:00:00", "6.1", "83.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 02:00:00", "6.3", "80.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 03:00:00", "5.3", "85.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 04:00:00", "5.3", "82.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 05:00:00", "4.9", "85.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 06:00:00", "3.7", "95.0", "0.0", "0.0", "20.0", "20.0"]]}

    response = client.post('/mbajk/predict', json=data)
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'predictions' in response_data


def test_predict_endpoint_with_invalid_data(client):
    # length = 11
    data = {"location": "DVORANA TABOR",
            "latitude": 46.54994670178013,
            "longitude": 15.635611927857214,
            "data": [["2024-03-20 19:00:00", "9.0", "67.0", "0.0", "0.0", "20.0", "18.0"],
                     ["2024-03-20 20:00:00", "7.5", "77.0", "0.0", "0.0", "20.0", "20.0"],
                     ["2024-03-20 21:00:00", "8.3", "74.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-20 22:00:00", "8.0", "74.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-20 23:00:00", "7.4", "76.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 00:00:00", "19.85689270315523", "72.56875492576894",
                      "20.103236357274433", "0.15397251817059357", "21.878957915831663",
                      "12.469072212579226"],
                     ["2024-03-21 01:00:00", "6.1", "83.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 02:00:00", "6.3", "80.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 03:00:00", "5.3", "85.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 04:00:00", "5.3", "82.0", "0.0", "0.0", "20.0", "19.0"],
                     ["2024-03-21 05:00:00", "4.9", "85.0", "0.0", "0.0", "20.0", "19.0"]]}

    response = client.post('/mbajk/predict', json=data)
    assert response.status_code == 400
    response_data = response.get_json()
    assert 'Invalid array length. Expected length is 12.' in response_data


def test_predict_endpoint_with_invalid_request(client):
    response = client.post('/mbajk/predict')
    assert response.status_code == 400
    response_data = response.get_json()
    assert 'error' in response_data


def test_fetch_last_12_rows_file_not_found(client):
    response = client.post('/data', json={'location': 'non_existent_file'})
    assert response.status_code == 404
    assert b"File 'non_existent_file.csv' not found." in response.data


def test_fetch_last_12_rows_success(client):
    response = client.post('/data', json={'location': 'DVORANA TABOR'})
    assert response.status_code == 200
    assert len(response.json["data"]) == 12
