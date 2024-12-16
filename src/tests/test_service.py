import pytest
from src.serve.app import app as flask_app  # Import the Flask app instance

@pytest.fixture
def client():
    """Set up the Flask test client."""
    with flask_app.test_client() as client:
        yield client


def test_live_data_endpoint_with_valid_location(client):
    """Test the /live-data endpoint with a valid location."""
    valid_location = {"location": "DVORANA TABOR"}

    response = client.post('/live-data', json=valid_location)
    assert response.status_code == 200
    response_data = response.get_json()
    assert "available_bike_stands" in response_data
    assert "available_bikes" in response_data
    assert "timestamp" in response_data


def test_live_data_endpoint_with_invalid_location(client):
    """Test the /live-data endpoint with an invalid location."""
    invalid_location = {"location": "INVALID LOCATION"}

    response = client.post('/live-data', json=invalid_location)
    assert response.status_code == 404
    response_data = response.get_json()
    assert "error" in response_data
    assert "not found in live data" in response_data["error"]


def test_predict_endpoint_with_valid_data(client):
    """Test the /mbajk/predict endpoint with valid data."""
    valid_data = {
        "location": "DVORANA TABOR",
        "data": [
            ["2024-12-15 19:00:00", "46.549946", "15.635611", "5", "15", "8.0", "60", "6.0", "7.5", "0.0"],
            ["2024-12-15 20:00:00", "46.549946", "15.635611", "5", "14", "7.5", "65", "5.8", "7.2", "0.0"],
            ["2024-12-15 21:00:00", "46.549946", "15.635611", "4", "16", "7.0", "70", "5.5", "7.0", "0.0"],
            ["2024-12-15 22:00:00", "46.549946", "15.635611", "4", "17", "6.5", "75", "5.2", "6.8", "0.0"],
            ["2024-12-15 23:00:00", "46.549946", "15.635611", "4", "13", "6.0", "80", "4.9", "6.5", "0.0"],
            ["2024-12-16 00:00:00", "46.549946", "15.635611", "3", "10", "5.5", "85", "4.6", "6.2", "0.0"],
            ["2024-12-16 01:00:00", "46.549946", "15.635611", "3", "12", "5.0", "90", "4.3", "6.0", "0.0"],
            ["2024-12-16 02:00:00", "46.549946", "15.635611", "3", "14", "4.5", "92", "4.0", "5.8", "0.0"],
            ["2024-12-16 03:00:00", "46.549946", "15.635611", "3", "16", "4.0", "95", "3.7", "5.6", "0.0"],
            ["2024-12-16 04:00:00", "46.549946", "15.635611", "2", "15", "3.5", "98", "3.4", "5.4", "0.0"],
            ["2024-12-16 05:00:00", "46.549946", "15.635611", "2", "17", "3.0", "100", "3.1", "5.2", "0.0"],
            ["2024-12-16 06:00:00", "46.549946", "15.635611", "2", "20", "2.5", "102", "2.8", "5.0", "0.0"]
        ]
    }

    response = client.post('/mbajk/predict', json=valid_data)
    assert response.status_code == 200
    response_data = response.get_json()
    assert "predictions" in response_data

    # Check that the output has 7 predictions, as per the model design
    assert len(response_data["predictions"]) == 7


def test_predict_endpoint_with_invalid_data(client):
    """Test the /mbajk/predict endpoint with invalid data."""
    invalid_data = {
        "location": "DVORANA TABOR",
        "data": [
            ["2024-12-15 19:00:00", "46.549946", "15.635611", "5", "15", "8.0", "60", "6.0", "7.5", "0.0"]
        ]  # Too few rows
    }

    response = client.post('/mbajk/predict', json=invalid_data)
    assert response.status_code == 400
    response_data = response.get_json()
    assert "error" in response_data
    assert "Invalid array length" in response_data["error"]


def test_predict_endpoint_with_invalid_location(client):
    """Test the /mbajk/predict endpoint with a non-existent model."""
    invalid_location_data = {
        "location": "INVALID LOCATION",
        "data": [
            ["2024-12-15 19:00:00", "46.549946", "15.635611", "5", "15", "8.0", "60", "6.0", "7.5", "0.0"]
        ] * 12
    }

    response = client.post('/mbajk/predict', json=invalid_location_data)
    assert response.status_code == 404
    response_data = response.get_json()
    assert "error" in response_data
    assert "Model for location" in response_data["error"]


def test_data_endpoint_with_valid_location(client):
    """Test the /data endpoint with a valid location."""
    valid_location = {"location": "DVORANA TABOR"}

    response = client.post('/data', json=valid_location)
    assert response.status_code == 200
    response_data = response.get_json()
    assert "data" in response_data
    assert len(response_data["data"]) == 12


def test_data_endpoint_with_invalid_location(client):
    """Test the /data endpoint with an invalid location."""
    invalid_location = {"location": "INVALID LOCATION"}

    response = client.post('/data', json=invalid_location)
    assert response.status_code == 404
    response_data = response.get_json()
    assert "error" in response_data
    assert "not found" in response_data["error"]


def test_data_endpoint_without_body(client):
    """Test the /data endpoint without a request body."""
    response = client.post('/data')
    assert response.status_code == 400
    response_data = response.get_json()
    assert "error" in response_data