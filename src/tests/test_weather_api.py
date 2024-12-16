import requests

def test_weather_api():
    """Test if the Weather API is reachable."""
    url = "https://api.open-meteo.com/v1/forecast?latitude=46.56&longitude=15.64&hourly=temperature_2m"
    response = requests.get(url)
    assert response.status_code == 200, f"Weather API returned status code {response.status_code}"