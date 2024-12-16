import requests

def test_mbajk_api():
    """Test if the MBajk API is reachable."""
    url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(url)
    assert response.status_code == 200, f"MBajk API returned status code {response.status_code}"