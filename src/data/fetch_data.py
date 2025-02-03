import os
import requests
import pandas as pd
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console (useful for GitHub Actions)
    ],
)

# Direct paths for bike and weather data
bike_raw_dir = './data/raw/mbajk'
weather_raw_dir = './data/raw/weather'

# Ensure directories exist
os.makedirs(bike_raw_dir, exist_ok=True)
os.makedirs(weather_raw_dir, exist_ok=True)

# Define constants
API_CALL_TIMEOUT = 10  # 10-second timeout for API calls
API_CALL_DELAY = 2     # 2-second delay between calls
MAX_RETRIES = 3        # Number of retries for failed API calls


def retry_request(url, retries=MAX_RETRIES):
    """
    Perform a GET request with retry logic.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=API_CALL_TIMEOUT)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for URL: {url}. Error: {e}")
            time.sleep(API_CALL_DELAY)  # Delay before retrying
    logging.error(f"Failed to fetch data after {retries} attempts. URL: {url}")
    return None


def fetch_bike_data(api_url):
    """
    Fetch raw bike station data and save each station's full data in a separate CSV file.
    """
    logging.info("Fetching bike data from API...")
    response = retry_request(api_url)
    if response and response.status_code == 200:
        bike_data = response.json()
        logging.info("Successfully fetched bike data.")

        for station in bike_data:
            station_name = station["name"]  # Keep the original station name
            csv_file_path = os.path.join(bike_raw_dir, f"{station_name}.csv")

            station["position_lat"] = station["position"]["lat"]
            station["position_lng"] = station["position"]["lng"]
            station["timestamp"] = datetime.fromtimestamp(station["last_update"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            del station["position"]

            station_df = pd.DataFrame([station])

            if not os.path.exists(csv_file_path):
                station_df.to_csv(csv_file_path, index=False)
                logging.info(f"Saved new bike data for station: {station_name}")
            else:
                original_df = pd.read_csv(csv_file_path)
                if station["timestamp"] not in original_df["timestamp"].values:
                    updated_df = pd.concat([original_df, station_df]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                    updated_df.to_csv(csv_file_path, index=False)
                    logging.info(f"Updated bike data for station: {station_name}")

            # Delay between API calls
            time.sleep(API_CALL_DELAY)
    else:
        logging.error("Failed to fetch bike data.")


def fetch_weather_data(lat, lng, station_name):
    """
    Fetch weather data for a given location and save it in a CSV file for the station.
    """
    logging.info(f"Fetching weather data for station: {station_name}")
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation&forecast_days=1"
    )

    response = retry_request(weather_url)
    if response and response.status_code == 200:
        weather_data = response.json()

        if "hourly" in weather_data and "time" in weather_data["hourly"]:
            hourly_data = weather_data["hourly"]

            weather_df = pd.DataFrame({
                "timestamp": hourly_data["time"],
                "temperature": hourly_data.get("temperature_2m", [None] * len(hourly_data["time"])),
                "humidity": hourly_data.get("relative_humidity_2m", [None] * len(hourly_data["time"])),
                "dew_point": hourly_data.get("dew_point_2m", [None] * len(hourly_data["time"])),
                "apparent_temperature": hourly_data.get("apparent_temperature", [None] * len(hourly_data["time"])),
                "precipitation": hourly_data.get("precipitation", [None] * len(hourly_data["time"]))
            })

            # 🔴 **Odstranjeno** zapolnjevanje manjkajočih vrednosti (prej se je `dropna()` izvajal)
            # weather_df.dropna(how='all', subset=["temperature", "humidity", "dew_point", "apparent_temperature", "precipitation"], inplace=True)

            if not weather_df.empty:
                weather_file_path = os.path.join(weather_raw_dir, f"{station_name}.csv")
                if not os.path.exists(weather_file_path):
                    weather_df.to_csv(weather_file_path, index=False)
                    logging.info(f"Saved new weather data for station: {station_name}")
                else:
                    existing_df = pd.read_csv(weather_file_path)
                    updated_df = pd.concat([existing_df, weather_df]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                    updated_df.to_csv(weather_file_path, index=False)
                    logging.info(f"Updated weather data for station: {station_name}")
            else:
                logging.warning(f"No valid weather data to save for {station_name}")
        else:
            logging.warning(f"No valid hourly data for {station_name}. Response: {weather_data}")
    else:
        logging.error(f"Failed to fetch weather data for {station_name}.")

    time.sleep(API_CALL_DELAY)


def fetch_data():
    """
    Fetch both bike and weather data and save them in raw format.
    """
    logging.info("Starting data fetching process...")
    bike_api_url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    fetch_bike_data(bike_api_url)

    bike_files = [file for file in os.listdir(bike_raw_dir) if file.endswith('.csv')]
    for bike_file in bike_files:
        bike_file_path = os.path.join(bike_raw_dir, bike_file)
        bike_df = pd.read_csv(bike_file_path)

        if not bike_df.empty:
            station_name = os.path.splitext(bike_file)[0]
            lat = bike_df["position_lat"].iloc[0]
            lng = bike_df["position_lng"].iloc[0]

            fetch_weather_data(lat, lng, station_name)
    logging.info("Data fetching process completed.")


if __name__ == "__main__":
    fetch_data()