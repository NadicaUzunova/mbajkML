import os
import pandas as pd

# Direct paths for raw and processed data
raw_bike_dir = './data/raw/mbajk'
processed_bike_dir = './data/processed/mbajk'
raw_weather_dir = './data/raw/weather'
processed_weather_dir = './data/processed/weather'
combined_dir = './data/processed/combined'

# Ensure directories exist
os.makedirs(processed_bike_dir, exist_ok=True)
os.makedirs(processed_weather_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)

print("Directories set up. Starting processing...")

# Preprocess bike data
for csv_file in os.listdir(raw_bike_dir):
    if csv_file.endswith('.csv'):
        print(f"Processing bike file: {csv_file}")
        file_path = os.path.join(raw_bike_dir, csv_file)
        bike_df = pd.read_csv(file_path)

        # Odstranimo neželene stolpce (latitude, longitude)
        bike_df = bike_df.drop(columns=["latitude", "longitude"], errors="ignore")

        # Convert timestamps
        bike_df['last_update'] = pd.to_datetime(bike_df['last_update'], unit='ms')
        bike_df['timestamp'] = bike_df['last_update'].dt.floor('H')  # Aggregate to hourly level

        # Rename and select columns
        bike_df = bike_df.rename(columns={
            'latitude': 'position_lat',
            'longitude': 'position_lng'
        })
        bike_df = bike_df[['timestamp', 'position_lat', 'position_lng', 'available_bike_stands', 'available_bikes']]

        # Aggregate by hour
        bike_df = bike_df.groupby('timestamp', as_index=False).mean()

        # Save processed bike data
        processed_file_path = os.path.join(processed_bike_dir, csv_file)
        if os.path.exists(processed_file_path):
            print(f"Appending to existing bike file: {processed_file_path}")
            existing_bike_df = pd.read_csv(processed_file_path, parse_dates=['timestamp'])
            bike_df = pd.concat([existing_bike_df, bike_df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        bike_df.to_csv(processed_file_path, index=False)
        print(f"Saved processed bike data to: {processed_file_path}")

# Preprocess weather data
for csv_file in os.listdir(raw_weather_dir):
    if csv_file.endswith('.csv'):
        print(f"Processing weather file: {csv_file}")
        file_path = os.path.join(raw_weather_dir, csv_file)
        weather_raw = pd.read_csv(file_path)

        # Ensure weather timestamps are datetime objects
        weather_raw['timestamp'] = pd.to_datetime(weather_raw['timestamp'])

        # Save processed weather data
        processed_file_path = os.path.join(processed_weather_dir, csv_file)
        if os.path.exists(processed_file_path):
            print(f"Appending to existing weather file: {processed_file_path}")
            existing_weather_df = pd.read_csv(processed_file_path, parse_dates=['timestamp'])
            weather_raw = pd.concat([existing_weather_df, weather_raw]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        weather_raw.to_csv(processed_file_path, index=False)
        print(f"Saved processed weather data to: {processed_file_path}")

# Combine bike and weather data
for csv_file in os.listdir(processed_bike_dir):
    if csv_file.endswith('.csv'):
        print(f"Combining data for file: {csv_file}")
        bike_data_path = os.path.join(processed_bike_dir, csv_file)
        weather_data_path = os.path.join(processed_weather_dir, csv_file)

        if os.path.exists(weather_data_path):
            bike_df = pd.read_csv(bike_data_path, parse_dates=['timestamp'])

            # Odstranimo neželene stolpce (latitude, longitude)
            bike_df = bike_df.drop(columns=["latitude", "longitude"], errors="ignore")

            weather_df = pd.read_csv(weather_data_path, parse_dates=['timestamp'])

            # Merge on timestamp
            merged_df = pd.merge(bike_df, weather_df, on='timestamp', how='inner')

            # Ensure columns are in the correct order
            merged_df = merged_df[[
                'timestamp', 'position_lat', 'position_lng', 'available_bike_stands', 'available_bikes',
                'temperature', 'humidity', 'dew_point', 'apparent_temperature', 'precipitation'
            ]]

            # Save combined data
            combined_file_path = os.path.join(combined_dir, csv_file)
            if os.path.exists(combined_file_path):
                print(f"Appending to existing combined file: {combined_file_path}")
                existing_combined_df = pd.read_csv(combined_file_path, parse_dates=['timestamp'])
                merged_df = pd.concat([existing_combined_df, merged_df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            merged_df.to_csv(combined_file_path, index=False)
            print(f"Saved combined data to: {combined_file_path}")
        else:
            print(f"Weather data not found for {csv_file}. Skipping...")