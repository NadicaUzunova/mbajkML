import os
import pandas as pd

# Directories for raw and processed data
raw_bike_dir = "./data/raw/mbajk"
processed_bike_dir = "./data/processed/mbajk"
raw_weather_dir = "./data/raw/weather"
processed_weather_dir = "./data/processed/weather"
combined_dir = "./data/processed/combined"
merged_dir = "./data/processed/merged"

# Ensure directories exist
os.makedirs(processed_bike_dir, exist_ok=True)
os.makedirs(processed_weather_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)
os.makedirs(merged_dir, exist_ok=True)  # 🔹 Dodano: ustvari mapo merged

print("📂 Directories set up. Starting processing...")

# 🚲 **Preprocess bike data**
for csv_file in os.listdir(raw_bike_dir):
    if csv_file.endswith(".csv"):
        print(f"🔹 Processing bike file: {csv_file}")
        file_path = os.path.join(raw_bike_dir, csv_file)
        bike_df = pd.read_csv(file_path)

        # Remove unnecessary columns
        bike_df = bike_df.drop(columns=["latitude", "longitude"], errors="ignore")

        # Convert timestamps
        bike_df["last_update"] = pd.to_datetime(bike_df["last_update"], unit="ms")
        bike_df["timestamp"] = bike_df["last_update"].dt.floor("H")  # Aggregate to hourly level

        # Rename and select columns
        bike_df = bike_df.rename(columns={"latitude": "position_lat", "longitude": "position_lng"})
        bike_df = bike_df[["timestamp", "position_lat", "position_lng", "available_bike_stands", "available_bikes"]]

        # Aggregate by hour
        bike_df = bike_df.groupby("timestamp", as_index=False).mean()

        # Save processed bike data
        processed_file_path = os.path.join(processed_bike_dir, csv_file)
        bike_df.to_csv(processed_file_path, index=False)
        print(f"✅ Saved processed bike data to: {processed_file_path}")

# 🌦️ **Preprocess weather data**
for csv_file in os.listdir(raw_weather_dir):
    if csv_file.endswith(".csv"):
        print(f"🔹 Processing weather file: {csv_file}")
        file_path = os.path.join(raw_weather_dir, csv_file)
        weather_raw = pd.read_csv(file_path)

        # Convert timestamp
        weather_raw["timestamp"] = pd.to_datetime(weather_raw["timestamp"])

        # Save processed weather data
        processed_file_path = os.path.join(processed_weather_dir, csv_file)
        weather_raw.to_csv(processed_file_path, index=False)
        print(f"✅ Saved processed weather data to: {processed_file_path}")

# 🔄 **Combine bike and weather data**
for csv_file in os.listdir(processed_bike_dir):
    if csv_file.endswith(".csv"):
        print(f"🔹 Combining data for: {csv_file}")
        bike_data_path = os.path.join(processed_bike_dir, csv_file)
        weather_data_path = os.path.join(processed_weather_dir, csv_file)

        if os.path.exists(weather_data_path):
            bike_df = pd.read_csv(bike_data_path, parse_dates=["timestamp"])
            weather_df = pd.read_csv(weather_data_path, parse_dates=["timestamp"])

            # Merge on timestamp
            merged_df = pd.merge(bike_df, weather_df, on="timestamp", how="inner")

            # Ensure columns are in the correct order
            merged_df = merged_df[
                ["timestamp", "position_lat", "position_lng", "available_bike_stands", "available_bikes",
                 "temperature", "humidity", "dew_point", "apparent_temperature", "precipitation"]
            ]

            # Save combined data
            combined_file_path = os.path.join(combined_dir, csv_file)
            merged_df.to_csv(combined_file_path, index=False)
            print(f"✅ Saved combined data to: {combined_file_path}")
        else:
            print(f"⚠️ Weather data not found for {csv_file}. Skipping...")

# 🛠️ **Merge all processed station files into one final dataset**
output_file = os.path.join(merged_dir, "data.csv")

csv_files = [f for f in os.listdir(combined_dir) if f.endswith(".csv")]
df_list = []
for file in csv_files:
    file_path = os.path.join(combined_dir, file)
    df = pd.read_csv(file_path)

    if not df.empty:
        df_list.append(df)  # 🔹 Skip empty files

# Create final merged dataset
if df_list:
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"✅ Merged {len(csv_files)} station files into {output_file}")
else:
    print(f"⚠️ No valid files to merge in {combined_dir}.")