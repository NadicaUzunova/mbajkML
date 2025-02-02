import pandas as pd
import os

# Paths (nova mapa za train/test podatke iz current_data)
complete_dir = './data/processed/current_data'  
train_dir = './data/processed/train_test_from_current/train'
test_dir = './data/processed/train_test_from_current/test'

# Ustvari mape, če ne obstajajo
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

test_size_ratio = 0.1  # 10% najstarejših podatkov gre v test set

for csv_file in os.listdir(complete_dir):
    if csv_file.endswith('.csv'):
        file_path = os.path.join(complete_dir, csv_file)
        df = pd.read_csv(file_path)

        # Preveri, ali je datoteka prazna
        if df.empty:
            print(f"⚠️ Warning: {csv_file} is empty. Skipping...")
            continue

        # Preveri, ali ima timestamp stolpec
        if "timestamp" not in df.columns:
            print(f"⚠️ Warning: {csv_file} is missing 'timestamp' column. Skipping...")
            continue

        # Sort by timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")  # Razvrsti po času

        # Razdelitev podatkov
        test_size = max(1, int(len(df) * test_size_ratio))  # Vsaj ena vrstica v testu
        test_data = df.iloc[:test_size]  # Najstarejših 10% v test set
        train_data = df.iloc[test_size:]  # Ostalo v train set

        # Shrani train in test podatke v **ločeno mapo** za current_data
        train_data.to_csv(os.path.join(train_dir, csv_file), index=False)
        test_data.to_csv(os.path.join(test_dir, csv_file), index=False)

        print(f"✅ Train/Test split completed for station: {csv_file} (Train: {len(train_data)}, Test: {len(test_data)})")

print("🚀 Train/Test split complete for all stations!")