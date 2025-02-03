import os
import pandas as pd

# Paths
merged_data_path = "./data/processed/merged/current_data.csv"
train_output_path = "./data/processed/train_test_merged/train.csv"
test_output_path = "./data/processed/train_test_merged/test.csv"

# Ensure output directories exist
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

# Load merged dataset
df = pd.read_csv(merged_data_path, parse_dates=["timestamp"])

# Sort by timestamp
df = df.sort_values(by="timestamp")

# Split data (10% latest for testing)
test_size = int(len(df) * 0.1)
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

# Save train & test datasets
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print(f"✅ Split merged dataset: {len(train_df)} train samples, {len(test_df)} test samples.")