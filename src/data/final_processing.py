import pandas as pd
import os

# # Final Preprocessing and Train-Test Split
# complete_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'combined')
# train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'train')
# test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'test')

# Paths
complete_dir = './data/processed/combined'
train_dir = './data/processed/train'
test_dir = './data/processed/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

test_size_ratio = 0.1

for csv_file in os.listdir(complete_dir):
    if csv_file.endswith('.csv'):
        file_path = os.path.join(complete_dir, csv_file)
        complete_df = pd.read_csv(file_path)

        # Sort by timestamp if a timestamp column exists
        if "timestamp" in complete_df.columns:
            complete_df["timestamp"] = pd.to_datetime(complete_df["timestamp"])
            complete_df = complete_df.sort_values("timestamp")

        # Calculate split index
        test_size = int(len(complete_df) * test_size_ratio)
        train_data = complete_df[:-test_size]
        test_data = complete_df[-test_size:]

        # Save train and test data
        train_data.to_csv(os.path.join(train_dir, csv_file), index=False)
        test_data.to_csv(os.path.join(test_dir, csv_file), index=False)