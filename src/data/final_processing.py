import pandas as pd
import os

# # Final Preprocessing and Train-Test Split
# complete_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'combined')
# train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'train')
# test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'test')

complete_dir = './data/processed/combined'
train_dir = './data/processed/train'
test_dir = './data/processed/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

window_size = 12
amount_of_data = 2 * 24 + 12

for csv_file in os.listdir(complete_dir):
    if csv_file.endswith('.csv'):
        file_path = os.path.join(complete_dir, csv_file)
        complete_df = pd.read_csv(file_path)

        train_data = complete_df[:-amount_of_data]
        test_data = complete_df[-amount_of_data:]

        train_data.to_csv(os.path.join(train_dir, csv_file), index=False)
        test_data.to_csv(os.path.join(test_dir, csv_file), index=False)