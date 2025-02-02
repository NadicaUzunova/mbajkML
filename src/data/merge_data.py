import os
import pandas as pd

# Directories
combined_dir = "./data/processed/combined"
merged_dir = "./data/processed/merged"

# Ensure directories exist
os.makedirs(merged_dir, exist_ok=True)

# Output file
output_file = os.path.join(merged_dir, "data.csv")

# Collect all CSV files
csv_files = [f for f in os.listdir(combined_dir) if f.endswith(".csv")]
df_list = []

for file in csv_files:
    file_path = os.path.join(combined_dir, file)
    df = pd.read_csv(file_path)

    if not df.empty:
        df_list.append(df)

# Merge datasets
if df_list:
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"✅ Merged {len(csv_files)} station files into {output_file}")
else:
    print(f"⚠️ No valid files to merge in {combined_dir}.")