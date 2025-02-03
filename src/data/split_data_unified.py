import os
import pandas as pd
import argparse

def split_data(input_path, output_train, output_test, test_size_ratio=0.1):
    """Razdeli podatke na train in test glede na časovne žige."""
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    if df.empty:
        print(f"⚠️ Warning: {input_path} is empty. Skipping...")
        return
    
    df = df.sort_values(by="timestamp")
    test_size = max(1, int(len(df) * test_size_ratio))
    
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"✅ Split {input_path}: Train ({len(train_df)}), Test ({len(test_df)})")

def main():
    parser = argparse.ArgumentParser(description="Split data into train and test sets.")
    parser.add_argument("--input", required=True, help="Path to input CSV file or folder.")
    parser.add_argument("--output_train", required=True, help="Path to train dataset.")
    parser.add_argument("--output_test", required=True, help="Path to test dataset.")
    parser.add_argument("--test_size_ratio", type=float, default=0.1, help="Fraction of data for testing.")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        for csv_file in os.listdir(args.input):
            if csv_file.endswith('.csv'):
                input_path = os.path.join(args.input, csv_file)
                output_train = os.path.join(args.output_train, csv_file)
                output_test = os.path.join(args.output_test, csv_file)
                split_data(input_path, output_train, output_test, args.test_size_ratio)
    else:
        split_data(args.input, args.output_train, args.output_test, args.test_size_ratio)

if __name__ == "__main__":
    main()