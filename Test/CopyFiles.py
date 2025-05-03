import os
import pandas as pd
import shutil

source_folder = "/Users/muyeedahmed/Library/CloudStorage/GoogleDrive-ma234@njit.edu/My Drive/Research/Dataset"
destination_folder = "../Datasets/"

# Make sure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(source_folder, filename)
        try:
            df = pd.read_csv(file_path)
            if 'target' not in df.columns:
                print(f"Skipping {filename}: 'target' column not found.")
                continue
            
            row_count = len(df)
            unique_classes = df['target'].nunique()
            print(filename, row_count, unique_classes)

            if 1000 <= row_count <= 5500 and unique_classes == 2:
                shutil.copy(file_path, os.path.join(destination_folder, filename))
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
