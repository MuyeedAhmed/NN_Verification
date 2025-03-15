import os
import pandas as pd

directory = "/Users/muyeedahmed/Desktop/Gitcode/AD_Attack/Dataset"

files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in files:
    file_path = os.path.join(directory, file)
    try:
        df = pd.read_csv(file_path)
        print(f"{file}: {df.shape}")
    except Exception as e:
        print(f"Error reading {file}: {e}")
