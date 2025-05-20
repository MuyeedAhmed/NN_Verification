import pandas as pd
import os
import numpy as np

def compare_files(folder_path, prefixes, i1, i2):
    for prefix in prefixes:
        f1 = os.path.join(folder_path, f"{prefix}_{i1}.npy")
        f2 = os.path.join(folder_path, f"{prefix}_{i2}.npy")
        cont1 = np.load(f1)
        cont2 = np.load(f2)
        if not os.path.exists(f1) or not os.path.exists(f2):
            return "Path not found"
        if not np.array_equal(cont1, cont2):
            return "Different"
        break
    return "Same"

def generate_summary(input_csv, weights_base_path, output_csv):
    df = pd.read_csv(input_csv)
    grouped = df.groupby('Dataset')
    output_data = []

    for name, group in grouped:
        if 0 not in group['Iteration'].values or 1 not in group['Iteration'].values:
            continue

        iter_group = group[group['Iteration'].isin([0, 1])]
        iter_group = iter_group.sort_values(by='Iteration')

        acc_0 = iter_group[iter_group['Iteration'] == 0]['Accuracy'].values[0]
        acc_1 = iter_group[iter_group['Iteration'] == 1]['Accuracy'].values[0]

        delta = (
            'Increased' if acc_1 > acc_0 else
            'Decreased' if acc_1 < acc_0 else
            'Same'
        )

        folder_name = name.replace('.csv', '')
        folder_path = os.path.join(weights_base_path, folder_name)

        weights_status = compare_files(folder_path, ['W1', 'W2', 'W3'], 0, 1)
        biases_status = compare_files(folder_path, ['b1', 'b2', 'b3'], 0, 1)

        output_data.append({
            'Dataset': name,
            'n': iter_group['n'].iloc[0],
            'col_size': iter_group['col_size'].iloc[0],
            'Accuracy_0': acc_0,
            'Accuracy_1': acc_1,
            'Delta_1': delta,
            'WeightChange_0_1': weights_status,
            'BiasChange_0_1': biases_status,
        })

    summary_df = pd.DataFrame(output_data)
    summary_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = "Stats/RecursiveCheck_Accuracy.csv"
    weights_base_path = "Weights"
    output_csv = "Stats/RecursiveCheck_Accuracy_Summary.csv"
    generate_summary(input_csv, weights_base_path, output_csv)

