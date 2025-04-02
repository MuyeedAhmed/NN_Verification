import pandas as pd
import matplotlib.pyplot as plt
import os

input_dir = "Stats/"
output_dir = "Graph/Figures"
os.makedirs(output_dir, exist_ok=True)

threshold = 1e-8


filename = "Result_44_15_Any.csv"
df = pd.read_csv(input_dir+filename)
df = df[df['Threshold'] == threshold]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

x = df['Flip_Count']
x_indices = range(len(x))


ax1.bar(x_indices, df['Max_Abs_magn'], color='skyblue')
ax1.set_title('Max Absolute Magnitude by Flip Count')
ax1.set_ylabel('Max_Abs_magn')
ax1.grid(axis='y')

ax2.bar(x_indices, df['Geomean_magn'], color='lightgreen')
ax2.set_title('Geometric Mean Magnitude by Flip Count')
ax2.set_xlabel('Flip Count')
ax2.set_ylabel('Geomean_magn')
ax2.set_xticks(ticks=x_indices)
ax2.set_xticklabels(df['Flip_Count'])
ax2.grid(axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/FlipCountVS_{filename}_{threshold}.pdf")
plt.close()
