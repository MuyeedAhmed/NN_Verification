import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

input_dir = "Stats"
output_dir = "Graph/Figures"
os.makedirs(output_dir, exist_ok=True)

columns_to_plot = [
    "Max_Abs_Diff",
    "Sum_Abs_Diff",
    "Geomean_Diff",
    "Median_Diff"
]

for filepath in glob.glob(os.path.join(input_dir, "*.csv")):
    df = pd.read_csv(filepath)
    
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    try:
        for ax, col in zip(axes, columns_to_plot):
            df.boxplot(column=col, by="Threshold", grid=False, ax=ax)
            ax.set_title(f"{col}")
            ax.set_xlabel("Threshold/Tolerance")
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
    except:
        continue    
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}.png")
    plt.close()
