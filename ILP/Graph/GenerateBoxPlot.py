import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

input_dir = "Stats"
output_dir = "Graph/Figures"
os.makedirs(output_dir, exist_ok=True)

clipped = True

columns_to_plot = [
    "Max_Abs_magn",
    "Sum_Abs_magn",
    "Geomean_magn",
    "Median_magn"
]

for filepath in glob.glob(os.path.join(input_dir, "*.csv")):
    df = pd.read_csv(filepath)
    
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    try:
        for ax, col in zip(axes, columns_to_plot):
            if clipped == False:
                df.boxplot(column=col, by="Threshold", grid=False, ax=ax)
            else:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                temp_df = df.copy()
                temp_df[col] = temp_df[col].clip(lower=lower_bound, upper=upper_bound)

                temp_df.boxplot(column=col, by="Threshold", grid=False, ax=ax)
                
            ax.set_title(f"{col}")
            ax.set_xlabel("Threshold/Tolerance")
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
    except:
        print(f"Error processing {filepath}: {e}")
        continue    
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}_Clipped.pdf")
    plt.close()
