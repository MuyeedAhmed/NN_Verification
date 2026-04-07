import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def GetDelta(df):
    df = df[df["Method"] == "CMC"].copy()

    df["CMC_Label"] = (
        df["CMC_Type"].astype(str) + "_" +
        df["Misclassification_Count"].astype(str)
    )
    agg_s3 = df.groupby(
        ["Dataset", "Training_Type", "CMC_Label"]
    ).agg(
        S3_Test_mean=("S3_Test_acc", "mean"),
        S3_Val_mean=("S3_Val_acc", "mean")
    ).reset_index()

    baseline = df.groupby(
        ["Dataset", "Training_Type"]
    ).agg(
        S1_Test_mean=("S1_Test_acc", "mean"),
        S1_Val_mean=("S1_Val_acc", "mean")
    ).reset_index()
    merged = pd.merge(
        agg_s3,
        baseline,
        on=["Dataset", "Training_Type"]
    )

    merged["Delta"] = merged["S3_Test_mean"] - merged["S1_Test_mean"]
    merged["Delta_Val"] = merged["S3_Val_mean"] - merged["S1_Val_mean"]
    merged = merged.rename(columns={'S3_Test_mean': 'Training+CMC', 'S1_Test_mean': 'Standalone'})
    merged.drop(columns=["S3_Val_mean", "S1_Val_mean"], inplace=True)

    merged.to_csv("Stats/Summary_Delta.csv", index=False)
    
    merged_filtered = merged.copy()
    merged_filtered.loc[merged_filtered["Delta_Val"] < 0, "Delta"] = np.nan
    merged_filtered.to_csv("Stats/Summary_Delta_Filtered.csv", index=False)

    return merged, merged_filtered

def GetBestCMCperDataset(df):
    bestCMCs = pd.DataFrame(columns=df.columns)
    for ttype in df["Training_Type"].unique():
        for dataset in df["Dataset"].unique():
            subset = df[(df["Training_Type"] == ttype) & (df["Dataset"] == dataset)]
            best_row = subset.loc[subset["Delta_Val"].idxmax()]

            bestCMCs = pd.concat([bestCMCs, best_row.to_frame().T], ignore_index=True)

    return bestCMCs


def PlotBestCMCperDataset(bestCMCs):
    plt.figure(figsize=(10, 6))
    
    label_map = {
        "Any_1": "Any 1", 
        "Any_10": "Any 10", 
        "Correct_1": "Correct 1", 
        "Correct_10": "Correct 10"
    }
    bestCMCs["CMC_Label"] = bestCMCs["CMC_Label"].map(label_map).fillna(bestCMCs["CMC_Label"])

    dataset_order = (
        bestCMCs.groupby("Dataset")["Delta"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    ax = sns.barplot(
        data=bestCMCs,
        y='Dataset',
        x='Delta',
        hue='Training_Type',
        order=dataset_order
    )
    
    hue_order = [l.get_label() for l in ax.legend_.get_lines()]
    if not hue_order:
        hue_order = list(bestCMCs['Training_Type'].unique())

    for i, container in enumerate(ax.containers):
        current_hue = hue_order[i]
        
        for j, bar in enumerate(container):
            if bar is None or bar.get_height() == 0:
                continue
                
            dataset = dataset_order[j]
            label_row = bestCMCs[(bestCMCs['Dataset'] == dataset) & 
                                 (bestCMCs['Training_Type'] == current_hue)]
            
            if not label_row.empty:
                label_text = str(label_row['CMC_Label'].item())
                delta_val = bar.get_width()
                y_pos = bar.get_y() + bar.get_height() / 2
                
                x_pos = max(delta_val, 0.01)

                ax.annotate(
                    label_text,
                    xy=(x_pos, y_pos),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va='center',
                    ha='left',
                    fontsize=9,
                    fontweight='bold'
                )

    ax.set_xlabel(r'Accuracy Gain (Training$_{\text{CMC}}$ - Standalone)', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlim(-0.5, 3.5)
    
    plt.legend(title='Training Type', loc='lower right')
    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/Best_CMC_Delta.pdf')
    plt.show()


def PlotDelta(merged):
    merged = merged.copy() 
    g = sns.catplot(
        data=merged,
        x="Dataset",
        y="Delta",
        hue="CMC_Label",
        col="Training_Type",
        col_wrap=2,
        kind="bar",
        height=4,
        aspect=1,
        legend=True
    )    
    g.tick_params(axis='x', rotation=90)
    g.set_titles("{col_name}")
    g.set_axis_labels("Dataset", "Accuracy Gain (Training+CMC - Standalone)")

    for ax in g.axes.flat:
        ax.axhline(0, linestyle="--", color='black', linewidth=1)

    sns.move_legend(g, bbox_to_anchor=(1.0, 0.5), loc='center left')

    plt.tight_layout()
    os.makedirs("Figs", exist_ok=True)
    plt.savefig("Figs/Delta_Plot.pdf")

def PlotDelta_Sorted(merged):
    merged = merged.copy()
    os.makedirs("Figs", exist_ok=True)

    label_map = {
        "Any_1": "Any 1", 
        "Any_10": "Any 10", 
        "Correct_1": "Correct 1", 
        "Correct_10": "Correct 10"
    }

    for ttype in merged["Training_Type"].unique():
        df = merged[merged["Training_Type"] == ttype].copy()
        df["CMC_Label"] = df["CMC_Label"].map(label_map).fillna(df["CMC_Label"])
        order = (
            df.groupby("Dataset")["Delta"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        plt.figure(figsize=(6, 4))

        ax = sns.barplot(
            data=df,
            y="Dataset",
            x="Delta",
            hue="CMC_Label",
            order=order
        )
        ax.axvline(0, linestyle="--", color="black", linewidth=1)

        ax.set_title(f"{ttype}", fontsize=14)
        ax.set_xlabel(r'Accuracy Gain (Training$_{\text{CMC}}$ - Standalone)', fontsize=14)
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.legend(title="CMC Types", loc="lower right", fontsize=14, title_fontsize=14)

        plt.tight_layout()
        plt.savefig(f"Figs/Delta_{ttype}.pdf")
        plt.close()


def PlotStandaloneComparison(merged):
    df_plot = merged[['Dataset', 'Training_Type', 'Standalone']].drop_duplicates()

    df_plot = df_plot.sort_values(['Dataset', 'Standalone'], ascending=[True, False])
    plt.figure(figsize=(12, 10))

    ax = sns.barplot(
        data=df_plot,
        y='Dataset',
        x='Standalone',
        hue='Training_Type',
    )
    ax.set_xlabel('Standalone Accuracy (%)', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_xlim(60, 100)

    plt.legend(title='Training Type', loc='lower right')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/Standalone_Accuracies.pdf')
    plt.show()



if __name__ == "__main__":
    try:
        df = pd.read_csv("../Stats/Summary.csv")
    except FileNotFoundError:
        df = pd.read_csv("Stats/Summary.csv")
    
    merged, merged_filtered = GetDelta(df)
    # bestCMCs = GetBestCMCperDataset(merged_filtered)
    # PlotBestCMCperDataset(bestCMCs)
    # print(bestCMCs)
    # PlotDelta_Sorted(merged)
    PlotDelta_Sorted(merged_filtered)
    # PlotStandaloneComparison(merged)
