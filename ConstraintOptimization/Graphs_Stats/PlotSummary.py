import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    return merged

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

if __name__ == "__main__":
    try:
        df = pd.read_csv("../Stats/Summary.csv")
    except FileNotFoundError:
        df = pd.read_csv("Stats/Summary.csv")
    
    merged = GetDelta(df)
    PlotDelta(merged)
