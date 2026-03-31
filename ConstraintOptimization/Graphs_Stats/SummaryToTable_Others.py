import pandas as pd

def StandardVsSingle(df, training_type, ctype):
    latex_rows = []
    
    for _, row in df.iterrows():
        dataset = row["Dataset"]
        values = row.drop("Dataset").astype(float)
        sorted_vals = values.sort_values(ascending=False)
        best = sorted_vals.iloc[0]
        second = sorted_vals.iloc[1]

        formatted = []
        for v in values:
            if v == best:
                formatted.append(f"\\textbf{{\\textcolor{{green}}{{{v:.2f}}}}}")
            elif v == second:
                formatted.append(f"\\textcolor{{green!80}}{{{v:.2f}}}")
            else:
                formatted.append(f"{v:.2f}")

        latex_rows.append(dataset + " & " + " & ".join(formatted) + " \\\\")

    header = f"Dataset & Standard ERM & Standard ERM + & {training_type} & {training_type} + \\\\\n"
    header2 = f" & & $\\text{{CMC}}_{{{ctype}}}$ & & $\\text{{CMC}}_{{{ctype}}}$ \\\\"
    
    latex_table = r"""
\begin{table}[t]
\caption{Comparison of CMC with other methods across datasets}
\label{tab:cmc-comparison}
\begin{center}
\begin{tabular}{lcccc}
\toprule
""" + header + header2 + r"""
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
"""
    print(latex_table)



def CompareByCType(df, ctype="Any", MiscCount = 1):
    df_r = df[df["Training_Type"] == "ERP"]
    df_awp = df[df["Training_Type"] == "AWP"]
    df_sap = df[df["Training_Type"] == "SAP"]

    df_awp = df_awp[(df_awp["Method"] == "CMC") & (df_awp["CMC_Type"] == ctype) & (df_awp["Misclassification_Count"] == MiscCount)]
    df_awp = df_awp.groupby("Dataset").mean(numeric_only=True).reset_index()

    df_sap = df_sap[(df_sap["Method"] == "CMC") & (df_sap["CMC_Type"] == ctype) & (df_sap["Misclassification_Count"] == MiscCount)]
    df_sap = df_sap.groupby("Dataset").mean(numeric_only=True).reset_index()


    df_r = df_r[(df_r["Method"] == "CMC") & (df_r["CMC_Type"] == ctype) & (df_r["Misclassification_Count"] == MiscCount)]
    df_r = df_r.groupby("Dataset").mean(numeric_only=True).reset_index()

    merged_df_awp = pd.merge(df_awp, df_r, on="Dataset", suffixes=("_AWP", "_R"))
    output_columns_awp = ["Dataset", "S1_Test_acc_R", "S3_Test_acc_R", "S1_Test_acc_AWP", "S3_Test_acc_AWP"]
    merged_df_awp = merged_df_awp[output_columns_awp]

    merged_df_sap = pd.merge(df_sap, df_r, on="Dataset", suffixes=("_SAP", "_R"))
    output_columns_sap = ["Dataset", "S1_Test_acc_R", "S3_Test_acc_R", "S1_Test_acc_SAP", "S3_Test_acc_SAP"]
    merged_df_sap = merged_df_sap[output_columns_sap]

    StandardVsSingle(merged_df_awp, "AWP", f"{ctype}\_{MiscCount}")
    StandardVsSingle(merged_df_sap, "SAP", f"{ctype}\_{MiscCount}")

def FindTopTechniques(df):
    df_cmc = df[df["Method"] == "CMC"]
    datasets = df_cmc["Dataset"].unique()
    
    for dataset in datasets:
        df_d = df_cmc[df_cmc["Dataset"] == dataset]
        results = []
        
        standalone = df_d.groupby("Training_Type")["S1_Test_acc"].mean().reset_index()
        for _, row in standalone.iterrows():
            results.append({
                "name": f"{row['Training_Type']}",
                "acc": row["S1_Test_acc"]
            })
            
        improved = df_d.groupby(["Training_Type", "CMC_Type", "Misclassification_Count"])["S3_Test_acc"].mean().reset_index()
        for _, row in improved.iterrows():
            results.append({
                "name": f"{row['Training_Type']}+{row['CMC_Type']}\_{int(row['Misclassification_Count'])}",
                "acc": row["S3_Test_acc"]
            })
            
        results.sort(key=lambda x: x["acc"], reverse=True)
        # value = next(res['acc'] for res in results if res['name'] == 'SAP_Standalone')
        # print(value)
        print(f"{dataset} & " + " & ".join(f"{res['name']} ({res['acc']:.2f}\\%)" for res in results[:3]) + " & " + f"{next(res['acc'] for res in results if res['name'] == 'AWP'):.2f}\\% & {next(res['acc'] for res in results if res['name'] == 'SAP'):.2f}\\% \\\\")
        # print(f"\n--- Dataset: {dataset} ---")
        # for rank, res in enumerate(results[:3], 1):
        #     print(f"Rank {rank}: {res['name']} ({res['acc']:.2f}%)")


df = pd.read_csv("Stats/Summary copy.csv")
# FindTopTechniques(df)
CompareByCType(df, ctype="Any", MiscCount=10)