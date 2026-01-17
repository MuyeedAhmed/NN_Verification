import pandas as pd


def Summarize(df):
    datasets = df["Dataset"].unique()
    new_df = pd.DataFrame()
    for dataset in datasets:
        df_train = df[(df["Dataset"] == dataset) & (df["Method"] == "Train")]
        df_RAF_C1 = df[(df["Dataset"] == dataset) & (df["Method"] == "RAF_C1")]
        df_RAF_C10 = df[(df["Dataset"] == dataset) & (df["Method"] == "RAF_C10")]
        df_RAF_A1 = df[(df["Dataset"] == dataset) & (df["Method"] == "RAF_A1")]
        df_RAF_A10 = df[(df["Dataset"] == dataset) & (df["Method"] == "RAF_A10")]
        df_RAB0 = df[(df["Dataset"] == dataset) & (df["Method"] == "RAB0")]
        df_F_C1 = df[(df["Dataset"] == dataset) & (df["Method"] == "F_C1")]
        df_F_C10 = df[(df["Dataset"] == dataset) & (df["Method"] == "F_C10")]
        df_F_A1 = df[(df["Dataset"] == dataset) & (df["Method"] == "F_A1")]
        df_F_A10 = df[(df["Dataset"] == dataset) & (df["Method"] == "F_A10")]
        df_B0 = df[(df["Dataset"] == dataset) & (df["Method"] == "B0")]

        summary = [{
            "Dataset": dataset,
            "Train Train Loss": df_train["Train Loss"].mean(),
            "Train Train Acc": df_train["Train Acc"].mean(),
            "Train Test Acc": df_train["Test Acc"].mean(),
            "RAF_C1 Train Loss": df_RAF_C1["Train Loss"].mean(),
            "RAF_C1 Train Acc": df_RAF_C1["Train Acc"].mean(),
            "RAF_C1 Test Acc": df_RAF_C1["Test Acc"].mean(),
            "RAF_C10 Train Loss": df_RAF_C10["Train Loss"].mean(),
            "RAF_C10 Train Acc": df_RAF_C10["Train Acc"].mean(),
            "RAF_C10 Test Acc": df_RAF_C10["Test Acc"].mean(),
            "RAF_A1 Train Loss": df_RAF_A1["Train Loss"].mean(),
            "RAF_A1 Train Acc": df_RAF_A1["Train Acc"].mean(),
            "RAF_A1 Test Acc": df_RAF_A1["Test Acc"].mean(),
            "RAF_A10 Train Loss": df_RAF_A10["Train Loss"].mean(),
            "RAF_A10 Train Acc": df_RAF_A10["Train Acc"].mean(),
            "RAF_A10 Test Acc": df_RAF_A10["Test Acc"].mean(),
            "RAB0 Train Loss": df_RAB0["Train Loss"].mean(),
            "RAB0 Train Acc": df_RAB0["Train Acc"].mean(),
            "RAB0 Test Acc": df_RAB0["Test Acc"].mean(),
            "F_C1 Train Loss": df_F_C1["Train Loss"].mean(),
            "F_C1 Train Acc": df_F_C1["Train Acc"].mean(),
            "F_C1 Test Acc": df_F_C1["Test Acc"].mean(),
            "F_C10 Train Loss": df_F_C10["Train Loss"].mean(),
            "F_C10 Train Acc": df_F_C10["Train Acc"].mean(),
            "F_C10 Test Acc": df_F_C10["Test Acc"].mean(),
            "F_A1 Train Loss": df_F_A1["Train Loss"].mean(),
            "F_A1 Train Acc": df_F_A1["Train Acc"].mean(),
            "F_A1 Test Acc": df_F_A1["Test Acc"].mean(),
            "F_A10 Train Loss": df_F_A10["Train Loss"].mean(),
            "F_A10 Train Acc": df_F_A10["Train Acc"].mean(),
            "F_A10 Test Acc": df_F_A10["Test Acc"].mean(),
            "B0 Train Loss": df_B0["Train Loss"].mean(),
            "B0 Train Acc": df_B0["Train Acc"].mean(),
            "B0 Test Acc": df_B0["Test Acc"].mean()
        }]
        summary = pd.DataFrame(summary)
        new_df = pd.concat([new_df, summary], ignore_index=True)
    return new_df   

def GetTables(summary, method):
    for row in summary.iterrows():
        Init_train_loss = row[1][f"Train Train Loss"]
        Init_train_acc = row[1][f"Train Train Acc"]*100
        Init_test_acc = row[1][f"Train Test Acc"]*100

        G_train_loss = row[1][f"{method} Train Loss"]
        G_train_acc = row[1][f"{method} Train Acc"]*100
        G_test_acc = row[1][f"{method} Test Acc"]*100

        RAG_train_loss = row[1][f"RA{method} Train Loss"]
        RAG_train_acc = row[1][f"RA{method} Train Acc"]*100
        RAG_test_acc = row[1][f"RA{method} Test Acc"]*100

        # TAGD table
        print(f"{row[1]['Dataset']} & {Init_train_acc:.2f} & {Init_train_loss:.2f} & {G_train_acc:.2f} &  {G_train_loss:.2f} & {Init_test_acc:.2f} & {G_test_acc:.2f} & {(G_test_acc-Init_test_acc)} \\\\")
        # STC Test Acc
        # print(f"{row[1]['Dataset']} & {Init_test_acc:.2f} & {G_test_acc:.2f} & {(G_test_acc-Init_test_acc)} \\\\")

        # Change 1 classification
        print(f"{row[1]['Dataset']} & {Init_train_acc:.2f} & {Init_test_acc:.2f} & {G_train_acc:.2f} &  {G_test_acc:.2f} & {RAG_train_acc:.2f} &  {RAG_test_acc:.2f}  & {(G_train_acc-Init_train_acc):.3f} & {(G_test_acc-Init_test_acc):.3f}  & {(RAG_train_acc-Init_train_acc):.3f} & {(RAG_test_acc-Init_test_acc):.3f} \\\\")

        # Change All
        # print(f"{row[1]['Dataset']} & {(RAG_test_acc-Init_test_acc):.3f}")
        # print(f"{(RAG_test_acc-Init_test_acc):.3f}")
        
        

if __name__ == "__main__":
    df = pd.read_csv("Stats/Summary.csv")
    summary = Summarize(df)
    # print(summary)
    GetTables(summary, "B0")
    GetTables(summary, "F_A1")
    # GetTables(summary, "F_A10")
    # GetTables(summary, "F_C1")
    # GetTables(summary, "F_C10")
    