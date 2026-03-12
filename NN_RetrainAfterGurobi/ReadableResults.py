import pandas as pd



def Summarize_TAGD(df):
    datasets = df["Dataset"].unique()
    new_df = pd.DataFrame()
    for dataset in datasets:
        grouped = (
            df[df["Dataset"] == dataset]
            .groupby("Method", as_index=False)
            .agg(
                train_loss_mean=("Train Loss", "mean"),
                train_acc_mean=("Train Acc", "mean"),
                test_acc_mean=("Test Acc", "mean"),
            )
        )
        dataset_result = {
            "Dataset": dataset,
            "Init_training_acc": grouped.loc[grouped["Method"] == "Train", "train_acc_mean"].iloc[0]* 100.0,
            "Init_training_loss": grouped.loc[grouped["Method"] == "Train", "train_loss_mean"].iloc[0],
            "TAGD_Training_acc": grouped.loc[grouped["Method"] == "B0", "train_acc_mean"].iloc[0]* 100.0,
            "TAGD_Training_loss": grouped.loc[grouped["Method"] == "B0", "train_loss_mean"].iloc[0],
            "Init_test_acc": grouped.loc[grouped["Method"] == "Train", "test_acc_mean"].iloc[0]* 100.0,
            "TAGD_test_acc": grouped.loc[grouped["Method"] == "B0", "test_acc_mean"].iloc[0]* 100.0,
            "TAGD_test_acc_delta": (grouped.loc[grouped["Method"] == "B0", "test_acc_mean"].iloc[0] - grouped.loc[grouped["Method"] == "Train", "test_acc_mean"].iloc[0]) * 100.0,
        }
        new_df = pd.concat([new_df, pd.DataFrame([dataset_result])], ignore_index=True)
    return new_df


def Summarize(df):
    datasets = df["Dataset"].unique()
    new_df = pd.DataFrame()
    for dataset in datasets:
        grouped = (
            df[df["Dataset"] == dataset]
            .groupby("Method", as_index=False)
            .agg(
                # train_loss_mean=("Train Loss", "mean"),
                train_acc_mean=("Train Acc", "mean"),
                test_acc_mean=("Test Acc", "mean"),
            )
        )
        # print(grouped)
        init_test_acc = grouped.loc[grouped["Method"] == "Train", "test_acc_mean"].iloc[0]
        # print(f"Dataset: {dataset}, Train Test Acc: {init_test_acc}")
        grouped["Delta_test_acc_vs_init"] = (grouped["test_acc_mean"] - init_test_acc) * 100.0
        # df_delta = grouped[grouped["Method"] != "Train"]
        df_raf = grouped[grouped["Method"].str.contains("RAF", na=False)]
        # print(df_raf)
        result = (
            df_raf
            .set_index("Method")["Delta_test_acc_vs_init"]
            .to_frame()
            .T
        )
        result["Dataset"] = dataset
        # print(result)
        # break

    
        new_df = pd.concat([new_df, result], ignore_index=True)
    return new_df   
    
        

if __name__ == "__main__":
    df = pd.read_csv("Stats/Summary.csv")
    # summary = Summarize(df)
    # for row in summary.itertuples():
    #     print(f"{row.Dataset} & {row.RAF_A1:.2f} & {row.RAF_A10:.2f} & {row.RAF_C1:.2f} & {row.RAF_C10:.2f} \\\\")

    summary = Summarize_TAGD(df)
    for row in summary.itertuples():
        print(f"{row.Dataset} & {row.Init_training_acc:.2f} & {row.Init_training_loss:.2f} & {row.TAGD_Training_acc:.2f} & {row.TAGD_Training_loss:.2f} & {row.Init_test_acc:.2f} & {row.TAGD_test_acc:.2f} & {row.TAGD_test_acc_delta:.2f} \\\\")

    # print(summary)
    