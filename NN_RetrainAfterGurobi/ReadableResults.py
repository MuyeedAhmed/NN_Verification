import pandas as pd


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
    summary = Summarize(df)
    print(summary)
    