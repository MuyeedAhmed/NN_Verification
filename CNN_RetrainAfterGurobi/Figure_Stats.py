import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def FillStatsFileWithTrain():
    raf_folder_path = "Stats/RAF_CrossVal_All"
    rab_folder_path = "Stats/RAB_CrossVal_All"

    Stats_folder = "Stats"

    raf_log_files = [f for f in os.listdir(raf_folder_path) if f.endswith('_log.csv') and "gurobi" not in f]
    rab_log_files = [f for f in os.listdir(rab_folder_path) if f.endswith('_log.csv') and "gurobi" not in f]

    for file in raf_log_files:
        dataset_name = file.split('_log')[0]
        raf_file_path = os.path.join(raf_folder_path, file)
        rab_file_path = os.path.join(rab_folder_path, file)
        
        if not os.path.exists(rab_file_path):
            print(f"RAB file not found for {dataset_name}.")
            continue
        raf_history = pd.read_csv(raf_file_path)
        rab_history = pd.read_csv(rab_file_path)
        if "Phase" not in rab_history.columns:
            rab_history = pd.read_csv(rab_file_path, header=None)
            rab_history.columns = ['Run', 'Phase', 'Epoch', 'Train_loss', 'Train_acc']
        if "Phase" not in raf_history.columns:
            raf_history = pd.read_csv(raf_file_path, header=None)
            raf_history.columns = ['Run', 'Phase', 'Epoch', 'Train_loss', 'Train_acc']

        if raf_history[raf_history['Phase'] == 'Train'].empty:
            print(f"RAF {dataset_name}. Size: {raf_history.shape}")
            rab_train_history = rab_history[(rab_history['Phase'] == 'Train') | (rab_history['Phase'] == 'ResumeTrain') | (rab_history['Phase'] == 'Train_Test') | (rab_history['Phase'] == 'ResumeTrain_Test')]
            if not rab_train_history.empty:
                raf_history = pd.concat([raf_history, rab_train_history], ignore_index=True)
                raf_history.to_csv(raf_file_path, index=False)
            print(f"RAF {dataset_name}. New size: {raf_history.shape}")
        elif rab_history[rab_history['Phase'] == 'Train'].empty:
            print(f"RAB {dataset_name}. Size: {rab_history.shape}")
            raf_train_history = raf_history[(raf_history['Phase'] == 'Train') | (raf_history['Phase'] == 'ResumeTrain') | (raf_history['Phase'] == 'Train_Test') | (raf_history['Phase'] == 'ResumeTrain_Test')]
            if not raf_train_history.empty:
                rab_history = pd.concat([rab_history, raf_train_history], ignore_index=True)
                rab_history.to_csv(rab_file_path, index=False)
            print(f"RAB {dataset_name}. New size: {rab_history.shape}")
        else:
            print(f"Both RAF and RAB have Train Phase for {dataset_name}.")


def SummarizeStatsPerDataset(df, dataset_name, run_id):
    train_df = df[df['Phase'] == 'Train']
    resume_df = df[df['Phase'] == 'ResumeTrain']
    gurobi_df = df[df['Phase'] == 'GurobiEdit']
    
    S1_train_acc = train_df['Train_acc'].iloc[-1]
    S1_train_loss = train_df['Train_loss'].iloc[-1]
    
    S1_test_acc = df[df['Phase'] == 'Train_Test']['Train_acc'].iloc[0]
    S1_test_loss = df[df['Phase'] == 'Train_Test']['Train_loss'].iloc[0]
    
    if not resume_df.empty:
        S2_train_acc = resume_df['Train_acc'].iloc[-1]
        S2_train_loss = resume_df['Train_loss'].iloc[-1]
        S2_test_acc = df[df['Phase'] == 'ResumeTrain_Test']['Train_acc'].iloc[0]
        S2_test_loss = df[df['Phase'] == 'ResumeTrain_Test']['Train_loss'].iloc[0]

    if df[df['Phase'] == 'Gurobi_Complete_Eval_Train'].empty:
        return pd.DataFrame() 
    S3_Start_train_acc = df[df['Phase'] == 'Gurobi_Complete_Eval_Train']['Train_acc'].iloc[0]
    S3_Start_train_loss = df[df['Phase'] == 'Gurobi_Complete_Eval_Train']['Train_loss'].iloc[0]
    S3_Start_test_acc = df[df['Phase'] == 'Gurobi_Complete_Eval_Val']['Train_acc'].iloc[0]
    S3_Start_test_loss = df[df['Phase'] == 'Gurobi_Complete_Eval_Val']['Train_loss'].iloc[0]


    S3_train_acc = gurobi_df['Train_acc'].iloc[-1]
    S3_train_loss = gurobi_df['Train_loss'].iloc[-1]
    S3_test_loss = df[df['Phase'] == 'GurobiEdit_Test']['Train_loss'].iloc[0]
    S3_test_acc = df[df['Phase'] == 'GurobiEdit_Test']['Train_acc'].iloc[0]
    
    
    summary_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Run ID': [run_id],
        'S1_Train_Acc': [S1_train_acc],
        'S1_Train_Loss': [S1_train_loss],
        'S1_Test_Acc': [S1_test_acc],
        'S1_Test_Loss': [S1_test_loss],
        'S2_Train_Acc': [S2_train_acc] if 'S2_train_acc' in locals() else [S1_train_acc],
        'S2_Train_Loss': [S2_train_loss] if 'S2_train_loss' in locals() else [S1_train_loss],
        'S2_Test_Acc': [S2_test_acc] if 'S2_test_acc' in locals() else [S1_test_acc],
        'S2_Test_Loss': [S2_test_loss] if 'S2_test_loss' in locals() else [S1_test_loss],
        'S3_Start_Train_Acc': [S3_Start_train_acc],
        'S3_Start_Train_Loss': [S3_Start_train_loss],
        'S3_Start_Test_Acc': [S3_Start_test_acc],
        'S3_Start_Test_Loss': [S3_Start_test_loss],
        'S3_Train_Acc': [S3_train_acc],
        'S3_Train_Loss': [S3_train_loss],
        'S3_Test_Acc': [S3_test_acc],
        'S3_Test_Loss': [S3_test_loss]
    })

    return summary_df


def plot_training_history(df, dataset_name, run_id, fig_folder):
    train_df = df[df['Phase'] == 'Train']
    gurobi_df = df[df['Phase'] == 'GurobiEdit']
    gurobi_eval_df = df[df['Phase'] == 'Gurobi_Complete_Eval']

    train_end_epoch = train_df['Epoch'].max()
    gurobi_df = gurobi_df.copy()
    gurobi_df['Shifted_Epoch'] = gurobi_df['Epoch'] + train_end_epoch - 1
    

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_df['Epoch'], train_df['Train_acc'], color="blue", label='Train Accuracy (Train Phase)')
    plt.plot(train_df['Epoch'], train_df['Val_acc'], color="green", label='Val Accuracy (Train Phase)')

    plt.plot(gurobi_df['Shifted_Epoch'], gurobi_df['Train_acc'], color="blue",linestyle='--', label='Train Accuracy (GurobiEdit)')
    plt.plot(gurobi_df['Shifted_Epoch'], gurobi_df['Val_acc'], color="green",linestyle='--', label='Val Accuracy (GurobiEdit)')

    eval_epoch = train_end_epoch
    plt.plot(eval_epoch, gurobi_eval_df['Train_acc'].iloc[0], 'b*', markersize=12, label='Gurobi Eval Train Acc')
    plt.plot(eval_epoch, gurobi_eval_df['Val_acc'].iloc[0], 'g*', markersize=12, label='Gurobi Eval Val Acc')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(train_df['Epoch'], train_df['Train_loss'], color="blue", label='Train Loss (Train Phase)')
    plt.plot(train_df['Epoch'], train_df['Val_loss'], color="green", label='Val Loss (Train Phase)')

    plt.plot(gurobi_df['Shifted_Epoch'], gurobi_df['Train_loss'], color="blue", linestyle='--', label='Train Loss (GurobiEdit)')
    plt.plot(gurobi_df['Shifted_Epoch'], gurobi_df['Val_loss'], color="green", linestyle='--', label='Val Loss (GurobiEdit)')

    eval_epoch = train_end_epoch
    plt.plot(eval_epoch, gurobi_eval_df['Train_loss'].iloc[0], 'b*', markersize=12, label='Gurobi Eval Train Loss')
    plt.plot(eval_epoch, gurobi_eval_df['Val_loss'].iloc[0], 'g*', markersize=12, label='Gurobi Eval Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_folder}/{dataset_name}_{run_id}.pdf')
    # plt.show()


def ResultAllFile(Test):
    folder_path = f"Stats/{Test}_CrossVal_All"
    fig_folder = f"Figures/{Test}_CrossVal_All"
    Stats_folder = "Stats"
    # if not os.path.exists(fig_folder):
    #     os.makedirs(fig_folder)
    log_files = [f for f in os.listdir(folder_path) if f.endswith('_log.csv') and "gurobi" not in f]
    Stats_summary = pd.DataFrame()
    for file in log_files:
        dataset_name = file.split('_log')[0]
        file_path = os.path.join(folder_path, file)
        history = pd.read_csv(file_path)
        dataset_stats_summary = pd.DataFrame()
        for run_id in history['Run'].unique():
            run_history = history[history['Run'] == run_id]
            if not run_history.empty:
                # plot_training_history(run_history, dataset_name, run_id, fig_folder)
                summary_df = SummarizeStatsPerDataset(run_history, dataset_name, run_id)
                dataset_stats_summary = pd.concat([dataset_stats_summary, summary_df], ignore_index=True)
                
        if not dataset_stats_summary.empty:
            
            avg_summary = pd.DataFrame({
                'Dataset': [dataset_name],
                'Run ID': "Average",
                'S1_Train_Acc': dataset_stats_summary["S1_Train_Acc"].mean(),
                'S1_Train_Loss': dataset_stats_summary["S1_Train_Loss"].mean(),
                'S1_Test_Acc': dataset_stats_summary["S1_Test_Acc"].mean(),
                'S1_Test_Loss': dataset_stats_summary["S1_Test_Loss"].mean(),
                'S2_Train_Acc': dataset_stats_summary["S2_Train_Acc"].mean() if 'S2_Train_Acc' in dataset_stats_summary else None,
                'S2_Train_Loss': dataset_stats_summary["S2_Train_Loss"].mean() if 'S2_Train_Loss' in dataset_stats_summary else None,
                'S2_Test_Acc': dataset_stats_summary["S2_Test_Acc"].mean() if 'S2_Test_Acc' in dataset_stats_summary else None,
                'S2_Test_Loss': dataset_stats_summary["S2_Test_Loss"].mean() if 'S2_Test_Loss' in dataset_stats_summary else None,
                'S3_Start_Train_Acc': dataset_stats_summary["S3_Start_Train_Acc"].mean(),
                'S3_Start_Train_Loss': dataset_stats_summary["S3_Start_Train_Loss"].mean(),
                'S3_Start_Test_Acc': dataset_stats_summary["S3_Start_Test_Acc"].mean(),
                'S3_Start_Test_Loss': dataset_stats_summary["S3_Start_Test_Loss"].mean(),
                'S3_Train_Acc': dataset_stats_summary["S3_Train_Acc"].mean(),
                'S3_Train_Loss': dataset_stats_summary["S3_Train_Loss"].mean(),
                'S3_Test_Acc': dataset_stats_summary["S3_Test_Acc"].mean(),
                'S3_Test_Loss': dataset_stats_summary["S3_Test_Loss"].mean()
            })
            
            dataset_stats_summary = pd.concat([dataset_stats_summary, avg_summary], ignore_index=True)
        Stats_summary = pd.concat([Stats_summary, dataset_stats_summary], ignore_index=True)

    Stats_summary.to_excel(f'{Stats_folder}/ResultAllFile_{Test}.xlsx', index=False)              
        

def SummarizeAllFiles(Test):
    results = pd.read_excel(f"Stats/ResultAllFile_{Test}.xlsx")

    dataset_names = results['Dataset'].unique()
    summary_df = pd.DataFrame()
    all_diffs = pd.DataFrame()

    for dataset_name in dataset_names:
        dataset_results = results[results['Dataset'] == dataset_name].copy()
        dataset_results["TrainDiff"] = dataset_results["S3_Train_Acc"] - dataset_results["S2_Train_Acc"]
        dataset_results["TestDiff"] = dataset_results["S3_Test_Acc"] - dataset_results["S2_Test_Acc"]

        summary_df = pd.concat([summary_df, dataset_results[dataset_results["Run ID"] == "Average"]], ignore_index=True)
        all_diffs = pd.concat([all_diffs, dataset_results], ignore_index=True)

    summary_df.to_excel(f"Stats/Summary_{Test}.xlsx", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.boxplot(data=all_diffs, x="Dataset", y="TrainDiff", ax=axes[0])
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title("Train Accuracy Difference (S3 - S2)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=all_diffs, x="Dataset", y="TestDiff", ax=axes[1])
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title("Test Accuracy Difference (S3 - S2)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"Figures/Boxplot_{Test}.pdf")
    plt.close()

def SummarizeAllFiles_NoReTraining(Test):
    results = pd.read_excel(f"Stats/ResultAllFile_{Test}.xlsx")

    dataset_names = results['Dataset'].unique()
    all_diffs = pd.DataFrame()

    for dataset_name in dataset_names:
        dataset_results = results[results['Dataset'] == dataset_name].copy()
        dataset_results["TrainDiff"] = dataset_results["S3_Start_Train_Acc"] - dataset_results["S1_Train_Acc"]
        dataset_results["TestDiff"] = dataset_results["S3_Start_Test_Acc"] - dataset_results["S1_Test_Acc"]

        all_diffs = pd.concat([all_diffs, dataset_results], ignore_index=True)

    fig, axes = plt.subplots(1, 1, figsize=(16, 6), sharey=True)


    sns.boxplot(data=all_diffs, x="Dataset", y="TestDiff", ax=axes)
    axes.axhline(0, color='black', linestyle='--', linewidth=1)
    axes.set_title("Test Accuracy Difference (AfterGurobi - BeforeGurobi)")
    axes.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"Figures/Boxplot_{Test}_NoReTraining.pdf")
    plt.close()

if __name__ == "__main__":
    # FillStatsFileWithTrain()
    ResultAllFile("RAF")
    ResultAllFile("RAB")
    SummarizeAllFiles("RAF")
    SummarizeAllFiles("RAB")
    SummarizeAllFiles_NoReTraining("RAF")
    SummarizeAllFiles_NoReTraining("RAB")

