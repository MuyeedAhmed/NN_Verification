import pandas as pd
import matplotlib.pyplot as plt
import os

def summarize_statistics(df, dataset_name, run_id):
    train_df = df[df['Phase'] == 'Train']
    resume_df = df[df['Phase'] == 'Resume']
    gurobi_df = df[df['Phase'] == 'GurobiEdit']
    gurobi_eval_df = df[df['Phase'] == 'Gurobi_Complete_Eval']

    train_end_epoch = train_df['Epoch'].max()
    
    S1_train_acc = train_df['Train_acc'].iloc[-1]
    S1_train_loss = train_df['Train_loss'].iloc[-1]
    S1_val_acc = train_df['Val_acc'].iloc[-1]
    S1_val_loss = train_df['Val_loss'].iloc[-1]
    
    if not resume_df.empty:
        S2_train_acc = resume_df['Train_acc'].iloc[-1]
        S2_train_loss = resume_df['Train_loss'].iloc[-1]
        S2_val_acc = resume_df['Val_acc'].iloc[-1]
        S2_val_loss = resume_df['Val_loss'].iloc[-1]

    S3_train_acc = gurobi_df['Train_acc'].iloc[-1]
    S3_train_loss = gurobi_df['Train_loss'].iloc[-1]
    S3_val_acc = gurobi_df['Val_acc'].iloc[-1]
    S3_val_loss = gurobi_df['Val_loss'].iloc[-1]

    gurobi_eval_acc_train = gurobi_eval_df['Train_acc'].iloc[0]
    gurobi_eval_loss_train = gurobi_eval_df['Train_loss'].iloc[0]
    gurobi_eval_acc_val = gurobi_eval_df['Val_acc'].iloc[0]
    gurobi_eval_loss_val = gurobi_eval_df['Val_loss'].iloc[0]

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Run ID': [run_id],
        'S1_Train_Acc': [S1_train_acc],
        'S1_Train_Loss': [S1_train_loss],
        'S1_Val_Acc': [S1_val_acc],
        'S1_Val_Loss': [S1_val_loss],
        'S2_Train_Acc': [S2_train_acc] if 'S2_train_acc' in locals() else [None],
        'S2_Train_Loss': [S2_train_loss] if 'S2_train_loss' in locals() else [None],
        'S2_Val_Acc': [S2_val_acc] if 'S2_val_acc' in locals() else [None],
        'S2_Val_Loss': [S2_val_loss] if 'S2_val_loss' in locals() else [None],
        'S3_Train_Acc': [S3_train_acc],
        'S3_Train_Loss': [S3_train_loss],
        'S3_Val_Acc': [S3_val_acc],
        'S3_Val_Loss': [S3_val_loss],
        'Gurobi_Eval_Train_Acc': [gurobi_eval_acc_train],
        'Gurobi_Eval_Train_Loss': [gurobi_eval_loss_train],
        'Gurobi_Eval_Val_Acc': [gurobi_eval_acc_val],
        'Gurobi_Eval_Val_Loss': [gurobi_eval_loss_val]
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



def save_training_history(history, file_path):
    """
    Saves the training history to a CSV file.
    Parameters:
    history (History): The training history object returned by model.fit().
    file_path (str): Path to save the CSV file.
    """
    # Convert history to a DataFrame
    history_df = pd.DataFrame(history.history)
    
    # Save the DataFrame to a CSV file
    history_df.to_csv(file_path, index=False)
    print(f"Training history saved to {file_path}")


def main():
    Test = "Test"
    folder_path = "Stats_"+ Test
    fig_folder = "Figures/" + Test
    Stats_folder = "Stats"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
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
                plot_training_history(run_history, dataset_name, run_id, fig_folder)
                summary_df = summarize_statistics(run_history, dataset_name, run_id)
                dataset_stats_summary = pd.concat([dataset_stats_summary, summary_df], ignore_index=True)
        if not dataset_stats_summary.empty:
            avg_summary = pd.DataFrame({
                'Dataset': [dataset_name],
                'Run ID': "Average",
                'S1_Train_Acc': dataset_stats_summary["S1_Train_Acc"].mean(),
                'S1_Train_Loss': dataset_stats_summary["S1_Train_Loss"].mean(),
                'S1_Val_Acc': dataset_stats_summary["S1_Val_Acc"].mean(),
                'S1_Val_Loss': dataset_stats_summary["S1_Val_Loss"].mean(),
                'S2_Train_Acc': dataset_stats_summary["S2_Train_Acc"].mean(),
                'S2_Train_Loss': dataset_stats_summary["S2_Train_Loss"].mean(),
                'S2_Val_Acc': dataset_stats_summary["S2_Val_Acc"].mean(),
                'S2_Val_Loss': dataset_stats_summary["S2_Val_Loss"].mean(),
                'S3_Train_Acc': dataset_stats_summary["S3_Train_Acc"].mean(),
                'S3_Train_Loss': dataset_stats_summary["S3_Train_Loss"].mean(),
                'S3_Val_Acc': dataset_stats_summary["S3_Val_Acc"].mean(),
                'S3_Val_Loss': dataset_stats_summary["S3_Val_Loss"].mean(),
                'Gurobi_Eval_Train_Acc': dataset_stats_summary["Gurobi_Eval_Train_Acc"].mean(),
                'Gurobi_Eval_Train_Loss': dataset_stats_summary["Gurobi_Eval_Train_Loss"].mean(),
                'Gurobi_Eval_Val_Acc': dataset_stats_summary["Gurobi_Eval_Val_Acc"].mean(),
                'Gurobi_Eval_Val_Loss': dataset_stats_summary["Gurobi_Eval_Val_Loss"].mean()
            })
            dataset_stats_summary = pd.concat([dataset_stats_summary, avg_summary], ignore_index=True)
        Stats_summary = pd.concat([Stats_summary, dataset_stats_summary], ignore_index=True)

    # Save the summary statistics to a CSV file
    Stats_summary.to_csv(f'{Stats_folder}/Stats_summary_{Test}.csv', index=False)                
        
    
if __name__ == "__main__":
    main()
    