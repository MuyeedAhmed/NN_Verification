import shutil
import sys
import os

def rename_checkpoints(dataset_name):
    checkpoint_dir = f"./checkpoints/{dataset_name}"
    
    for file in os.listdir(checkpoint_dir):
        if "RAB" in file:
            os.remove(os.path.join(checkpoint_dir, file))
    for run_id in range(1, 6):
        try:
            if os.path.exists(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_Resume.pth"):
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_Resume.pth", f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_0.pth")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias_Resume.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight_Resume.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias_Resume.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight_Resume.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight_0.pt")
                
                os.remove(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint.pth")
                os.remove(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias.pt")
                os.remove(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight.pt")
                os.remove(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias.pt")
                os.remove(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight.pt")
            else:
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight_0.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint.pth", f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_0.pth")
        
            if os.path.exists(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_GE_RAF.pth"):
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_GE_RAF.pth", f"./checkpoints/{dataset_name}/Run{run_id}_full_checkpoint_1.pth")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias_GE_RAF.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_bias_1.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight_GE_RAF.pt", f"./checkpoints/{dataset_name}/Run{run_id}_classifier_weight_1.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias_GE_RAF.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias_1.pt")
                os.rename(f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight_GE_RAF.pt", f"./checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight_1.pt")
        except FileNotFoundError:
            print(f"Files for Run {run_id} not found for renaming in {checkpoint_dir}. Skipping...")

rename_checkpoints(sys.argv[1])
