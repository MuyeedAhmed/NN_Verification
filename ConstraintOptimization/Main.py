import torch
import csv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm
import os
import sys
import time
import random
import numpy as np
from Utils.TrainModel import TrainModel
from Utils.GetModelsDatasets import GetDataset, GetModel


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return loss_sum / total, 100. * correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 300
    G_epoch = 100
    
    if len(sys.argv) <= 3:
        print("Usage: python STC.py <Method> <Dataset_Name> <Save_Checkpoint(Y/N)> <Misclassification_Count> <RAF_Type>")
        sys.exit(1)
    method = sys.argv[1]
    dataset_name = sys.argv[2]
    save_checkpoint = sys.argv[3] if len(sys.argv) > 3 else "N"
    misclassification_count = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    raf_type = sys.argv[5] if len(sys.argv) > 5 else "Any"

    os.makedirs(f"Stats/{method}", exist_ok=True)    
    
    if save_checkpoint == "N":
        from Utils.RunGurobi import MILP

    if method == "RAB":
        n_samples_gurobi = -1
        G_epoch = 0
        misclassification_count = 0
        raf_type = ""
        if dataset_name == "EMNIST":
            n_samples_gurobi = 5000
    elif method == "RAF":
        n_samples_gurobi = 1000
        
    
    if dataset_name == "CIFAR10":
        BatchSize = 128
        optimize = "SGD"
        learningRate = 0.1
        scheduler_type = "MultiStepLR"
    else:
        BatchSize = 64
        optimize = "Adam"
        learningRate = 0.01
        scheduler_type = "CosineAnnealingLR"
    
    
    train_dataset, test_dataset = GetDataset(dataset_name)
    
    # full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = int(len(train_dataset) * 0.8)
    val_size = int(len(train_dataset) * 0.2)
    total_size = train_size + val_size
    total_run = 5

    # for i in range(1, total_run + 1):
    i = 2
    model_t, model_g = GetModel(dataset_name, device=device)

    rng = np.random.default_rng(seed=i*42)
    all_indices = rng.permutation(total_size)

    new_train_indices = all_indices[:train_size]
    new_val_indices = all_indices[train_size:]

    train_subset = Subset(train_dataset, new_train_indices)
    val_subset = Subset(train_dataset, new_val_indices)

    train_loader = DataLoader(train_subset, batch_size=BatchSize, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BatchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    
    checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth"
    gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{method}_{raf_type}_{misclassification_count}.pth"

    if os.path.exists(checkpoint_dir) == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i, start_experiment=True)
        TM.run()
    
    if save_checkpoint == "Y":
        sys.exit()
    ##
    if os.path.exists(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint_GE_{method}.pth"):
        print(f"Checkpoint for run {i} already exists. Skipping Gurobi edit.")
        #loop ## continue
    
    results = []

    TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit", run_id=i)

    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_dir)
    else:
        checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    
    TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
    TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded model for run {i} from checkpoint.")

    TM_after_g.save_fc_inputs("Train")
    TM_after_g.save_fc_inputs("Val")

    print(f"Saved FC inputs for run {i}.")

    X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
    labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
    pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()

    loaded_inputs_gurobi = {
        "X_full": X_full,
        "labels_full": labels_full,
        "pred_full": pred_full,
        "X_val": X_val,
        "labels_val": labels_val,
        "pred_val": pred_val,
    }
    S1_Train_loss, S1_Train_acc = TM_after_g.evaluate("Train")
    S1_Val_loss, S1_Val_acc = TM_after_g.evaluate("Val")
    S1_Test_loss, S1_Test_acc = evaluate_loader(TM_after_g.model, test_loader, device)
    print("Training and Validation Accuracy before Gurobi optimization:", S1_Train_acc, S1_Val_acc)
    print("Training and Validation Accuracy of loaded inputs:", 
          (np.mean(loaded_inputs_gurobi["pred_full"] == loaded_inputs_gurobi["labels_full"]),
           np.mean(loaded_inputs_gurobi["pred_val"] == loaded_inputs_gurobi["labels_val"])))

    print("Loaded inputs for Gurobi optimization.")
    with open(TM_after_g.log_file, "a") as f:
        f.write(f"{method}_{raf_type}_{misclassification_count},,,,,,,\n")
    time0 = time.time()

    milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, loaded_inputs=loaded_inputs_gurobi)
    if method == "RAB":
        Gurobi_output = milp_instance.Optimize(Method="LowerConf")
    elif method == "RAF":
        if raf_type == "Correct":
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Correct")
        elif raf_type == "Any":
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Any")
        elif raf_type == "Incorrect":
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Incorrect")
        else:
            print(f"Unknown RAF type: {raf_type}. Exiting.")
            sys.exit(1)
    else:
        print(f"Unknown method: {method}. Exiting.")
        sys.exit(1)

    time1 = time.time()

    if Gurobi_output is None:
        print("Gurobi did not find a solution.")
        # loop # continue

    W_new, b_new = Gurobi_output
    TM_after_g.delete_fc_inputs()
    new_W = torch.tensor(W_new).to(model_g.classifier.weight.device)
    new_b = torch.tensor(b_new).to(model_g.classifier.bias.device)
    with torch.no_grad():
        TM_after_g.model.classifier.weight.copy_(new_W)
        TM_after_g.model.classifier.bias.copy_(new_b)
        
        
    torch.save({
        'epoch': TM_after_g.num_epochs,
        'model_state_dict': TM_after_g.model.state_dict(),
        'optimizer_state_dict': TM_after_g.optimizer.state_dict(),
        'scheduler_state_dict': TM_after_g.scheduler.state_dict()
    }, gurobi_checkpoint_dir)
    
    train_loss, train_acc = TM_after_g.evaluate("Train")
    val_loss, val_acc = TM_after_g.evaluate("Val")
    test_loss, test_acc = evaluate_loader(TM_after_g.model, test_loader, device)


    with open(TM_after_g.log_file, "a") as f:
        f.write(f"{i},{method},{raf_type},{misclassification_count},Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
        f.write(f"{i},{method},{raf_type},{misclassification_count},Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
        f.write(f"{i},{method},{raf_type},{misclassification_count},Gurobi_Complete_Eval_Test,-1,{test_loss},{test_acc}\n")
    
    
    if method == "RAF":
        TM_after_g.run()
    
        S3_Train_loss, S3_Train_acc = TM_after_g.evaluate("Train")
        S3_Val_loss, S3_Val_acc = TM_after_g.evaluate("Val")
        S3_Test_loss, S3_Test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

    else:
        S3_Train_loss, S3_Train_acc = -1, -1
        S3_Val_loss, S3_Val_acc = -1, -1
        S3_Test_loss, S3_Test_acc = -1, -1

    results.append({
        "Dataset": dataset_name,
        "Run": i,
        "Checkpoint": gurobi_checkpoint_dir,
        "Method": method,
        "RAF_Type": raf_type,
        "Misclassification_Count": int(misclassification_count),
        "S1_Train_loss": float(S1_Train_loss),
        "S1_Train_acc": float(S1_Train_acc),
        "S1_Val_loss": float(S1_Val_loss),
        "S1_Val_acc": float(S1_Val_acc),
        "S1_Test_loss": float(S1_Test_loss),
        "S1_Test_acc": float(S1_Test_acc),
        "S2_Train_loss": float(train_loss),
        "S2_Train_acc": float(train_acc),
        "S2_Val_loss": float(val_loss),
        "S2_Val_acc": float(val_acc),
        "S2_Test_loss": float(test_loss),
        "S2_Test_acc": float(test_acc),
        "S3_Train_loss": float(S3_Train_loss),
        "S3_Train_acc": float(S3_Train_acc),
        "S3_Val_loss": float(S3_Val_loss),
        "S3_Val_acc": float(S3_Val_acc),
        "S3_Test_loss": float(S3_Test_loss),
        "S3_Test_acc": float(S3_Test_acc),
        "Solve_Time": float(time1 - time0),
    })

    ''' End of the loop - Runs '''    

    csv_path = "Stats/Summary.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset","Run","Checkpoint","Method","RAF_Type","Misclassification_Count",
                                    "S1_Train_loss","S1_Train_acc","S1_Val_loss","S1_Val_acc","S1_Test_loss","S1_Test_acc",
                                    "S2_Train_loss","S2_Train_acc","S2_Val_loss","S2_Val_acc","S2_Test_loss","S2_Test_acc",
                                    "S3_Train_loss","S3_Train_acc","S3_Val_loss","S3_Val_acc","S3_Test_loss","S3_Test_acc",
                                    "Solve_Time"])
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

