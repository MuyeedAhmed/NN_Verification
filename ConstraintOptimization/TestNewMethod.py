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
    n_samples_gurobi = 40


    dataset_name = sys.argv[1]
    method = sys.argv[2]
    misclassification_count = 1

    # os.makedirs(f"Stats/{method}", exist_ok=True)    
    
    from Utils.RunGurobi import MILP

        
        
    
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
    # gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{method}_{cmc_type}_{misclassification_count}.pth"

    if os.path.exists(checkpoint_dir) == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i, start_experiment=True)
        TM.run()
    
    
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
    # with open(TM_after_g.log_file, "a") as f:
    #     f.write(f"{method}_{cmc_type}_{misclassification_count},,,,,,,\n")
    time0 = time.time()

    milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, loaded_inputs=loaded_inputs_gurobi)
    
    Gurobi_output = milp_instance.Optimize(Method="Swap", optimization_direction="minimize")
    # Gurobi_output = milp_instance.Optimize(Method="MisCls_Correct")
    time1 = time.time()

    if Gurobi_output is None:
        print("Gurobi did not find a solution.")
        # loop # continue

    # W_new, b_new = Gurobi_output
    # TM_after_g.delete_fc_inputs()
    # new_W = torch.tensor(W_new).to(model_g.classifier.weight.device)
    # new_b = torch.tensor(b_new).to(model_g.classifier.bias.device)
    # with torch.no_grad():
    #     TM_after_g.model.classifier.weight.copy_(new_W)
    #     TM_after_g.model.classifier.bias.copy_(new_b)
        
