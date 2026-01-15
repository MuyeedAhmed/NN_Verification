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
def ensemble_test_accuracy(models, test_loader, device):
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        probs_sum = None
        for m in models:
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        avg_probs = probs_sum / len(models)
        pred = avg_probs.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / total


def load_models(checkpoint_paths, dataset_name, device):
    models = []

    for path in checkpoint_paths:
        model, _ = GetModel(dataset_name, device=device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models



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
    return loss_sum / total, correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 300
    G_epoch = 0
    
    misclassification_counts = [1, 5, 10, 20]
    top_k = 10
    total_candidates = 20

    method = sys.argv[1] if len(sys.argv) > 1 else "RAB"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "MNIST"
    save_checkpoint = sys.argv[3] if len(sys.argv) > 3 else "N"
    if save_checkpoint == "N":
        from Utils.RunGurobi import MILP

    os.makedirs(f"Stats_Ensemble/", exist_ok=True)

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

    train_size = int(len(train_dataset) * 0.8)
    val_size = int(len(train_dataset) * 0.2)
    total_size = train_size + val_size

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

    if os.path.exists(checkpoint_dir) == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i, start_experiment=True)
        TM.log_file = f"Stats_Ensemble/{dataset_name}_nn_run_log.csv"
        TM.run()
    
    if save_checkpoint == "Y":
        sys.exit()

    results = []

    TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit", run_id=i)
    TM_after_g.log_file = f"Stats_Ensemble/{dataset_name}_nn_run_log.csv"
    
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

    convertVal = True
    if convertVal:
        X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
        labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
        pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()

        loaded_inputs_gurobi = {
            "X_full": X_val,
            "labels_full": labels_val,
            "pred_full": pred_val,
            "X_val": X_val,
            "labels_val": labels_val,
            "pred_val": pred_val,
        }
    else:
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

    print("Loaded inputs for Gurobi optimization.")
    
    S1_Train_loss, S1_Train_acc = TM_after_g.evaluate("Train")
    S1_Val_loss, S1_Val_acc = TM_after_g.evaluate("Val")
    S1_Test_loss, S1_Test_acc = evaluate_loader(TM_after_g.model, test_loader, device)
    
    results.append({
        "Candidate": -1,
        "Checkpoint": checkpoint_dir,
        "Train_loss": float(S1_Train_loss),
        "Train_acc": float(S1_Train_acc),
        "Val_loss": float(S1_Val_loss),
        "Val_acc": float(S1_Val_acc),
        "Test_loss": float(S1_Test_loss),
        "Test_acc": float(S1_Test_acc),
        "Solve_Time": -1.0,
    })
    csv_path = "Stats_Ensemble/Summary.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Candidate","Checkpoint","Train_loss","Train_acc","Val_loss","Val_acc","Test_loss","Test_acc","Solve_Time",])
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)
        
    timeLimit = 600.0
    for candidate in range(total_candidates):
        time0 = time.time()
        misclassification_count = misclassification_counts[candidate % len(misclassification_counts)]
        if candidate % 3 == 0:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=candidate, loaded_inputs=loaded_inputs_gurobi, timeLimit=timeLimit)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Correct")
        elif candidate % 3 == 1:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=candidate, loaded_inputs=loaded_inputs_gurobi, timeLimit=timeLimit)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Incorrect")
        elif candidate % 3 == 2:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=candidate, loaded_inputs=loaded_inputs_gurobi, timeLimit=timeLimit)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Any")
        # else:
        #     milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=-1, tol=1e-5, candidate=candidate, loaded_inputs=loaded_inputs_gurobi, timeLimit=timeLimit)
        #     Gurobi_output = milp_instance.Optimize(Method="LowerConf")

        time1 = time.time()
        
        if Gurobi_output is None:
            print("Gurobi did not find a solution.")
            continue

        W2_new, b2_new = Gurobi_output
        # TM_after_g.delete_fc_inputs()
        new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
        with torch.no_grad():
            TM_after_g.model.classifier.weight.copy_(new_W)
            TM_after_g.model.classifier.bias.copy_(new_b)

        # gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{candidate}.pth"
        # torch.save({
        #     'epoch': TM_after_g.num_epochs,
        #     'model_state_dict': TM_after_g.model.state_dict(),
        #     'optimizer_state_dict': TM_after_g.optimizer.state_dict(),
        #     'scheduler_state_dict': TM_after_g.scheduler.state_dict()
        # }, gurobi_checkpoint_dir)
        
        # train_loss, train_acc = TM_after_g.evaluate("Train")
        # val_loss, val_acc = TM_after_g.evaluate("Val")
        # test_loss, test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

        # with open(TM_after_g.log_file, "a") as f:
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Test,-1,{test_loss},{test_acc}\n")
        
        TM_after_g.run()

        old_path = f"./checkpoints/{self.dataset_name}/Run{self.run_id}_full_checkpoint_GE_RAF.pth"
        gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{candidate}.pth"
        
        os.rename(old_path, gurobi_checkpoint_dir)

        train_loss, train_acc = TM_after_g.evaluate("Train")
        val_loss, val_acc = TM_after_g.evaluate("Val")
        test_loss, test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

        results.append({
            "Candidate": candidate,
            "Checkpoint": gurobi_checkpoint_dir,
            "Train_loss": float(train_loss),
            "Train_acc": float(train_acc),
            "Val_loss": float(val_loss),
            "Val_acc": float(val_acc),
            "Test_loss": float(test_loss),
            "Test_acc": float(test_acc),
            "Solve_Time": float(time1 - time0),
        })
        print(f"[Run {i} cand {candidate}] val_acc={val_acc:.4f} test_acc={test_acc:.4f} time={time1 - time0:.1f}s")
    
    TM_after_g.delete_fc_inputs()
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Candidate","Checkpoint","Train_loss","Train_acc","Val_loss","Val_acc","Test_loss","Test_acc","Solve_Time",])
        for row in results:
            writer.writerow(row)

    results_sorted = sorted(
        results,
        key=lambda r: r["Val_acc"],
        reverse=True
    )

    top_k_paths = [r["Checkpoint"] for r in results_sorted[:top_k]]

    models = load_models(top_k_paths, dataset_name, device)

    ensemble_acc = ensemble_test_accuracy(
        models=models,
        test_loader=test_loader,
        device=device
    )

    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")