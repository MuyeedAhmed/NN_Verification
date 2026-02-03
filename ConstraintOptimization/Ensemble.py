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
from Utils.GetModelsDatasets import GetDataset, GetModel, GetHparams
from Utils.RunGurobi import MILP



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

    return (correct / total) * 100.0


def load_models(checkpoint_paths, dataset_name, device):
    models = []

    for path in checkpoint_paths:
        model, _ = GetModel(dataset_name, device=device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models


def add_relative_weight_noise_(model, rel=0.01, include_bias=False, eps=1e-12):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (not include_bias) and (name.endswith(".bias") or name == "bias"):
                continue
            s = p.std().clamp_min(eps)
            p.add_(rel * s * torch.randn_like(p))

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
    return loss_sum / total, (correct / total) * 100.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    os.makedirs(f"Stats_Ensemble/", exist_ok=True)
    ''' 
    Initialize parameters 
    '''
    initEpoch = 300
    G_epoch = 50
    
    # total_candidates = 30

    if len(sys.argv) < 3:
        print("Usage: python Ensemble.py <DatasetName> <Method> [<Retrain Y/N>] [<Misclassification Count>] [<N Samples Gurobi>] [<Time Limit>] [Noise Level] [Total Candidates] [Run ID]")
        sys.exit(1)
    dataset_name = sys.argv[1]
    method = sys.argv[2]
    retrain = sys.argv[3] if len(sys.argv) > 3 else "N"
    misclassification_count = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    n_samples_gurobi = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    timeLimit = float(sys.argv[6]) if len(sys.argv) > 6 else 600.0
    noise_level = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
    total_candidates = int(sys.argv[8]) if len(sys.argv) > 8 else total_candidates
    i = int(sys.argv[9]) if len(sys.argv) > 9 else 2

    top_k = 20
    os.makedirs(f"./checkpoints/{dataset_name}_Candidates/", exist_ok=True)

    BatchSize, optimize, learningRate, scheduler_type = GetHparams(dataset_name)
    
    ''' 
    DataLoader and Model 
    '''
    train_dataset, test_dataset = GetDataset(dataset_name)

    train_size = int(len(train_dataset) * 0.8)
    val_size = int(len(train_dataset) * 0.2)
    total_size = train_size + val_size

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
    
    '''
    Run NN Training
    '''
    checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth"
    if os.path.exists(checkpoint_dir) == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i, start_experiment=True)
        TM.log_file = f"NNRunLog/{dataset_name}_Ensemble.csv"
        TM.run()

    '''
    Run Gurobi Optimization and Evaluate Candidates
    '''
    TotalTime0 = time.time()
    TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit", run_id=i)
    TM_after_g.log_file = f"NNRunLog/{dataset_name}_Ensemble.csv"
    
    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_dir)
    else:
        checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    
    TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
    TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded model for run {i} from checkpoint.")

    S1_Train_loss, S1_Train_acc = TM_after_g.evaluate("Train")
    S1_Val_loss, S1_Val_acc = TM_after_g.evaluate("Val")
    S1_Test_loss, S1_Test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

    TM_after_g.save_fc_inputs("Train")
    X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    
    # Add small weight noise
    TM_after_g.model.eval()
    add_relative_weight_noise_(TM_after_g.model, rel=noise_level, include_bias=True)
    print("Added small weight noise to the model.")


    NoiseAdded_Train_loss, NoiseAdded_Train_acc = TM_after_g.evaluate("Train")
    NoiseAdded_Val_loss, NoiseAdded_Val_acc = TM_after_g.evaluate("Val")
    NoiseAdded_Test_loss, NoiseAdded_Test_acc = evaluate_loader(TM_after_g.model, test_loader, device)
    print(f"LOGG:\n After noise - Train Acc: {NoiseAdded_Train_acc:.4f}, Val Acc: {NoiseAdded_Val_acc:.4f}, Test Acc: {NoiseAdded_Test_acc:.4f}")

    TM_after_g.save_fc_inputs("Train")
    TM_after_g.save_fc_inputs("Val")

    print(f"Saved FC inputs for run {i}.")


    X_full_edited = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    # labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    # pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
    labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
    pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()

    loaded_inputs_gurobi = {
        "X_full": X_full,
        "X_full_edited": X_full_edited,
        "labels_full": labels_full,
        "pred_full": pred_full,
        "X_val": X_val,
        "labels_val": labels_val,
        "pred_val": pred_val,
    }

    print("Loaded inputs for Gurobi optimization.")

    results = []        
    for candidate in range(1, total_candidates+1):
        time0 = time.time()
        milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, loaded_inputs=loaded_inputs_gurobi, candidate=candidate, timeLimit=timeLimit)
        Gurobi_output = milp_instance.Optimize(Method=method, optimization_direction="minimize")
        time1 = time.time()
        
        if Gurobi_output is None:
            misclassification_count = misclassification_count - 1
            print("Gurobi did not find a solution.")
            continue

        W2_new, b2_new = Gurobi_output
        # TM_after_g.delete_fc_inputs()
        new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
        with torch.no_grad():
            TM_after_g.model.classifier.weight.copy_(new_W)
            TM_after_g.model.classifier.bias.copy_(new_b)
        
        if retrain == "N":
            gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}_Candidates/Run{i}_checkpoint_{method}_{retrain}_{candidate}.pth"
            torch.save({
                'epoch': TM_after_g.num_epochs,
                'model_state_dict': TM_after_g.model.state_dict(),
                'optimizer_state_dict': TM_after_g.optimizer.state_dict(),
                'scheduler_state_dict': TM_after_g.scheduler.state_dict()
            }, gurobi_checkpoint_dir)
        # else:
        #     TM_after_g.run()
        #     old_path = f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint_GE_{method}.pth" Need to fix checkpoint path
        #     gurobi_checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{method}_{retrain}_{candidate}.pth"
        
        #     os.rename(old_path, gurobi_checkpoint_dir)

        # with open(TM_after_g.log_file, "a") as f:
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
        #     f.write(f"{i},{candidate},Gurobi_Complete_Eval_Test,-1,{test_loss},{test_acc}\n")
        
        train_loss, train_acc = TM_after_g.evaluate("Train")
        val_loss, val_acc = TM_after_g.evaluate("Val")
        test_loss, test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

        result = {
            "Candidate": candidate,
            "Checkpoint": gurobi_checkpoint_dir,
            "Solve_Time": float(time1 - time0),
            "Train_loss": float(train_loss),
            "Train_acc": float(train_acc),
            "Val_loss": float(val_loss),
            "Val_acc": float(val_acc),
            "Test_loss": float(test_loss),
            "Test_acc": float(test_acc),
        }
        csv_path = f"Stats_Ensemble/Candidates_{dataset_name}.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        
        results.append(result)
        print(f"[Run {i} cand {candidate}] val_acc={val_acc:.4f} test_acc={test_acc:.4f} time={time1 - time0:.1f}s")
    
    TM_after_g.delete_fc_inputs()
    if len(results) == 0:
        print("No candidates were successfully optimized by Gurobi.")
        summary_results ={
            "Dataset": dataset_name,
            "Retrain": retrain,
            "Noise_Level": noise_level,
            "N_Samples_Gurobi": n_samples_gurobi,
            "Time_Limit": timeLimit,
            "Method": method,
            "Misclassification_Count": misclassification_count,
            "Time_Taken": float(time.time() - TotalTime0),
            "Train_loss": float(S1_Train_loss),
            "Train_acc": float(S1_Train_acc),
            "Val_loss": float(S1_Val_loss),
            "Val_acc": float(S1_Val_acc),
            "Test_loss": float(S1_Test_loss),
            "Test_acc": float(S1_Test_acc),
            "Ensemble_Test_acc": float(-1),
        }
        summary_path = "Stats_Ensemble/Summary.csv"

        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_results.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(summary_results)
        sys.exit(1)
    
    results_sorted = sorted(
        results,
        key=lambda r: r["Val_acc"],
        reverse=True
    )
    top_k = min(top_k, len(results_sorted))
    top_k_paths = [r["Checkpoint"] for r in results_sorted[:top_k]]

    models = load_models(top_k_paths, dataset_name, device)

    ensemble_acc = ensemble_test_accuracy(
        models=models,
        test_loader=test_loader,
        device=device
    )

    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    # with open(TM_after_g.log_file, "a") as f:
    #     f.write(f"Ensemble of top {top_k} models Test Accuracy: {ensemble_acc:.4f}\n")
    
    summary_results ={
        "Dataset": dataset_name,
        "Retrain": retrain,
        "Noise_Level": noise_level,
        "N_Samples_Gurobi": n_samples_gurobi,
        "Time_Limit": timeLimit,
        "Method": method,
        "Misclassification_Count": misclassification_count,
        "Time_Taken": float(time.time() - TotalTime0),
        "Train_loss": float(S1_Train_loss),
        "Train_acc": float(S1_Train_acc),
        "Val_loss": float(S1_Val_loss),
        "Val_acc": float(S1_Val_acc),
        "Test_loss": float(S1_Test_loss),
        "Test_acc": float(S1_Test_acc),
        "Ensemble_Test_acc": float(ensemble_acc),
    }
    summary_path = "Stats_Ensemble/Summary.csv"

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(summary_results)

    noiseed_results ={
        "Dataset": dataset_name,
        "Noise_Level": noise_level,
        "Train_loss": float(NoiseAdded_Train_loss),
        "Train_acc": float(NoiseAdded_Train_acc),
        "Val_loss": float(NoiseAdded_Val_loss),
        "Val_acc": float(NoiseAdded_Val_acc),
        "Test_loss": float(NoiseAdded_Test_loss),
        "Test_acc": float(NoiseAdded_Test_acc),
    }
    noise_summary_path = "Stats_Ensemble/NoiseAdded_Summary.csv"
    write_header = not os.path.exists(noise_summary_path)
    with open(noise_summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=noiseed_results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(noiseed_results)

