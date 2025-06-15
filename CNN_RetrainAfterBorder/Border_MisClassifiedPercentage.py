import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 7200


def GurobiBorder(dataset_name, X_full, labels_full, pred_full, n=-1, tol = 5e-6):
    X_full_size = X_full.shape[0]
    if X_full_size < n:
        return None
    if n == -1:
        X = X_full
        labels = labels_full
        pred = pred_full
    else:
        X = X_full[0:n]
        labels = labels_full[0:n]
        pred = pred_full[0:n]

    W1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_weight.pt", map_location=torch.device('cpu')).cpu().numpy()
    b1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_bias.pt", map_location=torch.device('cpu')).cpu().numpy()
    W2 = torch.load(f"checkpoints/{dataset_name}/classifier_weight.pt", map_location=torch.device('cpu')).cpu().numpy()
    b2 = torch.load(f"checkpoints/{dataset_name}/classifier_bias.pt", map_location=torch.device('cpu')).cpu().numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2

    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    model_g = gp.Model()
    # model_g.setParam("OutputFlag", 1)

    W2_offset = model_g.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
    b2_offset = model_g.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

    Z2_list = []
    max_min_diff = []

    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = model_g.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            model_g.addConstr(Z2[j] == expr)

        for k in range(l2_size):
            if k != label_max:
                model_g.addConstr(Z2[label_max] >= Z2[k] + tol, f"Z2_max_{s}_{k}")

        Z2_list.append(Z2)
        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
    model_g.setObjective(objective, GRB.MINIMIZE)
    model_g.addConstr(objective >= 0, "ObjectiveLowerBound")
    model_g.setParam('TimeLimit', timeLimit)
    model_g.optimize()

    if model_g.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and model_g.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off
        def relu(x): return np.maximum(0, x)
        def softmax(logits):
            e = np.exp(logits - np.max(logits))
            return e / np.sum(e)

        misclassified = 0
        ce_loss_target = 0
        ce_loss_pred = 0

        for i in range(n_samples):
            x = X[i]
            label = int(np.argmax(Z2_target[i]))
            a1 = relu(W1 @ x + b1)
            z2 = W2_new @ a1 + b2_new
            pred = np.argmax(z2)
            if pred != label:
                misclassified += 1

            pred_probs = softmax(z2)
            target_probs = softmax(Z2_target[i])
            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)
        if misclassified > 0:
            with open(f"Stats/B_MisC_Error_{dataset_name}.csv", "a") as f:
                f.write(f"Tol:{tol}\nMisclassified: {misclassified}\n")
            GurobiBorder(dataset_name, n=n, tol=tol+5e-6)
        
        W2_offsets_sum =  np.sum(np.abs(W2_off))
        b2_offsets_sum = np.sum(np.abs(b2_off))
        objective_value = model_g.ObjVal
        Avg_cross_Entropy_loss = ce_loss_target / n_samples
        Avg_cross_Entropy_loss_pred = ce_loss_pred / n_samples

        """ Full Dataset Statistics """
        Z1_full = np.maximum(0, X_full @ W1.T + b1)
        Z2_full_target = Z1_full @ W2.T + b2
        misclassified_full = 0
        for i in range(X_full_size):
            a1 = relu(W1 @ X_full[i] + b1)
            z2 = W2_new @ a1 + b2_new
            pred = np.argmax(z2)
            label = int(np.argmax(Z2_full_target[i]))
            if pred != label:
                misclassified_full += 1
            pred_probs = softmax(z2)
            target_probs = softmax(Z2_full_target[i])
            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)
        
        Avg_cross_Entropy_loss_full = ce_loss_target / X_full_size
        Avg_cross_Entropy_loss_pred_full = ce_loss_pred / X_full_size

        with open(f"Stats/B_MisC_{dataset_name}.csv", "a") as f:
            f.write(f"{X_full_size},{n},{tol},{model_g.Runtime},{objective_value},{W2_offsets_sum},{b2_offsets_sum},{Avg_cross_Entropy_loss},{Avg_cross_Entropy_loss_pred},{Avg_cross_Entropy_loss_full},{Avg_cross_Entropy_loss_pred_full},{misclassified},{misclassified_full}\n")

    
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off

        return [W2_new, b2_new]
    else:
        print("No solution found.")
        return None
        
        


if __name__ == "__main__":
    os.makedirs("Stats", exist_ok=True)
    n_samples_gurobi = -1
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "EMNIST"
    X_full = torch.load(f"checkpoints/{dataset_name}/fc_inputs.pt").numpy()
    labels_full = torch.load(f"checkpoints/{dataset_name}/fc_labels.pt").numpy()
    pred_full = torch.load(f"checkpoints/{dataset_name}/fc_preds.pt").numpy()
    with open(f"Stats/B_MisC_{dataset_name}.csv", "w") as f:
        f.write("DatasetSize,N_sample,Tol,RunTime,Objective Value,W2_offsets_sum,b2_offsets_sum,Avg_cross_Entropy_loss,Avg_cross_Entropy_loss_pred,Avg_cross_Entropy_loss_full,Avg_cross_Entropy_loss_pred_full,misclassified,misclassified_full\n")
    i = 1
    while i <= 200:
        Gurobi_output = GurobiBorder(dataset_name, X_full, labels_full, pred_full, n=i*1000)
        if i >= 10:
            i += 5
        else:
            i += 1
        if Gurobi_output is None:
            break

    