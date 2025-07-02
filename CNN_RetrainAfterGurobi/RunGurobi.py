import torch
torch.set_default_dtype(torch.float64)

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
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from medmnist import PathMNIST

from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG


timeLimit = 3600



def GurobiBorder(dataset_name, store_file_name, run_id, n=-1, tol = 1e-5):
    X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    X_full_size = X_full.shape[0]
    if n == -1:
        X = X_full
        labels_gt = labels_full
        pred_checkpoint = pred_full
    else:
        X = X_full[0:n]
        labels_gt = labels_full[0:n]
        pred_checkpoint = pred_full[0:n]

    W1 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight.pt", map_location=torch.device('cpu')).numpy()
    b1 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias.pt", map_location=torch.device('cpu')).numpy()
    W2 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_classifier_weight.pt", map_location=torch.device('cpu')).numpy()
    b2 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_classifier_bias.pt", map_location=torch.device('cpu')).numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2
    
    pred_target = np.argmax(Z2_target, axis=1)
    # print(f"W2 dtype: {W2.dtype}, b2 dtype: {b2.dtype}")
    # print("Size of X:", X.shape)
    # print("Size of W2:", W2.shape)
    # print("Size of b2:", b2.shape)
    print("Mismatch: ", sum(pred_checkpoint != pred_target))

    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    gurobi_model = gp.Model()
    W2_offset = gurobi_model.addVars(*W2.shape, lb=-20, ub=20, name="W2_offset")
    b2_offset = gurobi_model.addVars(l2_size, lb=-20, ub=20, name="b2_offset")

    Z2_list = []
    max_min_diff = []

    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = gurobi_model.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            gurobi_model.addConstr(Z2[j] == expr)

        for k in range(l2_size):
            if k != label_max:
                gurobi_model.addConstr(Z2[label_max] >= Z2[k] + tol, f"Z2_max_{s}_{k}")

        Z2_list.append(Z2)
        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
    gurobi_model.setObjective(objective, GRB.MINIMIZE)
    # gurobi_model.addConstr(objective >= 0, "ObjectiveLowerBound")
    gurobi_model.setParam('TimeLimit', timeLimit)
    gurobi_model.optimize()

    if gurobi_model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and gurobi_model.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])

        W2_new = (W2 + W2_off)
        b2_new = (b2 + b2_off)
        # print(f"W2 dtype: {W2.dtype}, b2 dtype: {b2.dtype}")
        # print(f"W2_new dtype: {W2_new.dtype}, b2_new dtype: {b2_new.dtype}")
        def softmax(x):
            x = x - np.max(x, axis=1, keepdims=True)
            e_x = np.exp(x)
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        A1 = np.maximum(0, X @ W1.T + b1)
        Z2_pred_gurobi = A1 @ W2_new.T + b2_new
        predictions_gurobi = np.argmax(Z2_pred_gurobi, axis=1)
        misclassified_mask = predictions_gurobi != pred_checkpoint
        misclassified = np.sum(misclassified_mask)
        accuracy_gurobi = np.sum(predictions_gurobi == labels_gt) / len(labels_gt) * 100
        
        if misclassified > 0:
            with open(f"Stats/RAB_CrossVal_All/{dataset_name}_gurobi_log_tol.csv", "a") as f:
                f.write(f"Tol:{tol}\nMisclassified: {misclassified}\n")
            GurobiBorder(dataset_name, store_file_name, run_id, n=n, tol=tol+5e-6)

        print(f"Total misclassified samples: {misclassified}")
        with open(f"Stats/RAB_CrossVal_All/{dataset_name}_gurobi_log.csv", "a") as f:
            f.write(f"-----\nRun ID: {run_id}\n\n")
            f.write("-------Weight/Bias Offsets-------\n")
            f.write(f"W2 offsets: {np.sum(np.abs(W2_off))}\n")
            f.write(f"b2 offsets: {np.sum(np.abs(b2_off))}\n")
            f.write(f"Objective value: {gurobi_model.ObjVal}\n")
            f.write("------------------------------------\n\n")
            f.write(f"Misclassified: {misclassified}\n")
        
        X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
        labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
        pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()
        Z1_val = np.maximum(0, X_val @ W1.T + b1)
        Z2_val_pred = Z1_val @ W2_new.T + b2_new
        predictions_val = np.argmax(Z2_val_pred, axis=1)
        accuracy_val = np.sum(predictions_val == labels_val) / len(labels_val) * 100
        
        with open(store_file_name, "a") as f:
            f.write(f"{run_id},GurobiComplete_Train,-1,-1,{accuracy_gurobi}\n")
            f.write(f"{run_id},GurobiComplete_Val,-1,-1,{accuracy_val}\n")

        
        return [W2_new, b2_new]
    else:
        print("No solution found.")
        return None


def GurobiFlip(dataset_name, store_file_name, run_id, n=-1, tol = 1e-5, misclassification_count=1):
    X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    X_full_size = X_full.shape[0]
    if n == -1:
        X = X_full
        labels_gt = labels_full
        pred_checkpoint = pred_full
    else:
        X = X_full[0:n]
        labels_gt = labels_full[0:n]
        pred_checkpoint = pred_full[0:n]

    W1 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_fc_hidden_weight.pt", map_location=torch.device('cpu')).numpy()
    b1 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_fc_hidden_bias.pt", map_location=torch.device('cpu')).numpy()
    W2 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_classifier_weight.pt", map_location=torch.device('cpu')).numpy()
    b2 = torch.load(f"checkpoints/{dataset_name}/Run{run_id}_classifier_bias.pt", map_location=torch.device('cpu')).numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2
    
    pred_target = np.argmax(Z2_target, axis=1)
    # print(f"W2 dtype: {W2.dtype}, b2 dtype: {b2.dtype}")
    print("Size of X:", X.shape)
    print("Size of W2:", W2.shape)
    print("Size of b2:", b2.shape)
    print("Mismatch: ", sum(pred_checkpoint != pred_target))

    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    gurobi_model = gp.Model()
    # W2_offset = gurobi_model.addVars(*W2.shape, lb=-20, ub=20, name="W2_offset")
    # b2_offset = gurobi_model.addVars(l2_size, lb=-20, ub=20, name="b2_offset")
    W2_offset = gurobi_model.addVars(*W2.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W2_offset")
    b2_offset = gurobi_model.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b2_offset")

    Z2_list = []
    misclassified_flags = gurobi_model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")
    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = gurobi_model.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            gurobi_model.addConstr(Z2[j] == expr)
        violations = gurobi_model.addVars(l2_size, vtype=GRB.BINARY, name=f"violations_{s}")
        for k in range(l2_size):
            if k != label_max:
                gurobi_model.addConstr((violations[k] == 1) >> (Z2[label_max] <= Z2[k] - tol), name=f"violation_1flip_{s}_{k}")
                gurobi_model.addConstr((violations[k] == 0) >> (Z2[label_max] >= Z2[k] + tol), name=f"violation_0flip_{s}_{k}")
            else:
                gurobi_model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

        gurobi_model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) >= misclassified_flags[s])
        gurobi_model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) <= (l2_size - 1) * misclassified_flags[s])
    
    gurobi_model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == misclassification_count, name="exactly_one_misclassified")

    abs_W2 = gurobi_model.addVars(*W2.shape, lb=0, name="abs_W2")
    abs_b2 = gurobi_model.addVars(l2_size, lb=0, name="abs_b2")

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            gurobi_model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            gurobi_model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])
    for i in range(l2_size):
        gurobi_model.addConstr(abs_b2[i] >= b2_offset[i])
        gurobi_model.addConstr(abs_b2[i] >= -b2_offset[i])
    
    objective = (
        gp.quicksum(abs_W2[i, j] for i in range(W2.shape[0]) for j in range(W2.shape[1])) +
        gp.quicksum(abs_b2[i] for i in range(l2_size))
    )
    gurobi_model.setObjective(objective, GRB.MINIMIZE)
    gurobi_model.addConstr(objective >= 0, "ObjectiveLowerBound")
    gurobi_model.setParam('TimeLimit', timeLimit)
    gurobi_model.optimize()

    if gurobi_model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and gurobi_model.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])

        W2_new = (W2 + W2_off)
        b2_new = (b2 + b2_off)
        
        def softmax(x):
            x = x - np.max(x, axis=1, keepdims=True)
            e_x = np.exp(x)
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        A1 = np.maximum(0, X @ W1.T + b1)
        Z2_pred_gurobi = A1 @ W2_new.T + b2_new
        predictions_gurobi = np.argmax(Z2_pred_gurobi, axis=1)
        misclassified_mask = predictions_gurobi != pred_checkpoint
        misclassified = np.sum(misclassified_mask)
        accuracy_gurobi = np.sum(predictions_gurobi == labels_gt) / len(labels_gt) * 100
        
        if misclassified != misclassification_count:
            with open(f"Stats/RAF_CrossVal_All/{dataset_name}_gurobi_log_tol.csv", "a") as f:
                f.write(f"Tol:{tol}\nMisclassified: {misclassified}\n")
            GurobiFlip(dataset_name, store_file_name, run_id, n=n, tol=tol+5e-6, misclassification_count=misclassified_count)

        print(f"Total misclassified samples: {misclassified}")
        with open(f"Stats/RAF_CrossVal_All/{dataset_name}_gurobi_log.csv", "a") as f:
            f.write(f"-----\nRun ID: {run_id}\n\n")
            f.write("-------Weight/Bias Offsets-------\n")
            f.write(f"W2 offsets: {np.sum(np.abs(W2_off))}\n")
            f.write(f"b2 offsets: {np.sum(np.abs(b2_off))}\n")
            f.write(f"Objective value: {gurobi_model.ObjVal}\n")
            f.write("------------------------------------\n\n")
            f.write(f"Misclassified: {misclassified}\n")
        
        X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
        labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
        pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()
        Z1_val = np.maximum(0, X_val @ W1.T + b1)
        Z2_val_pred = Z1_val @ W2_new.T + b2_new
        predictions_val = np.argmax(Z2_val_pred, axis=1)
        accuracy_val = np.sum(predictions_val == labels_val) / len(labels_val) * 100
        
        with open(store_file_name, "a") as f:
            f.write(f"{run_id},GurobiComplete_Train,-1,-1,{accuracy_gurobi}\n")
            f.write(f"{run_id},GurobiComplete_Val,-1,-1,{accuracy_val}\n")

        if n != -1:
            A1_full = np.maximum(0, X_full @ W1.T + b1)
            Z2_pred_gurobi_full = A1_full @ W2_new.T + b2_new
            predictions_gurobi_full = np.argmax(Z2_pred_gurobi_full, axis=1)
            misclassified_mask_full = predictions_gurobi_full != pred_full
            misclassified_full = np.sum(misclassified_mask_full)
            accuracy_gurobi_full = np.sum(predictions_gurobi_full == labels_full)  / len(labels_full) * 100
            with open(f"Stats/RAF_CrossVal_All/{dataset_name}_gurobi_log.csv", "a") as f:
                f.write(f"Total samples in full dataset: {X_full_size}\n")
                f.write(f"Total misclassified samples in full dataset: {misclassified_full}\n")
                f.write(f"Accuracy on full dataset: {accuracy_gurobi_full}\n")

        return [W2_new, b2_new]
    else:
        print("No solution found.")
        return None
        
