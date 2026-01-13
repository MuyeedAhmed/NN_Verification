import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader, Subset
import os
import sys
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 3600

class MILP:
    def __init__(self, dataset_name, store_file_name, run_id, n=-1, tol = 1e-5, misclassification_count=0, candidate=0, loaded_inputs=None):
        self.dataset_name = dataset_name
        self.store_file_name = store_file_name
        self.run_id = run_id
        self.n = n
        self.tol = tol
        self.misclassification_count = misclassification_count
        self.candidate = candidate
        self.X_full, self.labels_full, self.pred_full, self.X_val, self.labels_val, self.pred_val = None, None, None, None, None, None

        if loaded_inputs is not None:
            self.X_full, self.labels_full, self.pred_full, self.X_val, self.labels_val, self.pred_val = loaded_inputs['X_full'], loaded_inputs['labels_full'], loaded_inputs['pred_full'], loaded_inputs['X_val'], loaded_inputs['labels_val'], loaded_inputs['pred_val']
        
        self.gurobi_model = gp.Model()

    
    
    def LoadInputs(self):
        if self.X_full is None:
            self.X_full = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_inputs_train.pt").numpy()
            self.labels_full = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_labels_train.pt").numpy()
            self.pred_full = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_preds_train.pt").numpy()
        X_full_size = self.X_full.shape[0]
        if self.n == -1:
            self.X = self.X_full
            self.labels_gt = self.labels_full
            self.pred_checkpoint = self.pred_full
        else:
            np.random.seed(self.candidate*42)
            idx = np.random.choice(X_full_size, size=self.n, replace=False)
            self.X = self.X_full[idx]
            self.labels_gt = self.labels_full[idx]
            self.pred_checkpoint = self.pred_full[idx]

        self.W = torch.load(f"checkpoints/{self.dataset_name}/Run{self.run_id}_classifier_weight.pt", map_location=torch.device('cpu')).numpy()
        self.b = torch.load(f"checkpoints/{self.dataset_name}/Run{self.run_id}_classifier_bias.pt", map_location=torch.device('cpu')).numpy()

        self.Z_target = self.X @ self.W.T + self.b
    
    def PrintShapes(self):
        pred_target = np.argmax(self.Z_target, axis=1)

        print("Size of X:", self.X.shape)
        print("Size of W:", self.W.shape)
        print("Size of b:", self.b.shape)
        print("Mismatch: ", sum(self.pred_checkpoint != pred_target))

    
    
    def Optimize(self, Method = "MisCls_Correct"):
        milp_log_file = f"Stats/{self.dataset_name}_log.csv"
        self.LoadInputs()
        self.PrintShapes()
        
        self.W_offset = self.gurobi_model.addVars(*self.W.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W_offset")
        self.b_offset = self.gurobi_model.addVars(self.W.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b_offset")
        
        # time0 = time.time()

        if Method == "MisCls_Correct":
            self.AddConstraints_MisCls(samples="Correct")
        elif Method == "MisCls_Incorrect":
            self.AddConstraints_MisCls(samples="Incorrect")
        elif Method == "MisCls_Any":
            self.AddConstraints_MisCls(samples="Any")
        elif Method == "LowerConf":
            self.AddConstraints_LowerConf()


        self.gurobi_model.setParam('TimeLimit', timeLimit)
        self.gurobi_model.optimize()

        if self.gurobi_model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and self.gurobi_model.SolCount > 0:
            # time1 = time.time()
            W_off = np.array([[self.W_offset[i, j].X for j in range(self.W.shape[1])] for i in range(self.W.shape[0])])
            b_off = np.array([self.b_offset[i].X for i in range(self.W.shape[0])])
            W_new = (self.W + W_off)
            b_new = (self.b + b_off)
            
            Z2_pred_gurobi = self.X @ W_new.T + b_new
            predictions_gurobi = np.argmax(Z2_pred_gurobi, axis=1)
            misclassified_mask = predictions_gurobi != self.pred_checkpoint
            misclassified = np.sum(misclassified_mask)
            accuracy_gurobi = np.sum(predictions_gurobi == self.labels_gt) / len(self.labels_gt) * 100
            
            if misclassified != self.misclassification_count:
                with open(f"Stats/Error_{self.dataset_name}_gurobi_log_tol.csv", "a") as f:
                    f.write(f"Tol:{self.tol}\nMisclassified: {misclassified}\n")
                # GurobiFlip_Correct(self.dataset_name, self.store_file_name, self.run_id, n=self.n, tol=self.tol+5e-6, misclassification_count=self.misclassification_count)

            print(f"Total misclassified samples: {misclassified}")
            
            if self.X_val is None:
                self.X_val = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_inputs_val.pt").numpy()
                self.labels_val = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_labels_val.pt").numpy()
                # self.pred_val = torch.load(f"checkpoints_inputs/{self.dataset_name}/fc_preds_val.pt").numpy()
            
            Z_val_pred = self.X_val @ W_new.T + b_new
            predictions_val = np.argmax(Z_val_pred, axis=1)
            
            accuracy_val = np.sum(predictions_val == self.labels_val) / len(self.labels_val) * 100
            Z_pred_gurobi_full = self.X_full @ W_new.T + b_new
            predictions_gurobi_full = np.argmax(Z_pred_gurobi_full, axis=1)
            misclassified_mask_full = predictions_gurobi_full != self.pred_full
            misclassified_full = np.sum(misclassified_mask_full)
            accuracy_gurobi_full = np.sum(predictions_gurobi_full == self.labels_full)  / len(self.labels_full) * 100
            with open(self.store_file_name, "a") as f:
                f.write(f"{self.run_id},GurobiComplete_Train,-1,-1,{accuracy_gurobi}\n")
                f.write(f"{self.run_id},GurobiComplete_Val,-1,-1,{accuracy_val}\n")

            if os.path.exists(milp_log_file) == False:
                with open(milp_log_file, "w") as f:
                    f.write("Method,RunID,Candidate,W_offset_sum,b_offset_sum,Objective_value,n,Misclassified,Accuracy_Full,Accuracy_Val,GlobalMisclassified\n")

            with open(milp_log_file, "a") as f:
                f.write(f"{Method},{self.run_id},{self.candidate},{np.sum(np.abs(W_off))},{np.sum(np.abs(b_off))},{self.gurobi_model.ObjVal},{self.n},{misclassified},{accuracy_gurobi_full},{accuracy_val},{misclassified_full}\n")
            # time2 = time.time()
            
            # with open("TimeLogs.txt", "a") as f:
            #     f.write("Time: GurobiOptimize_{}_{}\t{:.2f}\n".format(Method, self.candidate, time1 - time0))
            #     f.write("Time: GurobiEval_{}_{}\t{:.2f}\n".format(Method, self.candidate, time2 - time1))
            # print(f"Gurobi Optimization Time: {time1 - time0:.3f} seconds")
            # print(f"Gurobi Evaluation Time: {time2 - time1:.3f} seconds")

            return [W_new, b_new]
        else:
            print("No solution found.")
            return None



    def AddConstraints_MisCls(self, samples = "Correct"):
        n_samples = len(self.X)
        l1_size = self.W.shape[1]
        layer_size = self.W.shape[0]
    
        misclassified_flags = self.gurobi_model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")
        for s in range(n_samples):
            label_max = int(np.argmax(self.Z_target[s]))
            label_min = int(np.argmin(self.Z_target[s]))
            A1_fixed = self.X[s]
            Z = self.gurobi_model.addVars(layer_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Z_{s}")
            for j in range(layer_size):
                expr = gp.LinExpr()
                for i in range(l1_size):
                    expr += (self.W[j, i] + self.W_offset[j, i]) * A1_fixed[i]
                expr += self.b[j] + self.b_offset[j]
                self.gurobi_model.addConstr(Z[j] == expr)
            
            # For correct or incorrect flips
            isCorrect = (self.labels_gt[s] == label_max)
            if samples == "Any" or ((samples == "Correct") == isCorrect):
                violations = self.gurobi_model.addVars(layer_size, vtype=GRB.BINARY, name=f"violations_{s}")
                for k in range(layer_size):
                    if k != label_max:
                        self.gurobi_model.addConstr((violations[k] == 1) >> (Z[label_max] <= Z[k] - self.tol), name=f"violation_1flip_{s}_{k}")
                        self.gurobi_model.addConstr((violations[k] == 0) >> (Z[label_max] >= Z[k] + self.tol), name=f"violation_0flip_{s}_{k}")
                    else:
                        self.gurobi_model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

                self.gurobi_model.addConstr(gp.quicksum(violations[k] for k in range(layer_size)) >= misclassified_flags[s])
                self.gurobi_model.addConstr(gp.quicksum(violations[k] for k in range(layer_size)) <= (layer_size - 1) * misclassified_flags[s])
            else:
                for k in range(layer_size):
                    if k != label_max:
                        self.gurobi_model.addConstr(Z[label_max] >= Z[k] + self.tol, name=f"violation_0flip_{s}_{k}")
                self.gurobi_model.addConstr(misclassified_flags[s] == 0, name=f"misclassified_flag_{s}")
        
        self.gurobi_model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == self.misclassification_count, name="exactly_one_misclassified")

        abs_W = self.gurobi_model.addVars(*self.W.shape, lb=0, name="abs_W")
        abs_b = self.gurobi_model.addVars(layer_size, lb=0, name="abs_b")

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.gurobi_model.addConstr(abs_W[i, j] >= self.W_offset[i, j])
                self.gurobi_model.addConstr(abs_W[i, j] >= -self.W_offset[i, j])
        for i in range(layer_size):
            self.gurobi_model.addConstr(abs_b[i] >= self.b_offset[i])
            self.gurobi_model.addConstr(abs_b[i] >= -self.b_offset[i])
        
        objective = (
            gp.quicksum(abs_W[i, j] for i in range(self.W.shape[0]) for j in range(self.W.shape[1])) +
            gp.quicksum(abs_b[i] for i in range(layer_size))
        )
        self.gurobi_model.setObjective(objective, GRB.MINIMIZE)
        self.gurobi_model.addConstr(objective >= 0, "ObjectiveLowerBound")
        
    def AddConstraints_LowerConf(self):
        n_samples = len(self.X)
        l1_size = self.W.shape[1]
        layer_size = self.W.shape[0]
    
        max_min_diff = []
        for s in range(n_samples):
            label_max = int(np.argmax(self.Z_target[s]))
            label_min = int(np.argmin(self.Z_target[s]))
            A1_fixed = self.X[s]
            Z = self.gurobi_model.addVars(layer_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Z_{s}")
            for j in range(layer_size):
                expr = gp.LinExpr()
                for i in range(l1_size):
                    expr += (self.W[j, i] + self.W_offset[j, i]) * A1_fixed[i]
                expr += self.b[j] + self.b_offset[j]
                self.gurobi_model.addConstr(Z[j] == expr)
            for k in range(layer_size):
                if k != label_max:
                    self.gurobi_model.addConstr(Z[label_max] >= Z[k] + self.tol, f"Z2_max_{s}_{k}")

            max_min_diff.append(Z[label_max] - Z[label_min])
        objective = gp.quicksum(max_min_diff)
        self.gurobi_model.setObjective(objective, GRB.MINIMIZE)

