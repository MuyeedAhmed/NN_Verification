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
import argparse
import numpy as np
from Utils.TrainModel import TrainModel
from Utils.GetModelsDatasets import GetDataset, GetModel, GetHparams


def write_stage_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pass_test_loader = False

    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--training_type", default="Regular")  # Regular (ERM), AWP, SAM, RWP
    parser.add_argument("--method", default="S")         # TAGD, TAGDW, HTA, CMC, S
    parser.add_argument("--save_checkpoint", default="N")
    parser.add_argument("--misclassification_count", type=int, default=0)
    parser.add_argument("--cmc_type", default="")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--input_type", default="t")  # v/t
    parser.add_argument("--init_epochs", type=int, default=300, help="Max epochs for Model_init (until convergence)")
    parser.add_argument("--extra_epochs", type=int, default=50, help="Extra epochs run standalone after Model_init converges -> Model_init_50")
    parser.add_argument("--cmc_resume_epochs", type=int, default=100, help="Max epochs to resume training after CMC (until convergence) -> Model_init_CMC_AGAIN_UNTIL_CONVERGENCE")
    parser.add_argument("--cmc_extra_epochs", type=int, default=50, help="Extra epochs after the post-CMC convergence -> Model_init_CMC_AGAIN_UNTIL_CONVERGENCE_50")
    parser.add_argument("--gurobi_samples", type=int, default=1000, help="Number of samples used to build the CMC MILP (subset size ablation)")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    training_type = args.training_type
    method = args.method
    save_checkpoint = args.save_checkpoint
    misclassification_count = args.misclassification_count
    cmc_type = args.cmc_type
    i = args.run_id
    input_type = args.input_type

    if training_type not in ["Regular", "S", "ERM", "AWP", "SAM", "RWP"]:
        print(f"Unknown training type: {training_type}. Exiting.")
        sys.exit(1)

    if training_type == "Regular":
        # TrainModel.checkpoint_paths() maps anything that isn't AWP/SAM/RWP to "ERM";
        # normalize here so the paths built in this file (gurobi_checkpoint_dir, the
        # ./checkpoints_{training_type}_CO makedirs below) and the ones RunGurobi.py
        # reads from checkpoints_{training_type}/... agree with where TrainModel
        # actually saves the checkpoint.
        training_type = "ERM"

    if save_checkpoint == "N" or (method == "CMC" or method == "TAGD" or method == "TAGDW" or method == "HTA"):
            os.makedirs(f"./checkpoints_{training_type}/{dataset_name}_CO", exist_ok=True)
            from Utils.RunGurobi import MILP

    if method == "TAGD" or method == "TAGDW" or method == "HTA":
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")
        n_samples_gurobi = -1
        misclassification_count = 0
        cmc_type = ""
    elif method == "CMC":
        n_samples_gurobi = args.gurobi_samples
    elif method == "S":
        n_samples_gurobi = 0
        misclassification_count = 0
        cmc_type = ""
    if input_type == "v":
        n_samples_gurobi = 100

    print(f'Using device: {device}, dataset: {dataset_name}, training: {training_type}, method: {method}, input: {input_type}')

    BatchSize, optimize, learningRate, scheduler_type = GetHparams(dataset_name)

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

    stages_csv_path = "Stats/Summary_Stages.csv"

    def stage_row(stage, checkpoint_path, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
        return {
            "Dataset": dataset_name,
            "Run": i,
            "Training_Type": training_type,
            "Method": method,
            "CMC_Type": cmc_type,
            "Misclassification_Count": int(misclassification_count),
            "Gurobi_Samples": int(n_samples_gurobi),
            "Stage": stage,
            "Checkpoint": checkpoint_path,
            "Train_Loss": float(train_loss),
            "Train_Acc": float(train_acc),
            "Val_Loss": float(val_loss),
            "Val_Acc": float(val_acc),
            "Test_Loss": float(test_loss),
            "Test_Acc": float(test_acc),
        }

    # ---------------------------------------------------------------
    # Stage 1: Model_init - initial training until convergence
    # ---------------------------------------------------------------
    TM = TrainModel(training_type, dataset_name, model_t, train_loader, val_loader, device, test_loader=test_loader, num_epochs=args.init_epochs, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i)
    _, checkpoint_file, _, _ = TM.checkpoint_paths("", co_dir=False)
    if not os.path.exists(checkpoint_file):
        TM.run(early_stopping_patience=25, save_suffix="", co_dir=False)
    else:
        TM.load_model("", co_dir=False)

    train_loss, train_acc = TM.evaluate("Train")
    val_loss, val_acc = TM.evaluate("Val")
    test_loss, test_acc = TM.evaluate("Test")

    results_standalone = {
        "Dataset": dataset_name,
        "Run": i,
        "Training_Type": training_type,
        "Method": "S",
        "Train_Loss": float(train_loss),
        "Train_Acc": float(train_acc),
        "Val_Loss": float(val_loss),
        "Val_Acc": float(val_acc),
        "Test_Loss": float(test_loss),
        "Test_Acc": float(test_acc)
    }

    os.makedirs("Stats", exist_ok=True)
    csv_path_standalone = "Stats/Summary_S.csv"
    write_header = not os.path.exists(csv_path_standalone)
    with open(csv_path_standalone, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_standalone.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results_standalone)

    write_stage_row(stages_csv_path, stage_row("Model_init", checkpoint_file, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

    # ---------------------------------------------------------------
    # Stage 2: Model_init_50 - standalone extra epochs after Model_init
    # converged. Independent of SAM/RWP/ERM/AWP-specific Gurobi/CMC path.
    # ---------------------------------------------------------------
    TM50 = TrainModel(training_type, dataset_name, TM.model, train_loader, val_loader, device, test_loader=test_loader, num_epochs=args.extra_epochs, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train_Extra", run_id=i)
    _, checkpoint_file_50, _, _ = TM50.checkpoint_paths("_50", co_dir=False)
    if not os.path.exists(checkpoint_file_50):
        TM50.run(early_stopping_patience=None, save_suffix="_50", co_dir=False)
    else:
        TM50.load_model("_50", co_dir=False)

    train_loss_50, train_acc_50 = TM50.evaluate("Train")
    val_loss_50, val_acc_50 = TM50.evaluate("Val")
    test_loss_50, test_acc_50 = TM50.evaluate("Test")

    write_stage_row(stages_csv_path, stage_row("Model_init_50", checkpoint_file_50, train_loss_50, train_acc_50, val_loss_50, val_acc_50, test_loss_50, test_acc_50))

    if method == "S" or save_checkpoint == "Y":
        sys.exit()

    results = []

    tm_type = training_type + "_" + method
    if method == "CMC":
        tm_type += "_" + cmc_type

    # CMC checkpoints live in their own subfolder per (cmc_type, misclassification_count,
    # gurobi_samples) combination so ablations (Any/Correct/Incorrect x 1/10/... x subset
    # size) never collide or overwrite each other. TAGD/TAGDW/HTA have no such ablation
    # axes (cmc_type/misclassification_count are forced to ""/0 for them), so they keep
    # the flat ./checkpoints_{tt}/{dataset}_CO/ layout, distinguished by method suffix.
    if method == "CMC":
        co_subdir = f"{cmc_type}_{misclassification_count}_N{n_samples_gurobi}"
        edit_suffix = ""
    else:
        co_subdir = ""
        edit_suffix = f"_{method}"

    TM_after_g = TrainModel(tm_type, dataset_name, model_g, train_loader, val_loader, device, test_loader=test_loader, num_epochs=args.cmc_resume_epochs, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit", run_id=i)
    TM_after_g.load_model("", co_dir=False)
    print(f"Loaded model for run {i} from checkpoint.")

    S1_Train_loss, S1_Train_acc = TM_after_g.evaluate("Train")
    S1_Val_loss, S1_Val_acc = TM_after_g.evaluate("Val")
    S1_Test_loss, S1_Test_acc = TM_after_g.evaluate("Test")
    print("Training and Validation Accuracy before Gurobi optimization:", S1_Train_acc, S1_Val_acc, "Test Accuracy:", S1_Test_acc)

    # ---------------------------------------------------------------
    # Stage 3: Model_init_CMC - immediately after the Gurobi weight edit.
    # Skip the (potentially slow) MILP solve entirely if this exact
    # ablation config has already been solved and checkpointed.
    # ---------------------------------------------------------------
    _, stage3_checkpoint, _, _ = TM_after_g.checkpoint_paths(edit_suffix, co_dir=True, co_subdir=co_subdir)

    if not os.path.exists(stage3_checkpoint):
        TM_after_g.save_fc_inputs("Train")
        TM_after_g.save_fc_inputs("Val")

        print(f"Saved FC inputs for run {i}.")

        X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt", weights_only=True).numpy()
        labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt", weights_only=True).numpy()
        pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt", weights_only=True).numpy()
        X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt", weights_only=True).numpy()
        labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt", weights_only=True).numpy()
        pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt", weights_only=True).numpy()

        loaded_inputs_gurobi = {
            "X_full": X_full,
            "labels_full": labels_full,
            "pred_full": pred_full,
            "X_val": X_val,
            "labels_val": labels_val,
            "pred_val": pred_val,
        }
        print("Training and Validation Accuracy of loaded inputs:",
              (np.mean(loaded_inputs_gurobi["pred_full"] == loaded_inputs_gurobi["labels_full"]),
               np.mean(loaded_inputs_gurobi["pred_val"] == loaded_inputs_gurobi["labels_val"])))

        print("Loaded inputs for Gurobi optimization.")

        time0 = time.time()

        milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, training_type=training_type, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, loaded_inputs=loaded_inputs_gurobi, input_type=input_type)
        if method == "TAGD":
            Gurobi_output = milp_instance.Optimize(Method="LowerConf")
        elif method == "TAGDW":
            Gurobi_output = milp_instance.Optimize(Method="MaxPerturbation")
        elif method == "HTA":
            Gurobi_output = milp_instance.Optimize(Method="HTA")
        elif method == "CMC":
            if cmc_type == "Correct":
                Gurobi_output = milp_instance.Optimize(Method="MisCls_Correct")
            elif cmc_type == "Any":
                Gurobi_output = milp_instance.Optimize(Method="MisCls_Any")
            elif cmc_type == "Incorrect":
                Gurobi_output = milp_instance.Optimize(Method="MisCls_Incorrect")
            else:
                print(f"Unknown CMC type: {cmc_type}. Exiting.")
                sys.exit(1)
        else:
            print(f"Unknown method: {method}. Exiting.")
            sys.exit(1)

        time1 = time.time()
        solve_time = time1 - time0

        if Gurobi_output is None:
            print("Gurobi did not find a solution.")
            sys.exit(1)
        W_new, b_new = Gurobi_output
        TM_after_g.delete_fc_inputs()
        # Gurobi returns float64 numpy arrays; MPS (Apple Silicon) has no float64 support,
        # so match the model's own parameter dtype instead of inheriting numpy's.
        new_W = torch.tensor(W_new, dtype=model_g.classifier.weight.dtype).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b_new, dtype=model_g.classifier.bias.dtype).to(model_g.classifier.bias.device)
        with torch.no_grad():
            TM_after_g.model.classifier.weight.copy_(new_W)
            TM_after_g.model.classifier.bias.copy_(new_b)

        TM_after_g.save_model(0.0, save_suffix=edit_suffix, co_dir=True, co_subdir=co_subdir)
    else:
        print(f"Model_init_CMC checkpoint already exists at {stage3_checkpoint}, skipping the Gurobi solve.")
        TM_after_g.load_model(edit_suffix, co_dir=True, co_subdir=co_subdir)
        solve_time = -1.0

    train_loss, train_acc = TM_after_g.evaluate("Train")
    val_loss, val_acc = TM_after_g.evaluate("Val")
    test_loss, test_acc = TM_after_g.evaluate("Test")

    write_stage_row(stages_csv_path, stage_row("Model_init_CMC", stage3_checkpoint, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

    if method == "CMC":
        # -----------------------------------------------------------
        # Stage 4: Model_init_CMC_AGAIN_UNTIL_CONVERGENCE - resume
        # training after CMC, budget args.cmc_resume_epochs (default 100)
        # -----------------------------------------------------------
        ge_suffix = "_Resume"
        _, ge_checkpoint, _, _ = TM_after_g.checkpoint_paths(ge_suffix, co_dir=True, co_subdir=co_subdir)
        if not os.path.exists(ge_checkpoint):
            TM_after_g.run(early_stopping_patience=25, save_suffix=ge_suffix, co_dir=True, co_subdir=co_subdir)
        else:
            TM_after_g.load_model(ge_suffix, co_dir=True, co_subdir=co_subdir)

        S3_Train_loss, S3_Train_acc = TM_after_g.evaluate("Train")
        S3_Val_loss, S3_Val_acc = TM_after_g.evaluate("Val")
        S3_Test_loss, S3_Test_acc = TM_after_g.evaluate("Test")

        write_stage_row(stages_csv_path, stage_row("Model_init_CMC_AGAIN_UNTIL_CONVERGENCE", ge_checkpoint, S3_Train_loss, S3_Train_acc, S3_Val_loss, S3_Val_acc, S3_Test_loss, S3_Test_acc))

        # -----------------------------------------------------------
        # Stage 5: Model_init_CMC_AGAIN_UNTIL_CONVERGENCE_50 - standalone
        # extra epochs (default 50) after stage 4 converges
        # -----------------------------------------------------------
        TM_after_g_extra = TrainModel(tm_type, dataset_name, TM_after_g.model, train_loader, val_loader, device, test_loader=test_loader, num_epochs=args.cmc_extra_epochs, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit_Extra", run_id=i)
        ge_extra_suffix = "_Resume_Extra"
        _, ge_extra_checkpoint, _, _ = TM_after_g_extra.checkpoint_paths(ge_extra_suffix, co_dir=True, co_subdir=co_subdir)
        if not os.path.exists(ge_extra_checkpoint):
            TM_after_g_extra.run(early_stopping_patience=None, save_suffix=ge_extra_suffix, co_dir=True, co_subdir=co_subdir)
        else:
            TM_after_g_extra.load_model(ge_extra_suffix, co_dir=True, co_subdir=co_subdir)

        S4_Train_loss, S4_Train_acc = TM_after_g_extra.evaluate("Train")
        S4_Val_loss, S4_Val_acc = TM_after_g_extra.evaluate("Val")
        S4_Test_loss, S4_Test_acc = TM_after_g_extra.evaluate("Test")

        write_stage_row(stages_csv_path, stage_row("Model_init_CMC_AGAIN_UNTIL_CONVERGENCE_50", ge_extra_checkpoint, S4_Train_loss, S4_Train_acc, S4_Val_loss, S4_Val_acc, S4_Test_loss, S4_Test_acc))

    else:
        S3_Train_loss, S3_Train_acc = -1, -1
        S3_Val_loss, S3_Val_acc = -1, -1
        S3_Test_loss, S3_Test_acc = -1, -1

    results.append({
        "Dataset": dataset_name,
        "Run": i,
        "Checkpoint": stage3_checkpoint,
        "Training_Type": training_type,
        "Method": method,
        "CMC_Type": cmc_type,
        "Input_Type": input_type,
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
        "Solve_Time": float(solve_time),
    })

    ''' End of the loop - Runs '''

    csv_path = "Stats/Summary.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)
