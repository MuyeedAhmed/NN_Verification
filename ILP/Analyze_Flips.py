import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, l1, l2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(l2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

l1 = 4
l2 = 4
dataset_dir = "../../Dataset"
flipAnyDir = os.path.expanduser("~/Repositories/NN_Verification/FlipPercentage/Stats/Flip_Any_44.csv")
trainA_base = os.path.expanduser("~/Repositories/NN_Verification/FlipPercentage/Weights/FlipPercentage_l44_Flip1/TrainA")
trainC_base = os.path.expanduser("~/Repositories/NN_Verification/FlipPercentage/Weights/FlipPercentage_l44_Flip1/TrainC")
tsne_dir_base = "Stats/tsne_plots"
os.makedirs(tsne_dir_base, exist_ok=True)
summary_rows = []

flipAnyDf = pd.read_csv(flipAnyDir)
for idx, row in flipAnyDf.iterrows():
    dataset_file = row['Dataset']
    dataset_name = dataset_file.replace(".csv", "")
    mismatch = row['Mismatch']
    if mismatch != 1:
        continue
    print("================================================================================")
    print(f"Analyzing {dataset_name}")

    trainA_dir = os.path.join(trainA_base, dataset_name)
    trainC_dir = os.path.join(trainC_base, dataset_name)
    modelA_path = os.path.join(trainA_dir, "model.pth")
    modelC_path = os.path.join(trainC_dir, "model.pth")
    predsA_path = os.path.join(trainA_dir, "train_preds.npy")
    if not (os.path.exists(modelA_path) and os.path.exists(modelC_path) and os.path.exists(predsA_path)):
        print(f"Skipping {dataset_name}: missing models or predictions.")
        continue

    df = pd.read_csv(os.path.join(dataset_dir, dataset_file))
    X_full = df.iloc[:, :-1].values
    y_gt_full = df.iloc[:, -1].values
    if not (set(np.unique(y_gt_full)) <= {0, 1}):
        continue

    # === Split to train/val and scale as in training code ===
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_gt_full, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # === Gurobi subset selection: sampled from train, like in coworker code ===
    y_train_pred = np.round(np.load(predsA_path))
    X_train_gb, _, y_train_pred_gb, _ = train_test_split(X_train_scaled, y_train_pred, test_size=0.9, random_state=42)

    # --- Load models ---
    modelA = BinaryClassifier(X_train_scaled.shape[1], l1, l2)
    modelA.load_state_dict(torch.load(modelA_path))
    modelA.eval()
    modelC = BinaryClassifier(X_train_scaled.shape[1], l1, l2)
    modelC.load_state_dict(torch.load(modelC_path))
    modelC.eval()

    # --- Gurobi subset prediction ---
    X_train_gb_tensor = torch.tensor(X_train_gb, dtype=torch.float32)
    with torch.no_grad():
        gb_preds_old = modelA(X_train_gb_tensor).numpy().flatten()
        gb_preds_new = modelC(X_train_gb_tensor).numpy().flatten()
    gb_preds_old_binary = np.round(gb_preds_old)
    gb_preds_new_binary = np.round(gb_preds_new)
    gb_flipped_idxs = np.where(gb_preds_old_binary != gb_preds_new_binary)[0]

    N_1 = len(X_train_gb)

    # --- Full data prediction (scaled with train scaler!) ---
    X_full_scaled = scaler.transform(X_full)
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_old = modelA(X_full_tensor).numpy().flatten()
        y_pred_new = modelC(X_full_tensor).numpy().flatten()
    y_pred_old_binary = np.round(y_pred_old)
    y_pred_new_binary = np.round(y_pred_new)
    flip_idxs = np.where(y_pred_old_binary != y_pred_new_binary)[0]
    N_2 = len(X_full_scaled)
    k = len(flip_idxs)

    tsne_dir = os.path.join(tsne_dir_base, dataset_name)
    os.makedirs(tsne_dir, exist_ok=True)

    subset_labels = np.zeros(len(X_train_gb), dtype=int)
    if len(gb_flipped_idxs) > 0 and len(X_train_gb) > 1 and X_train_gb.shape[1]>1:
        subset_labels[gb_flipped_idxs[0]] = 1
        tsne_subset = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_train_gb)-1))
        subset_emb = tsne_subset.fit_transform(X_train_gb)
        plt.figure(figsize=(6,5))
        plt.scatter(subset_emb[subset_labels==0,0], subset_emb[subset_labels==0,1], color='gray', label="Subset (unchanged)")
        plt.scatter(subset_emb[subset_labels==1,0], subset_emb[subset_labels==1,1], color='red', label="Flipped (subset)", s=140, marker='*', edgecolor='black', linewidth=1.5)
        plt.title(f"{dataset_name}: Gurobi Subset (Red Star = Flipped)")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/subset_flipped.png")
        plt.close()
    else:
        print(f"{dataset_name}: No flipped points in Gurobi subset, skipping subset t-SNE.")

    # --- 2. t-SNE plot of full data, highlight all flips ---
    if N_2 > 1 and X_full_scaled.shape[1] > 1:
        full_flipped = (y_pred_old_binary != y_pred_new_binary).astype(int)
        tsne_full = TSNE(n_components=2, random_state=42, perplexity=min(30, N_2-1))
        full_emb = tsne_full.fit_transform(X_full_scaled)
        plt.figure(figsize=(6,5))
        plt.scatter(full_emb[full_flipped == 0, 0], full_emb[full_flipped == 0, 1], color='gray', label="Full (unchanged)", alpha=0.5, s=40, zorder=1)
        plt.scatter(full_emb[full_flipped == 1, 0], full_emb[full_flipped == 1, 1], color='blue', label="Flipped (all)", alpha=0.95, s=70, edgecolor='black', linewidth=0.5, zorder=2)
        plt.title(f"{dataset_name}: Full Data (Blue = Flipped)")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/full_flipped.png")
        plt.close()

    # --- 3. Plot change in loss for flipped points ---
    avg_flip_diff = median_flip_diff = avg_flip_diff_rel = median_flip_diff_rel = np.nan
    if len(flip_idxs) > 0:
        pred_probs_old = y_pred_old[flip_idxs]
        pred_probs_new = y_pred_new[flip_idxs]
        true_labels = y_gt_full[flip_idxs]
        epsilon = 1e-8
        loss_old = - (true_labels * np.log(pred_probs_old + epsilon) +
                      (1 - true_labels) * np.log(1 - pred_probs_old + epsilon))
        loss_new = - (true_labels * np.log(pred_probs_new + epsilon) +
                      (1 - true_labels) * np.log(1 - pred_probs_new + epsilon))
        loss_diff = loss_new - loss_old
        loss_diff_rel = (loss_new - loss_old) / (np.abs(loss_old) + epsilon)
        avg_flip_diff = np.mean(loss_diff)
        median_flip_diff = np.median(loss_diff)
        avg_flip_diff_rel = np.mean(loss_diff_rel)
        median_flip_diff_rel = np.median(loss_diff_rel)

        num_bins = 20
        all_bins = np.linspace(loss_diff.min(), loss_diff.max(), num_bins + 1)
        pos_diff = loss_diff[loss_diff >= 0]
        neg_diff = loss_diff[loss_diff < 0]
        plt.figure(figsize=(7, 5))
        n_pos, bins_pos, patches_pos = plt.hist(pos_diff, bins=all_bins, alpha=0.85, edgecolor='black', color='#3399ff')
        n_neg, bins_neg, patches_neg = plt.hist(neg_diff, bins=all_bins, alpha=0.85, edgecolor='black', color='#ff6666')
        for rect in patches_neg:
            rect.set_height(-abs(rect.get_height()))
            plt.gca().add_patch(rect)
        plt.axhline(0, color='black', linewidth=1.2)
        legend_handles = [
            mpatches.Patch(color='#3399ff', label="Positive (After > Before)"),
            mpatches.Patch(color='#ff6666', label="Negative (After < Before)")
        ]
        plt.legend(handles=legend_handles)
        plt.title(f"{dataset_name}: Split Histogram of Loss Difference (Flipped Points)")
        plt.xlabel("Loss Difference (After - Before)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/loss_change_flipped_hist.png")
        plt.close()
    else:
        print(f"No points were flipped in {dataset_name}, skipping loss change plot.")

    row = {
        "Dataset": dataset_name,
        "k": k,
        "N1": N_1,
        "N2": N_2,
        "1/N1": 1/N_1 if N_1 > 0 else np.nan,
        "k/N2": k/N_2 if N_2 > 0 else np.nan,
        "average_flip_difference": avg_flip_diff,
        "median_flip_difference": median_flip_diff,
        "average_flip_difference_relative": avg_flip_diff_rel,
        "median_flip_difference_relative": median_flip_diff_rel
    }
    summary_rows.append(row)

flip_stats_df = pd.DataFrame(summary_rows)
flip_stats_df.to_csv("Stats/flip_stats_summary.csv", index=False)
print("\nSaved all flip statistics to Stats/flip_stats_summary.csv")
