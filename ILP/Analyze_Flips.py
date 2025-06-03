import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
matplotlib.use('Agg')

# You may need to copy your BinaryClassifier definition here:
class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim, l1, l2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, l1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(l1),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(l2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(l2, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ---- Analysis Parameters ----
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

l1 = 4
l2 = 4
dataset_dir = "../../Dataset"
trainA_base = "Weights/FlipPercentage_l44_10pct/TrainA"
trainC_base = "Weights/FlipPercentage_l44_10pct/TrainC"
tsne_dir_base = "Stats/tsne_plots"
os.makedirs(tsne_dir_base, exist_ok=True)
summary_rows = []

# ---- Main Analysis ----
for dataset_file in os.listdir(dataset_dir):
    if not dataset_file.endswith(".csv"):
        continue
    dataset_name = dataset_file
    dataset_stub = dataset_file.replace(".csv", "")

    print(f"Analyzing {dataset_file}")
    # Paths to models
    trainA_dir = os.path.join(trainA_base, dataset_stub)
    trainC_dir = os.path.join(trainC_base, dataset_stub)
    modelA_path = os.path.join(trainA_dir, "model.pth")
    modelC_path = os.path.join(trainC_dir, "model.pth")
    predsA_path = os.path.join(trainA_dir, "train_preds.npy")
    # Check for required files
    if not (os.path.exists(modelA_path) and os.path.exists(modelC_path) and os.path.exists(predsA_path)):
        print(f"Skipping {dataset_file}: missing models or predictions.")
        continue
    
    df = pd.read_csv(os.path.join(dataset_dir, dataset_file))
    # Load full dataset
    if not (100 <= len(df) <= 400):
            continue
    X = df.iloc[:, :-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_gt = df.iloc[:, -1]
    if not (set(y_gt.unique()) <= {0, 1}):
            continue
    y_train_pred = np.round(np.load(predsA_path))

    X_train_gb, _, y_train_pred_gb, _ = train_test_split(X, y_train_pred, test_size=0.9, random_state=42)

    #subset_indices_path = os.path.join(trainA_dir, "subset_indices.npy")
    #if not os.path.exists(subset_indices_path):
    #    print(f"Skipping {dataset_file}: missing subset_indices.")
    #    continue
    #subset_indices = np.load(subset_indices_path)
    #X_train_gb = X[subset_indices]
    #y_train_pred_gb = y_train_pred[subset_indices]

    # Load models
    modelA = BinaryClassifier(X.shape[1], l1, l2)
    modelA.load_state_dict(torch.load(modelA_path))
    modelA.eval()
    modelC = BinaryClassifier(X.shape[1], l1, l2)
    modelC.load_state_dict(torch.load(modelC_path))
    modelC.eval()

    # Predictions before/after on full data
    with torch.no_grad():
        y_pred_old = modelA(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
        y_pred_old_binary = np.round(y_pred_old)

    with torch.no_grad():
        y_pred_new = modelC(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
        y_pred_new_binary = np.round(y_pred_new)

    

    flip_idxs_path = os.path.join(trainA_dir, "flip_idxs.npy")
    if not os.path.exists(flip_idxs_path):
        print(f"Skipping {dataset_file}: missing flipped_full_indices.")
        continue
    flip_idxs = np.load(flip_idxs_path)
    

    # -- Statistics --
    N_1 = len(X_train_gb)
    N_2 = len(X)
    k = np.sum(y_pred_old_binary != y_pred_new_binary)
    
    #summary_rows.append({
    #    "Dataset": dataset_file,
    #    "1/N1": 1/N_1,
    #    "k/N2": k/N_2
    #})

    # --- Make output dir for this dataset ---
    tsne_dir = os.path.join(tsne_dir_base, dataset_stub)
    os.makedirs(tsne_dir, exist_ok=True)

    # --- 1. t-SNE plot of Gurobi subset, highlight first flipped point ---
    flipped_indices = np.where((y_pred_old_binary != y_pred_new_binary))[0]
    if len(flip_idxs) > 0:
        subset_labels = np.zeros(len(X_train_gb), dtype=int)
        subset_labels[flip_idxs[0]] = 1
        tsne_subset = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_train_gb)-1))
        subset_emb = tsne_subset.fit_transform(X_train_gb)
        plt.figure(figsize=(6,5))
        plt.scatter(subset_emb[subset_labels==0,0], subset_emb[subset_labels==0,1],
                    color='gray', label="Subset (unchanged)")
        plt.scatter(subset_emb[flip_idxs[0],0], subset_emb[flip_idxs[0],1],
                    color='red', label="Flipped (subset)", s=140, marker='*', edgecolor='black', linewidth=1.5)
        plt.title(f"{dataset_file}: Gurobi Subset (Red Star = Flipped)")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/subset_flipped.png")
        plt.close()
    else:
        print(f"{dataset_file}: No flipped points in Gurobi subset, skipping subset t-SNE.")

    # --- 2. t-SNE plot of full data, highlight all flips + subset flip ---
    full_flipped = (y_pred_old_binary != y_pred_new_binary).astype(int)
    tsne_full = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    full_emb = tsne_full.fit_transform(X)

    # Find which index in full corresponds to first flipped in subset
    subset_mask = np.array([np.any(np.all(row == X_train_gb, axis=1)) for row in X])
    subset_indices = np.where(subset_mask)[0]
    if len(flip_idxs) > 0 and len(subset_indices) > flip_idxs[0]:
        flipped_full_idx = subset_indices[flip_idxs[0]]
    else:
        flipped_full_idx = None

    plt.figure(figsize=(6,5))
    plt.scatter(full_emb[full_flipped == 0, 0], full_emb[full_flipped == 0, 1],
                color='gray', label="Full (unchanged)", alpha=0.5, s=40, zorder=1)
    plt.scatter(full_emb[full_flipped == 1, 0], full_emb[full_flipped == 1, 1],
                color='blue', label="Flipped (all)", alpha=0.95, s=70, edgecolor='black', linewidth=0.5, zorder=2)
    if flipped_full_idx is not None:
        plt.scatter(full_emb[flipped_full_idx, 0], full_emb[flipped_full_idx, 1],
                    color='red', label="Subset Flipped", s=220, marker='*', edgecolor='black', linewidth=2.5, zorder=3)
    plt.title(f"{dataset_file}: Full Data (Blue = Flipped, Red Star = Subset Flip)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{tsne_dir}/full_flipped.png")
    plt.close()

    # --- 3. Plot change in loss for flipped points ---
    flipped_indices = np.where(full_flipped == 1)[0]
    if len(flipped_indices) > 0:
        pred_probs_old = y_pred_old[flipped_indices]
        pred_probs_new = y_pred_new[flipped_indices]
        true_labels = y_gt[flipped_indices]
        epsilon = 1e-8
        loss_old = - (true_labels * np.log(pred_probs_old + epsilon) +
                      (1 - true_labels) * np.log(1 - pred_probs_old + epsilon))
        loss_new = - (true_labels * np.log(pred_probs_new + epsilon) +
                      (1 - true_labels) * np.log(1 - pred_probs_new + epsilon))
        plt.figure(figsize=(6, 4))
        plt.plot(loss_old, label="Loss Before", marker='o')
        plt.plot(loss_new, label="Loss After", marker='x')
        plt.title(f"{dataset_file}: Loss Change on Flipped Points")
        plt.xlabel("Flipped Point Index")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/loss_change_flipped.png")
        plt.close()
        #=================================== Histogram next
        pred_probs_old = y_pred_old[flipped_indices]
        pred_probs_new = y_pred_new[flipped_indices]
        true_labels = y_gt[flipped_indices]
        epsilon = 1e-8
        loss_old = - (true_labels * np.log(pred_probs_old + epsilon) +
                    (1 - true_labels) * np.log(1 - pred_probs_old + epsilon))
        loss_new = - (true_labels * np.log(pred_probs_new + epsilon) +
                    (1 - true_labels) * np.log(1 - pred_probs_new + epsilon))
        loss_diff = loss_new - loss_old
        avg_flip_diff = np.mean(loss_diff)
        median_flip_diff = np.median(loss_diff)

        num_bins = 20
        all_bins = np.linspace(loss_diff.min(), loss_diff.max(), num_bins + 1)
        pos_diff = loss_diff[loss_diff >= 0]
        neg_diff = loss_diff[loss_diff < 0]

        plt.figure(figsize=(7, 5))

        # Plot positive
        n_pos, bins_pos, patches_pos = plt.hist(
            pos_diff, bins=all_bins, alpha=0.85, edgecolor='black',
            color='#3399ff'
        )
        # Plot negative (upside down)
        n_neg, bins_neg, patches_neg = plt.hist(
            neg_diff, bins=all_bins, alpha=0.85, edgecolor='black',
            color='#ff6666'
        )
        for rect in patches_neg:
            rect.set_height(-abs(rect.get_height()))
            plt.gca().add_patch(rect)
        plt.axhline(0, color='black', linewidth=1.2)

        # Manual legend
        legend_handles = [
            mpatches.Patch(color='#3399ff', label="Positive (After > Before)"),
            mpatches.Patch(color='#ff6666', label="Negative (After < Before)")
        ]
        plt.legend(handles=legend_handles)

        plt.title(f"{dataset_file}: Split Histogram of Loss Difference (Flipped Points)")
        plt.xlabel("Loss Difference (After - Before)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{tsne_dir}/loss_change_flipped_hist.png")
        plt.close()
    else:
        avg_flip_diff = np.nan
        median_flip_diff = np.nan
        print(f"No points were flipped in {dataset_file}, skipping loss change plot.")
    summary_rows.append({
    "Dataset": dataset_file,
    "1/N1": 1/N_1,
    "k/N2": k/N_2,
    "average_flip_difference": avg_flip_diff,
    "median_flip_difference": median_flip_diff
    })


# --- Save summary statistics ---

flip_stats_df = pd.DataFrame(summary_rows)
flip_stats_df.to_csv("Stats/flip_stats_summary.csv", index=False)
print("\nSaved all flip statistics to Stats/flip_stats_summary.csv")
