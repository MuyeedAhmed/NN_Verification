import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# import matplotlib
# matplotlib.use('Agg')

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

def main():
    l1 = 4
    l2 = 4
    dataset_dir = "../../Dataset"
    Test = "FP_l44_F1_10Start"
    StatsFile = f"Stats/{Test}.csv"
    trainA_base = f"Weights/{Test}/TrainA"
    trainC_base = f"Weights/{Test}/TrainC"
    # tsne_dir_base = "Figs/Tsne/"
    # hist_dir_base = "Figs/Line/"
    # os.makedirs(tsne_dir_base, exist_ok=True)
    # os.makedirs(hist_dir_base, exist_ok=True)

    Stats = pd.read_csv(StatsFile)
    Stats["F1"] = None
    # Stats["Accuracy"] = None

    for idx, row in Stats.iterrows():
        dataset_file = row['Dataset']
        dataset_name = dataset_file.replace(".csv", "")
        # if os.path.exists(f"{tsne_dir_base}/{dataset_name}.pdf"):
        #     print("TSNE plot already exists for dataset:", dataset_name)
        #     continue
        if dataset_name != "appendicitis":
            continue
        # mismatch = row['Mismatch']
        # if mismatch != 1:
        #     print("Mismatch not equal to 1 for dataset:", dataset_name)
        #     continue
        trainA_dir = os.path.join(trainA_base, dataset_name)
        trainC_dir = os.path.join(trainC_base, dataset_name)
        modelA_path = os.path.join(trainA_dir, "model.pth")
        modelC_path = os.path.join(trainC_dir, "model.pth")
        predsA_path = os.path.join(trainA_dir, "train_preds.npy")
        
        if os.path.exists(modelC_path) == False:
            print("Model C not found for dataset:", dataset_name)
            continue

        df = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        X_full = df.iloc[:, :-1].values
        y_gt_full = df.iloc[:, -1].values

        X_train, X_val, y_train, y_val = train_test_split(X_full, y_gt_full, test_size=0.1, random_state=(int(row['Seed']/10)*42))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        y_train_pred = np.round(np.load(predsA_path))

        Stats.at[idx, 'F1'] = f1_score(y_train, y_train_pred)
        # Stats.at[idx, 'Accuracy'] = accuracy_score(y_train, y_train_pred)

        # indices = np.arange(len(X_train_scaled))

        # X_train_gb, _, y_train_pred_gb, _, gb_orig_indices, _ = train_test_split(X_train_scaled, y_train_pred, indices, test_size=0.9, random_state=(int(row['Seed']/10)*42))

        # modelA = BinaryClassifier(X_train_scaled.shape[1], l1, l2)
        # modelA.load_state_dict(torch.load(modelA_path))
        # modelA.eval()
        # modelC = BinaryClassifier(X_train_scaled.shape[1], l1, l2)
        # modelC.load_state_dict(torch.load(modelC_path))
        # modelC.eval()

        # X_train_gb_tensor = torch.tensor(X_train_gb, dtype=torch.float32)
        # with torch.no_grad():
        #     gb_preds_old = modelA(X_train_gb_tensor).numpy().flatten()
        #     gb_preds_new = modelC(X_train_gb_tensor).numpy().flatten()
        # gb_preds_old_binary = np.round(gb_preds_old)
        # gb_preds_new_binary = np.round(gb_preds_new)
        # gb_flipped_idxs = np.where(gb_preds_old_binary != gb_preds_new_binary)[0]
        
        # flipped_idx_in_subset = gb_orig_indices[gb_flipped_idxs]

        # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        # with torch.no_grad():
        #     y_pred_old = modelA(X_train_tensor).numpy().flatten()
        #     y_pred_new = modelC(X_train_tensor).numpy().flatten()
        # y_pred_old_binary = np.round(y_pred_old)
        # y_pred_new_binary = np.round(y_pred_new)
        # flip_idxs = np.where(y_pred_old_binary != y_pred_new_binary)[0]
        
        # gb_flipped_orig_idxs = gb_orig_indices[gb_flipped_idxs]
        # flipped_array = np.full_like(y_train_pred, fill_value=-1)
        # flipped_array[gb_flipped_orig_idxs] = gb_flipped_orig_idxs
        # full_flipped = (y_pred_old_binary != y_pred_new_binary).astype(int)
    
        # """TSNE - prep"""
        # if not isinstance(full_flipped, np.ndarray) or full_flipped.dtype != bool:
        #     full_flipped = full_flipped.astype(bool)

        # red_mask = np.zeros_like(full_flipped, dtype=bool)
        # red_mask[gb_orig_indices[gb_flipped_idxs]] = True
        # black_mask = np.zeros_like(full_flipped, dtype=bool)
        # black_mask[gb_orig_indices] = True
        # black_mask[gb_orig_indices[gb_flipped_idxs]] = False
        # blue_mask = full_flipped & ~red_mask
        # grey_mask = ~(black_mask | red_mask | blue_mask)
        # try:
        #     tsne_full = TSNE(n_components=2, random_state=42, perplexity=min(30, X_train_scaled.shape[0]))
        #     full_emb = tsne_full.fit_transform(X_train_scaled)
        # except Exception as e:
        #     continue
        # """TSNE - plot"""
        # plt.figure(figsize=(6,5))
        # plt.scatter(full_emb[grey_mask, 0], full_emb[grey_mask, 1], color='gray', label="Unchanged", alpha=0.5, s=40, zorder=1)
        # plt.scatter(full_emb[black_mask, 0], full_emb[black_mask, 1], color='black', label="GB subset", alpha=0.8, s=40, marker='*', zorder=2)
        # plt.scatter(full_emb[blue_mask, 0], full_emb[blue_mask, 1], color='blue', label="Flipped (Full)", alpha=0.8, s=40, zorder=3)
        # plt.scatter(full_emb[red_mask, 0], full_emb[red_mask, 1], color='red', label="Flipped (GB subset)", s=80, marker='*', edgecolor='black', linewidth=1.5)
        # plt.title(dataset_name)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"{tsne_dir_base}/{dataset_name}.pdf")
        # plt.close()

        # """Line Plot - prep"""
        # epsilon = 1e-8
        # loss_old = - (y_train * np.log(y_pred_old + epsilon) +
        #         (1 - y_train) * np.log(1 - y_pred_old + epsilon))
        # loss_new = - (y_train * np.log(y_pred_new + epsilon) +
        #             (1 - y_train) * np.log(1 - y_pred_new + epsilon))
        # loss_diff = loss_new - loss_old
        # sorted_indices = np.argsort(loss_diff)
        # loss_diff_sorted = loss_diff[sorted_indices]
        # full_flipped = np.array(full_flipped)
        # flipped_indices = np.where(full_flipped == 1)[0]
        # flipped_positions = [np.where(sorted_indices == i)[0][0] for i in flipped_indices]
        # gb_flipped_global_idx = gb_orig_indices[gb_flipped_idxs[0]]
        # gb_flipped_sorted_pos = int(np.where(sorted_indices == gb_flipped_global_idx)[0][0])

        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_diff_sorted, label="Loss Difference (sorted)", color='blue')
        # plt.scatter(flipped_positions, loss_diff_sorted[flipped_positions],
        #             color='red', label="Flipped Points", zorder=3)
        # plt.scatter([gb_flipped_sorted_pos], [loss_diff_sorted[gb_flipped_sorted_pos]],
        #         color='green', label="GB Flipped Point", zorder=4, s=80, edgecolor='black')
        # plt.axhline(0, color='black', linestyle='--', linewidth=1)

        # plt.title(f"{dataset_name}: Split Histogram of Loss Difference (Flipped Points)")
        # plt.xlabel("Loss Difference (After - Before)")
        # plt.ylabel("Count")
        # plt.tight_layout()
        # plt.savefig(f"{hist_dir_base}/{dataset_name}.pdf")
        # plt.close()

        # """Line Plot - both curves"""
        # loss_old_sorted = loss_old[sorted_indices]
        # loss_new_sorted = loss_new[sorted_indices]

        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_old_sorted, label="Loss (Before)", color='orange', linewidth=1.5)
        # plt.plot(loss_new_sorted, label="Loss (After)", color='blue', linewidth=1.5)
        # plt.scatter(flipped_positions, loss_new_sorted[flipped_positions],
        #             color='red', label="Flipped (After)", marker='o', zorder=3)
        # plt.scatter([gb_flipped_sorted_pos], [loss_new_sorted[gb_flipped_sorted_pos]],
        #             color='green', label="GB Flip (After)", marker='o', s=80, edgecolor='black')

        # plt.xlabel("Sorted Sample Index (by Loss Diff)")
        # plt.ylabel("Binary Cross-Entropy Loss")
        # plt.title(f"{dataset_name}: Sorted Loss (Before vs After)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"{hist_dir_base}/{dataset_name}_loss_both.pdf")
        # plt.close()

    Stats.to_csv(f"Stats/{Test}_SummaryWithF1.csv", index=False)

if __name__ == "__main__":
    main()