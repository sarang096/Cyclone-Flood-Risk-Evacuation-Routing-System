"""
train_graphsage_hurdle.py — GraphSAGE flood depth + classification, local
===========================================================================
Adapted from graphsage_flood_depth (2).py (Colab notebook) to run locally
against graphs_v5/*.pt (built by build_graphs_v5.py from 0.05/
CSVs), instead of the Drive .npz archives the notebook was written for.

DROPPED ON PURPOSE (per instruction — these were EDA, not training):
  - the feature-contribution diagnostic block (univariate/LOFO/permutation
    importance) — orthogonal to training, very slow, and duplicative of
    what train_partial_obs_fixed.py already reported
  - Colab cells: torch install, drive.mount, DROP_FEATURES for ERA5/soil
    moisture columns (not present in this schema — nothing to drop)

KEPT UNCHANGED: the two-head GraphSAGE model, focal-loss classification +
masked-MSE depth loss, recall-driven checkpoint selection, evaluation,
F-beta threshold sweep, and the flood-detection metric emphasis.

DATA MAPPING (npz keys -> our .pt Data fields, both already match here):
    x, edge_index      unchanged (already torch.Tensor on the Data object)
    y                  -> data.y_flood     (SAR binary, unchanged name kept
                                              as y_flooded below for parity)
    depth              -> data.y_depth
    depth_mask         -> NOT a separate field in our schema. Our builder
                          already computes mask_depth = (sar==1) & assigned,
                          i.e. exactly the "wse actually filled" indicator
                          restricted to flooded rows. depth_trustworthy is
                          reconstructed as (y_flood==0) | mask_depth, which
                          is algebraically identical to the notebook's
                          (sar==0) | ((sar==1) & (wse==1)).

NOTE ON A DESIGN TENSION carried over unchanged from the pasted script:
  build_graphs.py's own design note #3 says wse_assigned should gate the
  DEPTH head only — masking the flood LABEL by it "discards ~70% of
  positives from the head that needs them most." This script uses ONE
  valid_mask for BOTH the classification loss and the depth loss, which
  reintroduces exactly that discard for the classification head. Left
  as-is since the ask was to run this script, not redesign it — but
  worth knowing before trusting the classification numbers.
"""

import argparse
import json
import os
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

torch.manual_seed(42)
np.random.seed(42)

HERE = os.path.dirname(__file__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs-dir", default=os.path.join(HERE, "graphs_v5"))
    p.add_argument("--checkpoint-dir", default=os.path.join(HERE, "checkpoints_graphsage"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--no-wse-mask", action="store_true",
                   help="disable the wse_assigned depth-trustworthy gate")
    p.add_argument("--flood-threshold-m", type=float, default=0.05)
    p.add_argument("--target-recall", type=float, default=0.90)
    return p.parse_args()


args = parse_args()
os.makedirs(args.checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

USE_WSE_MASK = not args.no_wse_mask
FLOOD_THRESHOLD_M = args.flood_threshold_m
TARGET_RECALL = args.target_recall

# porbandar dropped — 0.05/ has no porbandar CSVs, so build_graphs_v5.py
# never produced porbandar__*.pt
# bhubaneswar added 2026-08-23; held out on "dana" (its most recent event),
# same choice as puri, which shares dana/fani/titli/yaas with it.
TEST_CYCLONE_PER_CITY = {
    "chennai": "mandous",
    "kolkata": "remal",
    "puri": "dana",
    "bhubaneswar": "dana",
}

# ============================================================
# Load graphs
# ============================================================
with open(os.path.join(args.graphs_dir, "manifest.json")) as f:
    manifest = json.load(f)
FEATURE_NAMES = manifest["feature_names"]

train_graphs, test_graphs = [], []
train_names, test_names = [], []

for entry in manifest["graphs"]:
    stem = entry["file"][:-3]  # strip ".pt"
    p = os.path.join(args.graphs_dir, entry["file"])
    d = torch.load(p, weights_only=False)
    d.feature_names = FEATURE_NAMES
    # parity with the notebook's naming
    d.y_flooded = d.y_flood.float()

    if TEST_CYCLONE_PER_CITY.get(d.city, "").lower() == d.cyclone.lower():
        test_graphs.append(d); test_names.append(stem)
    else:
        train_graphs.append(d); train_names.append(stem)

print(f"\ntrain ({len(train_graphs)}): {train_names}")
print(f"test  ({len(test_graphs)}): {test_names}")
assert train_graphs and test_graphs, "split is empty -- check TEST_CYCLONE_PER_CITY"

for d, nm in zip(train_graphs + test_graphs, train_names + test_names):
    print(f"  {nm:22s} {d.num_nodes:>7,} nodes  {d.edge_index.shape[1]:>8,} edges  "
          f"flooded={float((d.y_flooded > 0.5).float().mean()):.4f}")

# ============================================================
# Validity masks + normalization
# ============================================================

def build_valid_mask(data, use_wse_mask=True):
    x = data.x
    y = data.y_depth
    sar = data.y_flooded

    feat_valid = ~torch.isnan(x).any(dim=1)
    label_valid = ~torch.isnan(y)

    if use_wse_mask:
        depth_trustworthy = (sar == 0) | data.mask_depth
    else:
        depth_trustworthy = torch.ones_like(y, dtype=torch.bool)

    return feat_valid & label_valid & depth_trustworthy


for data in train_graphs + test_graphs:
    data.valid_mask = build_valid_mask(data, USE_WSE_MASK)

train_x_valid = torch.cat([d.x[d.valid_mask] for d in train_graphs], dim=0)
feat_mean = train_x_valid.mean(dim=0)
feat_std = train_x_valid.std(dim=0)
feat_std[feat_std == 0] = 1.0


def normalize_and_clean(data):
    x = (data.x - feat_mean) / feat_std
    x = torch.nan_to_num(x, nan=0.0)
    data.x = x
    data.y_depth = torch.nan_to_num(data.y_depth, nan=0.0)
    return data


train_graphs = [normalize_and_clean(d) for d in train_graphs]
test_graphs = [normalize_and_clean(d) for d in test_graphs]

for d, name in zip(train_graphs, train_names):
    print(f"  {name}: {d.valid_mask.sum().item()}/{d.num_nodes} valid nodes")
for d, name in zip(test_graphs, test_names):
    print(f"  {name}: {d.valid_mask.sum().item()}/{d.num_nodes} valid nodes")

# ============================================================
# DataLoaders
# ============================================================
BATCH_SIZE = args.batch_size
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

# ============================================================
# Model
# ============================================================

class GraphSAGEHurdle(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )
        self.depth_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        cls_logit = self.cls_head(x).squeeze(-1)
        depth = self.depth_head(x).squeeze(-1)
        return cls_logit, depth


in_channels = train_graphs[0].x.shape[1]
model = GraphSAGEHurdle(in_channels=in_channels, hidden_channels=args.hidden,
                        num_layers=args.layers, dropout=args.dropout).to(device)
print(model)

# ============================================================
# Train/val split within TRAIN graphs
# ============================================================
random.seed(42)

train_graphs_shuffled = train_graphs.copy()
random.shuffle(train_graphs_shuffled)
n_val = max(1, len(train_graphs_shuffled) // 5)
val_split = train_graphs_shuffled[:n_val]
fit_split = train_graphs_shuffled[n_val:]

print(f"fit: {len(fit_split)} graphs | val (early stop): {len(val_split)} graphs | "
      f"test: {len(test_graphs)} graphs")

fit_loader = DataLoader(fit_split, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_split, batch_size=1, shuffle=False)

flooded_count = sum((d.y_flooded[d.valid_mask] == 1).sum().item() for d in fit_split)
dry_count = sum((d.y_flooded[d.valid_mask] == 0).sum().item() for d in fit_split)
FLOOD_WEIGHT = min(dry_count / max(flooded_count, 1), 20.0)
print(f"flooded: {flooded_count} | dry: {dry_count} | FLOOD_WEIGHT = {FLOOD_WEIGHT:.2f}")

# ============================================================
# Training loop — focal loss for classification, masked MSE for depth
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

EPOCHS = args.epochs
PATIENCE = args.patience

FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
DEPTH_LOSS_WEIGHT = 1.0


def focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def compute_metrics(cls_logit, true_flooded, mask):
    probs = torch.sigmoid(cls_logit[mask]).detach().cpu().numpy()
    true = true_flooded[mask].detach().cpu().numpy()

    pr_auc = average_precision_score(true, probs) if true.sum() > 0 else float("nan")

    pred_flooded = (probs >= 0.5).astype(int)
    tp = int(((pred_flooded == 1) & (true == 1)).sum())
    fp = int(((pred_flooded == 1) & (true == 0)).sum())
    fn = int(((pred_flooded == 0) & (true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return pr_auc, recall, precision


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_nodes = 0.0, 0
    all_logit, all_true_flooded, all_mask = [], [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad()

            cls_logit, depth = model(batch.x, batch.edge_index)
            mask = batch.valid_mask

            cls_loss = focal_loss(cls_logit[mask], batch.y_flooded[mask])
            depth_loss = F.mse_loss(depth[mask], batch.y_depth[mask])
            loss = cls_loss + (DEPTH_LOSS_WEIGHT * depth_loss)

            if train:
                loss.backward()
                optimizer.step()

            n = mask.sum().item()
            total_loss += loss.item() * n
            total_nodes += n

            all_logit.append(cls_logit.detach())
            all_true_flooded.append(batch.y_flooded.detach())
            all_mask.append(mask.detach())

    all_logit = torch.cat(all_logit)
    all_true_flooded = torch.cat(all_true_flooded)
    all_mask = torch.cat(all_mask)
    pr_auc, recall, precision = compute_metrics(all_logit, all_true_flooded, all_mask)

    return total_loss / total_nodes, pr_auc, recall, precision


history = {"train_loss": [], "val_loss": [], "train_recall": [], "val_recall": []}
best_val_pr_auc = -1.0
best_state = None
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_pr_auc, train_recall, train_prec = run_epoch(fit_loader, train=True)
    val_loss, val_pr_auc, val_recall, val_prec = run_epoch(val_loader, train=False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_recall"].append(train_recall)
    history["val_recall"].append(val_recall)

    scheduler.step(val_pr_auc)

    improved = val_pr_auc > best_val_pr_auc
    if improved:
        best_val_pr_auc = val_pr_auc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch == 1 or epoch % args.print_every == 0:
        print(f"epoch {epoch:4d} | train loss {train_loss:.4f} PR-AUC {train_pr_auc:.3f} "
              f"recall@.5 {train_recall:.3f} prec@.5 {train_prec:.3f} | "
              f"val loss {val_loss:.4f} PR-AUC {val_pr_auc:.3f} recall@.5 {val_recall:.3f} "
              f"prec@.5 {val_prec:.3f} | best val PR-AUC {best_val_pr_auc:.3f}")

    if epochs_no_improve >= PATIENCE:
        print(f"early stopping at epoch {epoch} (no PR-AUC improvement for {PATIENCE} epochs)")
        break

model.load_state_dict(best_state)
print(f"\nrestored best model -- val PR-AUC = {best_val_pr_auc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history["train_loss"], label="train loss")
axes[0].plot(history["val_loss"], label="val loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (focal + MSE)"); axes[0].legend()
axes[0].set_title("Training loss")

axes[1].plot(history["train_recall"], label="train recall")
axes[1].plot(history["val_recall"], label="val recall")
axes[1].axhline(TARGET_RECALL, color="gray", linestyle=":", label=f"target ({TARGET_RECALL})")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("recall (flooded segments)"); axes[1].legend()
axes[1].set_title(f"Recall @ {FLOOD_THRESHOLD_M}m threshold (model selection metric)")
plt.tight_layout()
curves_path = os.path.join(args.checkpoint_dir, "training_curves.png")
plt.savefig(curves_path, dpi=120)
print("saved ->", curves_path)

# ============================================================
# Evaluation on held-out cyclones
# ============================================================

def evaluate(loader, names, cls_threshold=0.5):
    model.eval()
    results = []
    all_prob, all_depth_pred, all_true_depth, all_true_flooded = [], [], [], []

    with torch.no_grad():
        for data, name in zip(loader, names):
            data = data.to(device)
            cls_logit, depth_pred = model(data.x, data.edge_index)
            mask = data.valid_mask

            prob = torch.sigmoid(cls_logit[mask]).cpu().numpy()
            d_pred = depth_pred[mask].cpu().numpy()
            t_depth = data.y_depth[mask].cpu().numpy()
            t_flooded = data.y_flooded[mask].cpu().numpy()

            mae = np.mean(np.abs(d_pred - t_depth))
            rmse = np.sqrt(np.mean((d_pred - t_depth) ** 2))
            ss_res = np.sum((t_depth - d_pred) ** 2)
            ss_tot = np.sum((t_depth - t_depth.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            pred_flooded = (prob >= cls_threshold).astype(int)
            tp = int(((pred_flooded == 1) & (t_flooded == 1)).sum())
            fp = int(((pred_flooded == 1) & (t_flooded == 0)).sum())
            fn = int(((pred_flooded == 0) & (t_flooded == 1)).sum())
            tn = int(((pred_flooded == 0) & (t_flooded == 0)).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
                  else float("nan"))
            pr_auc = average_precision_score(t_flooded, prob) if t_flooded.sum() > 0 else float("nan")

            results.append({
                "event": name, "n_nodes": mask.sum().item(),
                "MAE_m": mae, "RMSE_m": rmse, "R2": r2,
                "PR_AUC": pr_auc, "recall": recall, "precision": precision, "F1": f1,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            })
            all_prob.append(prob); all_depth_pred.append(d_pred)
            all_true_depth.append(t_depth); all_true_flooded.append(t_flooded)

    df = pd.DataFrame(results)

    all_prob = np.concatenate(all_prob)
    all_depth_pred = np.concatenate(all_depth_pred)
    all_true_depth = np.concatenate(all_true_depth)
    all_true_flooded = np.concatenate(all_true_flooded)

    overall_mae = np.mean(np.abs(all_depth_pred - all_true_depth))
    overall_rmse = np.sqrt(np.mean((all_depth_pred - all_true_depth) ** 2))
    ss_res = np.sum((all_true_depth - all_depth_pred) ** 2)
    ss_tot = np.sum((all_true_depth - all_true_depth.mean()) ** 2)
    overall_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    pred_flooded_all = (all_prob >= cls_threshold).astype(int)
    tp = int(((pred_flooded_all == 1) & (all_true_flooded == 1)).sum())
    fp = int(((pred_flooded_all == 1) & (all_true_flooded == 0)).sum())
    fn = int(((pred_flooded_all == 0) & (all_true_flooded == 1)).sum())
    overall_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else float("nan"))
    overall_pr_auc = (average_precision_score(all_true_flooded, all_prob)
                      if all_true_flooded.sum() > 0 else float("nan"))

    print(df.drop(columns=["TP", "FP", "FN", "TN"]).to_string(index=False))
    print(f"\nOVERALL depth accuracy:  MAE={overall_mae:.4f} m | RMSE={overall_rmse:.4f} m | R2={overall_r2:.4f}")
    print(f"OVERALL flood detection: PR-AUC={overall_pr_auc:.3f} | recall={overall_recall:.3f} | "
          f"precision={overall_precision:.3f} | F1={overall_f1:.3f}  (@ threshold {cls_threshold})")
    return df, (all_prob, all_depth_pred, all_true_depth, all_true_flooded)


test_metrics_df, (test_prob, test_depth_pred, test_true_depth, test_true_flooded) = \
    evaluate(test_loader, test_names)

# threshold sweep — F2-optimal cutoff
thresholds = np.linspace(0.0, 1.0, 101)
sweep_recall, sweep_precision = [], []
for th in thresholds:
    pred_flooded = (test_prob >= th).astype(int)
    tp = int(((pred_flooded == 1) & (test_true_flooded == 1)).sum())
    fp = int(((pred_flooded == 1) & (test_true_flooded == 0)).sum())
    fn = int(((pred_flooded == 0) & (test_true_flooded == 1)).sum())
    sweep_recall.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    sweep_precision.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)

sweep_recall = np.array(sweep_recall)
sweep_precision = np.array(sweep_precision)

BETA = 2.0
with np.errstate(divide="ignore", invalid="ignore"):
    f_beta = (1 + BETA**2) * sweep_precision * sweep_recall / (BETA**2 * sweep_precision + sweep_recall)

valid = ~np.isnan(f_beta)
best_idx = np.nanargmax(f_beta[valid]) if valid.any() else None

if best_idx is not None:
    best_idx = np.arange(len(thresholds))[valid][best_idx]
    recommended_threshold = thresholds[best_idx]
    print(f"F{BETA:.0f}-optimal threshold: {recommended_threshold:.3f} "
          f"(recall={sweep_recall[best_idx]:.3f}, precision={sweep_precision[best_idx]:.3f}, "
          f"F{BETA:.0f}={f_beta[best_idx]:.3f})")
else:
    recommended_threshold = 0.5
    print("No valid F-beta score found in sweep -- defaulting to 0.5")

meets_target = np.where(sweep_recall >= TARGET_RECALL)[0]
if len(meets_target) > 0:
    old_idx = meets_target[-1]
    print(f"(for reference, >= {TARGET_RECALL:.0%}-recall threshold: {thresholds[old_idx]:.3f} "
          f"-> recall={sweep_recall[old_idx]:.3f}, precision={sweep_precision[old_idx]:.3f})")

plt.figure(figsize=(7, 5))
plt.plot(sweep_recall, sweep_precision)
plt.scatter([sweep_recall[best_idx]], [sweep_precision[best_idx]], color="red", zorder=5,
            label=f"F{BETA:.0f}-optimal (th={recommended_threshold:.2f})")
plt.xlabel("recall"); plt.ylabel("precision"); plt.legend()
plt.title("Precision-recall tradeoff across thresholds")
pr_path = os.path.join(args.checkpoint_dir, "pr_tradeoff.png")
plt.savefig(pr_path, dpi=120)
print("saved ->", pr_path)

plt.figure(figsize=(6, 6))
plt.scatter(test_true_depth, test_depth_pred, alpha=0.15, s=5)
lims = [0, max(test_true_depth.max(), test_depth_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("actual flood_depth_m"); plt.ylabel("predicted flood_depth_m (depth head)")
plt.title("GraphSAGE hurdle model -- predicted vs actual depth (held-out cyclones)")
scatter_path = os.path.join(args.checkpoint_dir, "depth_scatter.png")
plt.savefig(scatter_path, dpi=120)
print("saved ->", scatter_path)

# ============================================================
# Save checkpoint + metrics
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = os.path.join(args.checkpoint_dir, f"graphsage_hurdle_{timestamp}.pt")

torch.save({
    "model_state_dict": best_state,
    "in_channels": in_channels,
    "feature_names": FEATURE_NAMES,
    "feat_mean": feat_mean,
    "feat_std": feat_std,
    "test_cyclone_per_city": TEST_CYCLONE_PER_CITY,
    "use_wse_mask": USE_WSE_MASK,
    "pos_weight": FLOOD_WEIGHT,
    "best_val_pr_auc": best_val_pr_auc,
    "recommended_cls_threshold": recommended_threshold if len(meets_target) > 0 else None,
    "target_recall": TARGET_RECALL,
}, ckpt_path)
print("saved checkpoint ->", ckpt_path)

metrics_path = os.path.join(args.checkpoint_dir, f"graphsage_hurdle_metrics_{timestamp}.csv")
test_metrics_df.to_csv(metrics_path, index=False)
print("saved metrics ->", metrics_path)
