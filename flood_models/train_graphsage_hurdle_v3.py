"""
train_graphsage_hurdle_v3.py — rotating leave-one-event-out on unit-level graphs
====================================================================================
v2 (train_graphsage_hurdle_v2.py) fixed the norm layer, mask sharing, and depth
target skew, but was still evaluated on ONE fixed held-out event per city (4
test folds total) — n=4 independent samples is a hard ceiling no architecture
change fixes. This script addresses that directly: it rotates EVERY one of the
15 city-events through the held-out slot in turn (true leave-one-event-out,
matching build_flood_units.py's own --eval-rotate practice and the LOBO
methodology used in the closest published analog), retrains a fresh model each
time, and reports mean +/- std across all 15 folds — a genuine k=15 evaluation
instead of k=4.

Runs on bitch/graphs_units_v1/*.pt (built by build_graphs_units.py from
build_flood_units.py's unit-level output) specifically because those graphs
are ~5-6x smaller by node count than the segment-level graphs_v5/, which is
what makes 15 independent training runs affordable in one session.

Same three v2 fixes carried over unchanged: LayerNorm (not BatchNorm), separate
cls_valid/depth_valid masks, log1p depth target. Same HistGBM tabular baseline,
now also computed per-fold for a fair rotating comparison.
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

HERE = os.path.dirname(__file__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs-dir", default=os.path.join(HERE, "graphs_units_v1"))
    p.add_argument("--out", default=os.path.join(HERE, "rotating_loeo_results.json"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n-val", type=int, default=3, help="train events reserved for early-stop val each fold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-folds", type=int, default=None,
                   help="run only the first N folds (for smoke testing)")
    return p.parse_args()


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
print("device:", device)

# ============================================================
# Load all graphs once (raw, unnormalized — normalization is
# refit per fold on that fold's own training events)
# ============================================================
with open(os.path.join(args.graphs_dir, "manifest.json")) as f:
    manifest = json.load(f)
FEATURE_NAMES = manifest["feature_names"]

all_graphs = {}
for entry in manifest["graphs"]:
    stem = entry["file"][:-3]
    d = torch.load(os.path.join(args.graphs_dir, entry["file"]), weights_only=False)
    d.y_flooded = d.y_flood.float()
    all_graphs[stem] = d

stems = sorted(all_graphs.keys())
print(f"{len(stems)} events total: {stems}")


def build_masks(data):
    feat_valid = ~torch.isnan(data.x).any(dim=1)
    cls_valid = feat_valid & ~torch.isnan(data.y_flooded)
    depth_trustworthy = (data.y_flooded == 0) | data.mask_depth
    depth_valid = feat_valid & ~torch.isnan(data.y_depth) & depth_trustworthy
    return cls_valid, depth_valid


for d in all_graphs.values():
    d.cls_valid, d.depth_valid = build_masks(d)


class GraphSAGEHurdle(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_channels // 2, 1))
        self.depth_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_channels // 2, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, edge_index))), p=self.dropout, training=self.training)
        return self.cls_head(x).squeeze(-1), self.depth_head(x).squeeze(-1)


FOCAL_ALPHA, FOCAL_GAMMA = 0.75, 2.0


def focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def cls_metrics(logits, true, mask):
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    t = true[mask].detach().cpu().numpy()
    pr = average_precision_score(t, probs) if t.sum() > 0 else float("nan")
    return pr


def run_epoch(model, loader, optimizer, train):
    model.train() if train else model.eval()
    all_logit, all_true, all_mask = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad()
            cls_logit, depth_log = model(batch.x, batch.edge_index)
            cmask, dmask = batch.cls_valid, batch.depth_valid
            cls_loss = focal_loss(cls_logit[cmask], batch.y_flooded[cmask])
            depth_loss = (F.mse_loss(depth_log[dmask], batch.y_depth_log[dmask])
                         if dmask.any() else torch.tensor(0.0, device=device))
            loss = cls_loss + depth_loss
            if train:
                loss.backward()
                optimizer.step()
            all_logit.append(cls_logit.detach()); all_true.append(batch.y_flooded.detach())
            all_mask.append(cmask.detach())
    return cls_metrics(torch.cat(all_logit), torch.cat(all_true), torch.cat(all_mask))


def run_fold(test_stem, fold_idx, n_folds):
    d_test_raw = all_graphs[test_stem]
    train_stems = [s for s in stems if s != test_stem]

    rng = random.Random(args.seed + fold_idx)
    val_stems = rng.sample(train_stems, min(args.n_val, len(train_stems) - 1))
    fit_stems = [s for s in train_stems if s not in val_stems]

    fit_raw = [all_graphs[s] for s in fit_stems]
    x_valid = torch.cat([d.x[d.cls_valid] for d in fit_raw], dim=0)
    feat_mean = x_valid.mean(dim=0)
    feat_std = x_valid.std(dim=0)
    feat_std[feat_std == 0] = 1.0

    def prep(d_raw):
        d = d_raw.clone()
        d.x = torch.nan_to_num((d_raw.x - feat_mean) / feat_std, nan=0.0)
        d.y_depth = torch.nan_to_num(d_raw.y_depth, nan=0.0)
        d.y_depth_log = torch.log1p(d.y_depth.clamp(min=0))
        d.cls_valid, d.depth_valid = d_raw.cls_valid, d_raw.depth_valid
        d.y_flooded = d_raw.y_flooded
        return d

    fit_split = [prep(all_graphs[s]) for s in fit_stems]
    val_split = [prep(all_graphs[s]) for s in val_stems]
    d_test = prep(d_test_raw)

    fit_loader = DataLoader(fit_split, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=1, shuffle=False)

    in_channels = fit_split[0].x.shape[1]
    model = GraphSAGEHurdle(in_channels, args.hidden, args.layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val, best_state, waited = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, fit_loader, optimizer, train=True)
        val_pr = run_epoch(model, val_loader, optimizer, train=False)
        if val_pr > best_val:
            best_val, waited = val_pr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        cls_logit, depth_log = model(d_test.x.to(device), d_test.edge_index.to(device))
    cmask, dmask = d_test.cls_valid, d_test.depth_valid
    prob = torch.sigmoid(cls_logit[cmask]).cpu().numpy()
    t_flooded = d_test.y_flooded[cmask].numpy()
    d_pred = torch.expm1(depth_log[dmask]).clamp(min=0).cpu().numpy()
    t_depth = d_test.y_depth[dmask].numpy()

    pr_auc = average_precision_score(t_flooded, prob) if t_flooded.sum() > 0 else float("nan")
    roc_auc = (roc_auc_score(t_flooded, prob)
              if 0 < t_flooded.sum() < len(t_flooded) else float("nan"))
    mae = float(np.mean(np.abs(d_pred - t_depth))) if len(t_depth) else float("nan")
    ss_res = float(np.sum((t_depth - d_pred) ** 2))
    ss_tot = float(np.sum((t_depth - t_depth.mean()) ** 2)) if len(t_depth) else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # tabular baseline, same fold's fit+val pool (all 14 train events), same test event
    Xtr_cls = np.concatenate([d.x[d.cls_valid].numpy() for d in fit_split + val_split])
    ytr_cls = np.concatenate([d.y_flooded[d.cls_valid].numpy() for d in fit_split + val_split]).astype(int)
    base_rate = ytr_cls.mean()
    sw = np.where(ytr_cls == 1, (1 - base_rate) / max(base_rate, 1e-9), 1.0)
    gbm_cls = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.1, random_state=42)
    gbm_cls.fit(Xtr_cls, ytr_cls, sample_weight=sw)
    p_base = gbm_cls.predict_proba(d_test.x[cmask].numpy())[:, 1]
    base_pr_auc = average_precision_score(t_flooded, p_base) if t_flooded.sum() > 0 else float("nan")
    base_roc_auc = (roc_auc_score(t_flooded, p_base)
                    if 0 < t_flooded.sum() < len(t_flooded) else float("nan"))

    Xtr_d = np.concatenate([d.x[d.depth_valid].numpy() for d in fit_split + val_split])
    ytr_d = np.concatenate([d.y_depth_log[d.depth_valid].numpy() for d in fit_split + val_split])
    base_mae, base_r2 = float("nan"), float("nan")
    if len(ytr_d) > 20 and dmask.sum() > 0:
        gbm_d = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.1, random_state=42)
        gbm_d.fit(Xtr_d, ytr_d)
        dp_base = np.expm1(gbm_d.predict(d_test.x[dmask].numpy())).clip(min=0)
        base_mae = float(np.mean(np.abs(dp_base - t_depth)))
        ss_res_b = float(np.sum((t_depth - dp_base) ** 2))
        base_r2 = 1 - ss_res_b / ss_tot if ss_tot > 0 else float("nan")

    result = {
        "test_event": test_stem, "n_cls": int(cmask.sum()), "n_depth": int(dmask.sum()),
        "base_rate": float(t_flooded.mean()),
        "sage_pr_auc": pr_auc, "sage_roc_auc": roc_auc, "sage_depth_mae": mae, "sage_depth_r2": r2,
        "gbm_pr_auc": base_pr_auc, "gbm_roc_auc": base_roc_auc,
        "gbm_depth_mae": base_mae, "gbm_depth_r2": base_r2,
        "best_val_pr_auc": best_val,
    }
    print(f"[{fold_idx+1:2d}/{n_folds}] {test_stem:22s} base={result['base_rate']:.4f}  "
          f"SAGE pr={pr_auc:.4f} roc={roc_auc:.4f} depthR2={r2:+.4f}  |  "
          f"GBM pr={base_pr_auc:.4f} roc={base_roc_auc:.4f} depthR2={base_r2:+.4f}")
    return result


fold_stems = stems if args.limit_folds is None else stems[:args.limit_folds]
results = []
for i, test_stem in enumerate(fold_stems):
    results.append(run_fold(test_stem, i, len(fold_stems)))

df = pd.DataFrame(results)
print("\n" + "=" * 100)
print(f"ROTATING LEAVE-ONE-EVENT-OUT — {len(results)} independent folds")
print("=" * 100)
print(df.to_string(index=False))

def summarize(col):
    v = df[col].dropna()
    return f"{v.mean():.4f} +/- {v.std():.4f}" if len(v) else "n/a"

print("\n--- mean +/- std across folds ---")
for label, col in [("SAGE PR-AUC", "sage_pr_auc"), ("SAGE ROC-AUC", "sage_roc_auc"),
                   ("SAGE depth R2", "sage_depth_r2"), ("SAGE depth MAE", "sage_depth_mae"),
                   ("GBM  PR-AUC", "gbm_pr_auc"), ("GBM  ROC-AUC", "gbm_roc_auc"),
                   ("GBM  depth R2", "gbm_depth_r2"), ("GBM  depth MAE", "gbm_depth_mae")]:
    print(f"  {label:16s} {summarize(col)}")

mean_base = df["base_rate"].mean()
print(f"\n  mean base rate across folds: {mean_base:.4f}")
print(f"  SAGE PR-AUC lift: {df['sage_pr_auc'].mean() / mean_base:.2f}x")
print(f"  GBM  PR-AUC lift: {df['gbm_pr_auc'].mean() / mean_base:.2f}x")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(args.out, "w") as f:
    json.dump({"results": results, "timestamp": timestamp,
              "graphs_dir": args.graphs_dir}, f, indent=2)
print(f"\nwrote {args.out}")
