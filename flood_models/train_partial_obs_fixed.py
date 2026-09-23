"""
train_partial_obs_fixed.py — corrected partial-observation flood inference
==========================================================================
Drop-in replacement for the %%writefile cell in partial_obs_colab.ipynb.
Same problem statement, same model. Fixes seven things that change either
the numbers or the conclusion.


WHAT CHANGED, AND WHY
---------------------

[1] mask_flood IS NOW RESPECTED (was ignored entirely)
    The original read g.y_flood and never looked at g.mask_flood. Nodes
    with no usable label were used as training targets, FED IN AS
    OBSERVED LABELS, and scored as ground truth. Your own Puri GEE
    header says that without wse_assigned this was "71% of SAR-flooded
    points" on Chennai/Mandous — FwDET never assigned a water surface
    and unmask(0) wrote a zero that is indistinguishable from dry.
    Now: only mask_flood nodes can be observed, supervised, or scored.
    Everything else is inert — it still passes messages (a road is a
    road) but never contributes a label or a loss term.

    SANITY CHECK: base rates printed below should match the ones from
    your audit run (Chennai/Mandous 0.1212, Kolkata/Remal 0.1249,
    Puri/Yaas 0.3190, ...). If they do not, something else is wrong.

[2] BLOCK MASKING ADDED (was random-only)
    The docstring motivates this work with "revisit gaps, look angle and
    urban layover" — every one of which produces CONTIGUOUS holes. The
    original simulated coverage with torch.rand(n) < frac, which
    scatters observations uniformly, so nearly every hidden segment
    keeps several observed neighbours. That is the best case for label
    propagation, not the stated case.
    Now both regimes run. Block masking grows BFS regions from seeds, so
    segments deep inside a hole have no observed neighbour at any hop
    and the method must fall back on features.
    QUOTE THE BLOCK NUMBER. The random number is an upper bound.

[3] TRANSDUCTIVE FEATURE-ONLY ARM ADDED (removes a confound)
    The original compared feat_clf (fit on TRAINING cities, inductive)
    against logreg_nbr (fit on the TEST event's observed nodes,
    transductive). Two things differ at once, so the 0.63 -> 0.813 gap
    cannot be attributed to neighbour labels alone. The new feat_trans
    arm is features-only fit transductively, which isolates it:
        feat_ind   -> feat_trans   = the transductive advantage
        feat_trans -> logreg_nbr   = the neighbour-label contribution

[4] EDGE DIRECTION MADE CONSISTENT (was contradictory)
    PyG SAGEConv propagates edge_index[0] -> edge_index[1], so a node
    aggregates from its IN-neighbours (upslope, in your flow graph).
    label_propagation and neighbour_label_features both did
    index_add_(0, src, ...[dst]), accumulating at src — aggregating from
    OUT-neighbours (downslope). Model and baselines were looking at
    opposite neighbourhoods on a directed graph.
    Now everything uses one symmetrised edge set. "Is this
    neighbourhood underwater" is a symmetric question even when the
    flow that caused it is not.

[5] PR-AUC REPORTED FOR EVERY ARM (was ROC-AUC headline, AP for two)
    Base rates here are 3-32%. ROC-AUC is inflated relative to PR-AUC
    in exactly that regime (Saito & Rehmsmeier 2015,
    DOI 10.1371/journal.pone.0118432), and your whole earlier analysis
    was in PR-AUC-over-base-rate lift. Both are printed; lift is
    computed against the base rate of the hidden set.

[6] MULTIPLE MASK DRAWS (was one fixed-seed draw)
    N_DRAWS masks per (event, coverage, regime), mean and std reported.
    A single draw at 3% base rate is noise.

[7] CONT_IDX IS NOW DERIVED FROM THE DATA, NOT HARDCODED (was silently
    wrong)
    The previous version fixed this at CONT_IDX = list(range(12)) with
    a comment claiming "0-11 are the continuous columns, 12+ are the
    hw_* one-hots" — verified against an OLDER 16-column audit set that
    no longer matches your v5 export (9 static + length_m + 11 rain =
    21 continuous columns, not 12). StandardScaler was silently leaving
    columns 12-20 unscaled: real-valued features sitting next to
    genuinely binary one-hot columns on a completely different scale,
    degrading the transductive logistic regressions and LayerNorm
    without ever raising an error.
    Rather than replace one hardcoded guess with another (I don't have
    your graph builder's exact column order to verify against),
    CONT_IDX is now INFERRED at runtime: any column with more than 2
    distinct values across every valid node, pooled over every event,
    is treated as continuous; everything else is treated as a one-hot
    indicator. This self-corrects if the schema changes again, and it
    prints what it found so you can sanity-check the count and the
    column range against what the graph builder actually wrote.


UNCHANGED AND STILL CORRECT
---------------------------
  - The leakage control. A node's own label is zeroed in its own input
    and the loss is taken on the complement of the fed set. Verified by
    inspection: lab = y * label_mask, and supervision is on ~label_mask.
  - LayerNorm over BatchNorm, for the reason given in the original
    docstring: BatchNorm would erase columns that are near-constant
    within a single event. Your feature audit found the rain_* columns
    have within-event spatial std around 0.05-0.10 versus 0.6-1.5 for
    terrain features — almost flat per event — so that bug would have
    been fatal for exactly the columns meant to carry storm-to-storm
    signal.
  - Leave-one-city-out for parameters, partial observation for the
    target event.


STILL WORTH CHECKING, NOT CHANGED HERE (need your input, not a guess)
----------------------------------------------------------------------
  - --data-root defaults to a different path (.../Cyclone_Dataset/
    graphs_flow) than the extraction script's output folder
    (cyclonefloodnewdataset/road_samples_v5). Confirm this points at
    graphs built from the current 23-band CSVs before trusting a run.
  - Under block masking, `sage` is evaluated from a model trained ONLY
    with uniform random coverage (torch.rand(n) < p during training) —
    it never sees block-shaped gaps during training, while feat_trans
    and logreg_nbr refit transductively on whatever draw they're given,
    block or random. A weak `sage` row under block masking could mean
    "never trained for this regime," not "doesn't help." If you want a
    clean comparison, block-mask a fraction of training draws too.
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv

FRACTIONS = (0.10, 0.25, 0.50, 0.75)
REGIMES = ("random", "block")


def load_events(root):
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(root, "*.pt"))):
        city, event = os.path.basename(p)[:-3].split("__")
        d = torch.load(p, map_location="cpu", weights_only=False)
        d.city, d.event = city, event
        out[city][event] = d
    return dict(out)


# ----------------------------------------------------------------------
# [7] continuous-column inference — replaces the hardcoded CONT_IDX
# ----------------------------------------------------------------------

def infer_cont_idx(all_graphs, max_categories=2):
    """Continuous columns = those with more than `max_categories` distinct
    values over every valid (mask_flood) node, pooled across every event.
    One-hot / binary indicator columns fail that test and are left out.

    Data-derived on purpose: a hardcoded index range silently goes stale
    the moment the graph builder's column order changes, exactly as
    happened with the previous CONT_IDX = range(12). This prints what it
    found so a schema change is visible immediately instead of silently
    degrading downstream scaling.
    """
    xs = []
    for g in all_graphs:
        v = g.mask_flood.bool()
        if v.any():
            xs.append(g.x[v].numpy())
    X = np.concatenate(xs, axis=0)

    n_unique = np.array([
        np.unique(np.round(X[:, i], 6)).size for i in range(X.shape[1])
    ])
    cont = [i for i in range(X.shape[1]) if n_unique[i] > max_categories]
    onehot = [i for i in range(X.shape[1]) if i not in cont]

    print(f"  {X.shape[1]} total feature columns -> "
          f"{len(cont)} continuous, {len(onehot)} one-hot/binary")
    if cont:
        print(f"    continuous column indices: {cont[0]}-{cont[-1]}"
              if cont == list(range(cont[0], cont[-1] + 1))
              else f"    continuous column indices (non-contiguous): {cont}")
    if not cont:
        raise ValueError(
            "infer_cont_idx found zero continuous columns — every column "
            "had <=2 distinct values. Check that mask_flood and x are "
            "populated as expected before training on this.")
    if cont != list(range(len(cont))):
        print("    NOTE: continuous columns are not a leading 0..k-1 block "
              "— one-hots are interleaved with them. The index list still "
              "scales the right columns either way, but this is worth "
              "confirming against the graph builder's intended layout.")
    return cont


# ----------------------------------------------------------------------
# [4] one symmetric edge set for model and baselines alike
# ----------------------------------------------------------------------

def symmetrize(edge_index):
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    keep = ei[0] != ei[1]
    ei = ei[:, keep]
    key = (ei[0].to(torch.int64) * (int(ei.max()) + 1) + ei[1].to(torch.int64)).numpy()
    _, uniq = np.unique(key, return_index=True)
    return ei[:, torch.as_tensor(np.sort(uniq))]


def sparse_adj(edge_index, n):
    s, d = edge_index[0].numpy(), edge_index[1].numpy()
    A = csr_matrix((np.ones(len(s), np.float32), (d, s)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


# ----------------------------------------------------------------------
# [2] coverage simulation — random and contiguous
# ----------------------------------------------------------------------

def sample_observed(A, valid_idx, frac, n, regime, rng, n_seeds=40):
    """Return a boolean array of OBSERVED nodes, a subset of valid_idx."""
    target = int(round(frac * len(valid_idx)))
    if target < 1:
        target = 1

    if regime == "random":
        obs = np.zeros(n, bool)
        obs[rng.choice(valid_idx, target, replace=False)] = True
        return obs

    # block: grow contiguous observed regions, so the UNOBSERVED
    # complement is also contiguous — real SAR holes, not confetti
    valid = np.zeros(n, bool)
    valid[valid_idx] = True
    obs = np.zeros(n, bool)
    obs[rng.choice(valid_idx, min(n_seeds, len(valid_idx)), replace=False)] = True

    guard = 0
    while obs[valid].sum() < target and guard < 10_000:
        guard += 1
        nxt = (A @ obs.astype(np.float32)) > 0
        new = nxt & ~obs & valid
        if not new.any():
            pool = valid_idx[~obs[valid_idx]]
            if len(pool) == 0:
                break
            obs[rng.choice(pool, min(n_seeds, len(pool)), replace=False)] = True
            continue
        obs |= new

    got = np.where(obs & valid)[0]
    if len(got) > target:
        obs = np.zeros(n, bool)
        obs[rng.choice(got, target, replace=False)] = True
    return obs


# ----------------------------------------------------------------------
# Model (unchanged)
# ----------------------------------------------------------------------

class SageLabel(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.3, aggr="mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim + 2
        for _ in range(layers):
            self.convs.append(SAGEConv(d, hidden, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden))
            d = hidden
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, y, label_mask):
        lab = (y.float() * label_mask.float()).unsqueeze(1)
        h = torch.cat([x, lab, label_mask.float().unsqueeze(1)], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, edge_index))), p=self.dropout,
                          training=self.training)
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------------
# Baselines — all on the symmetrised graph now
# ----------------------------------------------------------------------

def label_propagation(A, y, obs, n, iters=30, alpha=0.9):
    deg = np.asarray(A.sum(1)).ravel().clip(min=1)
    seed = float(y[obs].mean()) if obs.any() else 0.0
    f = np.where(obs, y, seed).astype(np.float32)
    init = f.copy()
    for _ in range(iters):
        f = alpha * ((A @ f) / deg) + (1 - alpha) * init
        f[obs] = y[obs]
    return f


def neighbour_label_features(A, y, obs, n):
    """[frac1, frac2, obs_share_1hop, obs_count_1hop] from OBSERVED nodes only."""
    om = obs.astype(np.float32)
    yo = (y * obs).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel().clip(min=1)

    c1 = A @ om
    s1 = A @ yo
    m1 = np.where(c1 > 0, s1 / np.maximum(c1, 1), 0.5)

    c2 = A @ c1
    s2 = A @ s1
    m2 = np.where(c2 > 0, s2 / np.maximum(c2, 1), 0.5)

    return np.column_stack([m1, m2, c1 / deg, c1]).astype(np.float32)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_sage(train_graphs, val_graphs, in_dim, args, device):
    model = SageLabel(in_dim, args.hidden, args.layers, args.dropout, args.aggr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # [1] pos_weight over VALID nodes only
    pos = sum(int(g["y"][g["valid"]].sum()) for g in train_graphs)
    neg = sum(int((g["y"][g["valid"]] == 0).sum()) for g in train_graphs)
    pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)
    print(f"    class balance over valid nodes: {pos:,} pos / {neg:,} neg "
          f"(pos_weight {float(pw):.2f})")

    best, best_state, waited = -1.0, None, 0
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_graphs)))
        rng.shuffle(order)
        tot = 0.0
        for i in order:
            g = train_graphs[i]
            n = g["x"].size(0)
            valid = g["valid"]
            p = rng.uniform(args.min_cov, args.max_cov)

            # [1] only VALID nodes may be fed as labels or supervised
            lm = (torch.rand(n) < p) & valid
            sup = (~lm) & valid
            if int(sup.sum()) == 0 or int(g["y"][sup].sum()) == 0:
                continue

            opt.zero_grad()
            out = model(g["x"].to(device), g["edge_index"].to(device),
                        g["y"].to(device), lm.to(device))
            loss = F.binary_cross_entropy_with_logits(
                out[sup.to(device)], g["y"].float().to(device)[sup.to(device)],
                pos_weight=pw)
            loss.backward()
            opt.step()
            tot += float(loss)

        if epoch % args.eval_every == 0:
            model.eval()
            aps = []
            with torch.no_grad():
                for g in val_graphs:
                    n = g["x"].size(0)
                    valid = g["valid"]
                    gen = torch.Generator().manual_seed(args.seed)
                    lm = (torch.rand(n, generator=gen) < 0.5) & valid
                    hid = ((~lm) & valid).numpy()
                    o = model(g["x"].to(device), g["edge_index"].to(device),
                              g["y"].to(device), lm.to(device))
                    p_ = torch.sigmoid(o).cpu().numpy()[hid]
                    yt = g["y"].numpy()[hid]
                    if yt.max() > 0:
                        # [5] select on AP, not ROC-AUC — matches the reported metric
                        aps.append(average_precision_score(yt, p_))
            va = float(np.mean(aps)) if aps else 0.0
            if va > best:
                best, waited = va, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"    epoch {epoch:4d} loss {tot/max(len(order),1):.4f} val AP {va:.4f} *")
            else:
                waited += 1
                if waited >= args.patience:
                    print(f"    early stop epoch {epoch} (best val AP {best:.4f})")
                    break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"device: {device}\n")

    events = load_events(args.data_root)
    results = {}

    # ---- [7] derive CONT_IDX from the data, once, before any fold ----
    print("=" * 80)
    print("INFERRING CONTINUOUS FEATURE COLUMNS (for StandardScaler)")
    print("=" * 80)
    all_graphs = [g for c in events for g in events[c].values()]
    CONT_IDX = infer_cont_idx(all_graphs)
    print()

    # ---- [1] base-rate sanity check against the audit run ----
    print("=" * 80)
    print("BASE RATES OVER VALID (mask_flood) NODES")
    print("These must match your audit output. If they do not, the graphs")
    print("in --data-root are not the ones you audited.")
    print("=" * 80)
    for c in sorted(events):
        for e in sorted(events[c]):
            g = events[c][e]
            v = g.mask_flood.bool()
            print(f"  {c}/{e:12s} valid {int(v.sum()):>8,} / {g.x.size(0):>8,}"
                  f"   base rate {float(g.y_flood[v].float().mean()):.4f}")

    for test_city in args.cities:
        if test_city not in events:
            continue
        print(f"\n{'='*80}\nFOLD: model never sees {test_city.upper()}\n{'='*80}")
        train_cities = [c for c in events if c != test_city]
        rng0 = random.Random(args.seed)
        val_pick = {c: rng0.choice(sorted(events[c])) for c in train_cities}

        tr_raw = [g for c in train_cities for e, g in sorted(events[c].items())
                  if e != val_pick[c]]
        va_raw = [events[c][val_pick[c]] for c in train_cities]
        te_raw = [g for _, g in sorted(events[test_city].items())]
        print(f"  train {len(tr_raw)} events, val {len(va_raw)}, test {len(te_raw)}")

        # scaler on training cities only, valid nodes only
        sc = StandardScaler().fit(np.concatenate(
            [g.x[g.mask_flood.bool()][:, CONT_IDX].numpy() for g in tr_raw]))

        def pack(gs):
            out = []
            for g in gs:
                x = g.x.clone()
                x[:, CONT_IDX] = torch.tensor(
                    sc.transform(x[:, CONT_IDX].numpy()), dtype=torch.float)
                ei = symmetrize(g.edge_index)
                out.append({"x": x, "edge_index": ei, "y": g.y_flood,
                            "valid": g.mask_flood.bool(), "event": g.event,
                            "A": sparse_adj(ei, x.size(0))})
            return out

        tp, vp, sp = pack(tr_raw), pack(va_raw), pack(te_raw)
        model, val_ap = train_sage(tp, vp, tp[0]["x"].size(1), args, device)

        # [3] inductive features-only, trained on other cities, valid nodes only
        Xtr = np.concatenate([g["x"].numpy()[g["valid"].numpy()] for g in tp])
        ytr = np.concatenate([g["y"].numpy()[g["valid"].numpy()] for g in tp])
        feat_ind = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)

        fold = {}
        for regime in REGIMES:
            for frac in FRACTIONS:
                print(f"\n  --- {regime} masking, coverage {frac:.0%} ---")
                print(f"    {'event':12s}{'base':>8s}{'feat_ind':>10s}{'feat_tr':>9s}"
                      f"{'lprop':>8s}{'lr_nbr':>9s}{'sage':>8s}   (AP; ROC in json)")
                per = {}
                for g in sp:
                    n = g["x"].size(0)
                    valid = g["valid"].numpy()
                    valid_idx = np.where(valid)[0]
                    X = g["x"].numpy()
                    y = g["y"].numpy().astype(np.float32)
                    A = g["A"]

                    acc = defaultdict(list)
                    for draw in range(args.draws):
                        r = np.random.default_rng(args.seed + 977 * draw)
                        obs = sample_observed(A, valid_idx, frac, n, regime, r,
                                              n_seeds=args.seeds)
                        hid = valid & ~obs
                        if hid.sum() < 50 or y[hid].sum() == 0 or y[obs].sum() == 0:
                            continue
                        yt = y[hid]
                        acc["base"].append(float(yt.mean()))

                        # inductive features only
                        acc["feat_ind"].append(
                            average_precision_score(yt, feat_ind.predict_proba(X[hid])[:, 1]))
                        acc["feat_ind_roc"].append(
                            roc_auc_score(yt, feat_ind.predict_proba(X[hid])[:, 1]))

                        # [3] transductive features only — isolates the confound
                        try:
                            ft = LogisticRegression(max_iter=1000, class_weight="balanced")
                            ft.fit(X[obs], y[obs])
                            pf = ft.predict_proba(X[hid])[:, 1]
                            acc["feat_trans"].append(average_precision_score(yt, pf))
                            acc["feat_trans_roc"].append(roc_auc_score(yt, pf))
                        except ValueError:
                            pass

                        # label propagation, no features no learning
                        pl = label_propagation(A, y, obs, n)[hid]
                        acc["label_prop"].append(average_precision_score(yt, pl))
                        acc["label_prop_roc"].append(roc_auc_score(yt, pl))

                        # transductive features + neighbour labels
                        nb = neighbour_label_features(A, y, obs, n)
                        Xn = np.concatenate([X, nb], axis=1)
                        try:
                            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
                            lr.fit(Xn[obs], y[obs])
                            pn = lr.predict_proba(Xn[hid])[:, 1]
                            acc["logreg_nbr"].append(average_precision_score(yt, pn))
                            acc["logreg_nbr_roc"].append(roc_auc_score(yt, pn))
                        except ValueError:
                            pass

                        # the model
                        model.eval()
                        with torch.no_grad():
                            o = model(g["x"].to(device), g["edge_index"].to(device),
                                      g["y"].to(device),
                                      torch.as_tensor(obs).to(device))
                        ps = torch.sigmoid(o).cpu().numpy()[hid]
                        acc["sage"].append(average_precision_score(yt, ps))
                        acc["sage_roc"].append(roc_auc_score(yt, ps))

                    if not acc["base"]:
                        continue
                    row = {k: float(np.mean(v)) for k, v in acc.items()}
                    row["sage_std"] = float(np.std(acc["sage"]))
                    per[g["event"]] = row
                    print(f"    {g['event']:12s}{row['base']:8.4f}"
                          f"{row['feat_ind']:10.4f}{row.get('feat_trans', float('nan')):9.4f}"
                          f"{row['label_prop']:8.4f}{row.get('logreg_nbr', float('nan')):9.4f}"
                          f"{row['sage']:8.4f}")
                fold[f"{regime}_{frac:.2f}"] = per
        fold["val_ap"] = val_ap
        results[test_city] = fold

    # ---------------- operating curves ----------------
    for regime in REGIMES:
        print(f"\n{'='*88}")
        print(f"OPERATING CURVE — {regime.upper()} MASKING   (mean PR-AUC on unobserved)")
        if regime == "random":
            print("  OPTIMISTIC BOUND. Scattered holes; nearly every hidden segment")
            print("  keeps observed neighbours. Not what SAR gaps look like.")
        else:
            print("  REALISTIC. Contiguous holes, as produced by revisit gaps,")
            print("  look angle and urban layover. THIS is the number to quote.")
        print("=" * 88)
        print(f"{'cover':>7s}{'base':>9s}{'feat_ind':>10s}{'feat_tr':>9s}{'lprop':>8s}"
              f"{'lr_nbr':>9s}{'sage':>8s}{'lift':>8s}{'nbr gain':>10s}{'learned':>9s}")
        for frac in FRACTIONS:
            k = f"{regime}_{frac:.2f}"
            acc = defaultdict(list)
            for city in results:
                for ev in results[city].get(k, {}).values():
                    for m in ("base", "feat_ind", "feat_trans", "label_prop",
                              "logreg_nbr", "sage"):
                        if m in ev and ev[m] == ev[m]:
                            acc[m].append(ev[m])
            if not acc["sage"]:
                continue
            b = np.mean(acc["base"])
            ft = np.mean(acc["feat_trans"]) if acc["feat_trans"] else float("nan")
            ln = np.mean(acc["logreg_nbr"]) if acc["logreg_nbr"] else float("nan")
            sg = np.mean(acc["sage"])
            print(f"{frac:7.0%}{b:9.4f}{np.mean(acc['feat_ind']):10.4f}{ft:9.4f}"
                  f"{np.mean(acc['label_prop']):8.4f}{ln:9.4f}{sg:8.4f}"
                  f"{sg/b:7.2f}x{ln-ft:+10.4f}{sg-ln:+9.4f}")

    print("""
  base       base rate of the hidden set — the floor.
  feat_ind   features, fit on OTHER cities. Your current pipeline.
  feat_tr    features, fit on THIS event's observed nodes.
             feat_ind -> feat_tr isolates the transductive advantage.
  lprop      pure diffusion of observed labels. No features, no learning.
  lr_nbr     features + observed-neighbour labels, transductive.
             feat_tr -> lr_nbr is the TRUE neighbour-label contribution,
             with the transductive confound removed.
  sage       GraphSAGE with label inputs, params from other cities only.
  learned    sage - lr_nbr: what the GNN adds over hand-built features.
             If this is <= 0, a logistic regression is doing the job and
             the GNN is not earning its complexity. That is reportable.
""")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/content/drive/MyDrive/Cyclone_Dataset/graphs_flow")
    p.add_argument("--cities", nargs="+",
                   default=["chennai", "kolkata", "porbandar", "puri"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--aggr", default="mean", choices=["mean", "max"])
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-cov", type=float, default=0.05)
    p.add_argument("--max-cov", type=float, default=0.80)
    p.add_argument("--draws", type=int, default=3, help="mask draws per cell")
    p.add_argument("--seeds", type=int, default=40, help="BFS seeds for block masking")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="partial_obs_results.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
