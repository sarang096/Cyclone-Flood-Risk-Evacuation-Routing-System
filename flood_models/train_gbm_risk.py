"""
train_gbm_risk.py — segment-level flood risk scorer (HistGBM), 15-fold rotating LOEO
=======================================================================================
Produces the per-road-segment flood risk scores that feed the routing layer.

WHY GBM AND NOT THE GNN: train_graphsage_hurdle_v3.py ran a 15-fold rotating
leave-one-event-out comparison at unit level. HistGBM beat GraphSAGE on
classification in 14 of 15 folds (PR-AUC 0.224 +/- 0.124 vs 0.166 +/- 0.116;
ROC-AUC 0.653 vs 0.600). Two independent evaluations (that one, and
train_partial_obs_fixed.py's learned = sage - lr_nbr) agree the graph structure
isn't earning its complexity on this data. So: ship the model that won.

WHY NO DEPTH HEAD: dropped deliberately. build_graphs.py's own design note #2
says the depth head "is partly re-learning FwDET's own arithmetic rather than
discovering physics... The FLOOD head is the genuine predictive contribution."
Depth was also sparse (50-1,700 usable nodes/event) and scored R2 ~ -0.15,
i.e. worse than predicting the mean. Binary flood risk only.

WHY SEGMENT LEVEL, NOT UNIT LEVEL: routing happens on road segments, so the
risk score has to exist per segment. The v3 numbers are unit-level
(flood_maj over ~8-segment clusters) and do NOT transfer to this population --
hence this script re-measures from scratch at segment granularity.

OUT-OF-FOLD SCORES: the risk score written for each event comes from a model
that never saw that event during training. That keeps the downstream routing
evaluation honest -- routing on in-sample predictions would flatter the result.

CALIBRATION CAVEAT: sample_weight is used to counter ~9:1 class imbalance,
which inflates predicted probabilities. Fine for routing (Dijkstra needs
relative ordering, and a monotone transform preserves it); NOT to be read as
literal flood probabilities.

    Input   graphs_v5/*.pt          (segment-level graphs, 15 events)
            graphs_v5/nodes_<city>.csv  (road_id, lon/lat per node)
    Output  risk_scores/<city>__<cyclone>_risk.csv   (per-segment risk, out-of-fold)
            risk_scores/loeo_metrics.csv             (per-fold honest metrics)
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(__file__)
GRAPHS_DIR = os.path.join(HERE, "graphs_v5")
OUT_DIR = os.path.join(HERE, "risk_scores")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(os.path.join(GRAPHS_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    events = {}
    for entry in manifest["graphs"]:
        stem = entry["file"][:-3]
        d = torch.load(os.path.join(GRAPHS_DIR, entry["file"]), weights_only=False)
        events[stem] = d

    stems = sorted(events)
    print(f"{len(stems)} events, {len(feature_names)} features")
    print("15-fold rotating leave-one-event-out (segment level)\n")

    rows = []
    for i, test_stem in enumerate(stems):
        d_test = events[test_stem]
        train_stems = [s for s in stems if s != test_stem]

        Xtr = np.concatenate([events[s].x[events[s].mask_flood].numpy() for s in train_stems])
        ytr = np.concatenate([events[s].y_flood[events[s].mask_flood].numpy()
                              for s in train_stems]).astype(int)

        base = ytr.mean()
        sw = np.where(ytr == 1, (1 - base) / max(base, 1e-9), 1.0)

        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                             learning_rate=0.08, random_state=42)
        clf.fit(Xtr, ytr, sample_weight=sw)

        m = d_test.mask_flood.numpy()
        Xte = d_test.x[d_test.mask_flood].numpy()
        yte = d_test.y_flood[d_test.mask_flood].numpy().astype(int)
        risk = clf.predict_proba(Xte)[:, 1]

        pr = average_precision_score(yte, risk)
        roc = roc_auc_score(yte, risk)
        lift = pr / max(yte.mean(), 1e-9)

        # out-of-fold risk score for every node in this event
        full_risk = np.full(d_test.num_nodes, np.nan)
        full_risk[m] = risk

        city = d_test.city
        nodes = pd.read_csv(os.path.join(GRAPHS_DIR, f"nodes_{city}.csv"))
        out = pd.DataFrame({
            "node_idx": np.arange(d_test.num_nodes),
            "road_id": nodes["road_id"],
            "highway_type": nodes["highway_type"],
            "longitude": nodes["longitude"],
            "latitude": nodes["latitude"],
            "risk": full_risk,
            "sar_flooded": d_test.y_flood.numpy(),
        })
        out.to_csv(os.path.join(OUT_DIR, f"{test_stem}_risk.csv"), index=False)

        rows.append({"event": test_stem, "n": int(m.sum()), "base_rate": float(yte.mean()),
                     "PR_AUC": pr, "ROC_AUC": roc, "lift": lift})
        print(f"[{i+1:2d}/15] {test_stem:22s} base={yte.mean():.4f}  "
              f"PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  lift={lift:.2f}x")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "loeo_metrics.csv"), index=False)

    print("\n" + "=" * 78)
    print("SEGMENT-LEVEL GBM — 15-fold rotating leave-one-event-out")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\n  PR-AUC   {df.PR_AUC.mean():.4f} +/- {df.PR_AUC.std():.4f}")
    print(f"  ROC-AUC  {df.ROC_AUC.mean():.4f} +/- {df.ROC_AUC.std():.4f}")
    print(f"  mean base rate {df.base_rate.mean():.4f}  ->  mean lift "
          f"{df.PR_AUC.mean()/df.base_rate.mean():.2f}x")
    print(f"\nwrote per-event risk scores to {OUT_DIR}/")


if __name__ == "__main__":
    main()
