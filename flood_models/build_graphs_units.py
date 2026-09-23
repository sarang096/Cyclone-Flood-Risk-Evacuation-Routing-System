"""
build_graphs_units.py — PyG graphs from build_flood_units.py's unit-level output
====================================================================================
Converts bitch/units_v2/ (produced by `build_flood_units.py --csv "0.05/*.csv"
--out ./units_v2`) into the same city__cyclone.pt Data schema build_graphs_v5.py
produces (x, edge_index, y_flood, mask_flood, y_depth, mask_depth), except nodes
are spatially-constrained flood UNITS (~8 contiguous road segments each) instead
of individual segments.

WHY: build_graphs_v5.py gives 15 graphs (one per city-event) with no way to get
more independent evaluation folds than "hold out one city" or "hold out one
event per city" — a hard n=4/n=15 ceiling no amount of architecture tuning
fixes. Units don't create more city-events, but they do two things that make a
much cheaper, genuinely more-independent evaluation possible:
  1. Denoise the label — flood_maj is a majority vote over ~8 segments instead
     of one noisy point sample, at nearly-unchanged base rate (build_flood_
     units.py's own diagnostics: unit-majority base rate tracks segment base
     rate closely, unlike flood_any which inflates it ~2-2.9x — see its
     BASE-RATE WARNING docstring).
  2. Shrink node count ~5-6x per graph, cheap enough to afford ROTATING which
     cyclone is held out per city and averaging (train_graphsage_hurdle_v3.py),
     instead of one fixed test event — that's the actual lever for more
     independent samples, not the unit conversion by itself.

LABEL CHOICE: flood_maj (majority vote), not flood_any (max) — see the WHY
section above and build_flood_units.py's own BASE-RATE WARNING.

DEPTH MASK: mask_depth = (flood_maj==1) & (depth_mask==1), i.e. only units
that are flooded by majority vote AND have at least one member segment with a
genuine FwDET-assigned WSE contributing to the unit's mean depth. This is the
unit-level equivalent of build_graphs.py's own y_flood/wse_assigned gating,
and matches exactly what train_graphsage_hurdle_v2.py's build_masks() expects
(it separately reconstructs depth_valid = (y_flood==0) | mask_depth).

    Input   <UNITS_DIR>/<city>_unit_graph.npz
            <UNITS_DIR>/<city>_<cyclone>_units.csv
    Output  <OUT_DIR>/<city>__<cyclone>.pt
            <OUT_DIR>/manifest.json
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

HERE = os.path.dirname(__file__)
UNITS_DIR = os.path.join(HERE, "units_v2")
OUT_DIR = os.path.join(HERE, "graphs_units_v1")
os.makedirs(OUT_DIR, exist_ok=True)

STATIC_FEATURES = [
    "hand_m", "slope_deg", "twi", "dtw_m", "dist_to_water_m",
    "impervious_frac", "veg_frac", "water_frac", "total_length_m",
]
DYNAMIC_FEATURES = [
    "rain_total_mm", "rain_peak_30min_mm", "rain_wet_hours",
    "rain_max_1h_mm", "rain_max_2h_mm", "rain_max_3h_mm",
    "rain_max_6h_mm", "rain_max_24h_mm",
    "rain_ante_24h_mm", "rain_ante_72h_mm", "rain_ante_7d_mm",
]
DERIVED_FEATURES = ["elev_rank", "degree", "hand_min_1hop", "hand_mean_1hop",
                    "elev_rank_mean_1hop"]


def neighbourhood_features(edge_index, n_nodes, hand, elev_rank):
    hand_min = hand.copy().astype(np.float64)
    hand_sum = np.zeros(n_nodes)
    elev_sum = np.zeros(n_nodes)
    cnt = np.zeros(n_nodes)

    s, d = edge_index[0], edge_index[1]
    np.minimum.at(hand_min, s, hand[d])
    np.add.at(hand_sum, s, hand[d])
    np.add.at(elev_sum, s, elev_rank[d])
    np.add.at(cnt, s, 1.0)

    safe = np.maximum(cnt, 1.0)
    hand_mean = np.where(cnt > 0, hand_sum / safe, hand)
    elev_mean = np.where(cnt > 0, elev_sum / safe, elev_rank)
    return hand_min, hand_mean, elev_mean


def main():
    unit_csvs = sorted(glob.glob(os.path.join(UNITS_DIR, "*_unit_graph.npz")))
    cities = [os.path.basename(p)[:-len("_unit_graph.npz")] for p in unit_csvs]
    print(f"cities: {cities}")

    manifest = []
    feature_names_ref = None

    for city in cities:
        print(f"\n{'-' * 62}\n{city}")
        gz = np.load(os.path.join(UNITS_DIR, f"{city}_unit_graph.npz"))
        n_units = len(gz["unit_id"])
        edge_index = gz["edge_index"].astype(np.int64)
        print(f"  {n_units:,} units, {edge_index.shape[1]:,} directed edges")

        event_paths = sorted(glob.glob(os.path.join(UNITS_DIR, f"{city}_*_units.csv")))
        event_paths = [p for p in event_paths if not p.endswith("_segment_to_unit.csv")]

        # static terrain (elevation_m, for elev_rank) taken from the first event
        first = pd.read_csv(event_paths[0]).set_index("unit_id").reindex(range(n_units))
        elev_rank = first["elevation_m"].rank(pct=True).to_numpy(np.float64)
        hand = first["hand_m"].to_numpy(np.float64)
        h_min, h_mean, e_mean = neighbourhood_features(edge_index, n_units, hand, elev_rank)
        degree = np.bincount(edge_index[0], minlength=n_units).astype(np.float64)

        for p in event_paths:
            ev = pd.read_csv(p).set_index("unit_id").reindex(range(n_units))
            cyclone = str(ev["cyclone"].iloc[0]).lower()

            feat_cols = [c for c in STATIC_FEATURES + DYNAMIC_FEATURES if c in ev.columns]
            missing = [c for c in STATIC_FEATURES + DYNAMIC_FEATURES if c not in ev.columns]
            if missing:
                print(f"  ({cyclone}) not in units.csv, skipped: {missing}")

            num = ev.reindex(columns=feat_cols).astype(np.float64).fillna(0.0).to_numpy()
            X = np.column_stack([num, elev_rank, degree, h_min, h_mean, e_mean])
            feature_names = feat_cols + DERIVED_FEATURES

            if feature_names_ref is None:
                feature_names_ref = feature_names
            elif feature_names != feature_names_ref:
                raise ValueError(f"{city}/{cyclone} feature mismatch:\n"
                                 f"  expected {feature_names_ref}\n  got {feature_names}")

            y_flood = ev["flood_maj"].fillna(0).to_numpy().astype(np.int64)
            mask_flood = np.ones(n_units, dtype=bool)  # aggregate_units always fills this

            y_depth = ev["flood_depth_m"].fillna(0).to_numpy().astype(np.float32)
            depth_labeled = (ev["depth_mask"].fillna(0).to_numpy() > 0)
            mask_depth = (y_flood == 1) & depth_labeled

            fc = np.bincount(y_flood, minlength=2)
            pos_pct = 100 * fc[1] / max(fc.sum(), 1)
            print(f"  {cyclone}: flood_maj base rate {pos_pct:.2f}% "
                  f"({fc[1]:,}/{fc.sum():,}) | depth usable {int(mask_depth.sum()):,}")

            data = Data(
                x=torch.tensor(X, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y_flood=torch.tensor(y_flood, dtype=torch.long),
                mask_flood=torch.tensor(mask_flood, dtype=torch.bool),
                y_depth=torch.tensor(y_depth, dtype=torch.float),
                mask_depth=torch.tensor(mask_depth, dtype=torch.bool),
            )
            data.city, data.cyclone, data.event = city, cyclone, cyclone
            stem = f"{city}__{cyclone}"
            torch.save(data, os.path.join(OUT_DIR, stem + ".pt"))

            manifest.append({
                "city": city, "cyclone": cyclone, "n_nodes": int(n_units),
                "n_edges": int(edge_index.shape[1]),
                "n_flooded": int(fc[1]), "pct_flooded": round(float(pos_pct), 3),
                "n_usable_depth": int(mask_depth.sum()), "file": stem + ".pt",
            })

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "task": "hurdle: y_flood (flood_maj, unit-level) + y_depth (metres)",
            "feature_names": feature_names_ref,
            "unit_source": UNITS_DIR,
            "label_choice": "flood_maj (majority vote) -- NOT flood_any, "
                            "which build_flood_units.py's own diagnostics show "
                            "inflates the positive rate ~2-2.9x via max() aggregation",
            "graphs": manifest,
        }, f, indent=2)

    mf = pd.DataFrame(manifest)
    print(f"\n{'=' * 62}")
    print(mf.to_string(index=False))
    print(f"\nWrote {len(manifest)} unit-level graphs to {OUT_DIR}/")
    print(f"Features ({len(feature_names_ref)}): {feature_names_ref}")


if __name__ == "__main__":
    main()
