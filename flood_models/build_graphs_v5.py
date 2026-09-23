"""
build_graphs_v5.py — cyclone flood road-risk graph construction
================================================================
Adapted from cyclone_graphs.ipynb's build_graphs.py cell to match the
actual column schema of 0.05/*.csv (u/v/key instead of road_id,
lon/lat instead of longitude/latitude, length_m instead of
road_length_m, no rainfall_mm/effective_rain_mm/runoff_coeff/
curve_number — instead a richer per-point rain_* + terrain feature
set). Line-graph construction, hurdle labels (y_flood/y_depth with
their own masks), and edge attributes are UNCHANGED from the original.

    Input   <INPUT_DIR>/<city>_<cyclone>.csv
    Output  <OUTPUT_DIR>/<city>__<cyclone>.pt
            <OUTPUT_DIR>/nodes_<city>.csv
            <OUTPUT_DIR>/manifest.json
"""

import os
import re
import glob
import json
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ================================================================
# 0. CONFIG
# ================================================================

INPUT_DIR  = os.path.join(os.path.dirname(__file__), "0.05")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "graphs_v5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# terrain — MIN: the lowest point on the segment floods first
# gradients / context / rain — mean is representative
# labels — MAX: a road is impassable if ANY part of it floods
AGG_RULES = {
    "hand_m":            "min",
    "elevation_m":       "min",
    "slope_deg":         "mean",
    "twi":               "mean",
    "dtw_m":              "mean",
    "dist_to_water_m":    "mean",
    "impervious_frac":    "mean",
    "veg_frac":           "mean",
    "water_frac":         "mean",
    "rain_total_mm":          "mean",
    "rain_peak_30min_mm":     "mean",
    "rain_wet_hours":         "mean",
    "rain_max_1h_mm":         "mean",
    "rain_max_2h_mm":         "mean",
    "rain_max_3h_mm":         "mean",
    "rain_max_6h_mm":         "mean",
    "rain_max_24h_mm":        "mean",
    "rain_ante_24h_mm":       "mean",
    "rain_ante_72h_mm":       "mean",
    "rain_ante_7d_mm":        "mean",
    "road_length_m":     "mean",
    "longitude":         "mean",   # metadata only, not a feature
    "latitude":          "mean",   # metadata only, not a feature
    "sar_flood_binary":  "max",
    "flood_depth_m":     "max",
    "wse_assigned":      "max",
}

# Candidate numeric features. Constants are pruned automatically;
# rain_total_mm is protected as the one retained event-severity
# scalar even if it turns out to be spatially constant.
CANDIDATE_FEATURES = [
    "hand_m",
    "slope_deg",
    "elev_rank",            # derived: within-city percentile
    "twi",
    "dtw_m",
    "dist_to_water_m",
    "impervious_frac",
    "veg_frac",
    "water_frac",
    "road_length_m",
    "degree",               # derived: node degree
    "hand_min_1hop",        # derived: neighbourhood terrain
    "hand_mean_1hop",
    "elev_rank_mean_1hop",
    "rain_total_mm",         # event-severity scalar (protected)
    "rain_peak_30min_mm",
    "rain_wet_hours",
    "rain_max_1h_mm",
    "rain_max_2h_mm",
    "rain_max_3h_mm",
    "rain_max_6h_mm",
    "rain_max_24h_mm",
    "rain_ante_24h_mm",
    "rain_ante_72h_mm",
    "rain_ante_7d_mm",
]
PROTECTED_FEATURES = {"rain_total_mm"}

DERIVED_FEATURES = {"degree", "hand_min_1hop", "hand_mean_1hop",
                    "elev_rank_mean_1hop", "elev_rank"}

EXCLUDED = {"sar_flood_binary", "flood_depth_m", "wse_assigned",
            "longitude", "latitude", "elevation_m"}

HIGHWAY_KEEP = ["motorway", "trunk", "primary", "secondary",
                "tertiary", "residential", "unclassified", "service"]

MAX_JUNCTION_DEGREE = 20
CONSTANT_STD_EPS    = 1e-9

FNAME_RE = re.compile(r"(?P<city>[a-z]+)_(?P<cyclone>[a-z]+)\.csv$")


# ================================================================
# 1. LOAD
# ================================================================

def canonicalize_road_id(road_id):
    parts = str(road_id).split("_")
    if len(parts) < 3:
        return road_id
    u, v, k = parts[0], parts[1], "_".join(parts[2:])
    try:
        if int(u) > int(v):
            u, v = v, u
    except ValueError:
        if u > v:
            u, v = v, u
    return f"{u}_{v}_{k}"


def load_all_csvs(input_dir=INPUT_DIR):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs in {input_dir}")

    frames = []
    for p in paths:
        base = os.path.basename(p)
        m = FNAME_RE.search(base)
        if not m:
            print(f"  skip (name pattern): {base}")
            continue
        df = pd.read_csv(p, low_memory=False)
        need = {"u", "v", "key", "lon", "lat", "sar_flood_binary"}
        missing = need - set(df.columns)
        if missing:
            print(f"  skip (missing cols {sorted(missing)}): {base}")
            continue

        df["road_id"] = (df["u"].astype(str) + "_" + df["v"].astype(str)
                          + "_" + df["key"].astype(str))
        df = df.rename(columns={
            "lon": "longitude", "lat": "latitude", "length_m": "road_length_m",
        })
        df["city"] = df.get("city", pd.Series([m.group("city")] * len(df))).astype(str).str.lower()
        df["cyclone"] = df.get("cyclone", pd.Series([m.group("cyclone")] * len(df))).astype(str).str.lower()
        frames.append(df)
        print(f"  {base}: {len(df):,} rows")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n  {len(frames)} city-events, {len(all_df):,} sample points")

    before = all_df["road_id"].nunique()
    all_df["road_id"] = all_df["road_id"].map(canonicalize_road_id)
    after = all_df["road_id"].nunique()
    if before > after:
        pct = 100 * (before - after) / before
        print(f"  canonicalized road_id: {before:,} -> {after:,} unique "
              f"({pct:.1f}% were reverse-direction duplicates, now merged)")

    return all_df


def aggregate_to_segments(df):
    agg = {c: rule for c, rule in AGG_RULES.items() if c in df.columns}
    missing = [c for c in AGG_RULES if c not in df.columns]
    if missing:
        print(f"  (not in this CSV schema, skipped: {missing})")

    for c in ("highway_type", "osm_name"):
        if c in df.columns:
            agg[c] = "first"

    seg = df.groupby(["city", "cyclone", "road_id"], as_index=False).agg(agg)
    print(f"  {len(df):,} points -> {len(seg):,} segment-events")
    return seg


# ================================================================
# 2. LINE GRAPH + EDGE ATTRIBUTES
# ================================================================

def parse_endpoints(road_id):
    parts = str(road_id).split("_")
    return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)


def build_line_graph(segment_ids):
    idx_of = {rid: i for i, rid in enumerate(segment_ids)}

    incident = defaultdict(list)
    for rid in segment_ids:
        u, v = parse_endpoints(rid)
        if u is None:
            continue
        incident[u].append(idx_of[rid])
        if v != u:
            incident[v].append(idx_of[rid])

    src, dst, huge = [], [], 0
    for segs in incident.values():
        if len(segs) > MAX_JUNCTION_DEGREE:
            huge += 1
            continue
        for a, b in itertools.combinations(segs, 2):
            src += [a, b]
            dst += [b, a]

    if huge:
        print(f"  skipped {huge} junctions with >{MAX_JUNCTION_DEGREE} "
              f"incident segments (likely malformed geometry)")

    if not src:
        print("  NO EDGES — check road_id looks like 'u_v_key'")
        return (np.zeros((2, 0), np.int64), np.zeros(len(segment_ids), np.int64))

    edges = np.unique(np.vstack([src, dst]).T, axis=0).T.astype(np.int64)
    degree = np.bincount(edges[0], minlength=len(segment_ids)).astype(np.int64)

    n_iso = int((degree == 0).sum())
    md = degree.mean()
    print(f"  line graph: {len(segment_ids):,} nodes, {edges.shape[1]:,} "
          f"directed edges, mean degree {md:.2f}, {n_iso} isolated")
    if md > 8:
        print(f"  mean degree {md:.1f} is high for a road network "
              f"(expect ~3-4). Check road_id parsing for this city.")
    return edges, degree


def build_edge_attr(edge_index, hand, elev_rank, lon, lat):
    s, d = edge_index[0], edge_index[1]
    d_hand = (hand[d] - hand[s]).astype(np.float32)
    d_elev = (elev_rank[d] - elev_rank[s]).astype(np.float32)

    mlat = np.radians((lat[s] + lat[d]) / 2.0)
    dx = (lon[d] - lon[s]) * 111.32 * np.cos(mlat)
    dy = (lat[d] - lat[s]) * 110.57
    dist = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

    return np.vstack([d_hand, d_elev, dist]).T


def neighbourhood_features(edge_index, n_nodes, hand, elev_rank):
    hand_min = hand.copy().astype(np.float64)
    hand_sum = np.zeros(n_nodes)
    elev_sum = np.zeros(n_nodes)
    cnt      = np.zeros(n_nodes)

    s, d = edge_index[0], edge_index[1]
    np.minimum.at(hand_min, s, hand[d])
    np.add.at(hand_sum, s, hand[d])
    np.add.at(elev_sum, s, elev_rank[d])
    np.add.at(cnt, s, 1.0)

    safe = np.maximum(cnt, 1.0)
    hand_mean = np.where(cnt > 0, hand_sum / safe, hand)
    elev_mean = np.where(cnt > 0, elev_sum / safe, elev_rank)
    return hand_min, hand_mean, elev_mean


# ================================================================
# 3. LABELS  (hurdle: flood head + depth head)
# ================================================================

def build_labels(ev):
    depth = ev["flood_depth_m"].fillna(0).to_numpy()
    sar   = ev["sar_flood_binary"].fillna(0).to_numpy()

    y_flood    = sar.astype(np.int64)
    mask_flood = ev["sar_flood_binary"].notna().to_numpy()

    if "wse_assigned" in ev.columns:
        assigned = ev["wse_assigned"].fillna(0).to_numpy() > 0
    else:
        assigned = depth > 0
        print("    no wse_assigned column — depth mask falls back to "
              "depth>0, which discards true shallow readings")

    return y_flood, mask_flood, depth.astype(np.float32), (sar == 1) & assigned


# ================================================================
# 4. FEATURES
# ================================================================

def encode_highway(ev):
    cols = [f"hw_{k}" for k in HIGHWAY_KEEP + ["other"]]
    if "highway_type" not in ev.columns:
        return pd.DataFrame(0.0, index=ev.index, columns=cols, dtype=np.float32)

    ht = (ev["highway_type"].astype(str)
          .str.split(";").str[0].str.strip().str.lower()
          .str.replace(r"^\[|\]$|'", "", regex=True))
    ht = ht.where(ht.isin(HIGHWAY_KEEP), "other")

    dummies = pd.get_dummies(ht, prefix="hw").astype(np.float32)
    for c in cols:
        if c not in dummies.columns:
            dummies[c] = 0.0
    return dummies[cols]


def report_constants(seg, candidates):
    print("\nSpatial variance within graphs (std across nodes, "
          "averaged over events):")
    keep, dropped = [], []
    for c in candidates:
        if c not in seg.columns:
            if c in DERIVED_FEATURES:
                keep.append(c)
                print(f"  {c:22s} (derived later — kept)")
            else:
                print(f"  {c:22s} ABSENT from CSVs — skipped")
            continue
        s = seg.groupby(["city", "cyclone"])[c].std().mean()
        n = seg.groupby(["city", "cyclone"])[c].nunique().mean()
        const = (not np.isfinite(s)) or s < CONSTANT_STD_EPS
        tag = ""
        if const and c in PROTECTED_FEATURES:
            tag = "  CONSTANT — kept (event-severity scalar)"
            keep.append(c)
        elif const:
            tag = "  CONSTANT — dropped (redundant event ID)"
            dropped.append(c)
        else:
            keep.append(c)
        print(f"  {c:22s} std={s:12.6f}  uniq/graph={n:8.1f}{tag}")

    if dropped:
        print(f"\n  dropped {len(dropped)} constant feature(s): {dropped}")
    return keep


# ================================================================
# 5. MAIN
# ================================================================

def main():
    print("Loading CSVs...")
    df = load_all_csvs()

    print("\nAggregating to segments...")
    seg = aggregate_to_segments(df)

    print("\nImputing NaNs (city median)...")
    for col in AGG_RULES:
        if col in seg.columns and seg[col].isna().any():
            n = int(seg[col].isna().sum())
            seg[col] = seg.groupby("city")[col].transform(
                lambda s: s.fillna(s.median()))
            seg[col] = seg[col].fillna(seg[col].median())
            print(f"  {col}: filled {n:,}")

    if "elevation_m" in seg.columns:
        seg["elev_rank"] = seg.groupby("city")["elevation_m"].rank(pct=True)

    features = report_constants(seg, CANDIDATE_FEATURES)

    feature_names_ref, manifest = None, []

    for city, city_df in seg.groupby("city"):
        print(f"\n{'-' * 62}\n{city}")

        segment_ids = sorted(city_df["road_id"].unique())
        edge_index, degree = build_line_graph(segment_ids)

        first = (city_df.groupby("road_id").first().reindex(segment_ids))
        hand = first["hand_m"].to_numpy(np.float64)
        erank = (first["elev_rank"].to_numpy(np.float64)
                 if "elev_rank" in first.columns else np.zeros(len(segment_ids)))
        lon = first["longitude"].to_numpy(np.float64)
        lat = first["latitude"].to_numpy(np.float64)

        h_min, h_mean, e_mean = neighbourhood_features(
            edge_index, len(segment_ids), hand, erank)
        edge_attr = build_edge_attr(edge_index, hand, erank, lon, lat)

        nodes_csv = os.path.join(OUTPUT_DIR, f"nodes_{city.lower()}.csv")
        pd.DataFrame({
            "node_idx": np.arange(len(segment_ids)),
            "road_id": segment_ids,
            "osm_name": first.get("osm_name", pd.Series(index=segment_ids)).values,
            "highway_type": first.get("highway_type", pd.Series(index=segment_ids)).values,
            "longitude": lon, "latitude": lat,
        }).to_csv(nodes_csv, index=False)

        for cyclone, ev in city_df.groupby("cyclone"):
            ev = ev.set_index("road_id").reindex(segment_ids)
            ev["degree"] = degree
            ev["hand_min_1hop"] = h_min
            ev["hand_mean_1hop"] = h_mean
            ev["elev_rank_mean_1hop"] = e_mean

            num = ev.reindex(columns=features).astype(np.float64).fillna(0.0)
            hw = encode_highway(ev)
            X = pd.concat([num.reset_index(drop=True),
                           hw.reset_index(drop=True)], axis=1)
            feature_names = list(X.columns)

            if feature_names_ref is None:
                feature_names_ref = feature_names
            elif feature_names != feature_names_ref:
                raise ValueError(
                    f"{city}/{cyclone} feature mismatch.\n"
                    f"  expected: {feature_names_ref}\n  got: {feature_names}")

            print(f"  {cyclone}:")
            y_flood, mask_flood, y_depth, mask_depth = build_labels(ev)

            fc = np.bincount(y_flood[mask_flood], minlength=2)
            pos_pct = 100 * fc[1] / max(fc.sum(), 1)
            print(f"    flood head: {mask_flood.sum():,}/{len(ev):,} usable | "
                  f"dry={fc[0]:,} flooded={fc[1]:,} ({pos_pct:.2f}%)")

            if mask_depth.sum():
                dv = y_depth[mask_depth]
                print(f"    depth head: {mask_depth.sum():,} usable | "
                      f"mean {dv.mean():.3f}m  p95 {np.percentile(dv,95):.3f}m  "
                      f"max {dv.max():.3f}m")
            else:
                print(f"    depth head: 0 usable")

            if fc[1] == 0:
                print("    NO positive flood labels — this event carries "
                      "no signal; decide whether to keep it before folds")
            if mask_depth.sum() < 30:
                print("    <30 usable depth labels — depth head will be "
                      "near-unsupervised for this event")

            stem = f"{city.lower()}__{cyclone.lower()}"
            data = Data(
                x=torch.tensor(X.to_numpy(), dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y_flood=torch.tensor(y_flood, dtype=torch.long),
                mask_flood=torch.tensor(mask_flood, dtype=torch.bool),
                y_depth=torch.tensor(y_depth, dtype=torch.float),
                mask_depth=torch.tensor(mask_depth, dtype=torch.bool),
            )
            data.city, data.cyclone, data.event = city, cyclone, cyclone
            torch.save(data, os.path.join(OUTPUT_DIR, stem + ".pt"))

            manifest.append({
                "city": city, "cyclone": cyclone,
                "n_nodes": len(segment_ids),
                "n_edges": int(edge_index.shape[1]),
                "mean_degree": round(float(degree.mean()), 2),
                "n_usable_flood": int(mask_flood.sum()),
                "n_flooded": int(fc[1]),
                "pct_flooded": round(float(pos_pct), 3),
                "n_usable_depth": int(mask_depth.sum()),
                "file": stem + ".pt",
            })

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "task": "hurdle: y_flood (binary, SAR) + y_depth (metres, FwDET)",
            "feature_names": feature_names_ref,
            "edge_attr_names": ["delta_hand_m", "delta_elev_rank", "dist_km"],
            "split": "leave-one-city-out",
            "source_dir": INPUT_DIR,
            "graphs": manifest,
        }, f, indent=2)

    mf = pd.DataFrame(manifest)
    print(f"\n{'=' * 62}")
    print(mf.to_string(index=False))
    print(f"\nWrote {len(manifest)} graphs to {OUTPUT_DIR}/")
    print(f"Features ({len(feature_names_ref)}): {feature_names_ref}")

    dead = mf[mf.n_flooded == 0]
    thin = mf[(mf.n_flooded > 0) & (mf.pct_flooded < 0.5)]
    if len(dead):
        print(f"\n{len(dead)} event(s) with zero flood labels:")
        print(dead[["city", "cyclone"]].to_string(index=False))
    if len(thin):
        print(f"\n{len(thin)} event(s) below 0.5% flooded:")
        print(thin[["city", "cyclone", "pct_flooded"]].to_string(index=False))
    print("=" * 62)


if __name__ == "__main__":
    main()
