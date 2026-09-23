#!/usr/bin/env python3
"""
build_flood_units.py — group nearby road segments into "flood units" (mapping units)

WHY (literature context)
------------------------
In flood / landslide susceptibility modelling the choice of *mapping unit* is a
first-class modelling decision, not a preprocessing detail. The standard options
are grid cells, terrain-derived units (slope units / HRUs), and
administrative-or-street units; comparative studies repeatedly find the unit
choice changes model performance and factor importance more than the classifier
does (Reichenbach et al. 2018 review; Cama et al. 2016 on grid size; multi-scale
urban flood work using a catchment / street / grid tier system, RS 2026).

Two families are implemented here:

  1. "grid"   — snap segment midpoints to a fixed cell (default 90 m, i.e. the
                native resolution tier of your HAND/DEM stack). Cheap baseline,
                the classic grid-cell mapping unit.

  2. "region" — spatially-CONSTRAINED clustering (regionalization). Segments are
                clustered on terrain features under a hard contiguity constraint,
                so a unit is always a connected piece of the road network, never
                a set of look-alike streets on opposite sides of the city.
                This follows the SKATER / REDCAP / max-p family (Assuncao et al.
                2006; Guo 2008; Duque, Anselin & Rey 2012), which is what recent
                flood-zoning work uses when it needs contiguous homogeneous
                regions (SKATER for flood risk zoning, Geneva Papers 2025;
                homogeneous flash-flood regions, Land 2026 — that paper also
                shows unconstrained k-means fragments badly, which is exactly why
                the contiguity constraint is here).

                Implementation: minimum spanning tree over the road-adjacency
                graph with edge cost = feature dissimilarity, then size- and
                span-capped agglomeration along the MST (a scalable variant of
                SKATER's tree-edge removal, with max-p's minimum-size floor).

DESIGN NOTE THAT MATTERS FOR YOUR PIPELINE
------------------------------------------
Units are built ONCE PER CITY from *static* terrain features only, then every
cyclone event is aggregated onto that same fixed partition. If you clustered per
event, the units would differ between events and leave-one-event-out evaluation
would be meaningless.

BASE-RATE WARNING
-----------------
Aggregating labels with max() raises the positive rate (a unit is "flooded" if
any member segment is). That shifted your class balance ~1.55x last time and
confounded the comparison. This script writes BOTH `flood_any` and the
length-weighted `flood_frac`, and prints the base rate under each, so you can
hold class weighting fixed across the segment-level and unit-level runs.

USAGE
-----
  # one city, all its events
  python build_flood_units.py --csv "road_csvs/kolkata_*.csv" --out ./units

  # tune granularity
  python build_flood_units.py --csv "road_csvs/*.csv" --out ./units \
      --method region --max-size 10 --max-span-m 250

  # grid baseline at the DEM tier
  python build_flood_units.py --csv "road_csvs/*.csv" --out ./units \
      --method grid --cell-m 90

OUTPUTS (per city)
------------------
  <city>_segment_to_unit.csv   road_id -> unit_id   (broadcast predictions back
                               onto segments for routing; you keep full street
                               resolution downstream)
  <city>_<cyclone>_units.csv   one row per (unit, event), aggregated features
  <city>_unit_graph.npz        unit_id, lon, lat, edge_index (2, E) contiguity
                               graph over units — drop-in for GraphSAGE
  <city>_unit_report.json      diagnostics
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# --------------------------------------------------------------------------
# Feature groups. Static = fixed within a city across events -> used to build
# the partition. Dynamic = varies per event -> aggregated, never clustered on.
# --------------------------------------------------------------------------
STATIC_FEATURES = [
    "elevation_m", "slope_deg", "hand_m", "twi", "dtw_m",
    "dist_to_water_m", "impervious_frac", "veg_frac", "water_frac",
]

# Default feature set for the contiguity-constrained clustering. Terrain +
# drainage only: these are the variables that actually differentiate one street
# from its neighbour. Weather covariates are city-wide constants at your
# resolution, so clustering on them would produce noise-driven splits.
CLUSTER_FEATURES = ["hand_m", "elevation_m", "dtw_m", "twi", "slope_deg"]

DYNAMIC_FEATURES = [
    "rain_total_mm", "rain_peak_30min_mm", "rain_wet_hours",
    "rain_max_1h_mm", "rain_max_2h_mm", "rain_max_3h_mm",
    "rain_max_6h_mm", "rain_max_24h_mm",
    "rain_ante_24h_mm", "rain_ante_72h_mm", "rain_ante_7d_mm",
    "soil_moist_0_7cm", "soil_moist_7_28cm", "wind_max_ms",
    "rainfall_mm", "effective_rain_mm", "runoff_coeff", "curve_number",
    "rain_peak_intensity_mmhr", "rain_time_to_peak_hr",
    "rain_antecedent_7d_mm", "soil_moisture_antecedent",
    "era5_max_wind_ms", "era5_min_pressure_pa",
    "cyclone_track_dist_km", "cyclone_wind_exposure_ms",
]

EARTH_R = 6371000.0


# ==========================================================================
# I/O and segment construction
# ==========================================================================
def load_event(path):
    df = pd.read_csv(path, low_memory=False)
    need = {"u", "v", "key", "lon", "lat", "sar_flood_binary"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
    if "city" not in df.columns:
        df["city"] = os.path.basename(path).split("_")[0]
    if "cyclone" not in df.columns:
        df["cyclone"] = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    return df


def canonical_road_id(df):
    """Undirected segment id: (min(u,v), max(u,v), key).

    OSM stores each two-way street twice (u->v and v->u). Left uncanonicalized
    those become two rows with identical geometry, which both inflates the graph
    and leaks a perfect duplicate across any train/test split.
    """
    u = df["u"].to_numpy()
    v = df["v"].to_numpy()
    k = df["key"].to_numpy()
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    return (
        pd.Series(lo).astype(str) + "_" + pd.Series(hi).astype(str)
        + "_" + pd.Series(k).astype(str)
    ).to_numpy(), lo, hi


def points_to_segments(df):
    """Collapse sample points to one row per canonical road segment."""
    rid, lo, hi = canonical_road_id(df)
    df = df.assign(_rid=rid, _u=lo, _v=hi)

    num_cols = [c for c in df.columns
                if c in STATIC_FEATURES + DYNAMIC_FEATURES]
    agg = {c: "mean" for c in num_cols}
    agg.update({
        "lon": "mean", "lat": "mean",
        "_u": "first", "_v": "first",
        "sar_flood_binary": "max",       # any flooded sample -> segment flooded
    })
    if "length_m" in df.columns:
        agg["length_m"] = "mean"
    if "highway_type" in df.columns:
        agg["highway_type"] = "first"

    # depth only where the FwDET water-surface elevation was actually assigned
    if "flood_depth_m" in df.columns and "wse_assigned" in df.columns:
        df["_depth_valid"] = np.where(
            (df["wse_assigned"] > 0) & (df["flood_depth_m"] > 0),
            df["flood_depth_m"], np.nan)
        agg["_depth_valid"] = "mean"

    seg = df.groupby("_rid", sort=False).agg(agg).reset_index()
    seg = seg.rename(columns={"_rid": "road_id", "_depth_valid": "flood_depth_m"})
    if "length_m" not in seg.columns:
        seg["length_m"] = 1.0
    seg["length_m"] = seg["length_m"].fillna(1.0).clip(lower=1e-3)
    return seg


def local_xy(lon, lat):
    """Equirectangular projection to metres, accurate enough over one city."""
    lat0 = np.deg2rad(np.mean(lat))
    x = np.deg2rad(lon) * EARTH_R * np.cos(lat0)
    y = np.deg2rad(lat) * EARTH_R
    return x - x.min(), y - y.min()


# ==========================================================================
# Contiguity: two segments are neighbours if they share an intersection
# ==========================================================================
def build_adjacency(seg):
    """Return (src, dst) index arrays over seg rows. No network calls — the
    topology is already encoded in the OSM u/v node ids in the CSV."""
    node_to_segs = defaultdict(list)
    for i, (a, b) in enumerate(zip(seg["_u"].to_numpy(), seg["_v"].to_numpy())):
        node_to_segs[a].append(i)
        node_to_segs[b].append(i)

    src, dst = [], []
    for segs in node_to_segs.values():
        if len(segs) < 2 or len(segs) > 12:   # skip pathological mega-nodes
            continue
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                src.append(segs[a])
                dst.append(segs[b])
    if not src:
        return np.array([], dtype=int), np.array([], dtype=int)
    e = np.unique(np.stack([np.array(src), np.array(dst)]).T, axis=0)
    return e[:, 0], e[:, 1]


# ==========================================================================
# Union-find with size + spatial-span constraints
# ==========================================================================
class ConstrainedUF:
    """Union-find that refuses merges violating a size cap or a spatial-span cap.

    The span cap is what keeps units *spatially compact* — without it, MST
    agglomeration on feature similarity can string a unit out along a whole
    ridge line, which is not what "nearby segments" means.
    """

    def __init__(self, x, y, max_size, max_span_m):
        n = len(x)
        self.p = np.arange(n)
        self.sz = np.ones(n, dtype=np.int64)
        self.minx, self.maxx = x.copy(), x.copy()
        self.miny, self.maxy = y.copy(), y.copy()
        self.max_size = max_size
        self.max_span = max_span_m

    def find(self, a):
        p = self.p
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:      # path compression
            p[a], a = root, p[a]
        return root

    def _would_fit(self, ra, rb):
        if self.sz[ra] + self.sz[rb] > self.max_size:
            return False
        if self.max_span is not None:
            dx = max(self.maxx[ra], self.maxx[rb]) - min(self.minx[ra], self.minx[rb])
            dy = max(self.maxy[ra], self.maxy[rb]) - min(self.miny[ra], self.miny[rb])
            if np.hypot(dx, dy) > self.max_span:
                return False
        return True

    def union(self, a, b, force=False):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if not force and not self._would_fit(ra, rb):
            return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        self.minx[ra] = min(self.minx[ra], self.minx[rb])
        self.maxx[ra] = max(self.maxx[ra], self.maxx[rb])
        self.miny[ra] = min(self.miny[ra], self.miny[rb])
        self.maxy[ra] = max(self.maxy[ra], self.maxy[rb])
        return True


# ==========================================================================
# Clustering methods
# ==========================================================================
def cluster_grid(seg, cell_m):
    x, y = local_xy(seg["lon"].to_numpy(), seg["lat"].to_numpy())
    gx = np.floor(x / cell_m).astype(np.int64)
    gy = np.floor(y / cell_m).astype(np.int64)
    keys = gx * 1_000_003 + gy
    _, labels = np.unique(keys, return_inverse=True)
    return labels


def cluster_region(seg, src, dst, feature_cols, max_size, min_size, max_span_m,
                   verbose=True):
    """MST-based spatially-constrained regionalization.

    1. contiguity graph  (segments sharing an intersection)
    2. edge cost = Euclidean distance in standardized terrain-feature space
    3. minimum spanning forest  -> keeps only the cheapest way to connect
    4. agglomerate along MST edges cheapest-first, refusing any merge that
       breaks the size cap or the spatial-span cap  (SKATER-style tree pruning
       expressed as bottom-up agglomeration; O(E log E) instead of SKATER's
       repeated SSD-optimal edge search, which does not scale to 10^5 nodes)
    5. merge undersized units into their cheapest neighbour  (max-p floor)
    """
    n = len(seg)
    if len(src) == 0:
        return np.arange(n)

    F = seg[feature_cols].to_numpy(dtype=np.float64)
    F = np.nan_to_num(F, nan=np.nanmedian(F, axis=0))
    sd = F.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (F - F.mean(axis=0)) / sd

    cost = np.linalg.norm(Z[src] - Z[dst], axis=1)
    cost = np.maximum(cost, 1e-9)     # MST treats 0 as "no edge"

    mst = minimum_spanning_tree(
        coo_matrix((cost, (src, dst)), shape=(n, n))
    ).tocoo()
    order = np.argsort(mst.data)
    m_src, m_dst, m_cost = mst.row[order], mst.col[order], mst.data[order]
    if verbose:
        print(f"    contiguity edges {len(src):,} -> MST edges {len(m_src):,}")

    x, y = local_xy(seg["lon"].to_numpy(), seg["lat"].to_numpy())
    uf = ConstrainedUF(x, y, max_size, max_span_m)

    # pass 1: cheapest-first agglomeration under constraints
    for a, b in zip(m_src, m_dst):
        uf.union(a, b)

    # pass 2: max-p style minimum-size floor — absorb undersized units into
    # whichever neighbour the cheapest MST edge points at
    if min_size > 1:
        for a, b in zip(m_src, m_dst):
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            if uf.sz[ra] < min_size or uf.sz[rb] < min_size:
                uf.union(a, b, force=True)

    roots = np.array([uf.find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


# ==========================================================================
# Aggregation onto the fixed partition
# ==========================================================================
def aggregate_units(seg, labels, feature_cols):
    """Length-weighted aggregation of every feature onto units."""
    df = seg.copy()
    df["_lab"] = labels
    df["_w"] = df["length_m"].astype(float)
    g = df.groupby("_lab", sort=True)

    units = pd.DataFrame({
        "unit_id": np.sort(df["_lab"].unique()),
        "n_segments": g.size().to_numpy(),
        "total_length_m": g["_w"].sum().to_numpy(),
        "lon": g["lon"].mean().to_numpy(),
        "lat": g["lat"].mean().to_numpy(),
    })

    # length-weighted mean of every feature, vectorized (NaN-safe):
    #   sum(v_i * w_i) / sum(w_i over non-null v_i)
    cols = [c for c in feature_cols if c in df.columns]
    if cols:
        V = df[cols].apply(pd.to_numeric, errors="coerce")
        num = V.mul(df["_w"], axis=0).groupby(labels, sort=True).sum(min_count=1)
        den = V.notna().mul(df["_w"], axis=0).groupby(labels, sort=True).sum()
        wm = (num / den.replace(0, np.nan)).reset_index(drop=True)
        for c in cols:
            units[c] = wm[c].to_numpy()

    # ---- labels: BOTH conventions, so class weighting can be held fixed ----
    fl = df["sar_flood_binary"].fillna(0).to_numpy()
    df["_fl_w"] = fl * df["_w"]
    units["flood_any"] = g["sar_flood_binary"].max().to_numpy()
    units["flood_frac"] = (g["_fl_w"].sum() / g["_w"].sum()).to_numpy()
    units["flood_maj"] = (units["flood_frac"] >= 0.5).astype(int)
    units["n_flooded_segments"] = g["sar_flood_binary"].sum().to_numpy()

    # ---- depth: mean over members with a genuinely assigned WSE ----
    if "flood_depth_m" in df.columns:
        units["flood_depth_m"] = g["flood_depth_m"].mean().to_numpy()
        units["flood_depth_max_m"] = g["flood_depth_m"].max().to_numpy()
        units["n_depth_labeled"] = g["flood_depth_m"].count().to_numpy()
        units["depth_mask"] = (units["n_depth_labeled"] > 0).astype(int)

    return units


def unit_edge_index(labels, src, dst, n_units):
    """Contiguity graph over units, inherited from segment adjacency."""
    a, b = labels[src], labels[dst]
    keep = a != b
    a, b = a[keep], b[keep]
    e = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)]).T, axis=0)
    if len(e) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([
        np.concatenate([e[:, 0], e[:, 1]]),
        np.concatenate([e[:, 1], e[:, 0]]),
    ]).astype(np.int64)


# ==========================================================================
# Diagnostics — is the grouping actually keeping the signal?
# ==========================================================================
def diagnostics(seg, labels, feature_cols):
    n, k = len(seg), labels.max() + 1
    sizes = np.bincount(labels)
    fl = seg["sar_flood_binary"].fillna(0).to_numpy()

    # label purity: does a unit's membership agree on flooded/dry?
    dfp = pd.DataFrame({"lab": labels, "fl": fl})
    per = dfp.groupby("lab")["fl"].agg(["mean", "size"])
    pure = ((per["mean"] == 0) | (per["mean"] == 1)).mean()

    # eta^2: share of each feature's segment-level variance that survives
    # aggregation (1.0 = units capture it all, 0.0 = grouping destroyed it)
    eta = {}
    for c in feature_cols:
        if c not in seg.columns:
            continue
        v = pd.to_numeric(seg[c], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(v)) or np.nanstd(v) == 0:
            eta[c] = float("nan")
            continue
        tot = np.nanvar(v)
        gm = pd.DataFrame({"lab": labels, "v": v}).groupby("lab")["v"].transform("mean")
        within = np.nanvar(v - gm.to_numpy())
        eta[c] = float(1 - within / tot) if tot > 0 else float("nan")

    # spatial compactness
    x, y = local_xy(seg["lon"].to_numpy(), seg["lat"].to_numpy())
    d = pd.DataFrame({"lab": labels, "x": x, "y": y}).groupby("lab").agg(
        ["min", "max"])
    span = np.hypot(d[("x", "max")] - d[("x", "min")],
                    d[("y", "max")] - d[("y", "min")]).to_numpy()

    return {
        "n_segments": int(n),
        "n_units": int(k),
        "reduction_x": round(n / k, 2),
        "unit_size_mean": round(float(sizes.mean()), 2),
        "unit_size_median": int(np.median(sizes)),
        "unit_size_max": int(sizes.max()),
        "unit_span_m_mean": round(float(span.mean()), 1),
        "unit_span_m_p95": round(float(np.percentile(span, 95)), 1),
        "label_purity": round(float(pure), 4),
        "base_rate_segment": round(float(fl.mean()), 4),
        "base_rate_unit_any": round(float((per["mean"] > 0).mean()), 4),
        "base_rate_unit_majority": round(float((per["mean"] >= 0.5).mean()), 4),
        "variance_retained_eta2": {c: (None if np.isnan(v) else round(v, 3))
                                   for c, v in eta.items()},
    }


# ==========================================================================
# Optional: does the grouping actually help?
# ==========================================================================
def evaluate_grouping(seg, labels, feature_cols, block_m=2000.0, test_frac=0.25,
                      seed=0, label_rule="any", event_col=None, holdout=None):
    """Fair segment-level comparison of segment-model vs unit-model.

    Three things this gets right that a naive comparison does not:

    1. BOTH models are scored on the SAME segment-level test rows. The unit
       model's prediction is broadcast back onto its member segments before
       scoring. Comparing PR-AUC computed on 143k segment rows against PR-AUC
       computed on 25k unit rows compares two different populations and tells
       you nothing.
    2. The split is made at UNIT level over spatial blocks, so no unit straddles
       train and test — otherwise the unit model trains on rows containing its
       own test segments.
    3. Class weight is fixed once from the segment base rate and reused for both
       runs. Recomputing `balanced` per dataset confounds the grouping effect
       with a weighting change (max-aggregation raises the positive rate).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score, \
        precision_score, recall_score

    cols = [c for c in feature_cols if c in seg.columns
            and pd.to_numeric(seg[c], errors="coerce").std() > 0]
    X = seg[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = seg["sar_flood_binary"].fillna(0).to_numpy().astype(int)
    w = seg["length_m"].to_numpy(dtype=float)

    x, yy = local_xy(seg["lon"].to_numpy(), seg["lat"].to_numpy())
    ucent = pd.DataFrame({"lab": labels, "x": x, "y": yy}).groupby("lab").mean()

    if event_col is not None:
        # ---- leave-one-event-out: hold out one whole cyclone ---------------
        # This is the split that matches the deployment question ("new storm,
        # roads we already know"). A unit legitimately appears in both train
        # and test here, exactly as a segment does, so the comparison stays
        # symmetric between the two models.
        ev = seg[event_col].to_numpy()
        held = holdout if holdout is not None else sorted(pd.unique(ev))[-1]
        is_test = ev == held
        unit_is_test = None
    else:
        # ---- spatially blocked split, made at UNIT level -------------------
        blk = (np.floor(ucent["x"] / block_m).astype(int) * 100003
               + np.floor(ucent["y"] / block_m).astype(int))
        ublk = np.unique(blk)
        rng = np.random.default_rng(seed)
        test_blocks = set(rng.choice(ublk, size=max(1, int(len(ublk) * test_frac)),
                                     replace=False).tolist())
        unit_is_test = blk.isin(test_blocks)
        is_test = unit_is_test.to_numpy()[labels]
    tr, te = ~is_test, is_test
    if y[te].sum() == 0 or y[tr].sum() == 0:
        return {"error": "degenerate split — try another --eval-seed"}

    base = y[tr].mean()
    pos_w = (1 - base) / max(base, 1e-9)
    sw = np.where(y == 1, pos_w, 1.0)

    def fit_predict(Xtr, ytr, wtr, Xte):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr,
                                                   sample_weight=wtr)
        gb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=0
        ).fit(Xtr, ytr, sample_weight=wtr)
        return (lr.predict_proba(sc.transform(Xte))[:, 1],
                gb.predict_proba(Xte)[:, 1])

    # ---- A: segment-level model ------------------------------------------
    p_lr_seg, p_gb_seg = fit_predict(X[tr], y[tr], sw[tr], X[te])

    # ---- B: unit-level model, broadcast back onto segments ----------------
    dfu = pd.DataFrame(X, columns=cols)
    dfu["_lab"] = labels
    dfu["_y"], dfu["_w"] = y, w
    dfu["_yw"] = y * w
    dfu["_test"] = is_test
    if event_col is not None:
        dfu["_ev"] = seg[event_col].to_numpy()
        keys = ["_lab", "_ev"]
    else:
        keys = ["_lab"]

    gu = dfu.groupby(keys, sort=True)
    Xu = gu[cols].mean()
    frac = (gu["_yw"].sum() / gu["_w"].sum()).reindex(Xu.index)
    yu = ((frac > 0) if label_rule == "any" else (frac >= 0.5)).astype(int).to_numpy()
    u_is_test = (gu["_test"].max().reindex(Xu.index).to_numpy().astype(bool))
    swu = np.where(yu == 1, pos_w, 1.0)

    Xu_np = Xu.to_numpy()
    utr = ~u_is_test
    p_lr_all = np.zeros(len(Xu)); p_gb_all = np.zeros(len(Xu))
    p_lr_u, p_gb_u = fit_predict(Xu_np[utr], yu[utr], swu[utr], Xu_np[u_is_test])
    p_lr_all[u_is_test], p_gb_all[u_is_test] = p_lr_u, p_gb_u

    # map each TEST segment back to its unit row
    row_of = {k: i for i, k in enumerate(Xu.index)}
    if event_col is not None:
        seg_keys = list(zip(labels[te], dfu["_ev"].to_numpy()[te]))
    else:
        seg_keys = list(labels[te])
    idx = np.array([row_of[k] for k in seg_keys])
    p_lr_bc, p_gb_bc = p_lr_all[idx], p_gb_all[idx]

    def score(name, p):
        yt = y[te]
        return {
            "model": name,
            "PR_AUC": round(float(average_precision_score(yt, p)), 4),
            "ROC_AUC": round(float(roc_auc_score(yt, p)), 4),
            "recall@0.5": round(float(recall_score(yt, (p >= 0.5).astype(int),
                                                   zero_division=0)), 4),
            "precision@0.5": round(float(precision_score(yt, (p >= 0.5).astype(int),
                                                         zero_division=0)), 4),
        }

    return {
        "test_segments": int(te.sum()),
        "test_base_rate": round(float(y[te].mean()), 4),
        "fixed_pos_weight": round(float(pos_w), 2),
        "n_features": len(cols),
        "results": [
            score("segment-level  LR", p_lr_seg),
            score("segment-level GBM", p_gb_seg),
            score(f"unit-level  LR ({label_rule})", p_lr_bc),
            score(f"unit-level GBM ({label_rule})", p_gb_bc),
        ],
    }


def _print_eval(ev):
    if "error" in ev:
        print(f"  {ev['error']}")
        return
    print(f"  test: {ev['test_segments']:,} segments, "
          f"base rate {ev['test_base_rate']:.4f}, "
          f"fixed pos_weight {ev['fixed_pos_weight']}")
    print(f"  {'model':<28}{'PR-AUC':>9}{'ROC-AUC':>9}"
          f"{'recall@.5':>11}{'prec@.5':>9}{'lift':>8}")
    for r in ev["results"]:
        lift = r["PR_AUC"] / max(ev["test_base_rate"], 1e-9)
        print(f"  {r['model']:<28}{r['PR_AUC']:>9.4f}"
              f"{r['ROC_AUC']:>9.4f}{r['recall@0.5']:>11.4f}"
              f"{r['precision@0.5']:>9.4f}{lift:>8.2f}x")


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="glob for per-event road CSVs, e.g. 'road_csvs/*.csv'")
    ap.add_argument("--out", default="./units")
    ap.add_argument("--method", choices=["region", "grid"], default="region")
    ap.add_argument("--cell-m", type=float, default=90.0,
                    help="grid method: cell size in metres (default 90 = DEM tier)")
    ap.add_argument("--max-size", type=int, default=8,
                    help="region method: max segments per unit")
    ap.add_argument("--min-size", type=int, default=2,
                    help="region method: min segments per unit (max-p floor)")
    ap.add_argument("--max-span-m", type=float, default=250.0,
                    help="region method: max bounding-box diagonal of a unit")
    ap.add_argument("--cluster-features", nargs="*", default=CLUSTER_FEATURES)
    ap.add_argument("--eval", action="store_true",
                    help="run the fair segment-vs-unit comparison (spatially "
                         "blocked split, both scored on the same segment rows)")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--eval-block-m", type=float, default=2000.0)
    ap.add_argument("--eval-label", choices=["any", "majority"], default="any")
    ap.add_argument("--eval-rotate", action="store_true",
                    help="rotate EVERY cyclone as the held-out fold and average, "
                         "instead of testing one arbitrary event")
    ap.add_argument("--eval-split", choices=["auto", "event", "spatial"],
                    default="auto",
                    help="'event' = leave-one-cyclone-out (matches your "
                         "deployment question; needs >1 event per city); "
                         "'spatial' = blocked holdout within one event")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.csv!r}")
    os.makedirs(args.out, exist_ok=True)

    # ---- load every event, grouped by city -------------------------------
    by_city = defaultdict(list)
    for p in paths:
        print(f"[load] {p}")
        df = load_event(p)
        seg = points_to_segments(df)
        city = str(df["city"].iloc[0]).lower()
        cyc = str(df["cyclone"].iloc[0]).lower()
        seg["city"], seg["cyclone"] = city, cyc
        by_city[city].append((cyc, seg))
        print(f"       {len(df):,} points -> {len(seg):,} segments "
              f"(flood rate {seg['sar_flood_binary'].mean():.4f})")

    for city, events in by_city.items():
        print(f"\n=== {city}  ({len(events)} event(s)) ===")

        # ---- fixed partition from STATIC features, shared by all events ----
        # union of segments across events; static features are identical, so
        # the first event that contains a segment defines its geometry.
        base = pd.concat([s for _, s in events], ignore_index=True)
        base = base.drop_duplicates(subset="road_id", keep="first").reset_index(drop=True)
        print(f"  road network: {len(base):,} unique segments")

        src, dst = build_adjacency(base)
        print(f"  contiguity edges: {len(src):,}")

        cf = [c for c in args.cluster_features if c in base.columns]
        if args.method == "grid":
            labels = cluster_grid(base, args.cell_m)
            print(f"  grid @ {args.cell_m:.0f} m -> {labels.max()+1:,} units")
        else:
            print(f"  clustering on {cf}")
            labels = cluster_region(base, src, dst, cf,
                                    args.max_size, args.min_size, args.max_span_m)
            print(f"  regionalization -> {labels.max()+1:,} units")

        base["unit_id"] = labels
        seg2unit = base[["road_id", "unit_id", "lon", "lat", "length_m"]]
        seg2unit.to_csv(os.path.join(args.out, f"{city}_segment_to_unit.csv"),
                        index=False)

        mapping = dict(zip(base["road_id"], labels))
        n_units = int(labels.max() + 1)
        ei = unit_edge_index(labels, src, dst, n_units)
        np.savez_compressed(
            os.path.join(args.out, f"{city}_unit_graph.npz"),
            unit_id=np.arange(n_units),
            lon=base.groupby("unit_id")["lon"].mean().reindex(range(n_units)).to_numpy(),
            lat=base.groupby("unit_id")["lat"].mean().reindex(range(n_units)).to_numpy(),
            edge_index=ei,
        )
        print(f"  unit graph: {n_units:,} nodes, {ei.shape[1]:,} directed edges, "
              f"mean degree {ei.shape[1]/max(n_units,1):.2f}")

        feat_cols = [c for c in STATIC_FEATURES + DYNAMIC_FEATURES
                     if c in base.columns]

        # ---- aggregate each event onto the SAME partition ------------------
        for cyc, seg in events:
            lab = seg["road_id"].map(mapping)
            seg = seg[lab.notna()].copy()
            lab = lab[lab.notna()].to_numpy().astype(int)
            units = aggregate_units(seg, lab, feat_cols)
            units.insert(0, "city", city)
            units.insert(1, "cyclone", cyc)
            fp = os.path.join(args.out, f"{city}_{cyc}_units.csv")
            units.to_csv(fp, index=False)
            print(f"  [{cyc}] {len(seg):,} segments -> {len(units):,} unit-rows  "
                  f"seg base {seg['sar_flood_binary'].mean():.4f} | "
                  f"unit(any) {units['flood_any'].mean():.4f} | "
                  f"unit(maj) {units['flood_maj'].mean():.4f}")

        # ---- diagnostics on the largest event ------------------------------
        cyc, seg = max(events, key=lambda t: len(t[1]))
        lab = seg["road_id"].map(mapping)
        seg_d = seg[lab.notna()]
        rep = diagnostics(seg_d, lab[lab.notna()].to_numpy().astype(int), feat_cols)
        rep["method"] = args.method
        rep["diagnostic_event"] = cyc
        rep["params"] = {k: v for k, v in vars(args).items() if k != "csv"}
        with open(os.path.join(args.out, f"{city}_unit_report.json"), "w") as f:
            json.dump(rep, f, indent=2)

        print(f"\n  --- diagnostics ({cyc}) ---")
        print(f"  {rep['n_segments']:,} segments -> {rep['n_units']:,} units "
              f"({rep['reduction_x']}x reduction)")
        print(f"  unit size: median {rep['unit_size_median']}, max {rep['unit_size_max']}")
        print(f"  unit span: mean {rep['unit_span_m_mean']} m, "
              f"p95 {rep['unit_span_m_p95']} m")
        print(f"  label purity: {rep['label_purity']:.3f} "
              f"(share of units whose members all agree)")
        print(f"  base rate: segment {rep['base_rate_segment']:.4f} -> "
              f"unit-any {rep['base_rate_unit_any']:.4f} "
              f"({rep['base_rate_unit_any']/max(rep['base_rate_segment'],1e-9):.2f}x) | "
              f"unit-majority {rep['base_rate_unit_majority']:.4f}")
        print("  variance retained (eta^2, higher = signal survives grouping):")
        for c, v in rep["variance_retained_eta2"].items():
            if v is not None:
                print(f"      {c:<28} {v:.3f}")

        if args.eval:
            split = args.eval_split
            if split == "auto":
                split = "event" if len(events) > 1 else "spatial"
            if split == "event" and len(events) < 2:
                print("\n  [eval] only one event for this city — falling back "
                      "to a spatially blocked split")
                split = "spatial"

            if split == "event":
                allseg = pd.concat([s for _, s in events], ignore_index=True)
                allseg = allseg[allseg["road_id"].isin(mapping)].copy()
                elab = allseg["road_id"].map(mapping).to_numpy().astype(int)
                folds = (sorted(allseg["cyclone"].unique())
                         if args.eval_rotate
                         else [sorted(allseg["cyclone"].unique())[-1]])
                per_fold = []
                for held in folds:
                    print(f"\n  --- segment vs unit (leave-one-event-out, "
                          f"held out: {held}) ---")
                    e = evaluate_grouping(allseg, elab, feat_cols,
                                          label_rule=args.eval_label,
                                          event_col="cyclone", holdout=held)
                    e["holdout"] = held
                    per_fold.append(e)
                    _print_eval(e)
                ev = {"folds": per_fold}
                if len(per_fold) > 1:
                    print(f"\n  --- mean over {len(per_fold)} folds ---")
                    names = [r["model"] for r in per_fold[0]["results"]]
                    print(f"  {'model':<28}{'PR-AUC':>9}{'ROC-AUC':>9}"
                          f"{'lift':>8}")
                    for i, nm in enumerate(names):
                        pr = np.mean([f["results"][i]["PR_AUC"] for f in per_fold])
                        rc = np.mean([f["results"][i]["ROC_AUC"] for f in per_fold])
                        lift = np.mean([f["results"][i]["PR_AUC"] / f["test_base_rate"]
                                        for f in per_fold])
                        print(f"  {nm:<28}{pr:>9.4f}{rc:>9.4f}{lift:>8.2f}x")
                    ev["mean_PR_AUC"] = {
                        nm: round(float(np.mean([f["results"][i]["PR_AUC"]
                                                 for f in per_fold])), 4)
                        for i, nm in enumerate(names)}
            else:
                print(f"\n  --- segment vs unit comparison "
                      f"({cyc}, spatially blocked {args.eval_block_m:.0f} m) ---")
                ev = evaluate_grouping(
                    seg_d, lab[lab.notna()].to_numpy().astype(int), feat_cols,
                    block_m=args.eval_block_m, seed=args.eval_seed,
                    label_rule=args.eval_label)
                _print_eval(ev)
            ev["split"] = split
            rep["evaluation"] = ev
            with open(os.path.join(args.out, f"{city}_unit_report.json"), "w") as f:
                json.dump(rep, f, indent=2)

    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()