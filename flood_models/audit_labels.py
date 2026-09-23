#!/usr/bin/env python3
"""
audit_labels.py — is the ceiling weak features, or noisy labels?

Three independent experiments (NFA, GBM-beats-GraphSAGE, unit grouping) have now
landed on the same PR-AUC ceiling. Two explanations remain:

  (A) features are too coarse / collinear  -> fixable with better inputs
  (B) the SAR label is partly artifact     -> not fixable by any model

This script tests (B) directly, plus audits the feature columns for the nodata
signature that already forced you to drop soil_moist_* and wind_max_ms.

  1. CROSS-EVENT PERSISTENCE. Jaccard overlap between each pair of events'
     flooded-segment sets, against the overlap expected if the two events were
     independent at the same base rates. Terrain-driven flooding recurs in the
     same streets; a Jaccard near the independence baseline means the label is
     essentially re-rolled each event and carries little learnable structure.

  2. ARTIFACT SIGNATURE. Flood rate by impervious_frac decile, controlled within
     HAND decile. Sentinel-1 layover/shadow from dense buildings mimics low
     backscatter (= "water"). If flood rate rises with imperviousness *at fixed
     HAND*, that is a radar artifact, not water.

  3. NODATA AUDIT. Per-column share of exactly-zero values, and the ratio of
     high-frequency (neighbour-to-neighbour) variance to total variance. A
     coarse reanalysis field like era5_min_pressure_pa physically cannot vary
     between adjacent streets; if it does, those are masked cells, not values.

USAGE
  python audit_labels.py --csv "./*.csv"
  python audit_labels.py --csv "./*.csv" --city kolkata
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

SUSPECT_COLS = [
    "era5_min_pressure_pa", "era5_max_wind_ms", "soil_moisture_antecedent",
    "soil_moist_0_7cm", "soil_moist_7_28cm", "wind_max_ms", "curve_number",
    "cyclone_track_dist_km", "cyclone_wind_exposure_ms", "runoff_coeff",
    "rainfall_mm", "rain_max_24h_mm", "rain_ante_7d_mm",
    "hand_m", "elevation_m", "twi", "dtw_m", "impervious_frac",
]


def canon(df):
    u, v, k = df["u"].to_numpy(), df["v"].to_numpy(), df["key"].to_numpy()
    return (pd.Series(np.minimum(u, v)).astype(str) + "_"
            + pd.Series(np.maximum(u, v)).astype(str) + "_"
            + pd.Series(k).astype(str)).to_numpy()


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df["road_id"] = canon(df)
    stem = os.path.splitext(os.path.basename(path))[0]
    city = str(df["city"].iloc[0]).lower() if "city" in df else stem.split("_")[0]
    cyc = str(df["cyclone"].iloc[0]).lower() if "cyclone" in df else stem.split("_")[-1]
    agg = {c: "mean" for c in df.columns
           if c in SUSPECT_COLS or c in ("lon", "lat")}
    agg["sar_flood_binary"] = "max"
    seg = df.groupby("road_id", sort=True).agg(agg).reset_index()
    return city, cyc, seg


# ---------------------------------------------------------------- 1. persistence
def persistence(city, events):
    print(f"\n[1] cross-event persistence — {city}")
    print("    Jaccard of flooded-segment sets vs the independence baseline")
    print(f"    {'pair':<26}{'observed':>10}{'independent':>13}{'ratio':>8}")
    names = [c for c, _ in events]
    sets, rates = {}, {}
    for c, s in events:
        y = s.set_index("road_id")["sar_flood_binary"].fillna(0)
        sets[c] = set(y.index[y > 0])
        rates[c] = float(y.mean())

    ratios = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            obs = inter / union if union else np.nan
            pa, pb = rates[a], rates[b]
            exp = (pa * pb) / (pa + pb - pa * pb) if (pa + pb) > 0 else np.nan
            r = obs / exp if exp and exp > 0 else np.nan
            ratios.append(r)
            print(f"    {a+' / '+b:<26}{obs:>10.4f}{exp:>13.4f}{r:>8.2f}x")
    if ratios:
        m = np.nanmean(ratios)
        print(f"    mean ratio {m:.2f}x", end="  ")
        if m < 1.5:
            print("-> labels barely recur across events: little stable, "
                  "terrain-driven signal to learn.")
        elif m < 3:
            print("-> moderate recurrence; a real but partial terrain signal.")
        else:
            print("-> strong recurrence: flooding is location-driven and "
                  "learnable in principle.")


# ---------------------------------------------------------------- 2. artifacts
def artifact_signature(city, cyc, seg):
    if not {"impervious_frac", "hand_m"} <= set(seg.columns):
        return
    print(f"\n[2] artifact signature — {city}/{cyc}")
    print("    flood rate by impervious_frac quintile, within HAND quintile")
    d = seg[["sar_flood_binary", "impervious_frac", "hand_m"]].dropna()
    d = d.assign(
        hq=pd.qcut(d["hand_m"], 5, labels=False, duplicates="drop"),
        iq=pd.qcut(d["impervious_frac"], 5, labels=False, duplicates="drop"),
    )
    t = d.pivot_table(index="hq", columns="iq", values="sar_flood_binary",
                      aggfunc="mean")
    print("    rows = HAND quintile (0 = lowest/wettest), cols = imperviousness")
    print(t.round(4).to_string())
    slopes = []
    for _, row in t.iterrows():
        v = row.dropna().to_numpy()
        if len(v) >= 3:
            slopes.append(np.polyfit(np.arange(len(v)), v, 1)[0])
    if slopes:
        s = float(np.mean(slopes))
        print(f"    mean within-HAND trend across imperviousness: {s:+.4f} / quintile", end="  ")
        if s > 0.005:
            print("-> flood rate RISES with built-up density at equal HAND. "
                  "Consistent with layover/shadow being labelled as water.")
        elif s < -0.005:
            print("-> flood rate falls with built-up density; no layover "
                  "signature (drainage/permeability effect instead).")
        else:
            print("-> flat; no strong artifact signature.")


# ---------------------------------------------------------------- 3. nodata
def nodata_audit(city, cyc, seg):
    print(f"\n[3] nodata / resolution audit — {city}/{cyc}")
    print(f"    {'column':<28}{'%exactly 0':>12}{'%NaN':>8}{'HF var share':>14}")
    # true nearest spatial neighbour of each segment centroid
    from scipy.spatial import cKDTree
    lat0 = np.deg2rad(seg["lat"].mean())
    X = np.c_[np.deg2rad(seg["lon"]) * 6371000 * np.cos(lat0),
              np.deg2rad(seg["lat"]) * 6371000]
    _, nn = cKDTree(X).query(X, k=2)
    nn = nn[:, 1]

    for c in SUSPECT_COLS:
        if c not in seg.columns:
            continue
        v = pd.to_numeric(seg[c], errors="coerce").to_numpy(dtype=float)
        pz = float(np.mean(v == 0)) * 100
        pn = float(np.mean(np.isnan(v))) * 100
        tot = np.nanvar(v)
        hf = np.nanmean((v - v[nn]) ** 2) / 2 if tot > 0 else np.nan
        share = hf / tot if tot > 0 else np.nan
        flag = ""
        if pz > 10:
            flag += "  <- zeros"
        if not np.isnan(share) and share > 0.30 and c.startswith(("era5", "soil", "cyclone")):
            flag += "  <- too rough for a reanalysis field"
        print(f"    {c:<28}{pz:>11.1f}%{pn:>7.1f}%{share:>14.3f}{flag}")
    print("    HF var share = neighbour-to-neighbour variance / total variance.")
    print("    Coarse reanalysis inputs should sit near 0. Values above ~0.3")
    print("    mean the column is carrying mask edges, not measurements.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--city", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.csv!r}")

    by_city = defaultdict(list)
    for p in paths:
        city, cyc, seg = load(p)
        if args.city and city != args.city.lower():
            continue
        print(f"[load] {p}  -> {city}/{cyc}  {len(seg):,} segments  "
              f"flood {seg['sar_flood_binary'].mean():.4f}")
        by_city[city].append((cyc, seg))

    for city, events in by_city.items():
        print("\n" + "=" * 68)
        print(f"=== {city} ({len(events)} events)")
        print("=" * 68)
        if len(events) >= 2:
            persistence(city, events)
        else:
            print("\n[1] only one event for this city — persistence skipped")
        cyc, seg = events[0]
        artifact_signature(city, cyc, seg)
        nodata_audit(city, cyc, seg)


if __name__ == "__main__":
    main()