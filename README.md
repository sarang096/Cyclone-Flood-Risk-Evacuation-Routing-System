# Cyclone Road Flooding: GraphSAGE vs Gradient Boosting

Predicts which road segments flood during a cyclone, for **15 cyclone events across 4 Indian coastal cities**. Flood labels come from Sentinel-1 SAR flood extent. Each road segment gets terrain, land-cover and rainfall features and becomes a node in a road graph. The project then asks one question: **does modelling the road network as a graph (GraphSAGE) predict flooding better than a model that looks at each segment on its own (HistGradientBoosting)?**

Short answer: no. The per-segment terrain and rainfall features carry most of the signal, and gradient boosting beat GraphSAGE in 14 of 15 held-out storms.

## Data

| City | Cyclones | Road segments (nodes) | Edges |
|---|---|---|---|
| Chennai | Mandous, Nivar, Vardah | 200,883 | 786,490 |
| Kolkata | Amphan, Bulbul, Remal, Yaas | 143,160 | 545,096 |
| Bhubaneswar | Dana, Fani, Titli, Yaas | 47,990 | 186,134 |
| Puri | Dana, Fani, Titli, Yaas | 6,772 | 25,294 |

- **Label:** flooded or not, from Sentinel-1 SAR flood extent processed in Google Earth Engine and sampled onto OpenStreetMap road segments. Between 2.9% and 31.9% of segments are flooded, depending on the event.
- **Graph:** one node per road segment. Segments that meet at a junction are connected (mean degree ≈ 3.9). Edge attributes are the differences in HAND and elevation rank, and the distance between segments.
- **34 node features:**
  - **Terrain:** HAND, slope, elevation rank, TWI, depth to water table, distance to water.
  - **Land cover:** impervious, vegetation and water fractions.
  - **Rainfall:** event total, peak intensities from 30 min to 24 h, wet hours, and antecedent rain over 24 h, 72 h and 7 days.
  - **Road:** length, degree and OSM highway class.
  - **Neighbours:** HAND and elevation rank averaged over the neighbouring segments.

Full per-event statistics are in [`flood_models/manifest.json`](flood_models/manifest.json).

The per-event CSVs and graph files are over 1 GB, so they aren't in this repo.

## Results

All numbers are **leave-one-event-out**: each model is scored on a storm it never saw during training. PR-AUC is the headline metric because flooded segments are rare, and ROC-AUC looks optimistic when positives are rare. Lift is PR-AUC divided by the base rate.

### 1. GraphSAGE vs HistGBM (15 folds)

Run by `train_graphsage_hurdle_v3.py` on unit-level graphs, where each node is a contiguous group of about 8 road segments. Both models were trained and tested on exactly the same splits.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| GraphSAGE | 0.166 ± 0.116 | 0.600 ± 0.073 |
| **HistGradientBoosting** | **0.224 ± 0.124** | **0.653 ± 0.062** |

The mean base rate is 0.111. HistGBM had the higher PR-AUC in **14 of 15** folds. Adding message passing over the road graph did not beat the per-segment features alone. Per-fold numbers are in [`rotating_loeo_results.json`](flood_models/rotating_loeo_results.json).

### 2. Segment-level HistGBM risk scores

Run by `train_gbm_risk.py` directly on the road segments. For each event, the scores come from a model that never saw that event.

| Event | Segments | Base rate | PR-AUC | ROC-AUC | Lift |
|---|---|---|---|---|---|
| Bhubaneswar – Dana | 47,990 | 0.029 | 0.110 | 0.711 | 3.85 |
| Bhubaneswar – Fani | 47,990 | 0.046 | 0.149 | 0.728 | 3.28 |
| Bhubaneswar – Titli | 47,990 | 0.036 | 0.135 | 0.740 | 3.81 |
| Bhubaneswar – Yaas | 47,990 | 0.267 | 0.442 | 0.725 | 1.66 |
| Chennai – Mandous | 200,883 | 0.113 | 0.171 | 0.571 | 1.51 |
| Chennai – Nivar | 200,883 | 0.076 | 0.130 | 0.576 | 1.72 |
| Chennai – Vardah | 200,883 | 0.036 | 0.089 | 0.609 | 2.49 |
| Kolkata – Amphan | 143,160 | 0.068 | 0.112 | 0.534 | 1.65 |
| Kolkata – Bulbul | 143,160 | 0.076 | 0.147 | 0.626 | 1.94 |
| Kolkata – Remal | 143,160 | 0.125 | 0.211 | 0.626 | 1.69 |
| Kolkata – Yaas | 143,160 | 0.033 | 0.078 | 0.582 | 2.36 |
| Puri – Dana | 6,772 | 0.123 | 0.197 | 0.629 | 1.60 |
| Puri – Fani | 6,772 | 0.103 | 0.250 | 0.684 | 2.43 |
| Puri – Titli | 6,772 | 0.087 | 0.178 | 0.646 | 2.05 |
| Puri – Yaas | 6,772 | 0.319 | 0.494 | 0.675 | 1.55 |
| **Mean** | | **0.102** | **0.193** | **0.644** | **2.24** |

Source: [`loeo_metrics.csv`](flood_models/loeo_metrics.csv).

### 3. Masked-label experiment

SAR misses parts of a city because of revisit gaps, radar look angle and building layover. This experiment tests whether GraphSAGE can fill in flood labels for segments SAR never observed, given the labels it did observe (`train_partial_obs_fixed.py`).

Two masking regimes are compared:
- **Random:** observed labels are scattered across the city.
- **Block:** contiguous regions are hidden, which is how real SAR gaps look.

The table below is from a preliminary run on Chennai: the model was trained on the other cities, and the numbers are mean PR-AUC over Chennai's 3 events.

| Labels observed | Masking | Base rate | Features only | Label propagation | GraphSAGE |
|---|---|---|---|---|---|
| 10% | random | 0.075 | 0.139 | 0.226 | 0.104 |
| 50% | random | 0.075 | 0.143 | 0.435 | 0.107 |
| 75% | random | 0.075 | 0.143 | 0.501 | 0.110 |
| 10% | block | 0.074 | 0.125 | 0.092 | 0.100 |
| 50% | block | 0.070 | 0.113 | 0.111 | 0.091 |
| 75% | block | 0.068 | 0.110 | 0.158 | 0.090 |

- **Random gaps:** simple label propagation from observed neighbours does best.
- **Block gaps:** hidden segments have no observed neighbours, and every method falls back to about the features-only level.
- **GraphSAGE:** it did not beat the features-only baseline in either regime.

Full results, including 25% coverage, are in [`smoke_test_results.json`](flood_models/smoke_test_results.json).

### 4. Label audit

The feature-based and graph-based models all topped out at a similar PR-AUC. `audit_labels.py` checks whether that ceiling comes from weak features or noisy SAR labels. It does three things:
- Measures how consistently the same streets flood across storms.
- Looks for radar artifacts over dense buildings at fixed HAND.
- Checks feature columns for masked or no-data values.

## Pipeline

```
Sentinel-1 SAR flood extent + terrain + rainfall (Google Earth Engine)
  → sampled onto OSM road segments, one CSV per city-event
  → build_graphs_v5.py          segment-level PyG graphs (15 events)
  → build_flood_units.py        contiguous "flood units" (~8 segments each)
    build_graphs_units.py       unit-level PyG graphs
  → train_graphsage_hurdle*.py  GraphSAGE + HistGBM baseline (v1 → v3)
  → train_gbm_risk.py           segment-level HistGBM risk scores
  → train_partial_obs_fixed.py  masked-label experiment
  → audit_labels.py             label-noise audit
```

The Earth Engine export and road-sampling steps are not included in this repo.

## Repository layout

| File | What it does |
|---|---|
| `cyclone_graphs.ipynb` | Original graph-construction notebook |
| `flood_models/build_graphs_v5.py` | Per-event CSVs → segment-level graphs (`graphs_v5/`) |
| `flood_models/build_flood_units.py` | Groups segments into contiguous flood units (constrained regionalisation over the road graph) |
| `flood_models/build_graphs_units.py` | Flood units → unit-level graphs (`graphs_units_v1/`) |
| `flood_models/train_graphsage_hurdle.py` | GraphSAGE v1 |
| `flood_models/train_graphsage_hurdle_v2.py` | v2: LayerNorm, separate loss masks, HistGBM baseline on the same splits |
| `flood_models/train_graphsage_hurdle_v3.py` | v3: 15-fold rotating leave-one-event-out, GraphSAGE vs HistGBM |
| `flood_models/train_gbm_risk.py` | Segment-level HistGBM risk scores, leave-one-event-out |
| `flood_models/train_partial_obs_fixed.py` | Masked-label experiment |
| `flood_models/audit_labels.py` | Label-noise and feature audit |
| `flood_models/*.json`, `*.csv` | Results and graph statistics referenced above |

The GraphSAGE filenames say "hurdle" because the early models also had a flood-depth regression head. That head scored below a mean-only baseline (R² < 0), so it was dropped and the final work is flood/no-flood classification.

## Running

```bash
pip install -r requirements.txt
cd flood_models

python build_graphs_v5.py                                 # 0.05/*.csv       -> graphs_v5/
python build_flood_units.py --csv "0.05/*.csv" --out ./units_v2
python build_graphs_units.py                              # units_v2/        -> graphs_units_v1/
python train_graphsage_hurdle_v3.py                       # graphs_units_v1/ -> rotating_loeo_results.json
python train_gbm_risk.py                                  # graphs_v5/       -> risk_scores/, loeo_metrics.csv
python train_partial_obs_fixed.py --data-root <graphs dir>
python audit_labels.py --csv "0.05/*.csv"
```

Each script expects its data folders (`0.05/`, `graphs_v5/`, …) next to it.
