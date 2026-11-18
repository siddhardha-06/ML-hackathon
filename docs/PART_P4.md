# Part P4: EDA, FastAPI backend, evaluator UI, deployment & report

**Owner:** P4. **Code:** `eda/`, `api/`, `ui/`, `team/`, `README.md`, `requirements.txt`. **Results:** `results/eda/`.
**Report sections:** 2.2 (EDA plots, missing values, Folium map), 7 (API/UI/deployment), plus assembling the final PDF.

## What you must be able to explain

### 0. EDA findings (`eda/run_eda.py` → `results/eda/data_quality.md`, figures in `results/eda/figures/`, map in `results/eda/maps/`)
- 80 train trips with 208,417 GPS points; 10 student-test trips with 19,160 points. No NaNs and no
  out-of-order timestamps.
- 96.5% of steps are exactly 1 s. Gaps over 10 minutes occur in trips 11, 21, 42, 69, 78 and 90
  (the longest is 42.5 hours in trip 90).
- Median GPS-to-reference distance is 5.0 m on reliably labelled train points (GPS noise).
- Networks range from 299 to 155,003 nodes. There are no isolated nodes. The largest strongly connected
  component holds 95% of nodes on average.
- Reference routes are not always connected: 11 trips (train and student test) have consecutive route arcs
  that don't share a node.

- Plots: trip length and sampling interval, GPS-to-reference distance (noise), candidate recall vs radius,
  network size, unreliable labels per trip, trajectory examples. Every one is generated from the data by `eda/run_eda.py`.
- Run: `python -m eda.run_eda` (after P1's `python -m features.build_dataset`).

### 1. Two-laptop architecture
```
Laptop 2 (UI, Streamlit)                       Laptop 1 (API, FastAPI + model + road networks + references)
  GET /trips  ───────────────────────────────▶  lists folders in PATHSNAP_EVAL_DIR
  GET /predict/{trip_id}  ───────────────────▶  load trip → candidates (KD-tree) → LightGBM probabilities
                                                 → HMM/Viterbi → gap-filled route → score vs reference
  ◀── JSON: GPS trace, predicted route polyline, metrics, timings   (reference route never sent)
```
- The UI holds **no data and no model**. It only shows what the API returns, so outputs cannot be hardcoded.
- **Ground truth stays server-side.** `api/main.py::_score` derives the per-point reference labels on
  Laptop 1 and returns only accuracy numbers.
- Every request is logged in Laptop 1's console with the trip, Top-1, continuity, route F1 and time.
  Evaluators ask to see this log.

### 2. Endpoints (`api/main.py`)
| endpoint | returns |
|---|---|
| `GET /health` | model type, decoder parameters, eval folder, number of trips |
| `GET /trips` | `[{trip_id, n_obs, has_network}]` |
| `GET /predict/{trip_id}` | `gps`, `pred_route` (`[[lat, lon], ...]`), `metrics.hmm` / `metrics.classifier_only` (Top-1, continuity, route precision/recall/F1), `timings_s` |

Errors: unknown trip → **404**; trip without `.arcs` (e.g. student-test 00000000) → **422** with a clear message.

### 3. Metrics shown to the evaluator
- **Top-1 accuracy**: share of GPS points (within 100 m of the reference) whose predicted arc equals the reference arc.
- **Route continuity (per point)**: share of consecutive predictions that are the same arc or connected arcs.
- **Route continuity (reconstructed)**: the same check after filling skipped short arcs with shortest paths.
- **Route precision / recall / F1**: overlap between the set of predicted route arcs and the reference route arcs.

### 4. Measured locally (student-test trips, before final tuning)
- Trip 00000009: Top-1 91.4% (HMM) vs 88.0% (classifier only), route F1 0.94 vs 0.81, total 1.1 s.
- Trip 00000056: Top-1 96.5% vs 91.3%, route F1 0.96 vs 0.90, 4.1 s.

## Deployment checklist (do this before the evaluation)
1. Both laptops on the **same Wi-Fi/hotspot**. Run `ipconfig` on Laptop 1 and note its IPv4 address.
2. Laptop 1: `set PATHSNAP_EVAL_DIR=<folder with the 10 evaluator trips>`, then
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Allow Python through the Windows firewall (private network).
3. Laptop 2: `pip install -r ui/requirements.txt`, then `streamlit run ui/app.py`, and enter `http://<ip>:8000`.
4. Click through **all 10** trips once before the evaluator arrives. Screenshot the UI and the API console for the report.

## Commit scripts (`team/`)
- `bash team/commit_pX.sh "msg"` commits **only that member's folders**, under the git name of whoever runs it.
- Before pulling, it removes local untracked files that are byte-identical to what is already on GitHub,
  so a teammate's commit never clashes with your local copy. Nothing is lost because the content is identical.
- `--watch [min]` re-commits every N minutes. Files over 90 MB, the dataset, `.venv` and parquet caches are never committed.
