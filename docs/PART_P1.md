# Part P1: Data processing, labels, candidate generation & features

**Owner:** P1. **Code:** `common/config.py`, `common/geo.py`, `data/`, `features/`. **Results:** `results/p1/`.
**Report sections:** 2.2 questions 1-2 (preprocessing, extracted features and why); P4 covers the EDA plots.

## What you must be able to explain

### 1. How the raw data is organised (`data/loader.py`)
- Each trip folder has four headerless files: `.track` (lon, lat, t), `.nodes` (lon, lat; row = node id),
  `.arcs` (from_node, to_node; row = arc id) and `.route` (arc ids in travel order).
- **The route is not one label per GPS row.** For example, trip 00000001 has 1,315 GPS points but 525
  route arcs. This is the key fact behind the whole pipeline.
- Each trip ships **its own local road network**, and trips are spread across the world (Germany, Russia,
  USA, Japan, Brazil and more).
- `student_test/00000000` has **no `.arcs` file**, so it has no road network and cannot be map-matched. It is excluded.

### 2. Why we project to metres (`common/geo.py`)
- Lon/lat are angles. One degree of longitude is 111 km × cos(latitude), so Euclidean distance in degrees is wrong.
- Per trip, we use a local **equirectangular projection** centred on the trip's mean position:
  `x = R·cos(lat0)·Δlon`, `y = R·Δlat`. At trip scale (tens of km) the error is under 1%, far below
  GPS noise. It also needs no UTM-zone logic, which matters because trips are worldwide.

### 3. How a label is derived for each GPS point (`features/labels.py`)
- Problem: which route arc was GPS point *i* on?
- Solution: **monotone alignment with dynamic programming.** Each point is assigned to a route position j(i)
  that never goes backwards in time, minimising the total point-to-arc distance:
  `D[i,j] = cost(i,j) + min_{k≤j} D[i-1,k]`. The cost is capped at 200 m, so outliers cannot drag the
  alignment. Vectorised over j, it runs in O(N·R).
- Points more than **100 m** from their aligned arc are marked `label_ok = False`. That is **9.9% of train
  points**, almost all in trips 00000054 and 00000033. They are excluded from training and from accuracy metrics.
- Finding: in trip 00000054 the reference route ends after about 27 minutes, but the GPS keeps recording for
  about 4.5 hours, parked about 90 m from any road. The reference does not cover the whole trace, so
  "ground truth" is approximate.

### 4. Candidate generation (`features/candidates.py`)
- Every arc is densified into points every 20 m and stored in a **KD-tree**. A GPS point at perpendicular
  distance d from an arc always has a sample within d + 10 m, so querying radius R + 10 m and then computing
  the exact point-to-segment distance finds every arc within R.
- We keep the **K = 10 nearest arcs within R = 50 m**. This was chosen from the recall sweep
  (`results/p1/candidate_recall.csv`; plot in `results/eda/figures/candidate_recall.png`):

  | radius \ K | 2 | 4 | 10 | 24 |
  |---|---|---|---|---|
  | 25 m | 0.893 | 0.942 | 0.961 | 0.963 |
  | **50 m** | 0.906 | 0.958 | **0.979** | 0.984 |
  | 100 m | 0.912 | 0.964 | 0.988 | 0.994 |

  Going from 50 m to 100 m adds only about 1% recall but many more candidates, which means more HMM
  states and more confusion. The candidate miss rate is 2.1% on train (reliable labels) and 0.2% on student test.
- **Two-way streets are two arcs with identical geometry** (92% of arcs have a reverse twin), so distance
  alone cannot separate them. That is why the heading features exist.

### 5. Features (16 per candidate)
| feature | why |
|---|---|
| `dist`, `dist_minus_min`, `dist_ratio`, `cand_rank`, `n_cands` | closeness, absolute and relative to rival candidates |
| `heading_diff`, `cos_heading` | GPS travel direction (points i−2→i+2) vs arc direction; separates two-way twins |
| `speed`, `disp` | heading is meaningless when stopped (we set it neutral below 1 m of movement) |
| `proj_t`, `arc_len` | where on the arc the point projects |
| `dist_prev`, `dist_next` | is point i±3 also near this arc? Context that rejects side streets touched by noise |
| `out_deg_to`, `in_deg_from` | intersection complexity at the arc's ends |
| `dt_prev` | sampling gap before this point |

**Evidence from the final model** (`results/p2/figures/feature_importance.png`, LightGBM total gain): `dist_minus_min` is the strongest feature, followed by `heading_diff`, `cand_rank` and `cos_heading`. Relative closeness matters more than absolute distance (`dist` ranks near the bottom), and heading is what separates two-way twin arcs. `dt_prev` carries almost no signal, because 96.5% of steps are 1 s.

## How to run
```bash
python -m features.build_dataset     # ~15 s on 8 cores -> artifacts/*.parquet, results/p1/
```
