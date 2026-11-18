# Part P3: HMM/Viterbi route reconstruction & Folium maps

**Owner:** P3. **Code:** `hmm/`, `viz/`. **Results:** `results/p3/`, `artifacts/hmm_params.json`.
**Report sections:** 5 (HMM & Viterbi) and 6 (Folium).

## What you must be able to explain

### 1. Why an HMM at all
Picking the best candidate independently for each GPS point can jump between unconnected roads, for example
to a parallel street for a single second. An HMM adds two facts the classifier cannot see: **consecutive
positions must be connected in the road network**, and **the driven distance should match the distance the
GPS moved**.

### 2. Hidden states (report Q1)
At GPS observation *i*, the hidden state is "the vehicle is on candidate arc *c*, at position `proj_t` along
it". The state set of observation *i* is exactly its candidate list (up to 10 arcs within 50 m), so every
state maps one-to-one to an `arc_id`.

### 3. Emission probabilities (Q2)
- **classifier + HMM**: `log E = alpha · log(p_LightGBM)`, with p clipped at 1e-4. `alpha` sets how much we
  trust the classifier relative to the road network.
- **distance-only HMM** (a baseline without ML): `log E = −½ (dist / sigma)²`, a Gaussian GPS-noise model.

### 4. Transition probabilities (Q3)
- `route` = driving distance in the network from (arc a, t_a) to (arc b, t_b):
  - same arc: `|t_b − t_a| · len(a)`
  - different arcs: `(1 − t_a)·len(a) + shortest_path(end of a → start of b) + t_b·len(b)`
- `gc` = straight-line distance between the two GPS points.
- `log T = −|route − gc| / beta`. For real movement the driven distance is close to the straight-line
  distance. A detour through other streets, or a jump to a disconnected road, gets a large penalty.
- Shortest paths come from a **distance-limited Dijkstra** (limit = 2·gc + 300 m), cached per start node.
  With 1 Hz data the same nodes repeat constantly, so the cache makes decoding fast.

### 5. Viterbi recurrence
`V_i(b) = max_a [ V_{i−1}(a) + log T(a, b) ] + log E_i(b)`, keeping a back-pointer to the best `a`.
At the end of a chain, take `argmax V` and follow the back-pointers. Cost is O(N · K²) per trip (K ≤ 10).

### 6. Missing candidates, gaps and disconnected roads (Q4–Q5)
- A point with **no candidates** is skipped (prediction −1). The next point connects to the last point that
  had candidates.
- A **time gap larger than `gap_s`** breaks the chain: decoding restarts from emissions only, so a
  42-hour gap is never forced into one implausible path.
- If **every transition is impossible** (no road path within the search limit), the chain is also broken
  and restarted, instead of crashing or forcing a teleport.
- After decoding, consecutive distinct arcs that are not adjacent are **joined with the shortest path**
  (`fill_route`). At 1 Hz a car can cross a short arc between two GPS fixes, and filling it makes the
  reconstructed route continuous.

### 7. Decoder parameter tuning (no student-test or evaluator data)
12 train trips are held out. A copy of the final model is refit on the other 68 train trips, so the
probabilities used for tuning are out of sample. A grid search covers alpha × beta × gap_s for the
classifier HMM and sigma × beta for the distance HMM. Source: `results/p3/decoder_tuning.csv`,
`figures/decoder_tuning.png`.

### 8. Results (report Q6). Did HMM decoding help? Yes, on every trip.
Selected decoder (`artifacts/hmm_params.json`): classifier emission, `alpha=0.5`, `beta=10 m`, `gap_s=120 s`.
On the 12 tuning trips, Top-1 went from 0.876 (classifier only) to 0.937 (HMM). `alpha=0.5, beta=10` and
`alpha=1, beta=5` score identically because they are the same objective scaled by 2, so Viterbi returns the
same path. Only the product alpha·beta matters.

**Student test, 9 trips with a road network** (`results/p3/student_test_decoders_summary.csv`):

| decoder | Top-1 | continuity (per point) | continuity (reconstructed) | route precision | route recall | route F1 | s / trip |
|---|---|---|---|---|---|---|---|
| classifier only | 0.898 | 0.929 | 0.991 | 0.662 | 0.992 | 0.782 | 0.07 |
| distance-only HMM (no ML) | 0.930 | 0.974 | 1.000 | 0.934 | 0.991 | 0.960 | 0.77 |
| **classifier + HMM** | **0.947** | **0.977** | **1.000** | **0.958** | **0.992** | **0.974** | 0.80 |

- **The HMM improved Top-1 on all 9 trips**, by between 1.9 points (trip 52, which reaches 100%) and 11.4 points (trip 77).
- **Route precision is where the HMM matters most** (0.66 → 0.96). Classifier-only predictions flicker onto
  side streets and parallel roads for a second or two. Each flicker adds a wrong arc to the route, and the
  HMM's connectivity term removes them.
- **The ML emission beats the classic distance-only HMM** by 1.7 points of Top-1 and 1.4 points of route F1,
  so the classifier adds value on top of geometry.
- **Where it did not help as much (trip 00000077):** classifier + HMM reaches 0.842 while the distance-only
  HMM reaches 0.899. Of its 617 remaining errors, **482 (78%) are the reverse-direction twin of the correct
  arc**: the right road, wrong direction. They happen at low speed (median 2 m/s against 3.7 m/s for the
  trip), where GPS heading is noisy. The classifier confidently prefers the wrong direction from that heading,
  and the HMM follows it. The distance HMM ignores heading and settles direction through route continuity.
  On a map these errors are invisible because the geometry is identical. Possible fixes: a speed-dependent
  heading weight, or a higher beta for twin transitions.
- Per-point continuity is below 1 even for the HMM, because at 1 Hz a car can cross a short arc between two
  fixes. After `fill_route` inserts the shortest connecting path, the reconstructed route is continuous
  (0.9999). The small remainder comes from deliberate chain breaks at large time gaps.

### 9. Maps (report §6)
`results/p3/maps/route_trip_00000077.html` (largest gain), `route_trip_00000052.html` (smallest gain) and
`route_trip_00000009.html`. Each has raw GPS (blue), the road network (grey, toggle), the reference route
(green), classifier-only (orange dashed) and HMM (red), with a legend and layer control. Screenshot one for the PDF.

## How to run
```bash
python -m hmm.evaluate_hmm     # decoder tuning + student-test comparison + maps
```
