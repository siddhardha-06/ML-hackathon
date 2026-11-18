# PathSnap: technical report outline (assembled by P4; each owner writes their sections)

Convert to PDF when finished (for example, paste into Google Docs or Word and export). Every number must come
from a file in `results/`, and each table below names its source file. Write the explanations in your own
words, using `docs/PART_P*.md` as notes.

---

## 1. Team, repository and AI usage disclosure (P4)
- Team members and parts: P1 data and candidates · P2 metrics, models and tuning · P3 HMM and maps · P4 EDA, API, UI and report.
- GitHub: https://github.com/siddhardha-06/ML-hackathon
- **AI usage disclosure:** the team used an AI coding assistant to help write and debug code, design the
  pipeline and draft documentation. All results were produced by running the committed code on the
  provided data. *(Adjust the wording to match the event's AI policy.)*

## 2.2 Dataset understanding & EDA (P1 + P4)
1. **How did you preprocess the data?** Covers the local metric projection, time ordering, dropping
   non-finite rows, the route-to-GPS monotone alignment for labels, the 100 m reliability filter, and
   excluding trip 00000000 (no `.arcs`). Source: `docs/PART_P1.md` §2–3.
2. **Patterns and features.** Covers two-way twin arcs, GPS noise with a median of 5 m, parked trips beyond the
   route end (00000054, 00000033), gaps and sampling, and why each of the 16 features exists. Source:
   `results/eda/data_quality.md`, `docs/PART_P1.md` §5.
- Missing-value analysis: `results/eda/inventory.csv` (n_nan, has_* columns), `data_quality.md`.
- GPS trajectory plots: `results/eda/figures/trajectory_examples.png`, `trip_length_sampling.png`.
- Road-network visualisations: `results/eda/figures/network_size.png`, `results/eda/network_connectivity.csv`.
- Folium map (screenshot): `results/eda/maps/eda_trip_00000001.html`.
- Candidate recall table and plot: `results/p1/candidate_recall.csv`, `results/eda/figures/candidate_recall.png`.
- State the partition for every analysis (train or student_test).

## 3. Model experimentation & comparison (P2)
### 3.1 Overall comparison table
Source: `results/p2/model_comparison.csv` (CV macro-F1, CV Top-1, test Top-1, continuity, test macro-F1,
log loss, inference time).
### 3.2 One subsection per model (8 models): questions A1–2, B5–6, C7–9
Notes: `docs/PART_P2.md` §5. Hyperparameters: `model_zoo()` in `models/train_models.py`.
Loss and objective: log loss for LR, MLP and boosting; Gini impurity for RF and ET; no loss for KNN and NB
(NB is maximum likelihood).
### 3.3 Visualisations
`results/p2/figures/`: `cv_macro_f1.png`, `test_top1_f1.png`, `log_loss.png`, `confusion_matrices.png`,
`accuracy_vs_speed.png`, `mlp_history.png` (the only epoch-wise model), `lgbm_history.png`
(boosting-round history of the final model).

## 4. Bayesian hyperparameter optimisation (P2)
Questions 1–6. Sources: `tuning/optuna_search.py` (search space), `results/p2/optuna_trials.csv`,
`tuned_vs_untuned.csv`, `figures/optuna_history.png`, `optuna_param_importance.png`,
`artifacts/final_model_card.json`.

## 5. HMM & Viterbi route reconstruction (P3)
Questions 1–6. Sources: `hmm/viterbi.py` docstring, `results/p3/decoder_tuning.csv`,
`student_test_decoders_summary.csv`, `student_test_decoders_per_trip.csv`, `figures/decoder_comparison.png`,
`figures/decoder_tuning.png`. Compare classifier only, distance-only HMM and classifier + HMM, and discuss
the trips where the HMM did not help.

## 6. Geographical visualisation using Folium (P3)
Screenshot `results/p3/maps/route_trip_*.html` (GPS, road network, reference, classifier-only and HMM routes,
with legend and layer control).

## 7. API, UI & deployment (P4)
Screenshots of the UI on Laptop 2 and of the API console on Laptop 1 (one request per trip).
Architecture and endpoints: `docs/PART_P4.md`.

## 8. Conclusions & limitations (all)
- The reference routes are approximate: routes are disconnected in some trips, and some GPS traces run beyond the route.
- About 2% of points have no correct candidate within 50 m, which caps achievable Top-1.
- Two-way roads, parallel roads and parking areas are the main sources of error.
