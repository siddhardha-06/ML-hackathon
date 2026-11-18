# Part P2: Evaluation metrics, model experimentation & Bayesian optimisation

**Owner:** P2. **Code:** `common/metrics.py`, `models/`, `tuning/`. **Results:** `results/p2/`, `artifacts/final_model*`.
**Report sections:** 3 (model comparison, per-model questions, plots) and 4 (Bayesian optimisation).

## What you must be able to explain

### 1. The learning problem
- Each row is one (GPS point, candidate arc) pair with 16 features (built by P1). The label is 1 if the
  candidate is the point's reference arc. Positives make up about 15% of rows.
- **Primary metric: observation-level Top-1 accuracy.** For each GPS point, take the candidate with the
  highest predicted probability and check whether it is the reference arc. This is what route reconstruction
  consumes. Points whose true arc is missing from the candidates count as wrong, so the metric is end to end.
- Also reported: accuracy, precision, recall and macro-F1 at threshold 0.5 (row level), and **log loss**.
  Log loss matters because the HMM uses the probabilities themselves as emission scores, so calibration counts,
  not just ranking. Route continuity, training time and inference time are reported too.

### 2. Protocol (`models/train_models.py`)
- Training data: the 80 train trips, keeping only points with a reliable label. A fixed **30% sample of points
  per trip** (seed 42) is used so that slow models (KNN, MLP, RandomForest) are compared on exactly the same rows:
  362,181 candidate rows from 56,362 points.
- **5-fold GroupKFold with groups = trip_id.** No trip appears in both training and validation, which would
  leak because neighbouring points of one trip are nearly identical.
- Preprocessing (log1p of heavy-tailed features, then StandardScaler) sits **inside a pipeline**, so it is
  refit on each training fold only. Tree models use raw features.
- Each model is then refit on the whole sample and evaluated on the 10 student-test trips (9 have a road network).

### 3. Results (`results/p2/model_comparison.csv`)
| Model | Family | CV macro-F1 | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | Continuity | Train s | ms / 1k rows |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.903 | 0.857 | 0.841 | 0.895 | 0.135 | 0.906 | 0.8 | 0.3 |
| Gaussian NB | Probabilistic | 0.872 | 0.845 | 0.835 | 0.868 | 0.353 | 0.888 | 0.4 | 0.6 |
| KNN (k=25, distance) | Instance-based | 0.911 | 0.860 | 0.849 | 0.905 | 0.138 | 0.918 | 0.1 | 315 |
| Random Forest | Bagging | 0.925 | 0.874 | 0.869 | 0.922 | 0.090 | 0.924 | 48 | 3.1 |
| Extra Trees | Randomised trees | 0.917 | 0.867 | 0.857 | 0.911 | 0.100 | 0.919 | 15 | 3.8 |
| **LightGBM** | Boosting | **0.925** | **0.874** | **0.878** | **0.925** | **0.088** | 0.929 | 4.6 | 2.2 |
| XGBoost | Boosting | 0.924 | 0.874 | 0.875 | 0.923 | 0.088 | 0.930 | 4.3 | 1.8 |
| MLP (64, 32) | Neural network | 0.919 | 0.868 | 0.869 | 0.917 | 0.095 | 0.927 | 35 | 0.9 |

### 4. Why LightGBM
- It ties the best CV scores and is **best on student test for Top-1, macro-F1 and log loss**.
- It has the lowest log loss, so it gives the best-calibrated emission probabilities for the HMM.
- It trains in about 5 s and runs inference in about 2 ms per 1,000 rows. RandomForest is equally accurate but
  10× slower to train and its saved model is 200 MB. KNN is about 150× slower at inference.
- XGBoost is a near tie. We chose LightGBM for its slightly better student-test numbers and smaller tuning cost.

### 5. Per-model observations (use these for report §3.2, questions 1-9)
- **Logistic Regression** and **Gaussian NB** underfit. Their train and CV scores are both low (macro-F1 0.90
  and 0.87). The decision is not linear in the features: a small distance only matters when the heading also
  agrees. NB's independence assumption is badly violated because dist, dist_prev and dist_next are strongly
  correlated, which also gives it the worst log loss (0.35, over-confident).
- **KNN** works because similar feature vectors do share labels, but it has no notion of feature importance,
  and inference is slow because it must search all training rows.
- **Random Forest** and **Extra Trees** capture the interactions. Extra Trees' random split thresholds cost
  about 1 point of accuracy.
- **Boosting** (LightGBM and XGBoost) corrects errors sequentially and gives the best trade-off.
- **MLP** is competitive but slower and slightly worse. `results/p2/figures/mlp_history.png` is a real
  epoch-wise history (training loss and early-stopping validation accuracy).
- **Over- and underfitting**, from `model_comparison.csv` (train vs CV macro-F1):
  - **KNN memorises**: 1.000 vs 0.911. Distance-weighting gives each training row zero distance to itself.
  - **Random Forest and LightGBM overfit mildly**: about 0.96 vs 0.925. The gap is stable across folds, and test
    (0.922 / 0.925) matches CV, so they generalise to unseen trips.
  - **Logistic Regression and Gaussian NB underfit**: train ≈ CV ≈ test (0.90 / 0.87).
  - Per-fold Top-1 for LightGBM ranges from 0.840 to 0.907 (`cv_folds.csv`). The spread comes from which
    trips land in the fold, which is why grouping by trip matters.

### 5b. What the model relies on
**Evidence from the final model** (`results/p2/figures/feature_importance.png`, LightGBM total gain): `dist_minus_min` is the strongest feature, followed by `heading_diff`, `cand_rank` and `cos_heading`. Relative closeness matters more than absolute distance (`dist` ranks near the bottom), and heading is what separates two-way twin arcs. `dt_prev` carries almost no signal, because 96.5% of steps are 1 s.

### 6. Bayesian optimisation (`tuning/optuna_search.py`)
- **Optuna TPE sampler**, 30 trials. The objective is mean Top-1 over **3-fold GroupKFold** on the same train
  sample. Student-test and evaluator data are never used in the search.
- Search space: `n_estimators` 100–900, `learning_rate` 0.01–0.2 (log), `num_leaves` 15–255 (log),
  `min_child_samples` 5–300 (log), `subsample` 0.5–1, `colsample_bytree` 0.5–1, `reg_lambda` and `reg_alpha` 1e-3–10 (log).
- See `results/p2/optuna_trials.csv`, `tuned_vs_untuned.csv`, `figures/optuna_history.png`,
  `optuna_param_importance.png`, `lgbm_history.png` and `feature_importance.png`.
- Best parameters: `n_estimators=350, learning_rate=0.044, num_leaves=34, min_child_samples=33, subsample=0.54,
  colsample_bytree=0.53, reg_lambda=0.41, reg_alpha=0.004`. The tuned model is smaller and more regularised
  than the default (63 leaves, 400 rounds).
- Tuned vs untuned, on identical data (`tuned_vs_untuned.csv`):

  | | CV Top-1 | CV macro-F1 | CV log loss | Test Top-1 | Test macro-F1 | Test log loss | Inference ms/1k |
  |---|---|---|---|---|---|---|---|
  | untuned | 0.8746 | 0.9243 | 0.0979 | 0.8779 | 0.9253 | 0.0878 | 2.8 |
  | tuned | 0.8767 | 0.9245 | 0.0952 | 0.8751 | 0.9238 | 0.0883 | 2.4 |

  **Did tuning help?** A little on CV (+0.2 points Top-1, about 3% lower log loss) and inference got slightly
  faster, but student-test Top-1 dropped 0.3 points. With only 9 test trips that difference is within
  trip-to-trip noise (per-fold Top-1 varies by ±3 points). The honest conclusion: the default configuration
  was already near the optimum for these 16 features. Gains have to come from features and decoding, not
  hyperparameters. The search took 4.9 minutes for 30 trials.
- The final model is the tuned configuration refit on **all 1,209,597** reliable train rows
  (`artifacts/final_model.joblib`, `final_model_card.json`): student-test Top-1 0.8748, macro-F1 0.9239,
  log loss 0.0873, the best calibration of any model.

## How to run
```bash
python -m models.train_models        # ~6 min (8 models x 5 folds)
python -m tuning.optuna_search 30    # ~8 min
```
