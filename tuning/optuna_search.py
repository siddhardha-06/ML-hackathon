"""Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna TPE (P2).

Run from the project root (after models.train_models):   python -m tuning.optuna_search [n_trials]

Objective   maximise mean observation-level Top-1 accuracy over 3-fold GroupKFold (groups =
            trip_id) on the same 30 % observation sample of the 80 train trips used in the
            model comparison. Student-test and evaluator data are never touched during search.
Search space (why)
  n_estimators      100-900        more rounds vs. overfitting / inference time
  learning_rate     0.01-0.2 (log) step size; interacts with n_estimators
  num_leaves        15-255 (log)   tree complexity
  min_child_samples 5-300 (log)    leaf regularisation; protects against noisy labels
  subsample         0.5-1.0        row bagging (bagging_freq=1)
  colsample_bytree  0.5-1.0        feature bagging; 16 correlated features
  reg_lambda/alpha  1e-3-10 (log)  L2 / L1 regularisation
After search: refit tuned + untuned on the same sample -> compare on student-test; then refit the
tuned configuration on all reliable train rows -> artifacts/final_model.joblib.
"""
import json
import sys
import time

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.config import ARTIFACTS_DIR, RESULTS_DIR, SEED  # noqa: E402
from common.metrics import attach_predictions, top1_accuracy, top1_from_scores  # noqa: E402
from features.candidates import FEATURE_COLS  # noqa: E402
from models.dataset import load_partition, load_trips, xy  # noqa: E402
from models.train_models import TRAIN_OBS_FRAC, binary_metrics, evaluate, model_zoo  # noqa: E402
from common.metrics import sequence_report  # noqa: E402

OUT = RESULTS_DIR / "p2"
N_FOLDS = 3
BASE = dict(random_state=SEED, n_jobs=-1, verbose=-1)


def cv_top1(params, X, y, cands, obs, folds):
    scores, losses = [], []
    for a, b in folds:
        m = LGBMClassifier(**params, **BASE).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        fc = cands.iloc[b]
        fo = obs.merge(fc[["trip_id", "obs_idx"]].drop_duplicates(), on=["trip_id", "obs_idx"])
        scores.append(top1_accuracy(attach_predictions(fo, top1_from_scores(fc, p))))
        losses.append(binary_metrics(y[b], p)["log_loss"])
    return float(np.mean(scores)), float(np.mean(losses)), [float(s) for s in scores]


def main(n_trials=30):
    OUT.mkdir(parents=True, exist_ok=True)
    cands, obs = load_partition("train", TRAIN_OBS_FRAC)
    X, y = xy(cands)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, cands["trip_id"]))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        t0 = time.perf_counter()
        top1, ll, per_fold = cv_top1(params, X, y, cands, obs, folds)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("fold_top1", per_fold)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number:2d}: top1={top1:.4f} logloss={ll:.4f} ({time.perf_counter() - t0:.0f}s)")
        return top1

    t_search = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lgbm_top1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    search_s = time.perf_counter() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    print("best params:", best, "cv top1:", round(study.best_value, 4))

    # --- tuned vs untuned on identical data
    untuned_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63)
    te_c, te_o = load_partition("student_test")
    te_trips = load_trips("student_test")
    rows = []
    for name, params in (("LightGBM (untuned)", untuned_params), ("LightGBM (tuned)", best)):
        top1, ll, _ = cv_top1(params, X, y, cands, obs, folds)
        cv_f1 = []
        for a, b in folds:
            m = LGBMClassifier(**params, **BASE).fit(X[a], y[a])
            cv_f1.append(binary_metrics(y[b], m.predict_proba(X[b])[:, 1])["macro_f1"])
        t0 = time.perf_counter()
        m = LGBMClassifier(**params, **BASE).fit(X, y)
        fit_s = time.perf_counter() - t0
        te_m, _, te_obs = evaluate(m, te_c, te_o)
        seq = sequence_report(te_obs, te_trips)
        rows.append({"model": name, "cv_top1": top1, "cv_macro_f1": np.mean(cv_f1), "cv_log_loss": ll,
                     "test_top1": te_m["top1"], "test_macro_f1": te_m["macro_f1"],
                     "test_log_loss": te_m["log_loss"], "test_continuity": seq.continuity.mean(),
                     "train_time_s": fit_s, "infer_ms_per_1k_rows": te_m["infer_ms_per_1k_rows"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # --- final model on all reliable train rows, with a training history on one held-out fold
    full_c, _ = load_partition("train", 1.0)
    Xf, yf = xy(full_c)
    tr_idx, va_idx = next(GroupKFold(n_splits=5).split(Xf, yf, full_c["trip_id"]))
    hist_model = LGBMClassifier(**best, **BASE).fit(
        Xf[tr_idx], yf[tr_idx], eval_set=[(Xf[tr_idx], yf[tr_idx]), (Xf[va_idx], yf[va_idx])],
        eval_names=["train", "validation"], eval_metric="binary_logloss")
    ev = hist_model.evals_result_
    pd.DataFrame({"round": np.arange(1, len(ev["train"]["binary_logloss"]) + 1),
                  "train_logloss": ev["train"]["binary_logloss"],
                  "val_logloss": ev["validation"]["binary_logloss"]}).to_csv(OUT / "lgbm_history.csv", index=False)
    t0 = time.perf_counter()
    final = LGBMClassifier(**best, **BASE).fit(Xf, yf)
    final_fit_s = time.perf_counter() - t0
    joblib.dump(final, ARTIFACTS_DIR / "final_model.joblib")
    te_m, _, te_obs = evaluate(final, te_c, te_o)
    json.dump({"model": "LightGBM", "params": best, "features": FEATURE_COLS, "n_trials": n_trials,
               "objective": "mean 3-fold GroupKFold observation Top-1 (train trips)",
               "best_cv_top1": study.best_value, "search_seconds": search_s,
               "final_train_rows": int(len(Xf)), "final_fit_seconds": final_fit_s,
               "final_student_test": te_m},
              open(ARTIFACTS_DIR / "final_model_card.json", "w"), indent=2, default=float)
    json.dump(FEATURE_COLS, open(ARTIFACTS_DIR / "feature_columns.json", "w"))
    print(f"final model (all {len(Xf):,} rows): student-test top1={te_m['top1']:.4f} "
          f"macroF1={te_m['macro_f1']:.4f} logloss={te_m['log_loss']:.4f}; search took {search_s / 60:.1f} min")
    plots(trials, study)


def plots(trials, study):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials.number, trials.value, "o", alpha=.6, label="trial")
    ax.plot(trials.number, trials.value.cummax(), "-", color="red", label="best so far")
    ax.set(xlabel="trial", ylabel="CV Top-1", title="Optuna TPE optimisation history (LightGBM)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / "optuna_history.png", dpi=130); plt.close(fig)

    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(imp)[::-1], list(imp.values())[::-1])
    ax.set(xlabel="importance (fANOVA)", title="Hyperparameter importance")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "optuna_param_importance.png", dpi=130); plt.close(fig)

    h = pd.read_csv(OUT / "lgbm_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["round"], h.train_logloss, label="train")
    ax.plot(h["round"], h.val_logloss, label="validation (held-out trips)")
    ax.set(xlabel="boosting round", ylabel="log loss", title="Tuned LightGBM training history")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / "lgbm_history.png", dpi=130); plt.close(fig)

    final = joblib.load(ARTIFACTS_DIR / "final_model.joblib")
    gain = pd.Series(final.booster_.feature_importance("gain"), index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gain.index, gain.values)
    ax.set(xlabel="total gain", title="Final model feature importance")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "feature_importance.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
