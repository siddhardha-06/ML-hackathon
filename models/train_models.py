"""Train and compare candidate classifiers from different algorithm families (P2).

Run from the project root:   python -m models.train_models

Protocol
  * Training data: the 80 train trips. A fixed 30 % of observations per trip is sampled
    (seed 42) so that slow models (KNN, MLP, RF) are compared on exactly the same rows.
  * 5-fold GroupKFold with groups = trip_id: no trip is split between train and validation.
  * All preprocessing (log1p + StandardScaler) lives inside a pipeline, refit per fold.
  * Each model is then refit on the full sample and evaluated on the 10 student-test trips.

Primary selection metric: observation-level Top-1 accuracy on student-test trips, because
that is what the route reconstruction consumes (the best candidate per GPS point).
Macro-F1 (threshold 0.5) and log loss are reported too; log loss matters because the HMM
uses the predicted probabilities as emission scores, so calibrated probabilities help.

Outputs -> results/p2/
  model_comparison.csv, cv_folds.csv, test_per_trip_<model>.csv, figures/*.png
  artifacts/models/<model>.joblib
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.config import ARTIFACTS_DIR, RESULTS_DIR, SEED  # noqa: E402
from common.metrics import (attach_predictions, candidate_miss_rate, sequence_report,  # noqa: E402
                            top1_accuracy, top1_from_scores)
from models.dataset import load_partition, load_trips, scaled, xy  # noqa: E402

OUT = RESULTS_DIR / "p2"
FIG = OUT / "figures"
MODEL_DIR = ARTIFACTS_DIR / "models"
TRAIN_OBS_FRAC = 0.30
N_FOLDS = 5


def model_zoo():
    """name -> (family, config string, factory)."""
    return {
        "LogisticRegression": ("Linear", "C=1.0, lbfgs, max_iter=1000, log1p+scaling",
                               lambda: scaled(LogisticRegression(C=1.0, max_iter=1000))),
        "GaussianNB": ("Probabilistic", "var_smoothing=1e-9, log1p+scaling",
                       lambda: scaled(GaussianNB())),
        "KNN": ("Instance-based", "k=25, distance-weighted, euclidean, log1p+scaling",
                lambda: scaled(KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
        "RandomForest": ("Ensemble (bagging)", "300 trees, max_depth=18, min_samples_leaf=5",
                         lambda: RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                                        n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Ensemble (randomised trees)", "300 trees, max_depth=18, min_samples_leaf=5",
                       lambda: ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=SEED)),
        "LightGBM": ("Boosting", "400 rounds, lr=0.05, num_leaves=63",
                     lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                            random_state=SEED, n_jobs=-1, verbose=-1)),
        "XGBoost": ("Boosting", "400 rounds, lr=0.05, max_depth=7, hist",
                    lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                          tree_method="hist", random_state=SEED, n_jobs=-1,
                                          eval_metric="logloss")),
        "MLP": ("Neural network", "(64,32) ReLU, adam, early_stopping, log1p+scaling",
                lambda: scaled(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                             max_iter=200, random_state=SEED))),
    }


def binary_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yhat = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def evaluate(model, cands, obs):
    X, y = xy(cands)
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    infer_s = time.perf_counter() - t0
    m = binary_metrics(y, p)
    pred = top1_from_scores(cands, p)
    obs_p = attach_predictions(obs, pred)
    m["top1"] = top1_accuracy(obs_p)
    m["infer_ms_per_1k_rows"] = 1000 * infer_s / len(X) * 1000
    return m, p, obs_p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tr_c, tr_o = load_partition("train", TRAIN_OBS_FRAC)
    te_c, te_o_all = load_partition("student_test")
    te_trips = load_trips("student_test")
    X_tr, y_tr = xy(tr_c)
    groups = tr_c["trip_id"].to_numpy()
    print(f"train rows={len(X_tr):,} (obs={len(tr_o):,}, positives={y_tr.mean():.3f}) | "
          f"test rows={len(te_c):,} obs={len(te_o_all):,}")
    print(f"student-test candidate miss rate: {candidate_miss_rate(te_o_all):.4f}")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups))
    rows, fold_rows, cms = [], [], {}
    for name, (family, cfg, make) in model_zoo().items():
        print(f"\n=== {name}")
        cv = []
        for k, (a, b) in enumerate(folds):
            mdl = make()
            mdl.fit(X_tr[a], y_tr[a])
            fold_c = tr_c.iloc[b]
            fold_o = tr_o.merge(fold_c[["trip_id", "obs_idx"]].drop_duplicates(), on=["trip_id", "obs_idx"])
            m, _, _ = evaluate(mdl, fold_c, fold_o)
            m.update(model=name, fold=k)
            cv.append(m)
            fold_rows.append(m)
            print(f"  fold {k}: top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f}")
        cv = pd.DataFrame(cv)

        mdl = make()
        t0 = time.perf_counter()
        mdl.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        X_tr_fit_p = mdl.predict_proba(X_tr)[:, 1]
        train_m = binary_metrics(y_tr, X_tr_fit_p)
        te_m, te_p, te_obs = evaluate(mdl, te_c, te_o_all)
        seq = sequence_report(te_obs, te_trips)
        seq.to_csv(OUT / f"test_per_trip_{name}.csv", index=False)
        cms[name] = confusion_matrix(xy(te_c)[1], (te_p >= 0.5).astype(int))
        joblib.dump(mdl, MODEL_DIR / f"{name}.joblib")
        if name == "MLP":
            mlp = mdl[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(OUT / "mlp_history.csv", index=False)

        row = {
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1": cv.top1.mean(), "cv_top1_std": cv.top1.std(), "cv_log_loss": cv.log_loss.mean(),
            "train_macro_f1": train_m["macro_f1"], "train_log_loss": train_m["log_loss"],
            "test_accuracy": te_m["accuracy"], "test_precision": te_m["precision"],
            "test_recall": te_m["recall"], "test_macro_f1": te_m["macro_f1"],
            "test_log_loss": te_m["log_loss"], "test_top1": te_m["top1"],
            "test_continuity": seq.continuity.mean(), "test_route_f1": seq.route_f1.mean(),
            "train_time_s": train_s, "infer_ms_per_1k_rows": te_m["infer_ms_per_1k_rows"],
        }
        rows.append(row)
        print(f"  CV top1={row['cv_top1']:.4f}±{row['cv_top1_std']:.4f} | TEST top1={row['test_top1']:.4f} "
              f"macroF1={row['test_macro_f1']:.4f} logloss={row['test_log_loss']:.4f} "
              f"continuity={row['test_continuity']:.4f} train={train_s:.1f}s")

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "cv_folds.csv", index=False)
    json.dump({"train_obs_frac": TRAIN_OBS_FRAC, "n_folds": N_FOLDS, "seed": SEED,
               "student_test_miss_rate": candidate_miss_rate(te_o_all)},
              open(OUT / "experiment_config.json", "w"), indent=2)
    make_plots(comp, pd.DataFrame(fold_rows), cms)
    print("\n", comp[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss",
                      "test_continuity", "train_time_s", "infer_ms_per_1k_rows"]].round(4).to_string(index=False))


def make_plots(comp, folds, cms):
    names = comp.model.tolist()
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 4))
    data = [folds[folds.model == n].macro_f1.values for n in names]
    ax.boxplot(data, tick_labels=names)
    ax.set(ylabel="macro-F1 per fold", title="5-fold GroupKFold macro-F1 (train trips)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - .2, comp.test_top1, .4, label="Top-1 accuracy")
    ax.bar(x + .2, comp.test_macro_f1, .4, label="macro-F1")
    ax.set_xticks(x, names, rotation=30)
    ax.set(ylim=(min(comp.test_macro_f1.min(), comp.test_top1.min()) - .05, 1),
           title="Student-test: Top-1 accuracy and macro-F1")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - .2, comp.cv_log_loss, .4, label="CV")
    ax.bar(x + .2, comp.test_log_loss, .4, label="student-test")
    ax.set_xticks(x, names, rotation=30)
    ax.set(yscale="log", ylabel="log loss (lower is better)", title="Log loss")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comp.infer_ms_per_1k_rows, comp.test_top1)
    for _, r in comp.iterrows():
        ax.annotate(r.model, (r.infer_ms_per_1k_rows, r.test_top1), fontsize=8)
    ax.set(xscale="log", xlabel="inference ms per 1k candidate rows", ylabel="student-test Top-1",
           title="Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(FIG / "accuracy_vs_speed.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, n in zip(axes.ravel(), names):
        cm = cms[n]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9)
        ax.set(title=n, xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    fig.suptitle("Confusion matrices, student-test candidates (threshold 0.5)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=130); plt.close(fig)

    hist = OUT / "mlp_history.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(h.epoch, h.train_loss, color="tab:blue", label="train loss")
        ax1.set(xlabel="epoch", ylabel="train log loss")
        ax2 = ax1.twinx()
        ax2.plot(h.epoch, h.val_accuracy, color="tab:orange", label="validation accuracy")
        ax2.set_ylabel("validation accuracy (10 % hold-out)")
        fig.legend(loc="upper right"); ax1.set_title("MLP training history")
        fig.tight_layout(); fig.savefig(FIG / "mlp_history.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
