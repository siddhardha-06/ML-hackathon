"""Tune decoder parameters and compare classifier-only vs HMM decoding (P3).

Run from the project root (after tuning/optuna_search.py):   python -m hmm.evaluate_hmm

1. Decoder tuning on 12 TRAIN trips held out from a model refit on the other 68 train trips,
   so the probabilities used for tuning are out-of-sample (no student-test, no evaluator data).
2. Student-test comparison of three decoders using the final model:
     classifier-only argmax | distance-only HMM (no ML) | classifier + HMM
3. Folium maps for three student-test trips.

Outputs -> results/p3/ and artifacts/hmm_params.json
"""
import itertools
import json
import os
import time
from dataclasses import asdict

import joblib
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.config import ARTIFACTS_DIR, RESULTS_DIR, ROOT, SEED  # noqa: E402
from common.metrics import route_continuity, route_set_scores  # noqa: E402
from features.candidates import FEATURE_COLS  # noqa: E402
from hmm.pipeline import load_model  # noqa: E402
from hmm.viterbi import HMMParams, RoadGraph, classifier_only, decode, fill_route  # noqa: E402
from models.dataset import load_trips  # noqa: E402
from viz.maps import route_map  # noqa: E402

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
OUT = RESULTS_DIR / "p3"
N_TUNE_TRIPS = 12


def trip_scores(trip, obs, pred, graph=None):
    """Observation-level Top-1 + continuity of raw predictions; route-level scores on the
    gap-filled reconstructed route (same filling applied to every decoder)."""
    rel = obs["label_ok"].to_numpy()
    lab = obs["label_arc"].to_numpy()
    route = fill_route(graph or RoadGraph(trip), pred)
    return {"top1": float((pred[rel] == lab[rel]).mean()),
            "continuity": route_continuity(pred, trip.arcs),
            "route_continuity_filled": route_continuity(np.array(route), trip.arcs),
            **route_set_scores(route, trip.route)}


def run_decoder(trip, obs, cands, probs, params, kind):
    t0 = time.perf_counter()
    graph = RoadGraph(trip)
    if kind == "classifier only":
        pred = classifier_only(trip, cands, probs)
    else:
        pred = decode(trip, cands, probs, params, graph)
    s = trip_scores(trip, obs, pred, graph)
    s.update(trip_id=trip.trip_id, decoder=kind, seconds=time.perf_counter() - t0, **asdict(params))
    return s, pred


def tune(model):
    obs = pd.read_parquet(ARTIFACTS_DIR / "obs_train.parquet")
    cands = pd.read_parquet(ARTIFACTS_DIR / "cands_train.parquet")
    trips = load_trips("train")
    rng = np.random.default_rng(SEED)
    ok_trips = sorted(t for t, o in obs.groupby("trip_id") if o.label_ok.mean() > .9 and len(o) < 5000)
    tune_ids = sorted(rng.choice(ok_trips, N_TUNE_TRIPS, replace=False))
    fit_rows = cands[~cands.trip_id.isin(tune_ids)]
    fit_rows = fit_rows.merge(obs[obs.label_ok][["trip_id", "obs_idx"]], on=["trip_id", "obs_idx"])
    fit_rows = fit_rows.groupby("trip_id", group_keys=False).sample(frac=.3, random_state=SEED)
    m = clone(model).fit(fit_rows[FEATURE_COLS].to_numpy(np.float32), fit_rows["label"].to_numpy())
    data = {}
    for tid in tune_ids:
        c = cands[cands.trip_id == tid].reset_index(drop=True)
        data[tid] = (trips[tid], obs[obs.trip_id == tid].sort_values("obs_idx"), c,
                     m.predict_proba(c[FEATURE_COLS].to_numpy(np.float32))[:, 1])

    grid = [HMMParams("classifier", alpha=a, beta=b, gap_s=g)
            for a, b, g in itertools.product((0.5, 1.0, 2.0), (5.0, 10.0, 20.0, 40.0), (30.0, 120.0))]
    grid += [HMMParams("distance", sigma=s, beta=b, gap_s=120.0)
             for s, b in itertools.product((5.0, 10.0, 20.0), (5.0, 10.0, 20.0, 40.0))]
    jobs = [(tid, p) for p in grid for tid in tune_ids]
    res = Parallel(n_jobs=8)(delayed(run_decoder)(*data[tid][:3], data[tid][3], p, "hmm") for tid, p in jobs)
    df = pd.DataFrame([r[0] for r in res])
    keys = ["emission", "alpha", "sigma", "beta", "gap_s"]
    summary = df.groupby(keys)[["top1", "continuity", "route_f1", "seconds"]].mean().reset_index()
    summary = summary.sort_values("top1", ascending=False)
    summary.to_csv(OUT / "decoder_tuning.csv", index=False)
    clf_only = pd.DataFrame([run_decoder(*data[t][:3], data[t][3], HMMParams(), "classifier only")[0]
                             for t in tune_ids])
    print("tuning trips:", tune_ids)
    print("classifier-only on tuning trips: top1=%.4f continuity=%.4f" % (clf_only.top1.mean(), clf_only.continuity.mean()))
    print(summary.head(8).round(4).to_string(index=False))
    best_clf = summary[summary.emission == "classifier"].iloc[0]
    best_dist = summary[summary.emission == "distance"].iloc[0]
    to_p = lambda r: HMMParams(emission=r.emission, alpha=float(r.alpha), sigma=float(r.sigma),  # noqa: E731
                               beta=float(r.beta), gap_s=float(r.gap_s))
    return to_p(best_clf), to_p(best_dist), tune_ids


def evaluate_student_test(model, p_clf, p_dist):
    obs = pd.read_parquet(ARTIFACTS_DIR / "obs_student_test.parquet")
    cands = pd.read_parquet(ARTIFACTS_DIR / "cands_student_test.parquet")
    trips = load_trips("student_test")
    rows, preds = [], {}
    for tid in sorted(obs.trip_id.unique()):
        trip = trips[tid]
        o = obs[obs.trip_id == tid].sort_values("obs_idx")
        c = cands[cands.trip_id == tid].reset_index(drop=True)
        probs = model.predict_proba(c[FEATURE_COLS].to_numpy(np.float32))[:, 1]
        preds[tid] = {}
        for kind, p in (("classifier only", p_clf), ("distance HMM (no ML)", p_dist), ("HMM + Viterbi", p_clf)):
            s, pred = run_decoder(trip, o, c, probs, p, kind)
            rows.append(s)
            preds[tid][kind] = pred
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "student_test_decoders_per_trip.csv", index=False)
    summ = df.groupby("decoder")[["top1", "continuity", "route_continuity_filled", "route_precision",
                                  "route_recall", "route_f1", "seconds"]].mean()
    summ.to_csv(OUT / "student_test_decoders_summary.csv")
    print("\nSTUDENT TEST (9 trips with a road network):\n", summ.round(4).to_string())
    return df, preds, trips


def plots_and_maps(df, preds, trips):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    piv = df.pivot(index="trip_id", columns="decoder", values="top1")
    piv.plot.bar(ax=axes[0], rot=45, title="Top-1 accuracy per student-test trip")
    piv = df.pivot(index="trip_id", columns="decoder", values="continuity")
    piv.plot.bar(ax=axes[1], rot=45, title="Route continuity per student-test trip")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "decoder_comparison.png", dpi=130); plt.close(fig)

    tune = pd.read_csv(OUT / "decoder_tuning.csv")
    t = tune[tune.emission == "classifier"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (a, g), s in t.groupby(["alpha", "gap_s"]):
        s = s.sort_values("beta")
        ax.plot(s.beta, s.top1, marker="o", label=f"alpha={a}, gap={g:.0f}s")
    ax.set(xscale="log", xlabel="beta (m)", ylabel="mean Top-1 (12 train tuning trips)",
           title="Decoder parameter sensitivity")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "figures" / "decoder_tuning.png", dpi=130)
    plt.close(fig)

    gain = df.pivot(index="trip_id", columns="decoder", values="top1")
    gain = (gain["HMM + Viterbi"] - gain["classifier only"]).sort_values()
    picks = [gain.index[-1], gain.index[0], gain.index[len(gain) // 2]]
    for tid in dict.fromkeys(picks):
        m = route_map(trips[tid], {k: preds[tid][k] for k in ("classifier only", "HMM + Viterbi")})
        m.save(str(OUT / "maps" / f"route_trip_{tid}.html"))
    print("maps for trips:", list(dict.fromkeys(picks)))


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    model, _ = load_model()
    p_clf, p_dist, tune_ids = tune(model)
    json.dump(asdict(p_clf), open(ARTIFACTS_DIR / "hmm_params.json", "w"), indent=2)
    json.dump({"classifier_hmm": asdict(p_clf), "distance_hmm": asdict(p_dist), "tuning_trips": list(tune_ids)},
              open(OUT / "selected_decoder_params.json", "w"), indent=2)
    print("selected:", asdict(p_clf))
    df, preds, trips = evaluate_student_test(model, p_clf, p_dist)
    plots_and_maps(df, preds, trips)


if __name__ == "__main__":
    main()
