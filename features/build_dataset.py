"""Build the candidate datasets for train and student_test.

Run from the project root:   python -m features.build_dataset

Outputs
  artifacts/obs_<partition>.parquet     one row per GPS point (label arc, label distance, ...)
  artifacts/cands_<partition>.parquet   one row per (GPS point, candidate arc) + features + label
  results/p1/trip_summary_<partition>.csv
  results/p1/candidate_recall.csv       recall of the true arc for different radius / K (train only)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.config import (ARTIFACTS_DIR, CAND_MAX_K, CAND_RADIUS_M, LABEL_MAX_DIST_M, POOL_MAX_K,
                           POOL_RADIUS_M, RESULTS_DIR, ROOT)
from data.loader import list_trips, load_trip
from features.candidates import ArcIndex, build_features
from features.labels import label_trip

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")


def process_trip(trip_dir, partition):
    t0 = time.time()
    trip = load_trip(trip_dir, partition)
    n = len(trip.track)
    summary = {"trip_id": trip.trip_id, "partition": partition, "n_obs": n}
    if not trip.has_network:
        summary["note"] = "no .arcs file - skipped"
        return None, None, summary, None

    index = ArcIndex(trip)
    pool_arc, pool_d, pool_t = index.query(trip.gps_xy, POOL_RADIUS_M, POOL_MAX_K, kq=160)

    lab = label_trip(trip)
    obs = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": np.arange(n, dtype=np.int32),
                        "t": trip.t, "lon": trip.track[:, 0], "lat": trip.track[:, 1]})
    recall = None
    if lab is not None:
        label_arc, label_dist, route_pos = lab
        obs["label_arc"], obs["label_dist"], obs["route_pos"] = label_arc, label_dist, route_pos
        obs["label_ok"] = label_dist <= LABEL_MAX_DIST_M
        hit = pool_arc == label_arc[:, None]
        obs["label_rank"] = np.where(hit.any(1), hit.argmax(1), -1)
        obs["label_cand_dist"] = np.where(hit.any(1), pool_d[np.arange(n), hit.argmax(1)], np.inf)
        recall = obs[["label_rank", "label_cand_dist", "label_ok"]].copy()

    # final candidate set = first CAND_MAX_K pool entries within CAND_RADIUS_M (pool is sorted)
    arc, d, t = pool_arc[:, :CAND_MAX_K].copy(), pool_d[:, :CAND_MAX_K].copy(), pool_t[:, :CAND_MAX_K]
    far = d > CAND_RADIUS_M
    arc[far], d[far] = -1, np.inf
    feats = build_features(trip, index, arc, d, t)
    obs["n_cands"] = (arc >= 0).sum(1)
    if lab is not None:
        feats["label"] = (feats["arc_id"].to_numpy() == label_arc[feats["obs_idx"].to_numpy()]).astype(np.int8)
        obs["in_cands"] = (arc == label_arc[:, None]).any(1)

    summary.update({
        "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
        "n_route_arcs": 0 if trip.route is None else len(trip.route),
        "mean_cands": float(obs["n_cands"].mean()),
        "no_cand_rate": float((obs["n_cands"] == 0).mean()),
        "seconds": round(time.time() - t0, 2),
    })
    if lab is not None:
        summary.update({
            "label_dist_median": float(np.median(label_dist)),
            "label_dist_p95": float(np.percentile(label_dist, 95)),
            "unreliable_label_rate": float((~obs["label_ok"]).mean()),
            "candidate_miss_rate": float((~obs["in_cands"]).mean()),
        })
    return obs, feats, summary, recall


def recall_table(recalls):
    r = pd.concat(recalls, ignore_index=True)
    r = r[r["label_ok"]]
    rows = []
    for radius in (15, 25, 35, 50, 75, 100, 150):
        for k in (2, 4, 6, 8, 10, 16, 24):
            hit = (r["label_rank"] >= 0) & (r["label_rank"] < k) & (r["label_cand_dist"] <= radius)
            rows.append({"radius_m": radius, "k": k, "recall": hit.mean(), "miss_rate": 1 - hit.mean()})
    return pd.DataFrame(rows)


def main(partitions=("train", "student_test"), n_jobs=8):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "p1"
    out.mkdir(parents=True, exist_ok=True)
    for part in partitions:
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(process_trip)(d, part) for d in list_trips(part))
        obs = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
        feats = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
        summ = pd.DataFrame([r[2] for r in res])
        obs.to_parquet(ARTIFACTS_DIR / f"obs_{part}.parquet", index=False)
        feats.to_parquet(ARTIFACTS_DIR / f"cands_{part}.parquet", index=False)
        summ.to_csv(out / f"trip_summary_{part}.csv", index=False)
        print(f"[{part}] trips={len(summ)} obs={len(obs):,} candidate rows={len(feats):,} "
              f"positives={int(feats.get('label', pd.Series(dtype=int)).sum()):,} "
              f"miss_rate={(~obs['in_cands']).mean():.4f} time={time.time() - t0:.0f}s")
        if part == "train":
            rt = recall_table([r[3] for r in res if r[3] is not None])
            rt.to_csv(out / "candidate_recall.csv", index=False)
            print(rt.pivot(index="radius_m", columns="k", values="recall").round(4).to_string())


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "student_test"))
