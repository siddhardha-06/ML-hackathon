"""Metrics shared by model selection (P2), route reconstruction (P3) and the API (P4).

Observation-level Top-1 accuracy
  for every GPS point with a reliable label, take the predicted arc and compare it with the
  reference arc. Points whose true arc is not in the candidate set, or that have no
  candidates, count as wrong, so Top-1 is an end-to-end number, not inflated by misses.
Candidate miss rate
  share of reliable points whose reference arc is absent from the candidate set.
Route continuity
  share of consecutive predictions (a -> b) that are physically possible without a jump:
  same arc, or b starts where a ends (arcs[a].to == arcs[b].from).
Route-level precision / recall / F1
  set overlap between predicted arcs and reference route arcs (sequence-level quality).
"""
import numpy as np
import pandas as pd


def top1_from_scores(cands: pd.DataFrame, scores) -> pd.DataFrame:
    """Highest-scoring candidate per (trip_id, obs_idx) -> columns trip_id, obs_idx, pred_arc, pred_score."""
    d = cands[["trip_id", "obs_idx", "arc_id"]].copy()
    d["s"] = np.asarray(scores)
    d = d.sort_values(["trip_id", "obs_idx", "s"], ascending=[True, True, False], kind="stable")
    d = d.drop_duplicates(["trip_id", "obs_idx"])
    return d.rename(columns={"arc_id": "pred_arc", "s": "pred_score"})


def attach_predictions(obs: pd.DataFrame, pred: pd.DataFrame, col="pred_arc") -> pd.DataFrame:
    out = obs.merge(pred[["trip_id", "obs_idx", col]], on=["trip_id", "obs_idx"], how="left")
    out[col] = out[col].fillna(-1).astype(np.int64)
    return out


def top1_accuracy(obs_with_pred: pd.DataFrame, col="pred_arc") -> float:
    o = obs_with_pred[obs_with_pred["label_ok"]]
    return float((o[col] == o["label_arc"]).mean()) if len(o) else float("nan")


def candidate_miss_rate(obs: pd.DataFrame) -> float:
    o = obs[obs["label_ok"]]
    return float((~o["in_cands"]).mean()) if len(o) else float("nan")


def route_continuity(pred_arcs, arcs) -> float:
    p = np.asarray(pred_arcs)
    p = p[p >= 0]
    if len(p) < 2:
        return float("nan")
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_set_scores(pred_arcs, ref_route) -> dict:
    pred = set(int(x) for x in np.asarray(pred_arcs) if x >= 0)
    ref = set(int(x) for x in np.asarray(ref_route))
    tp = len(pred & ref)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def sequence_report(obs_with_pred: pd.DataFrame, trips: dict, col="pred_arc") -> pd.DataFrame:
    """Per-trip metrics. trips: trip_id -> Trip (needs .arcs and .route)."""
    rows = []
    for tid, o in obs_with_pred.groupby("trip_id", sort=True):
        trip = trips[tid]
        o = o.sort_values("obs_idx")
        rel = o[o["label_ok"]]
        rows.append({
            "trip_id": tid,
            "n_obs": len(o),
            "top1": float((rel[col] == rel["label_arc"]).mean()) if len(rel) else np.nan,
            "continuity": route_continuity(o[col].to_numpy(), trip.arcs),
            **route_set_scores(o[col].to_numpy(), trip.route),
        })
    return pd.DataFrame(rows)
