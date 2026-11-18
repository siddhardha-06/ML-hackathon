"""PathSnap API - runs on Laptop 1 (P4).

Start (from the project root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment
    PATHSNAP_EVAL_DIR   folder with the evaluation trips (one sub-folder per trip, same file
                        format as the training data). Defaults to the student_test partition.

Endpoints
    GET /health                 model + data status
    GET /trips                  the trips the evaluator can choose from
    GET /predict/{trip_id}      full pipeline on the server: candidates -> classifier -> HMM/Viterbi,
                                then accuracy against the server-side reference route.
The reference route / labels never leave the server: the response only carries the GPS
trace, the predicted route and the computed metrics.
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from common.config import PARTITIONS
from common.metrics import route_continuity, route_set_scores
from data.loader import list_trips, load_trip
from features.labels import label_trip
from hmm.pipeline import load_model, predict_trip
from hmm.viterbi import RoadGraph, arcs_to_polyline, fill_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", PARTITIONS["student_test"]))
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

MODEL, PARAMS = load_model()
log.info("model loaded: %s | decoder: %s", type(MODEL).__name__, PARAMS)
log.info("evaluation trips folder: %s", EVAL_DIR)


def _trip_dirs():
    return {d.name: d for d in list_trips("eval", root=EVAL_DIR)}


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "decoder": PARAMS.__dict__,
            "eval_dir": str(EVAL_DIR), "n_trips": len(_trip_dirs())}


@app.get("/trips")
def trips():
    out = []
    for tid, d in sorted(_trip_dirs().items()):
        has_arcs = (d / f"{tid}.arcs").exists()
        n = sum(1 for _ in open(d / f"{tid}.track"))
        out.append({"trip_id": tid, "n_obs": n, "has_network": has_arcs})
    log.info("GET /trips -> %d trips", len(out))
    return out


def _score(trip, pred, graph):
    """Accuracy against the server-side reference (never returned as raw labels)."""
    lab = label_trip(trip)
    route = fill_route(graph, pred)
    res = {"continuity": route_continuity(pred, trip.arcs),
           "route_continuity_filled": route_continuity(np.array(route), trip.arcs)}
    if lab is None:
        return res, route
    label_arc, label_dist, _ = lab
    ok = label_dist <= 100.0
    res["top1_accuracy"] = float((pred[ok] == label_arc[ok]).mean()) if ok.any() else None
    res["scored_points"] = int(ok.sum())
    res.update(route_set_scores(route, trip.route))
    return res, route


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    t0 = time.perf_counter()
    dirs = _trip_dirs()
    if trip_id not in dirs:
        log.warning("GET /predict/%s -> unknown trip", trip_id)
        raise HTTPException(404, f"unknown trip {trip_id}")
    trip = load_trip(dirs[trip_id], "eval")
    if not trip.has_network:
        log.warning("GET /predict/%s -> no road network (.arcs missing)", trip_id)
        raise HTTPException(422, f"trip {trip_id} has no .arcs file, so it cannot be map-matched")

    out = predict_trip(trip, MODEL, PARAMS)
    graph = RoadGraph(trip)
    hmm_m, route = _score(trip, out["pred_hmm"], graph)
    clf_m, _ = _score(trip, out["pred_clf"], graph)
    polyline, _ = arcs_to_polyline(trip, route)
    total = time.perf_counter() - t0

    step = max(1, len(trip.track) // 3000)                 # keep the payload small
    gps = [[float(a), float(b)] for b, a in trip.track[::step, :2]]
    resp = {
        "trip_id": trip_id,
        "n_obs": int(len(trip.track)),
        "n_obs_without_candidates": int((out["pred_hmm"] < 0).sum()),
        "gps": gps,
        "pred_route": polyline,
        "n_route_arcs": len(route),
        "metrics": {"hmm": hmm_m, "classifier_only": clf_m},
        "timings_s": {**{k: round(v, 3) for k, v in out["timings"].items()}, "total": round(total, 3)},
    }
    log.info("GET /predict/%s | obs=%d | HMM top1=%s cont=%.3f routeF1=%s | classifier top1=%s | %.2fs",
             trip_id, resp["n_obs"], _fmt(hmm_m.get("top1_accuracy")), hmm_m["continuity"],
             _fmt(hmm_m.get("route_f1")), _fmt(clf_m.get("top1_accuracy")), total)
    return resp


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"
