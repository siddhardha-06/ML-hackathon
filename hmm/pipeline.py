"""End-to-end inference for one trip: candidates -> classifier -> HMM/Viterbi (used by API + evaluation)."""
import json
import time

import joblib
import numpy as np

from common.config import ARTIFACTS_DIR
from features.candidates import FEATURE_COLS, candidates_for_trip
from hmm.viterbi import HMMParams, RoadGraph, classifier_only, decode


def load_model():
    """Final tuned model + decoder parameters saved by tuning/ and hmm/evaluate_hmm.py."""
    path = ARTIFACTS_DIR / "final_model.joblib"
    if not path.exists():
        path = ARTIFACTS_DIR / "models" / "LightGBM.joblib"
    model = joblib.load(path)
    params = HMMParams()
    cfg = ARTIFACTS_DIR / "hmm_params.json"
    if cfg.exists():
        params = HMMParams(**json.loads(cfg.read_text()))
    return model, params


def predict_trip(trip, model, params: HMMParams):
    timings = {}
    t0 = time.perf_counter()
    cands, _ = candidates_for_trip(trip)
    timings["candidates_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    probs = model.predict_proba(cands[FEATURE_COLS].to_numpy(np.float32))[:, 1] if len(cands) else np.array([])
    timings["classifier_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_clf = classifier_only(trip, cands, probs)
    pred_hmm = decode(trip, cands, probs, params, RoadGraph(trip))
    timings["viterbi_s"] = time.perf_counter() - t0
    return {"cands": cands, "probs": probs, "pred_clf": pred_clf, "pred_hmm": pred_hmm, "timings": timings}
