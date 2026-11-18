"""Load candidate tables for model training / evaluation (P2)."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common.config import ARTIFACTS_DIR, SEED
from data.loader import list_trips, load_trip
from features.candidates import FEATURE_COLS

# heavy-tailed features get log1p before scaling (only for scale-sensitive models)
LOG_COLS = ["dist", "dist_minus_min", "dist_ratio", "dist_prev", "dist_next", "arc_len",
            "speed", "disp", "dt_prev"]
LOG_IDX = [FEATURE_COLS.index(c) for c in LOG_COLS]


def _log_some(X):
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, LOG_IDX] = np.log1p(np.clip(X[:, LOG_IDX], 0, None))
    return X


def scaled(model):
    """log1p(heavy-tailed) -> StandardScaler -> model. Fitted inside each CV fold only."""
    return make_pipeline(FunctionTransformer(_log_some), StandardScaler(), model)


def load_partition(partition, obs_frac=1.0, seed=SEED):
    """Return cands (rows of reliable-label observations), obs table.

    obs_frac < 1 samples that share of observations per trip (all candidates of a sampled
    observation are kept, so per-observation Top-1 stays well defined)."""
    obs = pd.read_parquet(ARTIFACTS_DIR / f"obs_{partition}.parquet")
    cands = pd.read_parquet(ARTIFACTS_DIR / f"cands_{partition}.parquet")
    obs = obs[obs["label_ok"]]
    if obs_frac < 1.0:
        obs = obs.groupby("trip_id", group_keys=False).sample(frac=obs_frac, random_state=seed)
    keep = obs[["trip_id", "obs_idx"]]
    cands = cands.merge(keep, on=["trip_id", "obs_idx"], how="inner")
    cands = cands.sort_values(["trip_id", "obs_idx", "cand_rank"]).reset_index(drop=True)
    return cands, obs.reset_index(drop=True)


def xy(cands):
    return cands[FEATURE_COLS].to_numpy(np.float32), cands["label"].to_numpy(np.int8)


def load_trips(partition):
    return {d.name: load_trip(d, partition) for d in list_trips(partition)}
