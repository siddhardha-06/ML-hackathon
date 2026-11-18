"""Shared settings: paths, random seed and pipeline constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    for p in (ROOT / "Student folder", ROOT.parent / "Student folder"):
        if p.exists():
            return p
    return ROOT / "Student folder"


# Raw data is never committed. Override with env var PATHSNAP_DATA if it lives elsewhere.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", _default_data_dir()))
PARTITIONS = {
    "train": DATA_DIR / "train",
    "student_test": DATA_DIR / "student_test",
}
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42

# --- candidate generation (chosen from the recall sweep in results/p1/candidate_recall.csv)
CAND_RADIUS_M = 50.0        # arcs farther than this from a GPS point are not candidates
CAND_MAX_K = 10             # keep at most K nearest arcs per GPS point
POOL_RADIUS_M = 150.0       # wider pool used only for the recall sweep
POOL_MAX_K = 24
SAMPLE_SPACING_M = 20.0     # arc densification step for the KD-tree

# --- labels
LABEL_CAP_M = 200.0         # cap per-point cost in route alignment (robust to outliers)
LABEL_MAX_DIST_M = 100.0    # points farther than this from their reference arc = unreliable label

# --- features
HEADING_WINDOW = 2          # use points i-2 .. i+2 for heading / speed
CONTEXT_OFFSET = 3          # distance of point i±3 to the candidate arc
