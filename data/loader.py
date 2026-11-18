"""Raw trip loading.

Each trip folder <id>/ holds headerless whitespace files:
  <id>.track  lon lat t     one GPS observation per row (t in seconds)
  <id>.nodes  lon lat       road-network nodes, node_id = row index
  <id>.arcs   from to       directed road segments, arc_id = row index
  <id>.route  arc_id        reference route: arcs in travel order (NOT one per GPS row)
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.config import PARTITIONS
from common.geo import LocalProjection


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray                 # (N, 3) lon, lat, t
    nodes: np.ndarray                 # (M, 2) lon, lat
    arcs: np.ndarray | None           # (A, 2) from_node, to_node
    route: np.ndarray | None          # (R,) arc ids in travel order
    proj: LocalProjection = field(init=False)
    gps_xy: np.ndarray = field(init=False)
    node_xy: np.ndarray = field(init=False)

    def __post_init__(self):
        self.proj = LocalProjection(self.track[:, 0].mean(), self.track[:, 1].mean())
        self.gps_xy = np.column_stack(self.proj.to_xy(self.track[:, 0], self.track[:, 1]))
        self.node_xy = np.column_stack(self.proj.to_xy(self.nodes[:, 0], self.nodes[:, 1]))

    @property
    def t(self):
        return self.track[:, 2]

    @property
    def has_network(self):
        return self.arcs is not None and len(self.arcs) > 0

    def arc_segments(self):
        """(A, 2) start xy and (A, 2) end xy of every arc."""
        return self.node_xy[self.arcs[:, 0]], self.node_xy[self.arcs[:, 1]]


def _read(path: Path, dtype=float, ndmin=2):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir, partition: str = "") -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track")
    if track is None:
        raise FileNotFoundError(f"{tid}: missing .track")
    track = track[np.isfinite(track).all(axis=1)]
    track = track[np.argsort(track[:, 2], kind="stable")]     # guarantee time order
    nodes = _read(trip_dir / f"{tid}.nodes")
    arcs = _read(trip_dir / f"{tid}.arcs", dtype=np.int64)
    route = _read(trip_dir / f"{tid}.route", dtype=np.int64, ndmin=1)
    if arcs is not None and route is not None:
        route = route[(route >= 0) & (route < len(arcs))]
    return Trip(tid, partition, track, nodes, arcs, route)


def list_trips(partition: str, root=None):
    root = Path(root) if root else PARTITIONS[partition]
    return sorted(p for p in root.iterdir() if p.is_dir())


def iter_trips(partition: str):
    for d in list_trips(partition):
        yield load_trip(d, partition)
