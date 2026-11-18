"""Candidate generation and feature engineering.

Map matching is turned into a supervised problem:
  for every GPS observation i, generate a small set of candidate arcs near it,
  and train a classifier to score "is this candidate the true arc of point i?".

Candidate generation
  Every arc is densified into sample points every SAMPLE_SPACING_M metres and stored in a
  KD-tree. A point at perpendicular distance d from an arc always has an arc sample within
  d + spacing/2, so querying the tree with radius R + spacing/2 and then computing the
  exact point-to-segment distance finds every arc within R. Arcs are de-duplicated and the
  K nearest within R are kept. Both directions of a two-way street are separate arcs with
  identical geometry, so the heading feature is what tells them apart.

Features (per GPS point x candidate arc)
  dist, dist_minus_min, dist_ratio, cand_rank, n_cands   - geometric closeness, relative to rivals
  heading_diff, cos_heading                              - GPS travel direction vs arc direction
  speed, disp                                            - heading is unreliable when nearly stopped
  proj_t, arc_len                                        - where on the arc the point projects
  dist_prev, dist_next                                   - is point i±3 also close to this arc? (context)
  out_deg_to, in_deg_from                                - intersection complexity at the arc ends
  dt_prev                                                - sampling gap before this point
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common.config import (CAND_MAX_K, CAND_RADIUS_M, CONTEXT_OFFSET, HEADING_WINDOW,
                           SAMPLE_SPACING_M)
from common.geo import angle_diff_deg, bearing_deg, point_segment

FEATURE_COLS = [
    "dist", "dist_minus_min", "dist_ratio", "cand_rank", "n_cands",
    "heading_diff", "cos_heading", "speed", "disp",
    "proj_t", "arc_len", "dist_prev", "dist_next",
    "out_deg_to", "in_deg_from", "dt_prev",
]


class ArcIndex:
    """KD-tree over densified arc geometry of one trip's road network."""

    def __init__(self, trip, spacing=SAMPLE_SPACING_M):
        self.A, self.B = trip.arc_segments()
        vec = self.B - self.A
        self.L = np.hypot(vec[:, 0], vec[:, 1])
        self.bearing = bearing_deg(vec[:, 0], vec[:, 1])
        n_arcs = len(self.A)
        ns = np.maximum(np.ceil(self.L / spacing).astype(np.int64), 1) + 1
        self.arc_of = np.repeat(np.arange(n_arcs), ns)
        start = np.repeat(np.cumsum(ns) - ns, ns)
        t = (np.arange(ns.sum()) - start) / np.repeat(ns - 1, ns)
        pts = self.A[self.arc_of] + t[:, None] * vec[self.arc_of]
        self.tree = cKDTree(pts)
        self.spacing = spacing
        n_nodes = len(trip.node_xy)
        self.out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
        self.in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
        self.arcs = trip.arcs

    def query(self, xy, radius=CAND_RADIUS_M, k=CAND_MAX_K, kq=96):
        """Return arc ids, distances, projection t: arrays (N, k), sorted by distance, -1/inf padded."""
        n_samples = len(self.arc_of)
        kq = min(kq, n_samples)
        d_s, i_s = self.tree.query(xy, k=kq, distance_upper_bound=radius + self.spacing / 2 + 1.0)
        if kq == 1:
            d_s, i_s = d_s[:, None], i_s[:, None]
        found = np.isfinite(d_s)
        arc = np.where(found, self.arc_of[np.minimum(i_s, n_samples - 1)], 0)
        A, B = self.A[arc], self.B[arc]
        d, t = point_segment(xy[:, 0:1], xy[:, 1:2], A[..., 0], A[..., 1], B[..., 0], B[..., 1])
        d = np.where(found & (d <= radius), d, np.inf)
        arc = np.where(np.isfinite(d), arc, -1)

        order = np.lexsort((arc, d), axis=1)                # sort by distance, then arc id
        arc, d, t = (np.take_along_axis(a, order, 1) for a in (arc, d, t))
        dup = np.zeros_like(found)
        dup[:, 1:] = (arc[:, 1:] == arc[:, :-1]) & (arc[:, 1:] >= 0)
        d = np.where(dup, np.inf, d)
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        arc, d, t = (np.take_along_axis(a, order, 1) for a in (arc, d, t))
        arc = np.where(np.isfinite(d), arc, -1)
        if arc.shape[1] < k:                                  # tiny networks: pad to k columns
            pad = k - arc.shape[1]
            arc = np.pad(arc, ((0, 0), (0, pad)), constant_values=-1)
            d = np.pad(d, ((0, 0), (0, pad)), constant_values=np.inf)
            t = np.pad(t, ((0, 0), (0, pad)))
        return arc, d, t


def gps_motion(trip, w=HEADING_WINDOW):
    """Heading (deg), speed (m/s) and displacement (m) of each GPS point from points i-w..i+w."""
    xy, ts = trip.gps_xy, trip.t
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = xy[i1, 0] - xy[i0, 0], xy[i1, 1] - xy[i0, 1]
    disp = np.hypot(dx, dy)
    dt = np.maximum(ts[i1] - ts[i0], 1.0)
    return bearing_deg(dx, dy), disp / dt, disp


def build_features(trip, index: ArcIndex, arc, d, t):
    """Flatten (N, K) candidates of one trip into a feature DataFrame (one row per candidate)."""
    n, k = arc.shape
    xy = trip.gps_xy
    valid = arc >= 0
    a = np.where(valid, arc, 0)

    heading, speed, disp = gps_motion(trip)
    hd = angle_diff_deg(heading[:, None], index.bearing[a])
    hd = np.where(disp[:, None] < 1.0, 90.0, hd)           # stationary: direction unknown -> neutral

    n_c = valid.sum(1)
    dmin = np.where(n_c > 0, d[:, 0], np.nan)
    ctx = {}
    for name, off in (("dist_prev", -CONTEXT_OFFSET), ("dist_next", CONTEXT_OFFSET)):
        j = np.clip(np.arange(n) + off, 0, n - 1)
        A, B = index.A[a], index.B[a]
        ctx[name], _ = point_segment(xy[j, 0][:, None], xy[j, 1][:, None],
                                     A[..., 0], A[..., 1], B[..., 0], B[..., 1])
    dt_prev = np.diff(trip.t, prepend=trip.t[0])

    obs = np.broadcast_to(np.arange(n)[:, None], (n, k))
    rank = np.broadcast_to(np.arange(k)[None, :], (n, k))
    m = valid
    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": obs[m].astype(np.int32),
        "arc_id": arc[m].astype(np.int64),
        "dist": d[m],
        "dist_minus_min": (d - dmin[:, None])[m],
        "dist_ratio": (d / (dmin[:, None] + 1.0))[m],
        "cand_rank": rank[m].astype(np.int16),
        "n_cands": np.broadcast_to(n_c[:, None], (n, k))[m].astype(np.int16),
        "heading_diff": hd[m],
        "cos_heading": np.cos(np.radians(hd))[m],
        "speed": np.broadcast_to(speed[:, None], (n, k))[m],
        "disp": np.broadcast_to(disp[:, None], (n, k))[m],
        "proj_t": t[m],
        "arc_len": index.L[a][m],
        "dist_prev": ctx["dist_prev"][m],
        "dist_next": ctx["dist_next"][m],
        "out_deg_to": index.out_deg[index.arcs[a, 1]][m].astype(np.int16),
        "in_deg_from": index.in_deg[index.arcs[a, 0]][m].astype(np.int16),
        "dt_prev": np.broadcast_to(dt_prev[:, None], (n, k))[m],
    })
    return df


def candidates_for_trip(trip, radius=CAND_RADIUS_M, k=CAND_MAX_K):
    """Inference-time helper: candidates + features for a trip (no labels needed)."""
    index = ArcIndex(trip)
    arc, d, t = index.query(trip.gps_xy, radius, k)
    return build_features(trip, index, arc, d, t), index
