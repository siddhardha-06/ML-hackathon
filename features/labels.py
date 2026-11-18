"""Per-observation labels from the reference route.

The .route file lists the arcs the vehicle travelled, in order, but has no link to
individual GPS rows (a trip has thousands of GPS points but only hundreds of route
arcs). To train a classifier we need "which arc was GPS point i on?".

We solve this as a monotone alignment: assign each GPS point i to a route position
j(i) such that j never decreases over time (the vehicle moves forward along the route)
and the total point-to-arc distance is minimal. Dynamic programming:

    D[i, j] = c(i, j) + min_{k <= j} D[i-1, k]

where c(i, j) = min(distance(point i, route arc j), LABEL_CAP_M). Capping makes the
alignment robust to GPS outliers. Complexity O(N * R) with vectorised rows.
The label of point i is route[j(i)]; label_dist is its distance to that arc, which
we keep so unreliable labels (GPS far from the reference) can be filtered or reported.
"""
import numpy as np

from common.config import LABEL_CAP_M
from common.geo import point_segment


def align_route(gps_xy, seg_a, seg_b, cap=LABEL_CAP_M):
    """Return (route_pos, dist) per GPS point. seg_a/seg_b: (R,2) route arc endpoints."""
    n, r = len(gps_xy), len(seg_a)
    ax, ay, bx, by = seg_a[:, 0], seg_a[:, 1], seg_b[:, 0], seg_b[:, 1]
    back = np.empty((n, r), dtype=np.int16 if r < 32_000 else np.int32)
    idx = np.arange(r)

    def cost(i):
        d, _ = point_segment(gps_xy[i, 0], gps_xy[i, 1], ax, ay, bx, by)
        return np.minimum(d, cap)

    D = cost(0)
    back[0] = idx
    for i in range(1, n):
        run_min = np.minimum.accumulate(D)
        arg = np.maximum.accumulate(np.where(D <= run_min, idx, 0))   # prefix argmin
        back[i] = arg
        D = cost(i) + run_min

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(D))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _ = point_segment(gps_xy[:, 0], gps_xy[:, 1], ax[pos], ay[pos], bx[pos], by[pos])
    return pos, dist


def label_trip(trip):
    """Return (label_arc, label_dist, route_pos) per GPS point, or None if no route/network."""
    if not trip.has_network or trip.route is None or len(trip.route) == 0:
        return None
    A, B = trip.arc_segments()
    route = trip.route
    pos, dist = align_route(trip.gps_xy, A[route], B[route])
    return route[pos], dist, pos
