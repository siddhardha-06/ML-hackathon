"""HMM + Viterbi route reconstruction (P3).

Hidden states
  At GPS observation i the hidden state is "the vehicle is on candidate arc c, at the
  projected position proj_t along it". The state set of observation i is exactly its
  candidate list from features.candidates, so state <-> arc correspondence is preserved.

Emission  log P(obs_i | state c)
  'classifier' : alpha * log(p_clf(c))      p_clf = probability from the selected model
  'distance'   : -0.5 * (dist / sigma)^2    classic Gaussian GPS-noise model (no ML), baseline

Transition  log P(state_i = b | state_{i-1} = a)
  route = network driving distance from (arc a, t_a) to (arc b, t_b)
          same arc       : |t_b - t_a| * len(a)
          different arcs : (1 - t_a) len(a) + shortest_path(to(a), from(b)) + t_b len(b)
  gc    = straight-line distance between the two GPS points
  log T = -|route - gc| / beta      (Newson & Krumm 2009: real moves have route ~ gc)
  Unreachable within the search limit -> -inf (physically impossible jump).

Gaps and failures
  * observations without candidates are skipped (prediction -1) and the next observation
    connects to the last one that had candidates;
  * if the time gap exceeds gap_s seconds, or every transition is impossible, the chain is
    broken and decoding restarts from emissions only (separate sub-routes).

Viterbi recurrence:  V_i(b) = max_a [V_{i-1}(a) + log T(a,b)] + log E_i(b),  back-pointer argmax.
"""
import heapq
from dataclasses import dataclass

import numpy as np

from common.geo import point_segment


@dataclass
class HMMParams:
    emission: str = "classifier"   # 'classifier' or 'distance'
    alpha: float = 1.0             # weight of classifier log-probability
    sigma: float = 10.0            # GPS noise (m) for distance emission
    beta: float = 20.0             # transition tolerance (m)
    gap_s: float = 60.0            # break the chain on larger time gaps
    p_floor: float = 1e-4          # clip classifier probabilities
    max_search_m: float = 20_000.0


class RoadGraph:
    """Directed road graph in CSR form with a cached, distance-limited Dijkstra."""

    def __init__(self, trip):
        self.arcs = trip.arcs
        a, b = trip.node_xy[trip.arcs[:, 0]], trip.node_xy[trip.arcs[:, 1]]
        self.arc_len = np.hypot(*(b - a).T)
        n = len(trip.node_xy)
        order = np.argsort(trip.arcs[:, 0], kind="stable")
        self.indptr = np.searchsorted(trip.arcs[order, 0], np.arange(n + 1))
        self.nbr = trip.arcs[order, 1]
        self.w = self.arc_len[order]
        self.arc_of_edge = order
        self._cache = {}

    def arc_path(self, src: int, dst: int, limit: float = 3000.0):
        """Arc ids of the shortest path src -> dst (None if farther than limit)."""
        dist, parent = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                path = []
                while u != src:
                    k = parent[u]
                    path.append(int(self.arc_of_edge[k]))
                    u = int(self.arcs[self.arc_of_edge[k], 0])
                return path[::-1]
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(self.indptr[u], self.indptr[u + 1]):
                v, nd = int(self.nbr[k]), d + self.w[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v], parent[v] = nd, k
                    heapq.heappush(heap, (nd, v))
        return None

    def dist_from(self, src: int, limit: float) -> dict:
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        indptr, nbr, w = self.indptr, self.nbr, self.w
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v, nd = nbr[k], d + w[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) > 50_000:
            self._cache.clear()
        self._cache[src] = (limit, dist)
        return dist


def _transition(graph, prev_arc, prev_t, cur_arc, cur_t, gc, beta, max_search):
    """(K_prev, K_cur) matrix of log transition probabilities."""
    kp, kc = len(prev_arc), len(cur_arc)
    route = np.full((kp, kc), np.inf)
    limit = min(2.0 * gc + 300.0, max_search)
    L = graph.arc_len
    for i in range(kp):
        a, ta = prev_arc[i], prev_t[i]
        same = cur_arc == a
        route[i, same] = np.abs(cur_t[same] - ta) * L[a]
        rest = np.where(~same)[0]
        if len(rest) == 0:
            continue
        lead = (1.0 - ta) * L[a]
        dist = graph.dist_from(int(graph.arcs[a, 1]), limit)
        for j in rest:
            b = cur_arc[j]
            sp = dist.get(int(graph.arcs[b, 0]))
            if sp is not None:
                route[i, j] = lead + sp + cur_t[j] * L[b]
    with np.errstate(invalid="ignore"):
        return -np.abs(route - gc) / beta


def decode(trip, cands, probs=None, params: HMMParams = HMMParams(), graph: RoadGraph | None = None):
    """Viterbi decoding for one trip.

    cands: DataFrame of this trip's candidates (obs_idx, arc_id, dist, proj_t), any order.
    probs: classifier probability per row of cands (required for emission='classifier').
    Returns np.ndarray of predicted arc_id per GPS observation (-1 when no candidate).
    """
    n = len(trip.track)
    graph = graph or RoadGraph(trip)
    c = cands[["obs_idx", "arc_id", "dist", "proj_t"]].copy()
    if params.emission == "classifier":
        c["logE"] = params.alpha * np.log(np.clip(np.asarray(probs), params.p_floor, 1.0))
    else:
        c["logE"] = -0.5 * (c["dist"].to_numpy() / params.sigma) ** 2
    c = c.sort_values(["obs_idx", "dist"], kind="stable")
    groups = {i: g for i, g in c.groupby("obs_idx", sort=True)}

    pred = np.full(n, -1, dtype=np.int64)
    xy, ts = trip.gps_xy, trip.t
    steps = []                 # (obs_idx, arcs, back-pointer array or None for chain start)
    V = prev = None

    def flush():
        if not steps:
            return
        k = int(np.argmax(V))
        for obs_i, arcs_i, back in reversed(steps):
            pred[obs_i] = arcs_i[k]
            if back is not None:
                k = back[k]
        steps.clear()

    for i in range(n):
        g = groups.get(i)
        if g is None:
            continue
        arcs_i = g["arc_id"].to_numpy()
        t_i = g["proj_t"].to_numpy()
        e_i = g["logE"].to_numpy()
        if prev is None or ts[i] - ts[prev[0]] > params.gap_s:
            flush()
            V = e_i.copy()
            steps.append((i, arcs_i, None))
        else:
            gc = float(np.hypot(*(xy[i] - xy[prev[0]])))
            T = _transition(graph, prev[1], prev[2], arcs_i, t_i, gc, params.beta, params.max_search_m)
            S = V[:, None] + T
            back = np.argmax(S, axis=0)
            best = S[back, np.arange(len(arcs_i))]
            if not np.isfinite(best).any():          # every move impossible -> break the chain
                flush()
                V = e_i.copy()
                steps.append((i, arcs_i, None))
            else:
                V = best + e_i
                steps.append((i, arcs_i, back))
        prev = (i, arcs_i, t_i)
    flush()
    return pred


def classifier_only(trip, cands, probs):
    """Baseline: independent argmax of classifier probability per observation."""
    pred = np.full(len(trip.track), -1, dtype=np.int64)
    d = cands[["obs_idx", "arc_id"]].copy()
    d["p"] = np.asarray(probs)
    d = d.sort_values(["obs_idx", "p"], ascending=[True, False]).drop_duplicates("obs_idx")
    pred[d["obs_idx"].to_numpy()] = d["arc_id"].to_numpy()
    return pred


def fill_route(graph: RoadGraph, pred_arcs, max_fill_arcs=60):
    """Turn per-observation predictions into a continuous arc sequence.

    Consecutive duplicates are collapsed; when two consecutive arcs are not adjacent
    (a short arc was crossed between two GPS fixes) the shortest network path between
    them is inserted. Gaps that cannot be bridged stay as breaks."""
    seq = [int(a) for a in pred_arcs if a >= 0]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    if not seq:
        return []
    out = [seq[0]]
    for b in seq[1:]:
        a = out[-1]
        if graph.arcs[a, 1] != graph.arcs[b, 0]:
            path = graph.arc_path(int(graph.arcs[a, 1]), int(graph.arcs[b, 0]))
            if path is not None and len(path) <= max_fill_arcs:
                out.extend(path)
        out.append(b)
    return out


def arcs_to_polyline(trip, pred_arcs):
    """Collapse consecutive duplicate arcs and return [[lat, lon], ...] along the predicted route."""
    seq = [a for a in pred_arcs if a >= 0]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    pts = []
    for a in seq:
        f, t = trip.arcs[a]
        for node in (f, t):
            p = [float(trip.nodes[node, 1]), float(trip.nodes[node, 0])]
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts, seq


def snapped_points(trip, pred_arcs):
    """GPS points projected onto their predicted arc ([lat, lon] or None)."""
    out = []
    A, B = trip.arc_segments()
    for i, a in enumerate(pred_arcs):
        if a < 0:
            out.append(None)
            continue
        _, t = point_segment(trip.gps_xy[i, 0], trip.gps_xy[i, 1], A[a, 0], A[a, 1], B[a, 0], B[a, 1])
        x, y = A[a] + t * (B[a] - A[a])
        lon, lat = trip.proj.to_lonlat(x, y)
        out.append([float(lat), float(lon)])
    return out
