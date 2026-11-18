"""Exploratory data analysis (train + student_test partitions).

Run from the project root (after features.build_dataset):   python -m eda.run_eda

Outputs -> results/eda/   (reads P1 outputs from results/p1/)
  inventory.csv            files, sizes, row counts, gaps, extents per trip
  network_connectivity.csv nodes, arcs, isolated nodes, weak/strong components
  data_quality.md          short record of issues, assumptions and decisions
  figures/*.png            plots used in the report
  maps/eda_trip_<id>.html  interactive Folium map (GPS, network, reference route)
"""
from pathlib import Path

import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.config import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from data.loader import list_trips, load_trip  # noqa: E402

OUT = RESULTS_DIR / "eda"
P1RES = RESULTS_DIR / "p1"
FIG = OUT / "figures"
MAPS = OUT / "maps"


def inventory():
    rows, net = [], []
    for part in ("train", "student_test"):
        for d in list_trips(part):
            tid = d.name
            files = {ext: d / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
            trip = load_trip(d, part)
            dt = np.diff(trip.t)
            rows.append({
                "partition": part, "trip_id": tid,
                **{f"has_{e}": p.exists() for e, p in files.items()},
                **{f"kb_{e}": round(p.stat().st_size / 1024, 1) if p.exists() else 0 for e, p in files.items()},
                "n_obs": len(trip.track),
                "n_nan": int(np.isnan(np.loadtxt(files["track"])).sum()),
                "duration_s": trip.t[-1] - trip.t[0],
                "pct_dt_1s": float(np.mean(dt == 1)),
                "n_gaps_gt10s": int((dt > 10).sum()),
                "max_gap_s": float(dt.max()) if len(dt) else 0,
                "non_increasing_t": int((dt <= 0).sum()),
                "path_km": float(np.hypot(*np.diff(trip.gps_xy, axis=0).T).sum() / 1000),
                "lon_min": trip.track[:, 0].min(), "lon_max": trip.track[:, 0].max(),
                "lat_min": trip.track[:, 1].min(), "lat_max": trip.track[:, 1].max(),
                "n_route_arcs": 0 if trip.route is None else len(trip.route),
            })
            if trip.has_network:
                n = len(trip.nodes)
                a = trip.arcs
                g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n))
                n_weak, lab_w = connected_components(g, directed=True, connection="weak")
                n_strong, lab_s = connected_components(g, directed=True, connection="strong")
                used = np.zeros(n, bool)
                used[a.ravel()] = True
                route_cont = np.nan
                if trip.route is not None and len(trip.route) > 1:
                    r = trip.route
                    route_cont = float(np.mean(a[r[:-1], 1] == a[r[1:], 0]))
                twins = pd.MultiIndex.from_arrays([a[:, 0], a[:, 1]]).isin(
                    pd.MultiIndex.from_arrays([a[:, 1], a[:, 0]])).mean()
                net.append({
                    "partition": part, "trip_id": tid, "n_nodes": n, "n_arcs": len(a),
                    "isolated_nodes": int((~used).sum()),
                    "weak_components": int(n_weak),
                    "largest_weak_frac": float(np.bincount(lab_w).max() / n),
                    "strong_components": int(n_strong),
                    "largest_strong_frac": float(np.bincount(lab_s).max() / n),
                    "two_way_arc_frac": float(twins),
                    "self_loops": int((a[:, 0] == a[:, 1]).sum()),
                    "route_continuity": route_cont,
                })
            else:
                net.append({"partition": part, "trip_id": tid, "note": "no .arcs file"})
    return pd.DataFrame(rows), pd.DataFrame(net)


def plots(inv, net):
    FIG.mkdir(parents=True, exist_ok=True)
    obs_tr = pd.read_parquet(ARTIFACTS_DIR / "obs_train.parquet")
    recall = pd.read_csv(P1RES / "candidate_recall.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        ax[0].hist(inv.loc[inv.partition == part, "n_obs"], bins=30, alpha=.7, label=part, color=c)
    ax[0].set(xscale="log", xlabel="GPS observations per trip", ylabel="trips", title="Trip length")
    ax[0].legend()
    dts = []
    for d in list_trips("train"):
        t = np.loadtxt(d / f"{d.name}.track", usecols=2)
        dts.append(np.diff(t))
    dts = np.concatenate(dts)
    ax[1].hist(np.clip(dts, 0, 30), bins=30, color="tab:blue")
    ax[1].set(yscale="log", xlabel="seconds between GPS points (clipped at 30)", ylabel="count",
              title=f"Sampling interval (train): {np.mean(dts == 1):.1%} exactly 1 s")
    fig.tight_layout(); fig.savefig(FIG / "trip_length_sampling.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ld = obs_tr["label_dist"].clip(upper=200)
    ax.hist(ld, bins=100, color="tab:green")
    ax.axvline(100, color="r", ls="--", label="unreliable-label threshold (100 m)")
    ax.set(yscale="log", xlabel="distance GPS point -> aligned reference arc (m)", ylabel="points",
           title=f"GPS noise vs reference (train): median {obs_tr.label_dist.median():.1f} m")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "gps_to_reference_distance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in (2, 4, 6, 10, 24):
        s = recall[recall.k == k]
        ax.plot(s.radius_m, s.recall, marker="o", label=f"K={k}")
    ax.axvline(50, color="grey", ls=":")
    ax.set(xlabel="candidate radius (m)", ylabel="recall of true arc", title="Candidate recall (train, reliable labels)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "candidate_recall.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    nn = net.dropna(subset=["n_nodes"])
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        s = nn[nn.partition == part]
        ax.scatter(s.n_nodes, s.n_arcs, label=part, color=c, alpha=.7)
    ax.set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "network_size.png", dpi=130); plt.close(fig)

    summ = pd.read_csv(P1RES / "trip_summary_train.csv", dtype={"trip_id": str}).sort_values("unreliable_label_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(summ.trip_id, summ.unreliable_label_rate, color="tab:red")
    ax.set(ylabel="share of points > 100 m from reference", title="Unreliable labels per trip (train)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout(); fig.savefig(FIG / "unreliable_labels_per_trip.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, tid in zip(axes, ("00000001", "00000033", "00000054")):
        trip = load_trip(next(d for d in list_trips("train") if d.name == tid), "train")
        A, B = trip.arc_segments()
        lo, hi = trip.gps_xy.min(0) - 200, trip.gps_xy.max(0) + 200
        keep = ((A >= lo) & (A <= hi)).all(1)
        for a, b in zip(A[keep][::1], B[keep][::1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="lightgrey", lw=.5)
        r = trip.route
        for a, b in zip(A[r], B[r]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=2)
        ax.scatter(trip.gps_xy[:, 0], trip.gps_xy[:, 1], s=1, color="blue")
        ax.set(title=f"trip {tid} (train)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    fig.suptitle("GPS (blue), reference route (green), road network (grey)")
    fig.tight_layout(); fig.savefig(FIG / "trajectory_examples.png", dpi=130); plt.close(fig)


def folium_map(trip_id="00000001", partition="train"):
    MAPS.mkdir(parents=True, exist_ok=True)
    trip = load_trip(next(d for d in list_trips(partition) if d.name == trip_id), partition)
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    gps = folium.FeatureGroup("Raw GPS", show=True)
    for i in range(0, len(lat), 2):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="blue", fill=True, opacity=.7,
                            tooltip=f"t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    net = folium.FeatureGroup("Road network (near trip)", show=True)
    A, B = trip.arc_segments()
    lo, hi = trip.gps_xy.min(0) - 150, trip.gps_xy.max(0) + 150
    keep = np.where(((A >= lo) & (A <= hi)).all(1))[0][:6000]
    for i in keep:
        f, t = trip.arcs[i]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#888", weight=1, opacity=.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route", show=True)
    pts = [[trip.nodes[trip.arcs[r, 0], 1], trip.nodes[trip.arcs[r, 0], 0]] for r in trip.route]
    pts.append([trip.nodes[trip.arcs[trip.route[-1], 1], 1], trip.nodes[trip.arcs[trip.route[-1], 1], 0]])
    folium.PolyLine(pts, color="green", weight=5, opacity=.8, tooltip="reference route").add_to(ref)
    ref.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    path = MAPS / f"eda_trip_{trip_id}.html"
    m.save(str(path))
    return path


def quality_notes(inv, net):
    tr = inv[inv.partition == "train"]
    summ = pd.read_csv(P1RES / "trip_summary_train.csv", dtype={"trip_id": str})
    bad = summ[summ.unreliable_label_rate > 0.2].trip_id.tolist()
    lines = [
        "# Data-quality record (generated by eda/run_eda.py)",
        "",
        f"- Trips: train={len(tr)}, student_test={(inv.partition == 'student_test').sum()}. "
        f"Evaluator-test trips are not in this package.",
        f"- GPS observations: train={tr.n_obs.sum():,}; NaN values: {inv.n_nan.sum()}; "
        f"non-increasing timestamps: {inv.non_increasing_t.sum()}.",
        "- Missing files: " + ", ".join(
            f"{r.partition}/{r.trip_id} .{e}" for _, r in inv.iterrows()
            for e in ("track", "nodes", "arcs", "route") if not r[f"has_{e}"]) + " -> trip cannot be map-matched; excluded.",
        f"- Sampling: {(tr.pct_dt_1s * tr.n_obs).sum() / tr.n_obs.sum():.1%} of steps are 1 s. "
        f"Trips with a gap > 10 min: {', '.join(tr[tr.max_gap_s > 600].trip_id)}.",
        "- Each trip ships its own local road network; trips are spread worldwide, so distances use a "
        "per-trip local equirectangular projection in metres (error < 1 % at trip scale).",
        "- The .route file is an ordered arc list, not one label per GPS row. Per-point labels come from a "
        "monotone DP alignment of GPS points to route arcs (features/labels.py).",
        f"- Points more than 100 m from their aligned reference arc are flagged unreliable "
        f"({(~pd.read_parquet(ARTIFACTS_DIR / 'obs_train.parquet').label_ok).mean():.1%} of train points); "
        f"they are excluded from training and from accuracy metrics. Trips with > 20 % unreliable: {', '.join(bad)}.",
        "- Example: trip 00000054's reference route ends after ~27 min while GPS continues for ~4.5 h "
        "away from the network, so the reference route does not cover the whole trajectory.",
        f"- Reference routes are not always connected: {int((net.route_continuity < 1).sum())} trips have "
        "consecutive route arcs that do not share a node -> treat the reference as approximate.",
        "- Two-way streets appear as two arcs with identical geometry and opposite direction; only the "
        "travel direction (heading feature) separates them.",
    ]
    (OUT / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, net = inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)
    net.to_csv(OUT / "network_connectivity.csv", index=False)
    plots(inv, net)
    print("map:", folium_map())
    quality_notes(inv, net)
    print(inv.groupby("partition")[["n_obs", "duration_s", "path_km"]].describe().T.round(1).to_string())
    print(net.drop(columns=[c for c in ("note",) if c in net]).describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
