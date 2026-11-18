"""Folium maps: raw GPS, nearby road network, reference route, predicted route(s) (P3)."""
import folium
import numpy as np
from branca.element import MacroElement, Template

from hmm.viterbi import RoadGraph, arcs_to_polyline, fill_route

LAYER_STYLE = {
    "reference": dict(color="#2ca02c", weight=7, opacity=.55),
    "classifier only": dict(color="#ff7f0e", weight=3, opacity=.9, dash_array="6 6"),
    "HMM + Viterbi": dict(color="#d62728", weight=4, opacity=.9),
}


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, c in items)
    html = ('{% macro html(this, kwargs) %}<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
            'background:white;padding:8px 12px;border:1px solid #999;border-radius:6px;font:12px sans-serif">'
            f'<b>Legend</b>{rows}</div>{{% endmacro %}}')
    el = MacroElement()
    el._template = Template(html)
    return el


def route_map(trip, predictions: dict, show_reference=True, network_margin_m=120, max_network_arcs=5000):
    """predictions: layer name -> array of predicted arc per GPS observation."""
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles="OpenStreetMap")

    # GPS first so the route lines are drawn on top of the dots
    gps = folium.FeatureGroup("Raw GPS")
    step = max(1, len(lat) // 1500)
    for i in range(0, len(lat), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color="#1f77b4", fill=True,
                            opacity=.7, tooltip=f"obs {i}, t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    if trip.has_network:
        A, B = trip.arc_segments()
        lo, hi = trip.gps_xy.min(0) - network_margin_m, trip.gps_xy.max(0) + network_margin_m
        keep = np.where(((A >= lo) & (A <= hi)).all(1))[0][:max_network_arcs]
        net = folium.FeatureGroup("Road network", show=False)
        for i in keep:
            f, t = trip.arcs[i]
            folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                            color="#777", weight=1, opacity=.5).add_to(net)
        net.add_to(m)

    legend = [("raw GPS", "#1f77b4")]
    if show_reference and trip.route is not None:
        pts, _ = arcs_to_polyline(trip, trip.route)
        fg = folium.FeatureGroup("Reference route")
        folium.PolyLine(pts, tooltip="reference route", **LAYER_STYLE["reference"]).add_to(fg)
        fg.add_to(m)
        legend.append(("reference route", LAYER_STYLE["reference"]["color"]))

    graph = RoadGraph(trip) if predictions else None
    for name, pred in predictions.items():
        pts, _ = arcs_to_polyline(trip, fill_route(graph, pred))
        style = LAYER_STYLE.get(name, dict(color="#9467bd", weight=4))
        fg = folium.FeatureGroup(f"Predicted: {name}")
        if pts:
            folium.PolyLine(pts, tooltip=name, **style).add_to(fg)
        fg.add_to(m)
        legend.append((f"predicted - {name}", style["color"]))

    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(_legend(legend))
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m
