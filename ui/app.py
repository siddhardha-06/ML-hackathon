"""PathSnap evaluator UI - runs on Laptop 2 (P4).

Start (from the project root on Laptop 2):
    pip install -r ui/requirements.txt
    streamlit run ui/app.py
then enter Laptop 1's API address (e.g. http://192.168.1.10:8000) in the sidebar.
The UI holds no data or model: it lists trips from the API, sends the selected trip id,
and draws whatever the API returns.
"""
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.title("PathSnap: Map-Matching & Route Reconstruction")

api = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected: {health['model']} + HMM/Viterbi")
    trips = requests.get(f"{api}/trips", timeout=10).json()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach the API at {api}: {e}")
    st.stop()

labels = {t["trip_id"]: f"Trip {t['trip_id']}  ({t['n_obs']:,} GPS points)"
          + ("" if t["has_network"] else "  - no road network") for t in trips}
trip_id = st.sidebar.radio("Evaluation trips", list(labels), format_func=labels.get)
run = st.sidebar.button("Reconstruct route", type="primary", use_container_width=True)

if run:
    with st.spinner(f"API is processing trip {trip_id} ..."):
        r = requests.get(f"{api}/predict/{trip_id}", timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    hm, cm = res["metrics"]["hmm"], res["metrics"]["classifier_only"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    st.subheader(f"Trip {res['trip_id']}")
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", pct(hm.get("top1_accuracy")),
                None if hm.get("top1_accuracy") is None or cm.get("top1_accuracy") is None
                else f"{100 * (hm['top1_accuracy'] - cm['top1_accuracy']):+.1f} pts vs classifier only")
    c[1].metric("Route continuity", pct(hm.get("continuity")))
    c[2].metric("Route F1", pct(hm.get("route_f1")))
    c[3].metric("GPS points", f"{res['n_obs']:,}")
    c[4].metric("API time", f"{res['timings_s']['total']:.1f} s")

    gps, route = res["gps"], res["pred_route"]
    m = folium.Map(location=gps[len(gps) // 2], zoom_start=14)
    g = folium.FeatureGroup("Raw GPS")
    for p in gps[:: max(1, len(gps) // 1500)]:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, opacity=.7).add_to(g)
    g.add_to(m)
    if route:
        folium.PolyLine(route, color="#d62728", weight=5, opacity=.85, tooltip="predicted route").add_to(
            folium.FeatureGroup("Predicted route").add_to(m))
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st.caption("Blue dots: raw GPS observations | Red line: route reconstructed by classifier + HMM/Viterbi")
    components.html(m.get_root().render(), height=560)

    st.markdown("#### Classifier only vs classifier + HMM")
    rows = [("Top-1 accuracy", "top1_accuracy"), ("Route continuity (per-point)", "continuity"),
            ("Route continuity (reconstructed)", "route_continuity_filled"),
            ("Route precision", "route_precision"), ("Route recall", "route_recall"), ("Route F1", "route_f1")]
    st.table({"metric": [r[0] for r in rows], "classifier only": [pct(cm.get(k)) for _, k in rows],
              "classifier + HMM": [pct(hm.get(k)) for _, k in rows]})
    st.caption(f"Timings (s): {res['timings_s']} | points without candidates: {res['n_obs_without_candidates']}")
