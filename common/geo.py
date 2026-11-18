"""Geometry helpers.

Coordinates arrive as (lon, lat) degrees. Degrees are not metres (1 deg of longitude
shrinks with cos(latitude)), so every trip is projected to a local flat x/y plane in
metres using an equirectangular projection centred on the trip's mean GPS position:

    x = R * cos(lat0) * (lon - lon0)      y = R * (lat - lat0)      (radians)

Each trip covers at most a few tens of km, where this projection's distance error is
well under 1 %, far below GPS noise (~5-20 m). It is cheap, exact enough and needs no
UTM-zone handling, which matters because trips are spread all over the world.
"""
import numpy as np

R_EARTH = 6_371_008.8  # mean Earth radius, metres


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = R_EARTH * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = R_EARTH * np.pi / 180.0

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment(s) AB (broadcasting) and the position t in [0,1]
    of the closest point along AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a displacement in local x/y."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
