"""
sensing_integration/osm_loader.py
════════════════════════════════════════════════════════════════
OpenStreetMap data integration via Overpass API.

Provides:
  - Real building footprints (replaces procedural generation)
  - OSM landuse → surface material zone mapping
  - Road centrelines for street-canyon wind modelling
  - H/W (height-to-width) canyon ratio computation

Install optional dep:
    pip install overpy

Usage:
    loader = OSMLoader(site_lat=24.80, site_lon=120.97, radius_m=100)
    ok = loader.fetch()
    buildings = loader.get_buildings_local()
    mat_zones = loader.get_material_zones()
    roads     = loader.get_road_segments_local()
    hw        = compute_canyon_hw_ratios(sensor_pts, roads, buildings)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import overpy
    _OVERPY = True
except ImportError:
    _OVERPY = False

# ────────────────────────────────────────────────────────────────
# OSM tag → surface material mapping
# ────────────────────────────────────────────────────────────────

OSM_LANDUSE_TO_MATERIAL: Dict[str, str] = {
    # Green
    "grass":              "grass",
    "meadow":             "grass",
    "park":               "grass",
    "recreation_ground":  "grass",
    "village_green":      "grass",
    "cemetery":           "grass",
    "forest":             "grass",
    "orchard":            "grass",
    "allotments":         "grass",
    # Bare soil
    "farmland":           "bare_soil",
    "farmyard":           "bare_soil",
    "construction":       "bare_soil",
    "quarry":             "bare_soil",
    # Paved / urban
    "residential":        "concrete",
    "commercial":         "concrete",
    "retail":             "concrete",
    "institutional":      "concrete",
    "industrial":         "asphalt",
    # Water
    "basin":              "water",
    "reservoir":          "water",
    "salt_pond":          "water",
}

OSM_LEISURE_TO_MATERIAL: Dict[str, str] = {
    "park":            "grass",
    "garden":          "grass",
    "pitch":           "grass",
    "playground":      "grass",
    "golf_course":     "grass",
    "nature_reserve":  "grass",
    "swimming_pool":   "water",
}

OSM_NATURAL_TO_MATERIAL: Dict[str, str] = {
    "water":    "water",
    "wetland":  "water",
    "wood":     "grass",
    "grass":    "grass",
    "scrub":    "grass",
    "heath":    "bare_soil",
    "sand":     "bare_soil",
    "beach":    "bare_soil",
    "bare_rock":"bare_soil",
}

# Road widths by OSM highway tag [metres]
HIGHWAY_WIDTHS: Dict[str, float] = {
    "motorway":      28.0,
    "trunk":         20.0,
    "primary":       14.0,
    "secondary":     10.0,
    "tertiary":       8.0,
    "residential":    6.0,
    "unclassified":   6.0,
    "living_street":  5.0,
    "pedestrian":     4.0,
    "footway":        2.0,
    "cycleway":       2.0,
    "service":        4.0,
    "track":          4.0,
}

# Highway types that can form street canyons
CANYON_HIGHWAY_TYPES = frozenset({
    "primary", "secondary", "tertiary", "residential",
    "unclassified", "living_street", "pedestrian", "service",
})

# ────────────────────────────────────────────────────────────────
# Coordinate helpers
# ────────────────────────────────────────────────────────────────

_M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat_deg: float) -> float:
    return _M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _latlon_to_local(lat: float, lon: float,
                     origin_lat: float, origin_lon: float
                     ) -> Tuple[float, float]:
    """WGS84 → local XY in metres (origin = SW corner of bounding box).

    overpy returns node lat/lon as ``decimal.Decimal``; cast to float first
    so the mixed Decimal−float arithmetic below does not raise TypeError
    (previously swallowed by callers' bare except, silently dropping every
    building/landuse polygon and forcing a procedural fallback).
    """
    lat = float(lat); lon = float(lon)
    origin_lat = float(origin_lat); origin_lon = float(origin_lon)
    x = (lon - origin_lon) * _m_per_deg_lon(origin_lat)
    y = (lat - origin_lat) * _M_PER_DEG_LAT
    return x, y


def _parse_osm_number(tag, default: float = 0.0) -> float:
    """Robustly parse an OSM numeric tag to float.

    Handles the messy real-world variants: units ("18 m", "12m"),
    semicolon multi-values from merged ways ("11;10;12"), ranges
    ("3-5"), and commas. Returns ``default`` when nothing parses.
    """
    if tag is None:
        return default
    s = str(tag).strip().lower().replace("m", "").replace(",", ".")
    for sep in (";", "-", "~"):
        if sep in s:
            parts = [p for p in s.split(sep) if p.strip()]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p.strip()))
                except ValueError:
                    continue
            if vals:
                return sum(vals) / len(vals)   # mean of multi-value/range
            return default
    try:
        return float(s.strip())
    except ValueError:
        return default


def _shoelace_area(pts: List[Tuple[float, float]]) -> float:
    """Signed area of a 2-D polygon via shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return 0.5 * sum(
        xs[i] * ys[(i+1) % n] - xs[(i+1) % n] * ys[i]
        for i in range(n)
    )


# ────────────────────────────────────────────────────────────────
# OSMLoader
# ────────────────────────────────────────────────────────────────

class OSMLoader:
    """
    Download and parse OSM data within a bounding box via Overpass API.

    Parameters
    ----------
    site_lat, site_lon : WGS84 reference point — used as the ORIGIN of the
                         site-polygon coordinate system (i.e. site SW corner
                         when lat/lon from config is the SW corner).
    radius_m           : half-width of bounding box (metres)
    cache_result       : reuse fetched data within the session
    """

    def __init__(self,
                 site_lat:     float = 24.80,
                 site_lon:     float = 120.97,
                 radius_m:     float = 100.0,
                 cache_result: bool  = True):
        self.site_lat  = site_lat
        self.site_lon  = site_lon
        self.radius_m  = radius_m
        self._cache    = cache_result
        self._result   = None
        self._fetched  = False

        dlat = radius_m / _M_PER_DEG_LAT
        dlon = radius_m / _m_per_deg_lon(site_lat)
        self.bbox = (
            site_lat - dlat,   # south
            site_lon - dlon,   # west
            site_lat + dlat,   # north
            site_lon + dlon,   # east
        )
        # OSM internal local-coord origin = SW corner of bbox
        self.origin_lat = self.bbox[0]
        self.origin_lon = self.bbox[1]

        # Offset from OSM-local (0,0) to site-polygon (0,0).
        # site_lat/lon is the site-polygon origin (SW corner) in WGS84.
        # In OSM local coords that point is at (offset_x, offset_y).
        # Subtracting the offset converts OSM local → site-polygon coords.
        self._site_offset_x = (site_lon - self.origin_lon) * _m_per_deg_lon(self.origin_lat)
        self._site_offset_y = (site_lat - self.origin_lat) * _M_PER_DEG_LAT

    # ── Public API ──────────────────────────────────────────────

    def fetch(self, timeout: int = 30) -> bool:
        """
        Download OSM data from Overpass API.

        Returns True on success, False if network unavailable or overpy
        not installed (procedural fallback will be used automatically).
        """
        if self._fetched and self._cache:
            return self._result is not None

        if not _OVERPY:
            warnings.warn(
                "[OSMLoader] overpy not installed. "
                "Install with: pip install overpy\n"
                "Procedural geometry generation will be used instead."
            )
            self._fetched = True
            return False

        s, w, n, e = self.bbox
        query = f"""
        [out:json][timeout:{timeout}];
        (
          way["building"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["landuse"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["natural"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["highway"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          relation["building"]["type"="multipolygon"]
                  ({s:.6f},{w:.6f},{n:.6f},{e:.6f});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            api = overpy.Overpass()
            self._result  = api.query(query)
            self._fetched = True
            n_ways = len(self._result.ways)
            print(f"  [OSMLoader] {n_ways} ways fetched "
                  f"(S={s:.4f} W={w:.4f} N={n:.4f} E={e:.4f})")
            return True
        except Exception as exc:
            warnings.warn(
                f"[OSMLoader] Overpass query failed: {exc}\n"
                "Procedural geometry generation will be used."
            )
            self._result  = None
            self._fetched = True
            return False

    @property
    def has_data(self) -> bool:
        return self._result is not None and len(self._result.ways) > 0

    # ── Building extraction ─────────────────────────────────────

    def _to_site(self, local_pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Translate OSM-local coords → site-polygon coords."""
        dx, dy = self._site_offset_x, self._site_offset_y
        return [(x - dx, y - dy) for x, y in local_pts]

    def get_buildings_local(self, min_area_m2: float = 20.0) -> List[Dict]:
        """
        Return OSM buildings as site-local footprints.

        Returns
        -------
        list of {
            "footprint":     [[x, y], ...],  # local metres
            "height":        float,           # metres
            "floor_count":   int,
            "osm_id":        int,
            "building_type": str,
        }
        """
        if not self.has_data:
            return []

        buildings = []
        for way in self._result.ways:
            if "building" not in way.tags:
                continue
            try:
                osm_pts = [
                    _latlon_to_local(nd.lat, nd.lon,
                                     self.origin_lat, self.origin_lon)
                    for nd in way.nodes
                ]
                local_pts = self._to_site(osm_pts)
            except Exception:
                continue

            if len(local_pts) < 3:
                continue

            area = abs(_shoelace_area(local_pts))
            if area < min_area_m2:
                continue

            tags   = way.tags
            fl_tag = tags.get("building:levels", tags.get("levels", None))
            h_tag  = tags.get("height", tags.get("building:height", None))
            floors = _parse_osm_number(fl_tag, default=3.0)
            floors = int(round(floors)) if floors > 0 else 3
            h_val  = _parse_osm_number(h_tag, default=0.0)
            height = h_val if h_val > 0 else floors * 3.6
            btype  = tags.get("building", "yes")

            buildings.append({
                "footprint":       [list(pt) for pt in local_pts],
                "height":          height,
                "floor_count":     floors,
                "osm_id":          way.id,
                "building_type":   btype,
                # True only when height came from a real OSM tag (height /
                # building:height / building:levels), not the 3-floor guess.
                "height_from_tag": bool(h_tag) or bool(fl_tag),
            })

        print(f"  [OSMLoader] {len(buildings)} buildings extracted "
              f"(min area={min_area_m2} m²)")
        return buildings

    # ── Material zones ──────────────────────────────────────────

    def get_material_zones(self) -> Dict[str, List[List[List[float]]]]:
        """
        Return OSM land-use polygons grouped by surface material.

        Returns
        -------
        {
            "grass":   [[[x,y], ...], ...],
            "asphalt": [...],
            ...
        }
        """
        if not self.has_data:
            return {}

        zones: Dict[str, List] = {}
        for way in self._result.ways:
            tags = way.tags
            material = (
                OSM_LANDUSE_TO_MATERIAL.get(tags.get("landuse", ""))
                or OSM_NATURAL_TO_MATERIAL.get(tags.get("natural", ""))
                or OSM_LEISURE_TO_MATERIAL.get(tags.get("leisure", ""))
            )
            if material is None:
                hw = tags.get("highway", "")
                if hw in ("footway", "pedestrian", "path", "steps"):
                    material = "concrete"
                elif hw and hw not in ("motorway", "trunk", ""):
                    material = "asphalt"
            if material is None:
                continue

            try:
                osm_pts  = [
                    _latlon_to_local(nd.lat, nd.lon,
                                     self.origin_lat, self.origin_lon)
                    for nd in way.nodes
                ]
                local_pts = self._to_site(osm_pts)
            except Exception:
                continue

            if len(local_pts) < 3:
                continue

            zones.setdefault(material, []).append(
                [list(pt) for pt in local_pts]
            )

        n_zones = sum(len(v) for v in zones.values())
        print(f"  [OSMLoader] {n_zones} material zones extracted "
              f"({list(zones.keys())})")
        return zones

    # ── Road segments ───────────────────────────────────────────

    def get_road_segments_local(self) -> List[Dict]:
        """
        Return road centrelines with width estimates.

        Returns
        -------
        list of {
            "coords":  [[x1,y1],[x2,y2], ...],  # local metres
            "highway": str,
            "width_m": float,
            "name":    str,
        }
        """
        if not self.has_data:
            return []

        roads = []
        for way in self._result.ways:
            tags    = way.tags
            hw_type = tags.get("highway", "")
            if not hw_type:
                continue

            try:
                osm_pts  = [
                    _latlon_to_local(nd.lat, nd.lon,
                                     self.origin_lat, self.origin_lon)
                    for nd in way.nodes
                ]
                local_pts = self._to_site(osm_pts)
            except Exception:
                continue

            if len(local_pts) < 2:
                continue

            # Width: prefer explicit tag, else lookup table
            w_tag = tags.get("width", None)
            if w_tag:
                try:
                    w_m = float(str(w_tag).replace("m", "").strip())
                except ValueError:
                    w_m = HIGHWAY_WIDTHS.get(hw_type, 6.0)
            else:
                lanes_tag = tags.get("lanes", None)
                if lanes_tag:
                    try:
                        w_m = float(lanes_tag) * 3.5
                    except ValueError:
                        w_m = HIGHWAY_WIDTHS.get(hw_type, 6.0)
                else:
                    w_m = HIGHWAY_WIDTHS.get(hw_type, 6.0)

            roads.append({
                "coords":  [list(pt) for pt in local_pts],
                "highway": hw_type,
                "width_m": w_m,
                "name":    tags.get("name", ""),
            })

        print(f"  [OSMLoader] {len(roads)} road segments extracted")
        return roads


# ────────────────────────────────────────────────────────────────
# Street canyon H/W computation
# ────────────────────────────────────────────────────────────────

def compute_canyon_hw_ratios(sensor_pts:    np.ndarray,
                              roads:         List[Dict],
                              buildings:     List[Dict],
                              canyon_radius: float = 30.0) -> np.ndarray:
    """
    Compute H/W (height-to-width) canyon ratio for each sensor point.

    For sensors near a road segment the flanking buildings are found and:
        H/W = mean_flanking_height / road_width

    H/W ≈ 0   → open field (no canyon shielding)
    H/W ≈ 1   → moderate canyon
    H/W ≥ 2   → deep urban canyon (Oke 1988)

    Parameters
    ----------
    sensor_pts    : (N, 2) site-local coordinates
    roads         : output of OSMLoader.get_road_segments_local()
    buildings     : list of building dicts with "footprint" and "height"
    canyon_radius : search radius for flanking buildings [m]

    Returns
    -------
    hw_ratios : (N,) float32, clipped to [0, 5]
    """
    N         = len(sensor_pts)
    hw_ratios = np.zeros(N, dtype=np.float32)
    if not roads or not buildings:
        return hw_ratios

    # Pre-compute building centroids and heights
    bldg_data = []
    for b in buildings:
        fp = b.get("footprint", [])
        if len(fp) < 3:
            continue
        xs = [p[0] for p in fp]
        ys = [p[1] for p in fp]
        bldg_data.append((
            sum(xs) / len(xs),
            sum(ys) / len(ys),
            float(b.get("height", 10.0)),
        ))

    for road in roads:
        coords  = road["coords"]
        w_m     = max(float(road.get("width_m", 6.0)), 1.0)
        hw_type = road.get("highway", "")
        if hw_type not in CANYON_HIGHWAY_TYPES and hw_type:
            continue   # skip motorways / trunks

        for seg_i in range(len(coords) - 1):
            x1, y1 = coords[seg_i]
            x2, y2 = coords[seg_i + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len < 0.5:
                continue

            # Unit vectors along and normal to segment
            ux = (x2 - x1) / seg_len
            uy = (y2 - y1) / seg_len
            nx, ny = -uy, ux   # left normal

            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5  # midpoint

            # Find flanking buildings
            flanking_h = []
            for bx, by, bh in bldg_data:
                dx, dy = bx - mx, by - my
                dist_perp  = abs(dx * nx + dy * ny)
                dist_along = abs(dx * ux + dy * uy)
                if (dist_perp < canyon_radius and
                        dist_along < seg_len * 0.5 + canyon_radius):
                    flanking_h.append(bh)

            if not flanking_h:
                continue

            h_mean = float(np.mean(flanking_h))
            hw     = h_mean / w_m

            # Assign to nearby sensor points (weighted by distance)
            for i, pt in enumerate(sensor_pts):
                dx, dy = pt[0] - mx, pt[1] - my
                dist = math.hypot(dx, dy)
                if dist < canyon_radius:
                    weight = max(0.0, 1.0 - dist / canyon_radius)
                    hw_ratios[i] = max(hw_ratios[i], float(hw * weight))

    return np.clip(hw_ratios, 0.0, 5.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Canyon wind speed correction (Oke 1988 / Nicholson 1975)
# ────────────────────────────────────────────────────────────────

def canyon_wind_reduction(hw_ratio: np.ndarray) -> np.ndarray:
    """
    Reduction factor for wind speed inside a street canyon.

    Empirical relationship (Oke 1988):
      Isolated roughness flow:   H/W < 0.3   → factor ≈ 1.0
      Wake interference flow:    0.3–0.7     → factor ≈ 0.6–0.9
      Skimming flow:             H/W > 0.7   → factor ≈ 0.3–0.6

    Parameters
    ----------
    hw_ratio : (N,) H/W ratios

    Returns
    -------
    (N,) reduction factors in [0.15, 1.0]
    """
    # Piecewise linear fit to canyon wind experiments
    factor = np.where(
        hw_ratio < 0.3,
        1.0,
        np.where(
            hw_ratio < 1.5,
            1.0 - 0.5 * (hw_ratio - 0.3) / 1.2,
            0.5 * np.exp(-0.4 * (hw_ratio - 1.5)) + 0.15,
        ),
    )
    return np.clip(factor, 0.15, 1.0).astype(np.float32)
