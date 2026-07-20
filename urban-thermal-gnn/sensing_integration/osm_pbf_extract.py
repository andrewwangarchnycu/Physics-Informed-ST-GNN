"""
sensing_integration/osm_pbf_extract.py
════════════════════════════════════════════════════════════════
Offline OSM extraction from a local Geofabrik .osm.pbf using pyosmium.

Motivation: screening hundreds of candidate sites via the public Overpass
API is rate-limited and suffers multi-minute per-query latency. A single
regional .osm.pbf (downloaded once) lets us extract every building and
land-cover polygon in the study region in one pass, then serve any number
of sites from memory in milliseconds -- no network, no rate limits, fully
reproducible.

Scope / honesty:
  * Buildings and landuse/natural/leisure features stored as CLOSED-WAY
    polygons (the overwhelming majority in Taiwan OSM). Multipolygon
    RELATIONS (a small minority: a few large parks / building complexes)
    are not assembled here -- documented rather than silently dropped.
  * Building height is taken from a real OSM tag (height / building:height
    / building:levels) when present; otherwise flagged height_from_tag=False
    and left for a floor-count fallback downstream.

Usage:
    region = RegionOSM("data/osm/taiwan-latest.osm.pbf",
                       bbox=(23.95, 120.55, 25.10, 121.30))  # S,W,N,E
    region.load()
    b = region.buildings_local(lat, lon, radius_m=300)   # site-local metres
    m = region.materials_local(lat, lon, radius_m=300)
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import osmium
    from osmium.geom import WKBFactory
    _OSMIUM = True
except ImportError:
    _OSMIUM = False

from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from shapely.validation import make_valid

_M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat: float) -> float:
    return _M_PER_DEG_LAT * math.cos(math.radians(lat))


# Reuse the same landuse/natural/leisure -> material mapping as the live
# loader so offline and online paths agree.
from osm_loader import (OSM_LANDUSE_TO_MATERIAL, OSM_NATURAL_TO_MATERIAL,
                        OSM_LEISURE_TO_MATERIAL, _parse_osm_number)


def _material_of(tags) -> str | None:
    return (OSM_LANDUSE_TO_MATERIAL.get(tags.get("landuse", ""))
            or OSM_NATURAL_TO_MATERIAL.get(tags.get("natural", ""))
            or OSM_LEISURE_TO_MATERIAL.get(tags.get("leisure", "")))


class _Collector(osmium.SimpleHandler):
    """Collect building + material closed-way polygons within a bbox."""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        super().__init__()
        self.s, self.w, self.n, self.e = bbox
        self.wkbfab = WKBFactory()
        # buildings: list of (poly_lonlat, height, from_tag)
        self.buildings: List[Tuple[Polygon, float, bool]] = []
        # materials: list of (poly_lonlat, material_str)
        self.materials: List[Tuple[Polygon, str]] = []

    def _in_bbox(self, poly: Polygon) -> bool:
        cx, cy = poly.centroid.x, poly.centroid.y  # lon, lat
        return (self.w <= cx <= self.e) and (self.s <= cy <= self.n)

    def way(self, w):
        tags = w.tags
        is_bldg = "building" in tags
        material = None if is_bldg else _material_of(tags)
        if not is_bldg and material is None:
            return
        if not w.is_closed() or len(w.nodes) < 4:
            return
        # Build polygon directly from node locations (fast, no relation assembly)
        try:
            coords = [(n.location.lon, n.location.lat) for n in w.nodes
                      if n.location.valid()]
        except Exception:
            return
        if len(coords) < 4:
            return
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
                return
        except Exception:
            return
        if not self._in_bbox(poly):
            return

        if is_bldg:
            h_tag = tags.get("height", tags.get("building:height"))
            fl_tag = tags.get("building:levels", tags.get("levels"))
            from_tag = bool(h_tag) or bool(fl_tag)
            h = _parse_osm_number(h_tag, 0.0)
            if h <= 0:
                fl = _parse_osm_number(fl_tag, 0.0)
                h = fl * 3.6 if fl > 0 else 3 * 3.6
            self.buildings.append((poly, float(h), from_tag))
        else:
            self.materials.append((poly, material))


class RegionOSM:
    """Load a regional pbf once; serve site-local geometry from memory."""

    def __init__(self, pbf_path: str,
                 bbox: Tuple[float, float, float, float]):
        self.pbf_path = Path(pbf_path)
        self.bbox = bbox
        self._loaded = False
        self._bpolys: List[Polygon] = []
        self._bh: List[float] = []
        self._bft: List[bool] = []
        self._btree: STRtree | None = None
        self._mpolys: List[Polygon] = []
        self._mmat: List[str] = []
        self._mtree: STRtree | None = None

    def _cache_path(self) -> Path:
        s, w, n, e = self.bbox
        tag = f"{s:.2f}_{w:.2f}_{n:.2f}_{e:.2f}".replace(".", "p").replace("-", "m")
        return self.pbf_path.with_name(f".region_cache_{tag}.pkl")

    def load(self, use_cache: bool = True) -> "RegionOSM":
        if not _OSMIUM:
            raise ImportError("pyosmium not installed (pip install osmium)")
        if not self.pbf_path.exists():
            raise FileNotFoundError(self.pbf_path)

        cache = self._cache_path()
        if use_cache and cache.exists() and cache.stat().st_mtime >= self.pbf_path.stat().st_mtime:
            import pickle
            try:
                with open(cache, "rb") as f:
                    d = pickle.load(f)
                self._bpolys, self._bh, self._bft = d["bpolys"], d["bh"], d["bft"]
                self._mpolys, self._mmat = d["mpolys"], d["mmat"]
                self._btree = STRtree(self._bpolys) if self._bpolys else None
                self._mtree = STRtree(self._mpolys) if self._mpolys else None
                self._loaded = True
                print(f"  [RegionOSM] loaded cache: {len(self._bpolys)} buildings, "
                      f"{len(self._mpolys)} material polygons", flush=True)
                return self
            except Exception:
                pass

        print(f"  [RegionOSM] reading {self.pbf_path.name} ...", flush=True)
        col = _Collector(self.bbox)
        # locations=True populates node coordinates so closed ways -> polygons.
        col.apply_file(str(self.pbf_path), locations=True, idx="flex_mem")
        self._bpolys = [b[0] for b in col.buildings]
        self._bh     = [b[1] for b in col.buildings]
        self._bft    = [b[2] for b in col.buildings]
        self._mpolys = [m[0] for m in col.materials]
        self._mmat   = [m[1] for m in col.materials]
        self._btree = STRtree(self._bpolys) if self._bpolys else None
        self._mtree = STRtree(self._mpolys) if self._mpolys else None
        self._loaded = True
        n_ht = sum(self._bft)
        print(f"  [RegionOSM] {len(self._bpolys)} buildings "
              f"({n_ht} height-tagged), {len(self._mpolys)} material polygons "
              f"in region bbox", flush=True)
        if use_cache:
            import pickle
            try:
                with open(cache, "wb") as f:
                    pickle.dump({"bpolys": self._bpolys, "bh": self._bh,
                                 "bft": self._bft, "mpolys": self._mpolys,
                                 "mmat": self._mmat}, f)
                print(f"  [RegionOSM] cached region -> {cache.name}", flush=True)
            except Exception:
                pass
        return self

    # ── coord conversion: lon/lat -> local metres centred on (lat,lon) ──
    @staticmethod
    def _to_local(poly: Polygon, lat0: float, lon0: float) -> List[List[float]]:
        mlon = _m_per_deg_lon(lat0)
        ext = poly.exterior.coords if poly.geom_type == "Polygon" else poly.geoms[0].exterior.coords
        return [[(x - lon0) * mlon, (y - lat0) * _M_PER_DEG_LAT] for x, y in ext]

    def _query_box(self, lat: float, lon: float, radius_m: float):
        dlat = radius_m / _M_PER_DEG_LAT
        dlon = radius_m / _m_per_deg_lon(lat)
        return Point(lon, lat).buffer(max(dlat, dlon))  # rough lon/lat disc

    def buildings_local(self, lat: float, lon: float,
                        radius_m: float = 300.0,
                        min_area_m2: float = 20.0) -> List[Dict]:
        if not self._loaded or self._btree is None:
            return []
        qb = self._query_box(lat, lon, radius_m)
        out = []
        for idx in self._btree.query(qb):
            poly = self._bpolys[idx]
            if poly.centroid.distance(Point(lon, lat)) > qb.boundary.distance(Point(lon, lat)) + 1:
                pass
            local = self._to_local(poly, lat, lon)
            if len(local) < 4:
                continue
            lp = Polygon(local)
            if not lp.is_valid:
                lp = make_valid(lp)
            if lp.is_empty or lp.area < min_area_m2:
                continue
            # distance filter in metres from centre
            if lp.centroid.distance(Point(0, 0)) > radius_m:
                continue
            h = self._bh[idx]
            out.append({
                "footprint": [list(p) for p in local],
                "height": h,
                "floor_count": max(1, int(round(h / 3.6))),
                "osm_id": None,
                "building_type": "yes",
                "height_from_tag": self._bft[idx],
            })
        return out

    def materials_local(self, lat: float, lon: float,
                        radius_m: float = 300.0) -> Dict[str, List]:
        if not self._loaded or self._mtree is None:
            return {}
        qb = self._query_box(lat, lon, radius_m)
        zones: Dict[str, List] = {}
        for idx in self._mtree.query(qb):
            poly = self._mpolys[idx]
            local = self._to_local(poly, lat, lon)
            if len(local) < 4:
                continue
            lp = Polygon(local)
            if lp.is_empty:
                continue
            if lp.centroid.distance(Point(0, 0)) > radius_m + 50:
                continue
            zones.setdefault(self._mmat[idx], []).append([list(p) for p in local])
        return zones
