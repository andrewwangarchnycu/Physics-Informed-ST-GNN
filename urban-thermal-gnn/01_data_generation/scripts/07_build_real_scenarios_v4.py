"""
01_data_generation/scripts/07_build_real_scenarios_v4.py
════════════════════════════════════════════════════════════════
V4 real-geometry scenario builder.

Consumes the sites chosen by 06_select_real_sites_v4.py and turns each
one into a simulation scenario built from GENUINE geospatial data --
replacing the procedural Monte-Carlo geometry used up to V3:

  * Buildings  : real OSM footprints + heights (parsed from the per-zone
                 OSM cache written during selection -- no re-querying).
  * Vegetation : real ETH GlobalCanopyHeight (10 m) sampled on a grid;
                 canopy >= TREE_MIN_H m in open space becomes a tree with
                 its real measured height (source = "eth_canopy").
  * Surface    : real land-cover classes from OSM landuse/natural/leisure
                 polygons + building footprints, rasterised to a per-metre
                 land_cover_map feeding physically-meaningful albedo.
  * Month tag  : sites are round-robin assigned to June/July/August so the
                 300-scene set spans the real summer the CWB + IoT data
                 covers (temporal/seasonal diversity, not one typical day).

The output scenario dict matches the schema consumed by the simulation
runner (site_polygon, buildings, trees, land_cover_map, ...), so the
existing physics core can be reused, plus V4-only fields (site_lat/lon,
assigned_month, nearest IoT id) for the real-CWB / real-IoT step.

Usage:
    python 07_build_real_scenarios_v4.py \
        --sites   ../outputs/real_sites_v4/selected_real_sites.json \
        --cache   ../outputs/real_sites_v4/osm_cache \
        --out     ../outputs/real_simulations_v4
"""
from __future__ import annotations

import sys
import json
import math
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "sensing_integration"))

from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from shapely.validation import make_valid

from osm_pbf_extract import RegionOSM
from canopy_loader import CanopyHeightLoader

# ── config ──────────────────────────────────────────────────────
SITE_SIZE_M     = 80.0     # scenario footprint (matches selection)
CONTEXT_R_M     = 120.0    # keep buildings within this radius for shadowing
TREE_GRID_M     = 10.0     # canopy sampling grid (native ETH 10 m resolution)
TREE_MIN_H      = 3.0      # min canopy height [m] to count as a tree
TREE_MAX        = 80       # cap trees/scene (tallest kept) -> realistic + fast
ZONE_RADIUS_M   = 300.0    # must match selection (cache coord origin)
CANOPY_TIF      = _ROOT / "data" / "canopy" / "ETH_GlobalCanopyHeight_10m_2020_N24E120_Map.tif"

# Material -> UrbanGraph land-cover code (albedo lookup in the runner:
# {0/1: 0.30 built, 2: 0.40 soil, 3: 0.00 water, 4: 0.20 grass, 5: 0.00}).
MATERIAL_TO_LCCODE = {
    "built":      0,   # building roof / generic urban impervious
    "concrete":   0,
    "asphalt":    1,
    "bare_soil":  2,
    "water":      3,
    "grass":      4,
}

M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def build_scenario(site: dict, sid: int, region: RegionOSM,
                   canopy: CanopyHeightLoader, assigned_month: int) -> dict | None:
    lat, lon = site["lat"], site["lon"]

    osm_buildings = region.buildings_local(lat, lon, radius_m=ZONE_RADIUS_M)  # centred on (lat,lon)=(0,0)
    mat_zones = region.materials_local(lat, lon, radius_m=ZONE_RADIUS_M)

    # ── site polygon: 80x80 m centred on the cluster centre ──────
    h = SITE_SIZE_M / 2.0
    site_poly = box(-h, -h, h, h)

    # ── buildings: keep those within the context radius for shadows ─
    buildings = []
    placed = []
    for ob in osm_buildings:
        try:
            fp = Polygon(ob["footprint"])
            if not fp.is_valid:
                fp = make_valid(fp)
            if fp.is_empty or fp.area < 10.0:
                continue
            if fp.centroid.distance(Point(0, 0)) > CONTEXT_R_M:
                continue
            # Drop only near-identical duplicates (>80% mutual overlap);
            # real adjacent buildings share edges and must be KEPT.
            dup = False
            for p in placed:
                inter = fp.intersection(p).area
                if inter > 0.8 * min(fp.area, p.area):
                    dup = True
                    break
            if dup:
                continue
            hh = float(ob.get("height", 10.0))
            fl = int(ob.get("floor_count", max(1, round(hh / 3.6))))
            buildings.append({
                "footprint":  fp,
                "floors":     fl,
                "height":     hh,
                "centroid":   (fp.centroid.x, fp.centroid.y),
                "gfa":        fp.area * fl,
                "coverage":   fp.area,
                "shape_type": "osm",
                "osm_id":     ob.get("osm_id"),
                "height_from_tag": bool(ob.get("height_from_tag")),
            })
            placed.append(fp)
        except Exception:
            continue

    if len(buildings) < 1:
        return None

    bu = unary_union(placed) if placed else None
    open_space = site_poly.difference(bu) if bu is not None else site_poly

    # ── trees from real ETH canopy ───────────────────────────────
    # Sample at the native ETH grid resolution (~10 m) over the site plus a
    # modest buffer (the dominant canopy shading of an 80x80 m scene comes
    # from vegetation within ~40 m; far-field trees only shade at very low
    # sun). Each >=TREE_MIN_H pixel becomes one tree with its REAL measured
    # height; the total is capped (tallest kept) so a forested edge does not
    # explode into thousands of tree objects that would be both physically
    # double-counted and prohibitively slow in the shadow loop.
    tree_r = SITE_SIZE_M / 2.0 + 40.0    # ~80 m radius footprint of influence
    cand = []
    gxy = np.arange(-tree_r, tree_r + 1e-6, TREE_GRID_M)
    for gx in gxy:
        for gy in gxy:
            if math.hypot(gx, gy) > tree_r:
                continue
            p = Point(float(gx), float(gy))
            if bu is not None and bu.contains(p):
                continue
            ch = canopy.sample_at_local(float(gx), float(gy))
            if ch >= TREE_MIN_H:
                cand.append((float(ch), float(gx), float(gy)))
    n_real = len(cand)
    cand.sort(reverse=True)               # tallest first
    trees = [{"pos": (gx, gy), "height": ch, "radius": ch * 0.4,
              "source": "eth_canopy"}
             for ch, gx, gy in cand[:TREE_MAX]]

    # ── land-cover map: rasterise materials + buildings to 1 m grid ─
    land_cover_map = {}
    # ground materials first (lower priority)
    for material, polys in mat_zones.items():
        code = MATERIAL_TO_LCCODE.get(material)
        if code is None:
            continue
        for ring in polys:
            try:
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = make_valid(poly)
            except Exception:
                continue
            minx, miny, maxx, maxy = poly.bounds
            for xx in range(int(math.floor(max(minx, -h))), int(math.ceil(min(maxx, h))) + 1):
                for yy in range(int(math.floor(max(miny, -h))), int(math.ceil(min(maxy, h))) + 1):
                    if poly.contains(Point(xx, yy)):
                        land_cover_map[(xx, yy)] = code
    # buildings override as built(0) (roof)
    for fp in placed:
        minx, miny, maxx, maxy = fp.bounds
        for xx in range(int(math.floor(max(minx, -h))), int(math.ceil(min(maxx, h))) + 1):
            for yy in range(int(math.floor(max(miny, -h))), int(math.ceil(min(maxy, h))) + 1):
                if fp.contains(Point(xx, yy)):
                    land_cover_map[(xx, yy)] = MATERIAL_TO_LCCODE["built"]

    total_gfa = sum(b["gfa"] for b in buildings)
    total_cov = sum(b["coverage"] for b in buildings)
    site_area = site_poly.area

    return {
        "scenario_id":   sid,
        "site_polygon":  site_poly,
        "buildings":     buildings,
        "open_space":    open_space,
        "trees":         trees,
        "land_cover_map": land_cover_map or None,
        "far_actual":    total_gfa / site_area,
        "bcr_actual":    min(total_cov / site_area, 1.0),
        "total_gfa":     total_gfa,
        # ── V4 real-data provenance (for CWB/IoT step) ───────────
        "site_lat":      lat,
        "site_lon":      lon,
        "assigned_month": assigned_month,
        "n_canopy_pixels_detected": n_real,
        "n_trees_used":  len(trees),
        "n_height_tagged_buildings": sum(1 for b in buildings if b["height_from_tag"]),
        "nearest_iot_dist_m": site.get("nearest_iot_dist_m"),
        "source": "real_osm_eth_v4",
    }


def main(sites_json: str, pbf_path: str, out_dir: str, months: list[int]):
    doc = json.loads(Path(sites_json).read_text(encoding="utf-8"))
    sites = doc.get("sites", [])
    if not sites:
        print("[builder] no sites in JSON"); return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    rb = doc.get("region_bbox", {"lat_min": 23.95, "lon_min": 120.55,
                                 "lat_max": 25.10, "lon_max": 121.30})
    bbox = (rb["lat_min"], rb["lon_min"], rb["lat_max"], rb["lon_max"])
    region = RegionOSM(pbf_path, bbox).load()

    print(f"[builder] building {len(sites)} real-geometry scenarios ...")
    scenarios = []
    for i, site in enumerate(sites):
        # per-site canopy loader (local origin = this site's centre)
        canopy = CanopyHeightLoader(tif_path=str(CANOPY_TIF),
                                    site_lat=site["lat"], site_lon=site["lon"])
        month = months[i % len(months)]
        sc = build_scenario(site, i, region, canopy, month)
        canopy.close()
        if sc is not None:
            scenarios.append(sc)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sites)} processed -> {len(scenarios)} valid", flush=True)

    # honest summary
    nb = [len(s["buildings"]) for s in scenarios]
    nt = [s["n_trees_used"] for s in scenarios]
    ndet = [s["n_canopy_pixels_detected"] for s in scenarios]
    ntag = [s["n_height_tagged_buildings"] for s in scenarios]
    print(f"\n[builder] HONEST SUMMARY")
    print(f"  scenarios built : {len(scenarios)} / {len(sites)} sites")
    if scenarios:
        print(f"  buildings/scene : {min(nb)}-{max(nb)} (mean {np.mean(nb):.1f})")
        print(f"  trees used/scene: {min(nt)}-{max(nt)} (mean {np.mean(nt):.1f}) "
              f"[capped at {TREE_MAX}; raw canopy px mean {np.mean(ndet):.0f}]")
        print(f"  height-tagged buildings/scene: mean {np.mean(ntag):.1f}")
        for m in months:
            print(f"    month {m}: {sum(1 for s in scenarios if s['assigned_month']==m)} scenes")

    pkl = out / "scenarios_v4.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"  saved: {pkl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=str(_ROOT / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"))
    ap.add_argument("--pbf", default=str(_ROOT / "data" / "osm" / "taiwan-latest.osm.pbf"))
    ap.add_argument("--out", default=str(_ROOT / "01_data_generation" / "outputs" / "real_simulations_v4"))
    ap.add_argument("--months", default="6,7,8")
    args = ap.parse_args()
    months = [int(x) for x in args.months.split(",")]
    main(args.sites, args.pbf, args.out, months)
