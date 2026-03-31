"""
spatial_filtering.py
════════════════════════════════════════════════════════════════
Phase 1: GIS Spatial Filtering - [REMOVED_ZH:10] 250m [REMOVED_ZH:2]

[REMOVED_ZH:2]:
  1. [REMOVED_ZH:2] MOENV_iot_station.csv (11,000 [REMOVED_ZH:7])
  2. [REMOVED_ZH:2]Hsinchu Area[REMOVED_ZH:6] ([REMOVED_ZH:10])
  3. [REMOVED_ZH:2] 250m × 250m [REMOVED_ZH:4] ([REMOVED_ZH:2] UTM [REMOVED_ZH:2])
  4. [REMOVED_ZH:8] 1-3 [REMOVED_ZH:4]
  5. [REMOVED_ZH:2]：
     - filtered_station_ids.json ([REMOVED_ZH:3])
     - hsinchu_grid_250m.geojson ([REMOVED_ZH:4]and[REMOVED_ZH:2])
     - grid_station_mapping.json (grid_id → [station_ids])

Run:
  python spatial_filtering.py \
      --iot_station inputs/iot_data/MOENV_iot_station.csv \
      --grid_size 250 \
      --output outputs/spatial/
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# [REMOVED_ZH:6] (lat, lon)
HSINCHU_CENTER_LAT = 25.0330
HSINCHU_CENTER_LON = 120.9330

# UTM [REMOVED_ZH:4] ([REMOVED_ZH:4] zone 51)
UTM_ZONE_51_LON_BASE = 123.0
UTM_FALSE_EASTING = 500000
UTM_FALSE_NORTHING = 0
UTM_SCALE_FACTOR = 0.9996

# [REMOVED_ZH:4] ([REMOVED_ZH:5])
SEARCH_RADIUS_KM = 30.0  # [REMOVED_ZH:4] 30km


@dataclass
class Point2D:
    x: float
    y: float


def latlon_to_utm51(lat: float, lon: float) -> Tuple[float, float]:
    """
    [REMOVED_ZH:4]: WGS84 (lat, lon) → UTM Zone 51
    [REMOVED_ZH:2] Transverse Mercator [REMOVED_ZH:2]
    """
    # [REMOVED_ZH:4] (Zone 51 [REMOVED_ZH:5])
    lon_ref = 123.0
    e2 = 0.00669438  # WGS84 [REMOVED_ZH:7]

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon_ref_rad = math.radians(lon_ref)

    dlon = lon_rad - lon_ref_rad

    a = 6378137.0  # WGS84 [REMOVED_ZH:3]
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e2 * math.cos(lat_rad)**2
    A = math.cos(lat_rad) * dlon

    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
             - (3*e2/8 + 3*e2**2/32 - 45*e2**3/1024) * math.sin(2*lat_rad)
             + (15*e2**2/256 - 45*e2**3/1024) * math.sin(4*lat_rad)
             - (35*e2**3/3072) * math.sin(6*lat_rad))

    easting = UTM_FALSE_EASTING + UTM_SCALE_FACTOR * N * (
        A + (A**3/6) * (1 - T + C)
        + (A**5/120) * (1 - 5*T + 9*C + 4*C**2)
    )

    northing = UTM_FALSE_NORTHING + UTM_SCALE_FACTOR * (
        M + N * math.tan(lat_rad) * (
            (A**2/2) + (A**4/24) * (5 - T + 9*C + 4*C**2)
            + (A**6/720) * (61 - 58*T + T**2 + 600*C - 330*e2)
        )
    )

    return easting, northing


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """compute[REMOVED_ZH:3] Haversine [REMOVED_ZH:2] [km]"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def create_grid_250m(center_lat: float, center_lon: float,
                     radius_km: float, grid_size_m: int = 250
                     ) -> List[Dict]:
    """
    [REMOVED_ZH:7]，[REMOVED_ZH:4] 250m × 250m [REMOVED_ZH:2]
    [REMOVED_ZH:2] UTM [REMOVED_ZH:2]compute，[REMOVED_ZH:5]
    """
    # [REMOVED_ZH:5] UTM
    cx_utm, cy_utm = latlon_to_utm51(center_lat, center_lon)

    # compute[REMOVED_ZH:5] (UTM [REMOVED_ZH:2]: [REMOVED_ZH:1])
    grid_half_extent = (radius_km * 1000 / math.sqrt(2)) + grid_size_m

    grids = []
    grid_id = 0

    # [REMOVED_ZH:4] (UTM [REMOVED_ZH:4])
    x = cx_utm - grid_half_extent
    while x < cx_utm + grid_half_extent:
        y = cy_utm - grid_half_extent
        while y < cy_utm + grid_half_extent:
            # [REMOVED_ZH:4] (UTM)
            grid_center_x_utm = x + grid_size_m / 2
            grid_center_y_utm = y + grid_size_m / 2

            # [REMOVED_ZH:4] WGS84 ([REMOVED_ZH:5])
            # [REMOVED_ZH:9] ([REMOVED_ZH:5])
            # [REMOVED_ZH:2]: [REMOVED_ZH:10]
            dlon_utm = (grid_center_x_utm - UTM_FALSE_EASTING) / (UTM_SCALE_FACTOR * 111000)
            dlat_utm = (grid_center_y_utm - UTM_FALSE_NORTHING) / (UTM_SCALE_FACTOR * 111000)

            grid_lat = center_lat + dlat_utm
            grid_lon = center_lon + (dlon_utm / math.cos(math.radians(center_lat)))

            grids.append({
                "grid_id": grid_id,
                "center_lat": grid_lat,
                "center_lon": grid_lon,
                "center_utm_x": grid_center_x_utm,
                "center_utm_y": grid_center_y_utm,
                "size_m": grid_size_m,
            })
            grid_id += 1
            y += grid_size_m

        x += grid_size_m

    return grids


def filter_stations_by_distance(stations_df: pd.DataFrame,
                                center_lat: float, center_lon: float,
                                radius_km: float = SEARCH_RADIUS_KM) -> pd.DataFrame:
    """[REMOVED_ZH:7] radius_km [REMOVED_ZH:5]"""
    distances = stations_df.apply(
        lambda row: haversine_km(center_lat, center_lon,
                                row['lat'], row['lon']),
        axis=1
    )
    return stations_df[distances <= radius_km].reset_index(drop=True)


def assign_stations_to_grids(grids: List[Dict],
                            stations_df: pd.DataFrame,
                            k_nearest: int = 3) -> Dict[int, List[str]]:
    """
    [REMOVED_ZH:14]
    [REMOVED_ZH:2] UTM [REMOVED_ZH:2]compute[REMOVED_ZH:2] ([REMOVED_ZH:3])
    """
    grid_to_stations = {g["grid_id"]: [] for g in grids}

    # [REMOVED_ZH:12] UTM
    stations_utm = []
    for _, row in stations_df.iterrows():
        x_utm, y_utm = latlon_to_utm51(row['lat'], row['lon'])
        stations_utm.append((x_utm, y_utm, row['deviceId']))

    # [REMOVED_ZH:13]
    for x_utm, y_utm, station_id in stations_utm:
        min_dist = float('inf')
        nearest_grid_id = None

        for grid in grids:
            gx = grid['center_utm_x']
            gy = grid['center_utm_y']
            dist = math.sqrt((x_utm - gx)**2 + (y_utm - gy)**2)

            if dist < min_dist:
                min_dist = dist
                nearest_grid_id = grid['grid_id']

        if nearest_grid_id is not None:
            grid_to_stations[nearest_grid_id].append(station_id)

    # [REMOVED_ZH:5]
    grid_to_stations = {k: v for k, v in grid_to_stations.items() if v}

    return grid_to_stations


def save_results(grids: List[Dict],
                grid_to_stations: Dict[int, List[str]],
                output_dir: Path):
    """[REMOVED_ZH:6]"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. [REMOVED_ZH:6]
    all_station_ids = set()
    for station_list in grid_to_stations.values():
        all_station_ids.update(station_list)

    with open(output_dir / "filtered_station_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_station_ids)), f, indent=2)

    # 2. [REMOVED_ZH:2] GeoJSON
    features = []
    for grid in grids:
        if grid['grid_id'] in grid_to_stations:
            # compute[REMOVED_ZH:4] ([REMOVED_ZH:6] buffer)
            grid_size_deg = (grid['size_m'] / 111000)  # [REMOVED_ZH:1] → [REMOVED_ZH:1]

            feature = {
                "type": "Feature",
                "properties": {
                    "grid_id": grid['grid_id'],
                    "n_stations": len(grid_to_stations[grid['grid_id']]),
                    "station_ids": grid_to_stations[grid['grid_id']],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [grid['center_lon'], grid['center_lat']],
                }
            }
            features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "properties": {
            "title": "Hsinchu 250m Grid with IoT Sensors",
            "total_grids": len(features),
            "total_stations": len(all_station_ids),
        }
    }

    with open(output_dir / "hsinchu_grid_250m.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    # 3. [REMOVED_ZH:2]-[REMOVED_ZH:5]
    with open(output_dir / "grid_station_mapping.json", "w", encoding="utf-8") as f:
        mapping = {str(k): v for k, v in grid_to_stations.items()}
        json.dump(mapping, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Phase 1: GIS Spatial Filtering Complete")
    print(f"{'='*60}")
    print(f"  Total grids (with sensors):  {len(features)}")
    print(f"  Total stations (filtered):   {len(all_station_ids)}")
    print(f"  Avg stations per grid:       {len(all_station_ids)/max(1, len(features)):.1f}")
    print(f"  Search radius:               {SEARCH_RADIUS_KM} km around Hsinchu")
    print(f"\n  Output files:")
    print(f"    - {output_dir / 'filtered_station_ids.json'}")
    print(f"    - {output_dir / 'hsinchu_grid_250m.geojson'}")
    print(f"    - {output_dir / 'grid_station_mapping.json'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="GIS Spatial Filtering for IoT Sensors")
    parser.add_argument("--iot_station", type=str,
                       default="inputs/iot_data/MOENV_iot_station.csv",
                       help="Path to MOENV IoT station metadata CSV")
    parser.add_argument("--grid_size", type=int, default=250,
                       help="Grid size in meters (default: 250)")
    parser.add_argument("--radius_km", type=float, default=SEARCH_RADIUS_KM,
                       help=f"Search radius around Hsinchu center in km (default: {SEARCH_RADIUS_KM})")
    parser.add_argument("--output", type=str, default="outputs/spatial",
                       help="Output directory for results")

    args = parser.parse_args()

    # [REMOVED_ZH:7]
    print("\n[1/4] Reading IoT station metadata...")
    try:
        stations_df = pd.read_csv(args.iot_station, encoding='utf-8-sig')
        print(f"  [OK] Loaded {len(stations_df)} total stations from Taiwan")
    except Exception as e:
        print(f"  [ERROR] Error reading {args.iot_station}: {e}")
        return

    # [REMOVED_ZH:9]
    print(f"\n[2/4] Filtering stations within {args.radius_km} km of Hsinchu ({HSINCHU_CENTER_LAT}°, {HSINCHU_CENTER_LON}°)...")
    stations_filtered = filter_stations_by_distance(
        stations_df,
        HSINCHU_CENTER_LAT,
        HSINCHU_CENTER_LON,
        args.radius_km
    )
    print(f"  [OK] Filtered to {len(stations_filtered)} stations in search radius")

    # [REMOVED_ZH:4]
    print(f"\n[3/4] Generating {args.grid_size}m x {args.grid_size}m grid...")
    grids = create_grid_250m(
        HSINCHU_CENTER_LAT,
        HSINCHU_CENTER_LON,
        args.radius_km,
        args.grid_size
    )
    print(f"  [OK] Generated {len(grids)} grid cells")

    # [REMOVED_ZH:8]
    print(f"\n[4/4] Assigning stations to grids...")
    grid_to_stations = assign_stations_to_grids(grids, stations_filtered, k_nearest=3)

    # [REMOVED_ZH:4]
    output_dir = Path(args.output)
    save_results(grids, grid_to_stations, output_dir)


if __name__ == "__main__":
    main()
