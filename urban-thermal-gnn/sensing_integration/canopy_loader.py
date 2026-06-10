"""
sensing_integration/canopy_loader.py
════════════════════════════════════════════════════════════════
Meta HighResCanopyHeight GeoTIFF integration.
https://github.com/facebookresearch/HighResCanopyHeight

Provides 1m-resolution canopy height sampled from real raster data.
Falls back to zero (no canopy) when raster is unavailable.

Install optional deps:
    conda install -c conda-forge rasterio

Usage:
    loader = CanopyHeightLoader("data/canopy/canopy_tile.tif",
                                 site_lat=24.80, site_lon=120.97)
    trees = loader.enrich_trees(trees)          # replace random heights
    heights = loader.sample_local_points(pts)   # (N,) float32
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol as _rowcol
    _RASTERIO = True
except ImportError:
    _RASTERIO = False

# ────────────────────────────────────────────────────────────────
# Coordinate helpers
# ────────────────────────────────────────────────────────────────

_M_PER_DEG_LAT = 111_320.0   # metres per degree latitude (constant)


def _m_per_deg_lon(lat_deg: float) -> float:
    return _M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


# ────────────────────────────────────────────────────────────────
# CanopyHeightLoader
# ────────────────────────────────────────────────────────────────

class CanopyHeightLoader:
    """
    Sample canopy heights from Meta HighResCanopyHeight GeoTIFF.

    Parameters
    ----------
    tif_path      : path to canopy height GeoTIFF tile (may be None → fallback)
    site_lat      : WGS84 latitude of site local-coordinate origin (Y=0)
    site_lon      : WGS84 longitude of site local-coordinate origin (X=0)
    no_data_fill  : returned height when pixel is NoData (default 0.0)
    clip_max      : cap unrealistic canopy heights (default 50 m)
    """

    def __init__(self,
                 tif_path:     Optional[str] = None,
                 site_lat:     float         = 24.80,
                 site_lon:     float         = 120.97,
                 no_data_fill: float         = 0.0,
                 clip_max:     float         = 50.0):
        self.tif_path     = Path(tif_path) if tif_path else None
        self.site_lat     = site_lat
        self.site_lon     = site_lon
        self.no_data_fill = no_data_fill
        self.clip_max     = clip_max
        self._ds          = None   # rasterio.DatasetReader, lazy-opened

        if self.tif_path:
            if self.tif_path.exists():
                self._open()
            else:
                warnings.warn(
                    f"[CanopyLoader] GeoTIFF not found: {tif_path}\n"
                    "Will use fallback height (0 m).  "
                    "Download via download_canopy_tile() or manually from "
                    "https://github.com/facebookresearch/HighResCanopyHeight"
                )

    # ── Lifecycle ───────────────────────────────────────────────

    def _open(self) -> None:
        if not _RASTERIO:
            warnings.warn(
                "[CanopyLoader] rasterio not installed. "
                "Install with: conda install -c conda-forge rasterio\n"
                "Falling back to zero heights."
            )
            return
        try:
            self._ds = rasterio.open(self.tif_path)
            print(f"  [CanopyLoader] Opened {self.tif_path.name} "
                  f"| CRS={self._ds.crs} | res={self._ds.res[0]:.2f}m "
                  f"| shape={self._ds.height}×{self._ds.width}")
        except Exception as e:
            warnings.warn(f"[CanopyLoader] Cannot open GeoTIFF: {e}")
            self._ds = None

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __del__(self):
        self.close()

    @property
    def available(self) -> bool:
        """True if raster is open and ready to query."""
        return self._ds is not None

    # ── Coordinate conversion ───────────────────────────────────

    def _local_to_latlon(self, local_x: float, local_y: float
                          ) -> Tuple[float, float]:
        """Site-local XY (metres, origin = site SW corner) → WGS84."""
        lat = self.site_lat + local_y / _M_PER_DEG_LAT
        lon = self.site_lon + local_x / _m_per_deg_lon(self.site_lat)
        return lat, lon

    # ── Sampling ────────────────────────────────────────────────

    def sample_at_local(self, local_x: float, local_y: float) -> float:
        """Sample canopy height at a single local (x, y) coordinate."""
        if not self.available:
            return self.no_data_fill
        lat, lon = self._local_to_latlon(local_x, local_y)
        try:
            row, col = _rowcol(self._ds.transform, lon, lat)
            H, W = self._ds.height, self._ds.width
            if not (0 <= row < H and 0 <= col < W):
                return self.no_data_fill
            val = float(self._ds.read(1, window=((row, row+1), (col, col+1)))[0, 0])
            nd  = self._ds.nodata
            if nd is not None and val == nd:
                return self.no_data_fill
            return float(np.clip(val, 0.0, self.clip_max))
        except Exception:
            return self.no_data_fill

    def sample_local_points(self,
                             local_pts: List[Tuple[float, float]]) -> np.ndarray:
        """
        Batch-sample canopy height at N site-local coordinates.

        Parameters
        ----------
        local_pts : list of (x, y) in site-local metres

        Returns
        -------
        heights : (N,) float32
        """
        N = len(local_pts)
        heights = np.zeros(N, dtype=np.float32)
        if not self.available or N == 0:
            return heights

        lats = np.empty(N, dtype=float)
        lons = np.empty(N, dtype=float)
        m_per_lon = _m_per_deg_lon(self.site_lat)
        for i, (x, y) in enumerate(local_pts):
            lats[i] = self.site_lat + y / _M_PER_DEG_LAT
            lons[i] = self.site_lon + x / m_per_lon

        try:
            rows, cols = _rowcol(self._ds.transform, lons.tolist(), lats.tolist())
            H, W = self._ds.height, self._ds.width
            nd   = self._ds.nodata
            for i, (r, c) in enumerate(zip(rows, cols)):
                if 0 <= r < H and 0 <= c < W:
                    val = float(
                        self._ds.read(1, window=((r, r+1), (c, c+1)))[0, 0]
                    )
                    if nd is None or val != nd:
                        heights[i] = float(np.clip(val, 0.0, self.clip_max))
        except Exception as exc:
            warnings.warn(f"[CanopyLoader] Batch read failed: {exc}")

        return heights

    # ── Tree enrichment ─────────────────────────────────────────

    def enrich_trees(self, trees: List[dict]) -> List[dict]:
        """
        Replace procedurally-sampled tree heights with canopy-map values.

        Parameters
        ----------
        trees : list of {"pos": (x,y), "height": float, "radius": float}

        Returns
        -------
        Same list with "height" / "radius" updated; "source" tag added.
        """
        if not trees:
            return trees
        if not self.available:
            for t in trees:
                t.setdefault("source", "random_fallback")
            return trees

        pts       = [t["pos"] for t in trees]
        raster_h  = self.sample_local_points(pts)
        enriched  = []
        n_replaced = 0
        for tree, rh in zip(trees, raster_h):
            t2 = dict(tree)
            if rh >= 1.0:           # valid canopy pixel (≥ 1 m)
                t2["height"] = float(rh)
                t2["radius"] = float(rh) * 0.4
                t2["source"] = "canopy_map"
                n_replaced  += 1
            else:
                t2["source"] = "random_fallback"
            enriched.append(t2)

        print(f"  [CanopyLoader] Enriched {n_replaced}/{len(trees)} trees "
              "from raster.")
        return enriched


# ────────────────────────────────────────────────────────────────
# Download helper
# ────────────────────────────────────────────────────────────────

def download_canopy_tile(lat: float,
                          lon: float,
                          out_dir: str = "data/canopy") -> Optional[Path]:
    """
    Download a Meta/ETH HighResCanopyHeight 10m-resolution tile.

    Tiles are 1°×1° GeoTIFFs available from Zenodo:
    https://zenodo.org/records/8167679

    For the 1m high-resolution model, see:
    https://github.com/facebookresearch/HighResCanopyHeight#data-download

    Parameters
    ----------
    lat, lon : approximate site location (used to select tile)
    out_dir  : local directory for caching downloaded tiles

    Returns
    -------
    Path to local tile, or None on failure.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ETH tile naming: N24E120 (no underscore between lat/lon tags)
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))
    # Tiles cover 3-degree blocks; round down to nearest 3-degree boundary
    lat_3 = (lat_floor // 3) * 3
    lon_3 = (lon_floor // 3) * 3
    lat_tag = f"{'N' if lat_3 >= 0 else 'S'}{abs(lat_3):02d}"
    lon_tag = f"{'E' if lon_3 >= 0 else 'W'}{abs(lon_3):03d}"
    fname   = f"ETH_GlobalCanopyHeight_10m_2020_{lat_tag}{lon_tag}_Map.tif"
    local_fp = out_path / fname

    if local_fp.exists():
        print(f"  [CanopyLoader] Using cached tile: {local_fp}")
        return local_fp

    url = (
        "https://share.phys.ethz.ch/~pf/nlangdata/"
        "ETH_GlobalCanopyHeight_10m_2020_version1/3deg_cogs/"
        f"{fname}"
    )
    print(f"  [CanopyLoader] Downloading: {url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, local_fp)
        print(f"  [CanopyLoader] Saved: {local_fp}")
        return local_fp
    except Exception as e:
        warnings.warn(
            f"[CanopyLoader] Download failed: {e}\n"
            "Please download the tile manually from:\n"
            "  https://github.com/facebookresearch/HighResCanopyHeight"
        )
        return None
