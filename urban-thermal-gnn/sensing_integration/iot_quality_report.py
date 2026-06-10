"""
sensing_integration/iot_quality_report.py
════════════════════════════════════════════════════════════════
IoT sensor data quality assessment and visualisation.

Outputs (in out_dir):
  - coverage_heatmap.png      spatial sensor positions relative to site
  - data_completeness.png     hour × day completeness heat-map
  - temperature_overview.png  Ta time-series with anomaly flags
  - anomaly_summary.json      machine-readable quality stats
  - quality_report.html       combined HTML report

Usage:
    from sensing_integration.loader_iot import IotSensorLoader
    from sensing_integration.iot_quality_report import IoTQualityReport

    loader = IotSensorLoader(csv_path="iot.csv", site_lat=24.80, site_lon=120.97)
    df     = loader.load_and_clean(month=None, verbose=False)   # all months
    report = IoTQualityReport(df, site_lat=24.80, site_lon=120.97)
    stats  = report.generate(out_dir="outputs/iot_quality")
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")      # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False


# ────────────────────────────────────────────────────────────────
# IoTQualityReport
# ────────────────────────────────────────────────────────────────

class IoTQualityReport:
    """
    Data quality report for IoT sensor observations.

    Parameters
    ----------
    df          : pandas DataFrame with DatetimeIndex and columns [ta, rh, ...]
                  Optional columns: lat, lon, station_id
    site_lat    : WGS84 latitude of site centre
    site_lon    : WGS84 longitude of site centre
    ta_col      : column name for air temperature [°C]
    rh_col      : column name for relative humidity [%]
    iqr_thresh  : IQR multiplier for anomaly detection (default 3.0)
    months      : list of months to include in analysis (None = all)
    """

    def __init__(self,
                 df,
                 site_lat:   float       = 24.80,
                 site_lon:   float       = 120.97,
                 ta_col:     str         = "ta",
                 rh_col:     str         = "rh",
                 iqr_thresh: float       = 3.0,
                 months:     Optional[List[int]] = None):
        self.df         = df
        self.site_lat   = site_lat
        self.site_lon   = site_lon
        self.ta_col     = ta_col
        self.rh_col     = rh_col
        self.iqr_thresh = iqr_thresh
        self.months     = months

        # Filter to requested months
        if _PANDAS and months and df is not None and not df.empty:
            if hasattr(df.index, "month"):
                self.df = df[df.index.month.isin(months)]

    # ── Main entry point ─────────────────────────────────────────

    def generate(self, out_dir: str = "outputs/iot_quality") -> Dict:
        """
        Run all checks, generate plots and report.

        Returns
        -------
        dict with quality statistics (also written to anomaly_summary.json).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not _PANDAS:
            warnings.warn("[IoTQualityReport] pandas not installed — skipping report.")
            return {}

        if self.df is None or self.df.empty:
            warnings.warn("[IoTQualityReport] Empty DataFrame — nothing to analyse.")
            return {}

        stats: Dict = {
            "n_records":    len(self.df),
            "time_range":   {
                "start": str(self.df.index.min()) if hasattr(self.df.index, "min") else "?",
                "end":   str(self.df.index.max()) if hasattr(self.df.index, "max") else "?",
            },
        }

        stats["completeness"] = self._check_completeness()
        stats["anomalies"]    = self._detect_anomalies()
        stats["coverage"]     = self._assess_spatial_coverage()

        # Persist JSON
        json_path = out / "anomaly_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  [IoTQuality] anomaly_summary.json → {json_path}")

        # Generate plots
        if _MPL:
            self._plot_completeness(out / "data_completeness.png")
            self._plot_coverage(out / "coverage_heatmap.png")
            self._plot_temperature_ts(out / "temperature_overview.png")
            self._generate_html(out / "quality_report.html", stats, out)
        else:
            warnings.warn("[IoTQualityReport] matplotlib not installed — plots skipped.")

        print(f"  [IoTQuality] ✓ Report complete → {out}")
        return stats

    # ── Quality checks ───────────────────────────────────────────

    def _check_completeness(self) -> Dict:
        """Temporal completeness per column, plus gap detection."""
        df     = self.df
        result = {}

        for col in [self.ta_col, self.rh_col]:
            if col not in df.columns:
                continue
            total = len(df)
            valid = int(df[col].notna().sum())
            pct   = round(100.0 * valid / max(total, 1), 2)
            result[col] = {
                "total_records":    total,
                "valid_records":    valid,
                "completeness_pct": pct,
                "missing_pct":      round(100.0 - pct, 2),
            }

        # Gap detection on ta
        if self.ta_col in df.columns and hasattr(df.index, "to_series"):
            ta         = df[self.ta_col]
            is_missing = ta.isna()
            gap_starts = df.index[
                is_missing & ~is_missing.shift(1, fill_value=False)
            ].tolist()
            gap_ends = df.index[
                is_missing & ~is_missing.shift(-1, fill_value=False)
            ].tolist()
            result["gaps"] = [
                {"start": str(s), "end": str(e)}
                for s, e in zip(gap_starts[:30], gap_ends[:30])
            ]
            result["n_gaps"] = len(gap_starts)

        return result

    def _detect_anomalies(self) -> Dict:
        """IQR-based outlier flagging for Ta and RH."""
        df        = self.df
        anomalies = {}

        for col in [self.ta_col, self.rh_col]:
            if col not in df.columns:
                continue
            s   = df[col].dropna()
            if len(s) < 4:
                continue
            q1  = float(s.quantile(0.25))
            q3  = float(s.quantile(0.75))
            iqr = q3 - q1
            lo  = q1 - self.iqr_thresh * iqr
            hi  = q3 + self.iqr_thresh * iqr

            flag     = (s < lo) | (s > hi)
            bad_vals = s[flag]

            anomalies[col] = {
                "q1":            round(q1, 3),
                "q3":            round(q3, 3),
                "iqr":           round(iqr, 3),
                "lower_fence":   round(lo, 3),
                "upper_fence":   round(hi, 3),
                "n_anomalies":   int(flag.sum()),
                "anomaly_pct":   round(100.0 * flag.sum() / max(len(s), 1), 2),
                "anomaly_times": [str(t) for t in bad_vals.index[:50]],
                "anomaly_values":[round(float(v), 2) for v in bad_vals.values[:50]],
            }

        return anomalies

    def _assess_spatial_coverage(self) -> Dict:
        """Station count and distances from site centre."""
        df     = self.df
        result = {"n_stations": 1}

        if "station_id" in df.columns:
            stations = df["station_id"].dropna().unique()
            result["n_stations"]  = int(len(stations))
            result["station_ids"] = [str(s) for s in stations[:50]]

        if "lat" in df.columns and "lon" in df.columns:
            lats = df["lat"].dropna().unique()
            lons = df["lon"].dropna().unique()
            result["lat_range"] = [round(float(lats.min()), 5),
                                   round(float(lats.max()), 5)]
            result["lon_range"] = [round(float(lons.min()), 5),
                                   round(float(lons.max()), 5)]

            m_per_deg = 111_320.0
            cos_lat   = math.cos(math.radians(self.site_lat))
            offsets   = []
            for lt, ln in zip(lats[:20], lons[:20]):
                dx = (ln - self.site_lon) * m_per_deg * cos_lat
                dy = (lt - self.site_lat) * m_per_deg
                offsets.append({
                    "dx_m":   round(dx, 1),
                    "dy_m":   round(dy, 1),
                    "dist_m": round(math.hypot(dx, dy), 1),
                })
            result["station_offsets_m"] = offsets

        return result

    # ── Plots ────────────────────────────────────────────────────

    def _plot_completeness(self, out_path: Path) -> None:
        """Hour × day completeness heat-map for Ta and RH."""
        df = self.df
        if not hasattr(df.index, "hour"):
            return

        cols_avail = [c for c in [self.ta_col, self.rh_col] if c in df.columns]
        if not cols_avail:
            return

        ncols = len(cols_avail)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        fig.suptitle("IoT Data Completeness (hour × day-of-year)",
                     fontsize=12, fontweight="bold")

        cmaps  = ["RdYlGn", "Blues"]
        labels = {"ta": "Air Temp (Ta)", "rh": "Humidity (RH)"}

        for ax, col, cmap in zip(axes, cols_avail, cmaps):
            s   = df[col]
            piv = (
                s.groupby([s.index.dayofyear, s.index.hour])
                 .apply(lambda x: x.notna().mean() * 100)
                 .unstack(level=1)
            )
            im = ax.imshow(piv.T, aspect="auto", origin="lower",
                           cmap=cmap, vmin=0, vmax=100,
                           interpolation="nearest")
            ax.set_xlabel("Day of year")
            ax.set_ylabel("Hour of day")
            ax.set_title(f"{labels.get(col, col)}\n(% complete)")
            plt.colorbar(im, ax=ax, label="Completeness %")

        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [IoTQuality] Completeness plot → {out_path}")

    def _plot_coverage(self, out_path: Path) -> None:
        """Spatial map of sensor stations relative to site."""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("IoT Station Coverage Map", fontsize=12, fontweight="bold")
        ax.set_xlabel("East (m from site centre)")
        ax.set_ylabel("North (m from site centre)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Site outline (80 × 80 m)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (-40, -40), 80, 80,
            fill=False, edgecolor="navy", linewidth=2, linestyle="--",
            label="Site boundary 80×80 m"
        ))

        df = self.df
        if "lat" in df.columns and "lon" in df.columns:
            m_per_deg = 111_320.0
            cos_lat   = math.cos(math.radians(self.site_lat))
            lats      = df["lat"].dropna().unique()
            lons      = df["lon"].dropna().unique()
            for i, (lt, ln) in enumerate(zip(lats[:30], lons[:30])):
                x = (ln - self.site_lon) * m_per_deg * cos_lat
                y = (lt - self.site_lat) * m_per_deg
                ax.scatter(x, y, s=140, c="crimson", marker="^",
                           zorder=5, edgecolors="black", linewidths=0.6)
                ax.annotate(f"S{i+1}\n({x:.0f},{y:.0f})", (x, y),
                            textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color="black")
        else:
            ax.scatter([0], [0], s=250, c="orange", marker="*", zorder=5,
                       label="IoT station (lat/lon unknown)")
            ax.text(0, -5, "Location unknown", ha="center",
                    fontsize=9, color="grey")

        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [IoTQuality] Coverage map → {out_path}")

    def _plot_temperature_ts(self, out_path: Path) -> None:
        """Ta time-series with IQR anomaly flags."""
        df = self.df
        if self.ta_col not in df.columns:
            return

        ta = df[self.ta_col]
        if ta.dropna().empty:
            return

        q1  = float(ta.quantile(0.25))
        q3  = float(ta.quantile(0.75))
        iqr = q3 - q1
        lo  = q1 - self.iqr_thresh * iqr
        hi  = q3 + self.iqr_thresh * iqr
        bad = ta[(ta < lo) | (ta > hi)]

        fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle("Air Temperature Overview with Anomaly Detection",
                     fontsize=12, fontweight="bold")

        # Top: time-series
        ax0 = axes[0]
        ax0.plot(ta.index, ta.values, lw=0.5, color="steelblue",
                 alpha=0.8, label="Ta (observed)")
        if len(bad):
            ax0.scatter(bad.index, bad.values, color="red", s=25,
                        zorder=5,
                        label=f"Anomaly ({len(bad)} pts, IQR×{self.iqr_thresh})")
        ax0.axhline(lo, color="red",   lw=0.9, ls="--", alpha=0.5,
                    label=f"Fence: [{lo:.1f}, {hi:.1f}]°C")
        ax0.axhline(hi, color="red",   lw=0.9, ls="--", alpha=0.5)
        ax0.set_ylabel("Temperature (°C)")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.2)

        # Bottom: missing-data indicator
        ax1 = axes[1]
        is_miss = ta.isna().astype(float)
        ax1.fill_between(ta.index, is_miss, step="post",
                         color="salmon", alpha=0.7, label="Missing")
        ax1.set_ylim(0, 1.2)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["OK", "Missing"], fontsize=8)
        ax1.set_ylabel("Data gap")
        ax1.grid(True, alpha=0.2)

        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [IoTQuality] Temperature overview → {out_path}")

    # ── HTML report ──────────────────────────────────────────────

    def _generate_html(self, out_path: Path,
                       stats: Dict, asset_dir: Path) -> None:
        def _pct_span(v):
            clr = "green" if v >= 90 else ("orange" if v >= 70 else "red")
            return f'<span style="color:{clr}"><b>{v:.1f}%</b></span>'

        comp  = stats.get("completeness", {})
        anom  = stats.get("anomalies", {})
        cov   = stats.get("coverage", {})
        tr    = stats.get("time_range", {})

        comp_rows = ""
        for col, info in comp.items():
            if not isinstance(info, dict) or "completeness_pct" not in info:
                continue
            comp_rows += (
                f"<tr><td>{col}</td>"
                f"<td>{_pct_span(info['completeness_pct'])}</td>"
                f"<td>{info['valid_records']:,} / {info['total_records']:,}</td>"
                f"<td>{info['missing_pct']:.2f}%</td></tr>"
            )

        anom_rows = ""
        for col, info in anom.items():
            anom_rows += (
                f"<tr><td>{col}</td>"
                f"<td>{info['n_anomalies']}</td>"
                f"<td>{info['anomaly_pct']:.2f}%</td>"
                f"<td>[{info['lower_fence']:.1f}, {info['upper_fence']:.1f}]</td>"
                f"<td>IQR × {self.iqr_thresh}</td></tr>"
            )

        gaps = comp.get("gaps", [])
        gap_html = ""
        if gaps:
            gap_html = "<ul>" + "".join(
                f"<li>{g['start']} → {g['end']}</li>" for g in gaps[:15]
            ) + "</ul>"
            if len(gaps) > 15:
                gap_html += f"<p>… and {len(gaps)-15} more gaps.</p>"

        img = lambda name: (
            f'<img src="{name}" alt="{name}" '
            'style="max-width:100%;border:1px solid #ccc;border-radius:4px;margin:8px 0">'
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>IoT Quality Report</title>
<style>
  body  {{ font-family:Arial,sans-serif; margin:24px; background:#f4f6f9; color:#333; }}
  h1    {{ color:#2c3e50; }}
  h2    {{ color:#34495e; border-bottom:2px solid #bbb; padding-bottom:4px; }}
  .card {{ background:#fff; border-radius:8px; padding:18px;
           margin-bottom:18px; box-shadow:0 2px 6px rgba(0,0,0,.1); }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:12px; }}
  th,td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; font-size:13px; }}
  th    {{ background:#2c3e50; color:#fff; }}
  tr:nth-child(even) {{ background:#ecf0f1; }}
</style>
</head>
<body>
<h1>IoT Sensor Quality Report</h1>

<div class="card">
  <h2>Overview</h2>
  <p>Records: <b>{stats.get('n_records',0):,}</b> |
     Period: {tr.get('start','?')} → {tr.get('end','?')} |
     Stations: <b>{cov.get('n_stations','?')}</b></p>
  <p>Site: lat={self.site_lat:.4f}, lon={self.site_lon:.4f}</p>
</div>

<div class="card">
  <h2>Temporal Completeness</h2>
  <table>
    <tr><th>Variable</th><th>Completeness</th><th>Valid / Total</th>
        <th>Missing %</th></tr>
    {comp_rows}
  </table>
  {gap_html}
  {img("data_completeness.png")}
</div>

<div class="card">
  <h2>Anomaly Detection</h2>
  <table>
    <tr><th>Variable</th><th># Anomalies</th><th>Anomaly %</th>
        <th>Valid Range</th><th>Method</th></tr>
    {anom_rows}
  </table>
  {img("temperature_overview.png")}
</div>

<div class="card">
  <h2>Spatial Coverage</h2>
  {img("coverage_heatmap.png")}
</div>

</body></html>"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [IoTQuality] HTML report → {out_path}")


# ────────────────────────────────────────────────────────────────
# CLI convenience
# ────────────────────────────────────────────────────────────────

def main(iot_csv: str, out_dir: str = "outputs/iot_quality",
         months:  Optional[List[int]] = None,
         site_lat: float = 24.80,
         site_lon: float = 120.97) -> None:
    """Run quality report from command line."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from sensing_integration.loader_iot import IotSensorLoader
    loader = IotSensorLoader(csv_path=iot_csv,
                             site_lat=site_lat, site_lon=site_lon)
    df = loader.load_and_clean(month=None, verbose=True)

    report = IoTQualityReport(df, site_lat=site_lat, site_lon=site_lon,
                               months=months)
    report.generate(out_dir=out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IoT quality report")
    ap.add_argument("--iot_csv",  required=True)
    ap.add_argument("--out",      default="outputs/iot_quality")
    ap.add_argument("--lat",      type=float, default=24.80)
    ap.add_argument("--lon",      type=float, default=120.97)
    ap.add_argument("--months",   default="6,7,8,9",
                    help="Comma-separated months to include")
    args = ap.parse_args()
    m = [int(x) for x in args.months.split(",")]
    main(args.iot_csv, args.out, months=m,
         site_lat=args.lat, site_lon=args.lon)
