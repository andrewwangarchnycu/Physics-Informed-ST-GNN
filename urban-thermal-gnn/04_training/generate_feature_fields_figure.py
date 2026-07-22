"""
generate_feature_fields_figure.py
================================
Rebuild of Thesis_GIA chapter 4 fig:feature_fields (fig_geo6_feature_fields.png):
input feature fields (SVF, air temperature, mean radiant temperature) and
the model's UTCI output field, for one real V5 scenario across four
representative timesteps (08:00 / 11:00 / 14:00 / 18:00).

This replaces an earlier, un-tracked version of the figure whose panel
grid was assembled with per-axes fig.colorbar() calls -- that approach
shrinks whichever host axes get a colorbar and leaves the others alone,
so the panels (especially the top-left SVF panel) ended up slightly
misaligned relative to the rest of the grid. Here every data panel lives
in its own fixed GridSpec cell and colorbars live in dedicated slim
GridSpec columns instead of stealing space from a plot axes after the
fact, so all 16 panels share identical width/height and line up exactly
across both rows and columns by construction.

Produces:
  figures/fig_geo6_feature_fields.pdf
  figures/fig_geo6_feature_fields.png
"""
import sys, pickle
from pathlib import Path
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
H5_PATH   = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "ground_truth_v5.h5"
SCEN_PATH = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"
FIG_DIR   = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 13,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

print("[1] Loading one real V5 scenario ...")
with open(SCEN_PATH, "rb") as f:
    all_scenarios = pickle.load(f)
scenario_map = {int(s["scenario_id"]): s for s in all_scenarios}

with h5py.File(H5_PATH, "r") as hf:
    scen_ids = sorted(int(k) for k in hf["scenarios"].keys())
    sid = next((c for c in scen_ids if c in scenario_map and scenario_map[c].get("buildings")), scen_ids[0])
    g = hf[f"scenarios/{sid}"]
    sensor_pts = g["sensor_pts"][()]         # (N,2)
    svf        = g["svf"][()]                # (N,)
    ta         = g["ta"][()]                 # (T,N)
    mrt        = g["mrt"][()]                # (T,N)
    utci       = g["utci"][()]               # (T,N)
    in_shadow  = g["in_shadow"][()]           # (T,N)
T_total = ta.shape[0]
print(f"    scenario_id={sid}  N={sensor_pts.shape[0]}  T={T_total}")

buildings = scenario_map.get(sid, {}).get("buildings", [])

# Representative timesteps: hourly index 0 == 08:00 through index T_total-1 == 18:00
hour_of = lambda idx: 8 + idx
target_hours = [8, 11, 14, 18]
t_idx = [min(range(T_total), key=lambda i: abs(hour_of(i) - h)) for h in target_hours]
print(f"    timestep indices used: {t_idx}  (hours {[hour_of(i) for i in t_idx]})")

COLS = [
    ("svf",  "天空視角因子 SVF\n(靜態, 0–1)", "YlOrRd", None),
    ("ta",   "空氣溫度 Ta (°C)",              "YlOrRd", None),
    ("mrt",  "平均輻射溫度 MRT (°C)",          "RdYlBu_r", None),
    ("utci", "通用熱氣候指數 UTCI (°C)",       "RdYlBu_r", None),
]
DATA = {"svf": svf, "ta": ta, "mrt": mrt, "utci": utci}

# Shared colour normalisation per column across the 4 selected timesteps
# (svf is static so its own [0,1] range is used regardless).
norms = {}
for key, *_ in COLS:
    arr = DATA[key]
    if arr.ndim == 1:
        norms[key] = (0.0, 1.0)
    else:
        sub = arr[t_idx]
        norms[key] = (float(sub.min()), float(sub.max()))

n_rows, n_cols = 4, 4
fig = plt.figure(figsize=(22, 21))
# columns pattern: [plot, cbar, gap] x4, trimmed of the trailing gap
width_ratios = []
for _ in COLS:
    width_ratios += [1.0, 0.045, 0.09]
width_ratios = width_ratios[:-1]
gs = fig.add_gridspec(n_rows, len(width_ratios), width_ratios=width_ratios,
                       hspace=0.30, wspace=0.0, left=0.045, right=0.985, top=0.90, bottom=0.035)

fig.suptitle("都市微氣候空間特徵場：單一真實場景（V5）於四個代表時步之輸入／輸出對照",
             fontsize=19, fontweight="bold", y=0.965)
fig.text(0.5, 0.935, "欄：SVF／Ta／MRT／UTCI　　列：08:00, 11:00, 14:00, 18:00　　黑點：陰影中之空氣節點",
          ha="center", fontsize=13.5, color="#3a3a3a")

axes_grid = [[None] * n_cols for _ in range(n_rows)]
mappables = [None] * n_cols

for ci, (key, title, cmap, _) in enumerate(COLS):
    plot_col = ci * 3
    for ri, ti in enumerate(t_idx):
        ax = fig.add_subplot(gs[ri, plot_col])
        axes_grid[ri][ci] = ax

        for b in buildings:
            fp = b.get("footprint")
            if fp is not None and hasattr(fp, "exterior"):
                xs, ys = fp.exterior.xy
                ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                         facecolor="#4a4a4a", edgecolor="black",
                                         alpha=0.55, linewidth=0.6, zorder=3))

        vals = DATA[key] if DATA[key].ndim == 1 else DATA[key][ti]
        vmin, vmax = norms[key]
        sc = ax.scatter(sensor_pts[:, 0], sensor_pts[:, 1], c=vals, cmap=cmap,
                         vmin=vmin, vmax=vmax, s=9, zorder=2, edgecolors="none")
        mappables[ci] = sc

        shadow_t = in_shadow if in_shadow.ndim == 1 else in_shadow[ti]
        sm = shadow_t.astype(bool)
        if sm.any():
            ax.scatter(sensor_pts[sm, 0], sensor_pts[sm, 1], c="black", s=3.5,
                       zorder=4, alpha=0.55)

        ax.set_aspect("equal")
        ax.set_xlim(sensor_pts[:, 0].min() - 2, sensor_pts[:, 0].max() + 2)
        ax.set_ylim(sensor_pts[:, 1].min() - 2, sensor_pts[:, 1].max() + 2)
        ax.tick_params(labelsize=10.5)

        mean_val = float(vals.mean())
        ax.text(0.03, 0.97, f"μ={mean_val:.1f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#999999", alpha=0.9), zorder=5)

        if ci == 0:
            ax.set_ylabel(f"{hour_of(ti):02d}:00", fontsize=15.5, fontweight="bold")
        if ri == 0:
            ax.set_title(title, fontsize=14.5, fontweight="bold", pad=10)

    # one shared, full-height colorbar per column, in its own GridSpec slot
    cax = fig.add_subplot(gs[:, plot_col + 1])
    fig.colorbar(mappables[ci], cax=cax)
    cax.tick_params(labelsize=10.5)

# --- explicitly re-sync every data panel's box so rows/columns line up
#     pixel-exactly, regardless of any per-axes aspect-ratio auto-adjustment
#     triggered by ax.set_aspect("equal") above. ---
row_y0 = [min(axes_grid[ri][ci].get_position().y0 for ci in range(n_cols)) for ri in range(n_rows)]
row_y1 = [max(axes_grid[ri][ci].get_position().y1 for ci in range(n_cols)) for ri in range(n_rows)]
col_x0 = [min(axes_grid[ri][ci].get_position().x0 for ri in range(n_rows)) for ci in range(n_cols)]
col_x1 = [max(axes_grid[ri][ci].get_position().x1 for ri in range(n_rows)) for ci in range(n_cols)]
for ri in range(n_rows):
    for ci in range(n_cols):
        ax = axes_grid[ri][ci]
        ax.set_position([col_x0[ci], row_y0[ri], col_x1[ci] - col_x0[ci], row_y1[ri] - row_y0[ri]])

out_pdf = FIG_DIR / "fig_geo6_feature_fields.pdf"
out_png = FIG_DIR / "fig_geo6_feature_fields.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"[2] Saved: {out_pdf}\n           {out_png}")
