"""
generate_design_iteration_figure.py
================================
Design-iteration demonstration figure requested by the thesis committee:
"input building outline -> computation output -> design feedback/
adjustment" closed loop. Rather than a purely schematic illustration,
this figure uses two REAL non-dominated solutions from the route-B
(open plaza/courtyard) A/B/C NSGA-II experiment
(07_optimization/outputs/nsga2_route_B.json) that trade off building
massing against walkway thermal exposure, to demonstrate the kind of
design comparison/iteration a user actually performs when browsing the
Pareto front produced by UTCIOptimizer.

Produces:
  figures/fig_design_iteration.pdf
  figures/fig_design_iteration.png
"""
import sys, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OPT_DIR = _SCRIPT_DIR.parent / "07_optimization" / "outputs"

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

with open(OPT_DIR / "nsga2_route_B.json", "r", encoding="utf-8") as f:
    d = json.load(f)
pareto = d["final_pareto"]
route_wp = np.array(d["route_waypoints"])

# Iteration 1: best full-domain UTCI (denser, taller massing near path)
iter1 = min(pareto, key=lambda x: x["mean_utci"])
# Iteration 2: best walkway thermal exposure (redistributed massing, more shade on path)
iter2 = min(pareto, key=lambda x: x["walkway_exposure"])

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
panel_info = [
    (axes[0], iter1, "方案 1：以全域 UTCI 最小化為優先選擇", "#c9704f"),
    (axes[1], iter2, "方案 2：以人行動線熱暴露最小化為優先選擇", "#3f7fa6"),
]

for ax, sol, title, color in panel_info:
    ax.set_xlim(-2, 82)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 80, 80, facecolor="#f0ede8", edgecolor="#999999", linewidth=1.0))
    for b in sol["design"]["buildings"]:
        corner = (b["cx"] - b["w"]/2, b["cy"] - b["d"]/2)
        rect = Rectangle(corner, b["w"], b["d"], angle=b["rot"], rotation_point="center",
                          facecolor="#4a4a4a", edgecolor="black", alpha=0.75, linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        ax.text(b["cx"], b["cy"], f"{b['floors']}F", ha="center", va="center",
                 fontsize=7.5, color="white", zorder=4)
    for t in sol["design"]["trees"]:
        ax.add_patch(Circle((t["x"], t["y"]), t["radius"], facecolor="#7a9e6a", alpha=0.6, edgecolor="none", zorder=2))
    ax.plot(route_wp[:, 0], route_wp[:, 1], "-", color=color, linewidth=3.0, zorder=5)
    ax.plot(route_wp[:, 0], route_wp[:, 1], "o", color=color, markersize=7, zorder=6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    metrics_text = (f"全域平均 UTCI = {sol['mean_utci']:.2f}°C\n"
                     f"人行動線熱暴露 = {sol['walkway_exposure']:.4f}\n"
                     f"綠化率 = {sol['green_ratio']:.4f}    FAR = {sol['far']:.2f}")
    ax.text(40, -8, metrics_text, ha="center", va="top", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, linewidth=1.2))

# feedback loop arrow between the two panels
fig.text(0.5, 0.14, "設計者比較兩方案 → 依人行動線舒適度優先權重選擇方案 2\n（帕累托前緣提供之權衡資訊，取代單一「最優解」的線性輸出）",
          ha="center", fontsize=9.3, color="#333333",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f5f0", edgecolor="#888888"))

fig.suptitle("設計疊代示範：從建築輪廓輸入到計算輸出、再到設計回饋調整之閉環", fontsize=13, fontweight="bold", y=1.04)
fig.subplots_adjust(bottom=0.34, top=0.88, wspace=0.15)

out_pdf = FIG_DIR / "fig_design_iteration.pdf"
out_png = FIG_DIR / "fig_design_iteration.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
print(f"iter1 (best UTCI): mean_utci={iter1['mean_utci']}, walk={iter1['walkway_exposure']}")
print(f"iter2 (best walkway): mean_utci={iter2['mean_utci']}, walk={iter2['walkway_exposure']}")
