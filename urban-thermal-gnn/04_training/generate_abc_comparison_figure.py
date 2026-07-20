"""
generate_abc_comparison_figure.py
================================
Comparison figure for the A/B/C pedestrian-route NSGA-II experiment
(urban-thermal-gnn/07_optimization/run_abc_walkway_nsga2.py), addressing
the thesis committee's request for multiple design-alternative comparisons
(street frontage / open plaza / diagonal crossing) rather than a single
optimization result.

Reads:
  07_optimization/outputs/nsga2_route_{A,B,C}.json

Produces:
  figures/fig_abc_comparison.pdf
  figures/fig_abc_comparison.png
"""
import sys, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

ROUTES = ["A", "B", "C"]
COLORS = {"A": "#c9704f", "B": "#3f7fa6", "C": "#7a9e6a"}

data = {}
for r in ROUTES:
    with open(OPT_DIR / f"nsga2_route_{r}.json", "r", encoding="utf-8") as f:
        data[r] = json.load(f)

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.32)

# ── Row 1a: route geometry on the 80x80 m site (three small panels) ──
for k, r in enumerate(ROUTES):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(-2, 82)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), 80, 80, facecolor="#f0ede8", edgecolor="#999999", linewidth=1.0))

    d = data[r]
    wp = np.array(d["route_waypoints"])
    ax.plot(wp[:, 0], wp[:, 1], "-", color=COLORS[r], linewidth=3.0, zorder=5)
    ax.plot(wp[:, 0], wp[:, 1], "o", color=COLORS[r], markersize=7, zorder=6)

    # overlay the best-UTCI feasible design's buildings/trees for context
    pareto = d["final_pareto"]
    if pareto:
        best = min(pareto, key=lambda p: p["mean_utci"])
        for b in best["design"]["buildings"]:
            corner = (b["cx"] - b["w"]/2, b["cy"] - b["d"]/2)
            rect = plt.Rectangle(corner, b["w"], b["d"], angle=b["rot"],
                                  rotation_point="center",
                                  facecolor="#4a4a4a", edgecolor="black", alpha=0.55, linewidth=0.6, zorder=3)
            ax.add_patch(rect)
        for t in best["design"]["trees"]:
            ax.add_patch(plt.Circle((t["x"], t["y"]), t["radius"], facecolor="#7a9e6a",
                                     alpha=0.55, edgecolor="none", zorder=2))

    ax.set_title(f"路線 {r}：{d['route_name']}", fontsize=9.3, fontweight="bold")
    ax.axis("off")

# ── Row 1b removed; Row 2 left: convergence curves (mean UTCI) ──
ax = fig.add_subplot(gs[1, 0])
for r in ROUTES:
    hist = data[r]["history"]
    gens = [h["generation"] for h in hist]
    utci = [h["best_utci"] for h in hist]
    ax.plot(gens, utci, color=COLORS[r], linewidth=1.8, label=f"路線 {r}")
ax.set_xlabel("演化世代 (generation)", fontsize=9)
ax.set_ylabel("最佳全域平均 UTCI (°C)", fontsize=9)
ax.set_title("(a) 全域 UTCI 收斂曲線", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# ── Row 2 middle: convergence curves (walkway exposure) ──
ax = fig.add_subplot(gs[1, 1])
for r in ROUTES:
    hist = data[r]["history"]
    gens = [h["generation"] for h in hist]
    walk = [h["best_walkway_exposure"] for h in hist]
    ax.plot(gens, walk, color=COLORS[r], linewidth=1.8, label=f"路線 {r}")
ax.set_xlabel("演化世代 (generation)", fontsize=9)
ax.set_ylabel(r"最佳人行動線熱暴露 $\bar{\Phi}_{walkway}$", fontsize=9)
ax.set_title("(b) 人行動線熱暴露收斂曲線", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# ── Row 2 right: final Pareto front scatter (UTCI vs walkway exposure) ──
ax = fig.add_subplot(gs[1, 2])
for r in ROUTES:
    pareto = data[r]["final_pareto"]
    xs = [p["mean_utci"] for p in pareto]
    ys = [p["walkway_exposure"] for p in pareto]
    ax.scatter(xs, ys, color=COLORS[r], s=28, alpha=0.75, edgecolor="white", linewidth=0.4, label=f"路線 {r} (n={len(pareto)})")
ax.set_xlabel("全域平均 UTCI (°C)", fontsize=9)
ax.set_ylabel(r"人行動線熱暴露 $\bar{\Phi}_{walkway}$", fontsize=9)
ax.set_title("(c) 最終代帕累托前緣散點", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("A/B/C 三種人行路線配置之三目標 NSGA-II 比較實驗", fontsize=13.5, fontweight="bold", y=1.0)

out_pdf = FIG_DIR / "fig_abc_comparison.pdf"
out_png = FIG_DIR / "fig_abc_comparison.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")

# ── Print summary table for use in thesis prose ──
print("\nSummary (final generation, best feasible):")
for r in ROUTES:
    hist = data[r]["history"][-1]
    print(f"  Route {r}: best_utci={hist['best_utci']}  best_walk={hist['best_walkway_exposure']}  "
          f"best_green={hist['best_green']}  pareto_n={hist['pareto_count']}  "
          f"wall_time={data[r]['total_wall_time_s']}s")
