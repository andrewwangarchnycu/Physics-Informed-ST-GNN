"""
generate_nsga2_convergence_figure.py
=======================================
REAL NSGA-II convergence figure, built from an actual executed optimization
run (07_optimization/run_real_nsga2.py -> outputs/nsga2_real_run.json),
using the trained PIN-ST-GNN V2 checkpoint as the fitness evaluator (not a
schematic / fabricated curve).

Panel (a): best_utci and best_green_ratio per generation (2 real objectives
           currently wired into 07_optimization/fitness.py -- the 3rd
           walkway-exposure objective is specified but not yet integrated,
           see thesis Ch3 subsec:design_variables and Appendix F).
Panel (b): final-generation Pareto front (mean UTCI vs. green ratio) for
           the 19 non-dominated feasible individuals found.

Produces:
  figures/fig_nsga2_convergence.pdf
  figures/fig_nsga2_convergence.png
"""
import sys, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_JSON = _SCRIPT_DIR.parent / "07_optimization" / "outputs" / "nsga2_real_run.json"

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

with open(RUN_JSON, encoding="utf-8") as f:
    run = json.load(f)

hist = run["history"]
gens = [h["generation"] for h in hist]
best_utci = [h["best_utci"] for h in hist]
best_green = [h["best_green"] for h in hist]
n_feasible = [h["n_feasible"] for h in hist]
pareto_count = [h["pareto_count"] for h in hist]
pareto = run["final_pareto"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axA, axB = axes

# --- Panel (a): convergence curves ---
c_utci, c_green = "#c9463d", "#3a8f5b"
l1, = axA.plot(gens, best_utci, color=c_utci, linewidth=1.8, marker="o", markersize=2.5, label="最佳 UTCI（°C，越低越佳）")
axA.set_xlabel("世代（Generation）")
axA.set_ylabel("族群最佳全域平均 UTCI（°C）", color=c_utci)
axA.tick_params(axis="y", labelcolor=c_utci)
axA.set_title("(a) 收斂曲線（真實 GNN 適應度求值）")
axA.grid(alpha=0.25)

axA2 = axA.twinx()
l2, = axA2.plot(gens, best_green, color=c_green, linewidth=1.8, marker="s", markersize=2.5, label="最佳綠化率")
axA2.set_ylabel("族群最佳綠化率（Green Ratio）", color=c_green)
axA2.tick_params(axis="y", labelcolor=c_green)

axA.legend(handles=[l1, l2], loc="center right", fontsize=8, framealpha=0.9)

# --- Panel (b): final Pareto front scatter ---
utci_vals = [p["mean_utci"] for p in pareto]
green_vals = [p["green_ratio"] for p in pareto]
far_vals = [p["far"] for p in pareto]

sc = axB.scatter(utci_vals, green_vals, c=far_vals, cmap="viridis", s=55,
                  edgecolors="black", linewidths=0.5, zorder=3)
axB.set_xlabel("平均 UTCI（°C，越低越佳）")
axB.set_ylabel("綠化率（越高越佳）")
axB.set_title(f"(b) 第 50 代最終帕累托前緣（n={len(pareto)}）")
axB.grid(alpha=0.25)
cb = fig.colorbar(sc, ax=axB, shrink=0.85)
cb.set_label("FAR（容積率）", fontsize=8.5)

fig.suptitle("NSGA-II 真實收斂結果（PIN-ST-GNN V2 checkpoint，pop=40, gen=50, seed=42）",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_nsga2_convergence.pdf"
out_png = FIG_DIR / "fig_nsga2_convergence.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
print(f"Final gen: best_utci={best_utci[-1]}, best_green={best_green[-1]}, "
      f"n_feasible={n_feasible[-1]}, pareto_count={pareto_count[-1]}")
print(f"Pareto FAR range: {min(far_vals):.3f}-{max(far_vals):.3f}, "
      f"UTCI range: {min(utci_vals):.2f}-{max(utci_vals):.2f}, "
      f"green range: {min(green_vals):.4f}-{max(green_vals):.4f}")
