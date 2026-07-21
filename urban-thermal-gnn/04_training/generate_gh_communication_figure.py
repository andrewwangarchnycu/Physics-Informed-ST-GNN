"""
generate_gh_communication_figure.py
=====================================
Sequence diagram of the Grasshopper <-> FastAPI/WebSocket communication
architecture described in thesis Ch3 subsec:gh_integration, matching the
real endpoint structure implemented in 06_deployment/app.py (/ws endpoint,
"predict" and "optimize" actions, asyncio.Queue-bridged progress streaming
from NSGA2Optimizer.run_sync running in a ThreadPoolExecutor).

House style: matches fig_overview_pistgnn_white.png -- white background,
white lifeline-header boxes with black borders, black bold bilingual
labels, black message arrows (sync vs. async differentiated by line
style); colour reserved for small per-actor reference tags only.

Produces:
  figures/fig_gh_websocket_sequence.pdf
  figures/fig_gh_websocket_sequence.png
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, box, tag, ACCENTS, TEXT_MAIN, ARROW_COLOR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.3)

fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 12.2)
ax.axis("off")

# --- lifelines ---
LANES = {
    "GH":  1.4,
    "WS":  4.3,
    "GNN": 7.1,
    "OPT": 9.8,
}
LABELS = {
    "GH":  "Grasshopper\n（UTCIPredictor /\nUTCIOptimizer）",
    "WS":  "FastAPI\n/ws WebSocket\nEndpoint",
    "GNN": "PI-ST-GNN\n推理引擎",
    "OPT": "NSGA2Optimizer\n（ThreadPoolExecutor）",
}
LANE_TAG = {"GH": ACCENTS[2], "WS": ACCENTS[0], "GNN": ACCENTS[1], "OPT": ACCENTS[3]}

y_top, y_bot = 11.6, 0.6
for key, x in LANES.items():
    box(ax, x - 0.85, y_top - 0.55, 1.7, 0.75, LABELS[key], fontsize=8.3)
    tag(ax, x, y_top + 0.32, key, LANE_TAG[key], fontsize=7.3)
    ax.plot([x, x], [y_bot, y_top - 0.6], color="#7a7a7a",
             linewidth=1.3, linestyle=(0, (5, 3)), alpha=0.7, zorder=1)


def msg(y, x1, x2, text, dashed=False, fontsize=8.0, textabove=True):
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=11,
                         linewidth=1.3, color=ARROW_COLOR,
                         linestyle=(0, (4, 2)) if dashed else "-",
                         shrinkA=3, shrinkB=3, zorder=3)
    ax.add_patch(a)
    xm = (x1 + x2) / 2
    dy = 0.11 if textabove else -0.20
    ax.text(xm, y + dy, text, ha="center", va="bottom" if textabove else "top",
            fontsize=fontsize, color=TEXT_MAIN)


def actbox(x, y0, y1, w=0.16):
    ax.add_patch(Rectangle((x - w/2, y0), w, y1 - y0,
                            facecolor="#ececec", edgecolor="black", linewidth=0.9, zorder=2))


# ── Path A: single predict round-trip ──────────────────────────────────
y = 10.55
ax.text(0.05, y + 0.28, "(a) 幾何變更之即時推論回饋", fontsize=9.6, fontweight="bold",
        color=TEXT_MAIN, ha="left")
actbox(LANES["WS"], 9.6, y)
actbox(LANES["GNN"], 9.85, 10.05)
msg(y, LANES["GH"], LANES["WS"], "{action: \"predict\", geometry, id}")
y -= 0.55
msg(y, LANES["WS"], LANES["GNN"], "GNNInputBuilder.build() → tensor")
y -= 0.55
msg(y, LANES["GNN"], LANES["WS"], "UTCI 場（N_air × T=11），61.5 ms 平均延遲")
y -= 0.55
msg(y, LANES["WS"], LANES["GH"], "{action: \"predict_result\", utci_grid}")

# ── Path B: optimize job with async progress streaming ─────────────────
y = 7.95
ax.text(0.05, y + 0.55, "(b) NSGA-II 最佳化工作（非阻塞式進度串流）", fontsize=9.6,
        fontweight="bold", color=TEXT_MAIN, ha="left")

actbox(LANES["WS"], 3.05, y)
actbox(LANES["OPT"], 3.15, y - 0.15)

msg(y, LANES["GH"], LANES["WS"], "{action: \"optimize\", pop_size, n_gen, chromosome_config}")
y -= 0.55
msg(y, LANES["WS"], LANES["OPT"], "loop.run_in_executor(executor, run_sync, callback)")
y -= 0.55
ax.text(LANES["OPT"] + 0.05, y + 0.15, "for gen in 1..n_gen:", fontsize=7.8,
        color="#555555", style="italic", ha="left")
y -= 0.42
msg(y, LANES["OPT"], LANES["OPT"], "non-dominated sort +\ncrowding distance", textabove=False)
y -= 0.62
msg(y, LANES["OPT"], LANES["OPT"], "SBX crossover +\npolynomial mutation", textabove=False)
y -= 0.62
msg(y, LANES["OPT"], LANES["OPT"], "batch_evaluate()\n→ 呼叫 GNN 推理", textabove=False)
y -= 0.62
msg(y, LANES["OPT"], LANES["WS"], "callback(info) → asyncio.Queue.put()", dashed=True)
y -= 0.55
msg(y, LANES["WS"], LANES["GH"], "{action: \"optimize_progress\", generation, best_utci, pareto_count}", dashed=True)
y -= 0.65
ax.text((LANES["OPT"] + LANES["WS"]) / 2, y + 0.35, "── 逐代重複 ──", fontsize=7.6,
        color="#666666", ha="center", style="italic")
y -= 0.45
msg(y, LANES["OPT"], LANES["WS"], "run_sync() 返回 pareto_designs")
y -= 0.55
msg(y, LANES["WS"], LANES["GH"], "{action: \"optimize_complete\", pareto_designs}")

fig.suptitle("Grasshopper — FastAPI／WebSocket 非同步通訊架構時序圖",
             fontsize=12.2, fontweight="bold", y=0.995, color=TEXT_MAIN)

legend_elems = [
    Line2D([0], [0], color=ARROW_COLOR, lw=1.4, label="同步請求／回應（WebSocket 訊息）"),
    Line2D([0], [0], color=ARROW_COLOR, lw=1.4, linestyle=(0, (4, 2)), label="非同步進度回呼（asyncio.Queue 橋接執行緒）"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=2, fontsize=8.2, frameon=False, labelcolor=TEXT_MAIN)

fig.tight_layout(rect=[0, 0.02, 1, 0.98])

out_pdf = FIG_DIR / "fig_gh_websocket_sequence.pdf"
out_png = FIG_DIR / "fig_gh_websocket_sequence.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
