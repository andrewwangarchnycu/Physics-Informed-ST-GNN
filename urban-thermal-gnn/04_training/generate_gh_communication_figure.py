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

Revision note: labels shortened to short function-call notation (rather
than full JSON payloads) to reduce crowding; fonts enlarged; self-message
loops (internal NSGA-II processing steps on the OPT lane) are drawn as an
explicit loop bracket to the LEFT of the lane with their own text row, so
they no longer sit on top of the OPT activation bar or collide with the
adjacent asynchronous callback arrow.

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
apply_rcparams(11.5)

fig, ax = plt.subplots(figsize=(13.5, 10.5))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 14.6)
ax.axis("off")

# --- lifelines ---
LANES = {
    "GH":  1.6,
    "WS":  5.0,
    "GNN": 8.3,
    "OPT": 11.4,
}
LABELS = {
    "GH":  "Grasshopper\n（GH 元件）",
    "WS":  "FastAPI\nWebSocket 端點",
    "GNN": "PI-ST-GNN\n推理引擎",
    "OPT": "NSGA2Optimizer\n（背景執行緒）",
}
LANE_TAG = {"GH": ACCENTS[2], "WS": ACCENTS[0], "GNN": ACCENTS[1], "OPT": ACCENTS[3]}

y_top, y_bot = 13.9, 0.6
for key, x in LANES.items():
    box(ax, x - 1.0, y_top - 0.65, 2.0, 0.9, LABELS[key], fontsize=10.5)
    tag(ax, x, y_top + 0.42, key, LANE_TAG[key], fontsize=9.0)
    ax.plot([x, x], [y_bot, y_top - 0.7], color="#7a7a7a",
             linewidth=1.3, linestyle=(0, (5, 3)), alpha=0.7, zorder=1)


def msg(y, x1, x2, text, dashed=False, fontsize=10.0, textabove=True):
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.4, color=ARROW_COLOR,
                         linestyle=(0, (4, 2)) if dashed else "-",
                         shrinkA=3, shrinkB=3, zorder=3)
    ax.add_patch(a)
    xm = (x1 + x2) / 2
    dy = 0.13 if textabove else -0.15
    ax.text(xm, y + dy, text, ha="center", va="bottom" if textabove else "top",
            fontsize=fontsize, color=TEXT_MAIN, linespacing=1.3)


def selfmsg(y, x, text, fontsize=9.8, loop_w=0.5, loop_h=0.32):
    """Self-message on a single lane: a small loop bracket protruding to
    the LEFT of the lane (into open space), with its label to the left of
    the loop -- avoids drawing directly over the lane's activation bar."""
    x0 = x - loop_w
    a = FancyArrowPatch((x, y), (x0, y - loop_h), arrowstyle="-|>", mutation_scale=10,
                         linewidth=1.2, color=ARROW_COLOR,
                         connectionstyle="arc,angleA=180,angleB=270,armA=28,armB=28,rad=10",
                         shrinkA=2, shrinkB=2, zorder=3)
    ax.add_patch(a)
    ax.text(x0 - 0.18, y - loop_h / 2, text, ha="right", va="center",
            fontsize=fontsize, color=TEXT_MAIN, linespacing=1.3)


def actbox(x, y0, y1, w=0.16):
    ax.add_patch(Rectangle((x - w/2, y0), w, y1 - y0,
                            facecolor="#ececec", edgecolor="black", linewidth=0.9, zorder=2))


# ── Path A: single predict round-trip ──────────────────────────────────
y = 12.7
ax.text(0.05, y + 0.35, "(a) 幾何變更之即時推論回饋", fontsize=11.5, fontweight="bold",
        color=TEXT_MAIN, ha="left", zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
actbox(LANES["WS"], 11.55, y)
actbox(LANES["GNN"], 11.85, 12.1)
msg(y, LANES["GH"], LANES["WS"], "predict(geometry, id)")
y -= 0.68
msg(y, LANES["WS"], LANES["GNN"], "build_input() → tensor")
y -= 0.68
msg(y, LANES["GNN"], LANES["WS"], "UTCI 場（低延遲輸出）")
y -= 0.68
msg(y, LANES["WS"], LANES["GH"], "predict_result(utci_grid)")

# ── Path B: optimize job with async progress streaming ─────────────────
y = 9.55
ax.text(0.05, y + 0.68, "(b) NSGA-II 最佳化工作（非阻塞式進度串流）", fontsize=11.5,
        fontweight="bold", color=TEXT_MAIN, ha="left", zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

actbox(LANES["WS"], 3.05, y)
actbox(LANES["OPT"], 3.2, y - 0.2)

msg(y, LANES["GH"], LANES["WS"], "optimize(pop_size, n_gen, config)")
y -= 0.68
msg(y, LANES["WS"], LANES["OPT"], "run_in_executor(run_sync)")
y -= 0.62
ax.text(LANES["OPT"] - 0.15, y + 0.10, "for gen in 1..n_gen:", fontsize=9.6,
        color="#555555", style="italic", ha="right")
y -= 0.75
selfmsg(y, LANES["OPT"], "non-dominated sort +\ncrowding distance")
y -= 0.9
selfmsg(y, LANES["OPT"], "SBX crossover +\npolynomial mutation")
y -= 0.9
selfmsg(y, LANES["OPT"], "batch_evaluate()\n呼叫 GNN 推理")
y -= 0.95
msg(y, LANES["OPT"], LANES["WS"], "callback(info) → Queue.put()", dashed=True)
y -= 0.68
msg(y, LANES["WS"], LANES["GH"], "optimize_progress(generation, best_utci)", dashed=True)
y -= 0.72
ax.text((LANES["OPT"] + LANES["WS"]) / 2, y + 0.35, "── 逐代重複 ──", fontsize=9.6,
        color="#666666", ha="center", style="italic")
y -= 0.55
msg(y, LANES["OPT"], LANES["WS"], "run_sync() 返回 pareto_designs")
y -= 0.68
msg(y, LANES["WS"], LANES["GH"], "optimize_complete(pareto_designs)")

fig.suptitle("Grasshopper — FastAPI／WebSocket 非同步通訊架構時序圖",
             fontsize=14.5, fontweight="bold", y=0.995, color=TEXT_MAIN)

legend_elems = [
    Line2D([0], [0], color=ARROW_COLOR, lw=1.6, label="同步請求／回應（WebSocket 訊息）"),
    Line2D([0], [0], color=ARROW_COLOR, lw=1.6, linestyle=(0, (4, 2)), label="非同步進度回呼（asyncio.Queue 橋接執行緒）"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.015),
          ncol=2, fontsize=10.2, frameon=False, labelcolor=TEXT_MAIN)

fig.tight_layout(rect=[0, 0.02, 1, 0.98])

out_pdf = FIG_DIR / "fig_gh_websocket_sequence.pdf"
out_png = FIG_DIR / "fig_gh_websocket_sequence.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
