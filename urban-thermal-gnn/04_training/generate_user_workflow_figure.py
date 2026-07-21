"""
generate_user_workflow_figure.py
================================
Unified 1-8 step operation-workflow diagram for Thesis_GIA chapter 5
(subsec:user_workflow_overview), consolidating appendix appx:user_manual's
separate "preparation" (2 steps), "Flow A: UTCIPredictor" (4 steps) and
"Flow B: UTCIOptimizer" (3 steps) sub-flows into a single main-body figure,
addressing the committee's request for a numbered 1-8 step operation
diagram (data prep -> import -> convert -> read/analyze results) rather
than a plain bullet list.

House style: matches fig_overview_pistgnn_white.png -- white background,
white step boxes with black borders, black bold bilingual-adjacent text,
black arrows; colour reserved for small step-category corner tags only.

Produces:
  figures/fig_user_workflow.pdf
  figures/fig_user_workflow.png
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, tag, ACCENTS, TEXT_MAIN, BOX_EDGE, BOX_FILL, ARROW_COLOR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.5)

C_PREP = ACCENTS[3]
C_GEOM = ACCENTS[2]
C_PRED = ACCENTS[1]
C_OPT  = ACCENTS[0]
C_OUT  = "#5a5a5a"

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_xlim(0, 13)
ax.set_ylim(0, 13.2)
ax.axis("off")

def numbered_box(n, x, y, w, h, text, color, fontsize=8.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.09",
                        linewidth=1.2, edgecolor=BOX_EDGE, facecolor=BOX_FILL)
    ax.add_patch(b)
    ax.add_patch(Circle((x + 0.32, y + h - 0.32), 0.24, facecolor="white", edgecolor=BOX_EDGE, linewidth=1.3, zorder=5))
    ax.text(x + 0.32, y + h - 0.32, str(n), ha="center", va="center", fontsize=9.5, fontweight="bold", color=TEXT_MAIN, zorder=6)
    ax.text(x + w/2 + 0.15, y + h/2, text, ha="center", va="center", fontsize=fontsize, color=TEXT_MAIN, linespacing=1.25, fontweight="bold")
    tag(ax, x + w - 0.28, y + h - 0.28, "", color, fontsize=6)

def arrow(p1, p2, color=ARROW_COLOR, lw=1.4, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                         linewidth=lw, color=color, shrinkA=3, shrinkB=3)
    ax.add_patch(a)

# ── Step 1-3: one-time preparation + geometry (single row) ──
numbered_box(1, 0.3, 11.4, 3.7, 1.3, "啟動背景運算服務\n（後端伺服器，一次性）", C_PREP)
numbered_box(2, 4.3, 11.4, 3.7, 1.3, "安裝通訊套件\npip install websocket-client\n（Rhino Script Editor，一次性）", C_PREP)
numbered_box(3, 8.3, 11.4, 4.4, 1.3, "準備場地幾何：\n基地邊界、既有／設計建物、喬木位置與尺寸", C_GEOM, fontsize=8.3)
arrow((4.0, 12.05), (4.3, 12.05))
arrow((8.0, 12.05), (8.3, 12.05))

# ── Step 4: wire to Grasshopper component input ports ──
numbered_box(4, 4.55, 9.3, 4.1, 1.3, "接上 Grasshopper 元件輸入端口\n（表 gh_inputs 對照）", C_GEOM)
arrow((10.5, 11.4), (7.3, 10.6))

# ── Step 5: toggle run switch, branch A/B ──
numbered_box(5, 4.55, 7.2, 4.1, 1.3, "切換 run 開關為 True\n（送出計算請求）", C_GEOM)
arrow((6.6, 9.3), (6.6, 8.5))

# branch labels
ax.text(2.6, 6.75, "路徑 A：單次即時評估\n（UTCIPredictor）", ha="center", fontsize=9, fontweight="bold", color=TEXT_MAIN)
ax.text(10.6, 6.75, "路徑 B：多目標最佳化搜尋\n（UTCIOptimizer）", ha="center", fontsize=9, fontweight="bold", color=TEXT_MAIN)

arrow((5.2, 7.2), (2.6, 6.4), color=C_PRED)
arrow((8.0, 7.2), (10.6, 6.4), color=C_OPT)

# ── Step 6a / 6b ──
numbered_box("6a", 0.5, 5.1, 4.2, 1.15, "立即性回饋，快速回傳結果\n可邊調整設計邊即時觀察熱力圖", C_PRED, fontsize=8.3)
numbered_box("6b", 8.5, 5.1, 4.0, 1.15, "額外設定法規限制與搜尋規模\n（pop_size, n_gen）後啟動搜尋", C_OPT, fontsize=8.3)

arrow((2.6, 5.1), (2.6, 4.35))
arrow((10.5, 5.1), (10.5, 4.35))

# ── Step 7a / 7b ──
numbered_box("7a", 0.5, 3.15, 4.2, 1.1, "修改幾何後結果自動更新\n（run 保持 True 即持續運算）", C_PRED, fontsize=8.3)
numbered_box("7b", 8.5, 3.15, 4.0, 1.1, "背景執行緒搜尋，不阻塞 Rhino\n可隨時取消（cancel=True）", C_OPT, fontsize=8.3)

arrow((2.6, 3.15), (5.9, 1.85), color="#888888")
arrow((10.5, 3.15), (7.3, 1.85), color="#888888")

# ── Step 8: read/interpret results (shared) ──
numbered_box(8, 4.6, 0.9, 4.0, 1.35,
             "取得結果並解讀：\n熱力圖網格 / 數值指標 / 帕累托候選方案\n直接呈現於 Rhino 場景中，供設計調整回饋",
             C_OUT, fontsize=8.0)

legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PREP, markersize=12, label="一次性前置準備"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GEOM, markersize=12, label="場地幾何準備與元件連接"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PRED, markersize=12, label="路徑 A：單次即時評估"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OPT, markersize=12, label="路徑 B：多目標最佳化搜尋"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OUT, markersize=12, label="結果讀取與設計回饋"),
]
fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=8.5, frameon=False, labelcolor=TEXT_MAIN)

fig.suptitle("Grasshopper 元件操作流程（1--8 步驟）：資料準備 → 匯入 → 計算 → 結果讀取", fontsize=13, fontweight="bold", y=1.0, color=TEXT_MAIN)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_user_workflow.pdf"
out_png = FIG_DIR / "fig_user_workflow.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
