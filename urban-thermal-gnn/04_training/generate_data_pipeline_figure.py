"""
generate_data_pipeline_figure.py
==================================
Simple, academic-style data/system architecture diagram of the full
pipeline: raw GIS + sensor + weather data -> scenario generation ->
simulation -> heterogeneous graph -> PIN-ST-GNN -> NSGA-II / walkway design.

Clear text, restrained colour palette (5 stage colours max), boxes +
arrows only -- matches the "系統圖/資料架構圖" style requested.

Produces:
  figures/fig_data_pipeline_overview.pdf
  figures/fig_data_pipeline_overview.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Restrained 5-colour stage palette
C_GIS    = "#5b7fa6"   # blue -- data acquisition
C_SIM    = "#7a9e6a"   # green -- simulation
C_GRAPH  = "#b8895f"   # tan/orange -- graph construction
C_MODEL  = "#a15c5c"   # red -- model inference
C_DESIGN = "#8a6fa3"   # purple -- design decision

fig, ax = plt.subplots(figsize=(12.5, 7.2))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 7.2)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=9.5, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.88,
                        mutation_aspect=1)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, linespacing=1.35)
    return (x + w/2, y, x + w/2, y + h, x, y + h/2, x + w, y + h/2)

def arrow(p1, p2, color="#444444"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.3, color=color, shrinkA=2, shrinkB=2, zorder=1)
    ax.add_patch(a)

# ---- Row 1: GIS + sensor + weather acquisition ----
b1 = box(0.3, 5.7, 2.7, 1.1, "NLSC WMTS/WMS\n三服務融合建物擷取\n(TOPO01K+B5000+EMAP)", C_GIS)
b2 = box(3.3, 5.7, 2.7, 1.1, "IoT 微氣象感測器\n(環境部智慧城鄉網路)\n空間密度分析", C_GIS)
b3 = box(6.3, 5.7, 2.7, 1.1, "CWA 氣象站\n歷史實測資料\nIDW 空間降尺度校準", C_GIS)
b4 = box(9.3, 5.7, 2.9, 1.1, "參數化幾何生成\n(蒙地卡羅法, 300 場景)\nFAR/BCR 法規約束", C_GIS)

# ---- Row 2: simulation ----
b5 = box(2.5, 4.1, 4.0, 1.0, "Ladybug Tools / EnergyPlus\n全場域微氣候逐時模擬 (T=11)", C_SIM)
b6 = box(7.0, 4.1, 4.0, 1.0, "EPW 氣象檔生成\n地表溫度一階估算", C_SIM)

# ---- Row 3: graph construction ----
b7 = box(2.5, 2.6, 4.0, 1.0, "異質圖建構\n物件節點(7D) + 空氣節點(9D×11)\n5類關係型邊緣", C_GRAPH)
b8 = box(7.0, 2.6, 4.0, 1.0, "節點串接與索引偏移\n(+N_obj offset)", C_GRAPH)

# ---- Row 4: model inference ----
b9 = box(2.9, 1.1, 6.2, 1.0, "PIN-ST-GNN 推理引擎\nRGCN(×3) + Global Context + LSTM + 物理引導損失", C_MODEL)

# ---- Row 5: design decision ----
b10 = box(0.3, -0.3, 3.8, 1.0, "NSGA-II 多目標最佳化\n(UTCI / 人行動線 / 綠化率)", C_DESIGN)
b11 = box(4.4, -0.3, 3.8, 1.0, "熱舒適人行動線設計\n評估指標 Φ_walkway", C_DESIGN)
b12 = box(8.5, -0.3, 3.7, 1.0, "Grasshopper 介面\nFastAPI/WebSocket 即時回饋", C_DESIGN)

ax.set_ylim(-0.5, 7.0)

# arrows
arrow((1.65, 5.7), (4.0, 5.1))
arrow((4.65, 5.7), (4.5, 5.1))
arrow((7.65, 5.7), (6.5, 5.1))
arrow((10.75, 5.7), (8.5, 5.1))
arrow((4.5, 4.1), (4.5, 3.6))
arrow((9.0, 4.1), (9.0, 3.6))
arrow((6.5, 4.6), (7.0, 4.6))
arrow((4.5, 2.6), (4.5, 2.1))
arrow((9.0, 2.6), (9.0, 2.1))
arrow((6.5, 3.1), (7.0, 3.1))
arrow((4.5, 1.1), (5.0, 0.7))
arrow((6.0, 1.1), (6.3, 0.7))
arrow((9.0, 1.1), (10.35, 0.7))

# legend
legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GIS, markersize=12, label="第三章 §3.2/3.4 GIS 資訊建構"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_SIM, markersize=12, label="第三章 §3.3 都市形態模擬"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GRAPH, markersize=12, label="第三章 §3.5 異質圖建構"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MODEL, markersize=12, label="第三章 §3.6-3.7 PIN-ST-GNN 推理"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_DESIGN, markersize=12, label="第三章 §3.8 多目標最佳化與介面"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, fontsize=8.5, frameon=False)

fig.tight_layout()
out_pdf = FIG_DIR / "fig_data_pipeline_overview.pdf"
out_png = FIG_DIR / "fig_data_pipeline_overview.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
