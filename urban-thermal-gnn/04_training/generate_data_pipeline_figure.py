"""
generate_data_pipeline_figure.py
==================================
System/data-architecture diagram of the full pipeline: raw GIS + sensor +
weather data -> scenario generation -> simulation -> heterogeneous graph ->
PI-ST-GNN -> NSGA-II / walkway design.

House style: matches fig_overview_pistgnn_white.png -- white background,
light-gray rounded "zone" containers per pipeline stage, white component
boxes with black borders inside each zone, black arrows, bold bilingual
labels. Colour is reserved for the small stage-reference tags only.

Produces:
  figures/fig_data_pipeline_overview.pdf
  figures/fig_data_pipeline_overview.png
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, zone, box, arrow, tag, ACCENTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.3)

fig, ax = plt.subplots(figsize=(13, 9.4))
ax.set_xlim(0, 13)
ax.set_ylim(-0.6, 9.4)
ax.axis("off")

# ── Zone 1: data acquisition ───────────────────────────────────────────
zone(ax, 0.2, 7.2, 12.6, 2.0, "資料擷取", "Data Acquisition")
box(ax, 0.4, 7.35, 2.9, 1.15, "NLSC 三服務融合\n建物擷取\nTOPO01K+B5000+EMAP", 8.6)
box(ax, 3.5, 7.35, 2.9, 1.15, "IoT 微氣象感測器\n環境部智慧城鄉網路\n空間密度分析", 8.6)
box(ax, 6.6, 7.35, 2.9, 1.15, "CWA 氣象站\n歷史實測資料\nIDW 空間降尺度校準", 8.6)
box(ax, 9.7, 7.35, 2.9, 1.15, "OSM 建物足跡\n（V4，取代隨機生成）\nFAR/BCR 法規約束", 8.6)

# ── Zone 2: simulation ─────────────────────────────────────────────────
zone(ax, 0.2, 5.45, 12.6, 1.65, "都市形態模擬", "Urban Morphology Simulation")
box(ax, 2.9, 5.58, 3.6, 1.0, "物理式逐時微氣候模擬\nSVF＋陰影＋MRT＋風場（T=11）", 8.8)
box(ax, 6.9, 5.58, 3.6, 1.0, "EPW 氣象檔生成\n地表溫度一階估算", 8.8)

# ── Zone 3: graph construction ─────────────────────────────────────────
zone(ax, 0.2, 3.7, 12.6, 1.65, "異質圖建構", "Heterogeneous Graph Construction")
box(ax, 2.9, 3.83, 3.6, 1.0, "物件節點(7D)＋空氣節點(9D×11)\n5 類關係型邊緣", 8.8)
box(ax, 6.9, 3.83, 3.6, 1.0, "節點串接與索引偏移\n(+N_obj offset)", 8.8)

# ── Zone 4: model inference ─────────────────────────────────────────────
zone(ax, 0.2, 2.05, 12.6, 1.55, "PI-ST-GNN 推理", "PI-ST-GNN Inference")
box(ax, 2.5, 2.2, 7.6, 0.9, "RGCN(×3) ＋ Global Context ＋ LSTM ＋ 物理引導損失", 9.2)

# ── Zone 5: design decision ─────────────────────────────────────────────
zone(ax, 0.2, 0.0, 12.6, 1.85, "設計決策", "Design Decision")
box(ax, 0.4, 0.15, 3.9, 1.0, "NSGA-II 多目標最佳化\nUTCI／人行動線／綠化率", 8.6)
box(ax, 4.55, 0.15, 3.9, 1.0, "熱舒適人行動線設計\n評估指標 $\\Phi_{walkway}$", 8.6)
box(ax, 8.7, 0.15, 3.9, 1.0, "Grasshopper 介面\nFastAPI／WebSocket 即時回饋", 8.6)

# arrows between zones
arrow(ax, (6.5, 7.2), (6.5, 7.1))
arrow(ax, (6.5, 5.45), (6.5, 5.35))
arrow(ax, (6.5, 3.7), (6.5, 3.6))
arrow(ax, (6.5, 2.05), (6.5, 1.95))

# stage-reference tags (only colour accent, echoing the reference style's tag pills)
tag(ax, 12.35, 8.35, "§3.2/3.4", ACCENTS[0])
tag(ax, 12.35, 6.35, "§3.3", ACCENTS[1])
tag(ax, 12.35, 4.6, "§3.5", ACCENTS[2])
tag(ax, 12.35, 2.85, "§3.6-3.7", ACCENTS[3])
tag(ax, 12.35, 1.1, "§3.8", ACCENTS[4])

fig.suptitle("整合運算框架資料架構圖  Data Pipeline Overview", fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_data_pipeline_overview.pdf"
out_png = FIG_DIR / "fig_data_pipeline_overview.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
