"""
generate_gis_fusion_figure.py
================================
System diagram of the three-service GIS fusion building-vectorization
pipeline (TOPO01K + B5000 + EMAP -> confidence-scored building polygons).
Clear text, simple colour palette, matches academic system-diagram style.

Produces:
  figures/fig_gis_fusion_pipeline.pdf
  figures/fig_gis_fusion_pipeline.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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

C_TOPO = "#c9704f"   # pink/orange -- TOPO01K primary geometry+floor
C_B5000 = "#4a4a4a"  # dark grey -- B5000 line validation
C_EMAP = "#8a6fa3"   # purple -- EMAP area validation
C_PROC = "#5b7fa6"   # blue -- processing steps
C_OUT  = "#7a9e6a"   # green -- output

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(-0.6, 7.5)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=9, textcolor="white", alpha=0.9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.1, edgecolor=color, facecolor=color, alpha=alpha)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, linespacing=1.3)

def arrow(p1, p2, color="#444444", style="-|>", lw=1.3, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                         linewidth=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# --- Row 1: three data sources ---
box(0.3, 6.2, 3.6, 1.0, "TOPO01K (WMTS)\n粉色封閉曲線建物輪廓\n+ 樓高文字（唯一來源）", C_TOPO, fontsize=9)
box(4.2, 6.2, 3.6, 1.0, "B5000 (WMS)\n黑色細線封閉輪廓\n幾何獨立驗證來源", C_B5000, fontsize=9)
box(8.1, 6.2, 3.6, 1.0, "EMAP (WMS)\n淡紫灰色填色區域\n面積獨立驗證來源", C_EMAP, fontsize=9)

# --- Row 2: pixel-aligned fetch ---
box(2.5, 4.9, 7.0, 0.75, "相同 EPSG:3857 邊界框 / 相同像素尺寸取得柵格圖磚 -- 像素級對齊，無需重新取樣", C_PROC, fontsize=8.7)

arrow((2.1, 6.2), (5.0, 5.65))
arrow((6.0, 6.2), (6.0, 5.65))
arrow((9.9, 6.2), (7.0, 5.65))

# --- Row 3: TOPO01K processing chain ---
box(0.3, 3.6, 3.1, 0.8, "建物範圍重建\npink | ink 聯集遮罩", C_TOPO, fontsize=8.5)
box(0.3, 2.5, 3.1, 0.8, "粗細線分離\n形態學開運算去除 hatch", C_TOPO, fontsize=8.5)
box(0.3, 1.4, 3.1, 0.8, "子區塊輪廓萃取\ncv2 次像素等高線追蹤", C_TOPO, fontsize=8.5)
box(0.3, 0.3, 3.1, 0.8, "多字元群集樓高辨識\n逐區塊 OCR，非單點中心", C_TOPO, fontsize=8.5)

arrow((1.85, 4.9), (1.85, 4.4))
arrow((1.85, 3.6), (1.85, 3.3))
arrow((1.85, 2.5), (1.85, 2.2))
arrow((1.85, 1.4), (1.85, 1.1))

# --- Row 3-4: validation branches from B5000/EMAP ---
box(4.3, 2.15, 3.3, 0.85, "B5000 驗證遮罩\n黑線封閉區域 flood-fill", C_B5000, fontsize=8.5)
box(8.0, 2.15, 3.3, 0.85, "EMAP 驗證遮罩\n色彩容差比對", C_EMAP, fontsize=8.5)

arrow((5.5, 4.9), (5.7, 3.0), color=C_B5000)
arrow((8.5, 4.9), (9.4, 3.0), color=C_EMAP)

# --- Output: confidence-scored polygons ---
box(3.6, -0.4, 7.4, 1.3,
    "跨服務信心度評估\n像素重疊率比對 -> high / medium / low / unverified\n輸出：floor_text | floor_kind | floor_number | est_height_m | confidence",
    C_OUT, fontsize=9)

arrow((1.85, 0.3), (3.9, 0.3), color=C_TOPO)
arrow((5.95, 2.15), (6.3, 0.9), color=C_B5000)
arrow((9.65, 2.15), (8.5, 0.9), color=C_EMAP)

legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TOPO, markersize=12, label="TOPO01K 主幾何＋樓高（第三章 §3.4.2）"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_B5000, markersize=12, label="B5000 幾何驗證"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_EMAP, markersize=12, label="EMAP 面積驗證"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OUT, markersize=12, label="輸出：信心度分級建物多邊形"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, fontsize=8.3, frameon=False)

fig.suptitle("三服務融合建物向量化系統流程", fontsize=12, fontweight="bold", y=0.99)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_gis_fusion_pipeline.pdf"
out_png = FIG_DIR / "fig_gis_fusion_pipeline.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
