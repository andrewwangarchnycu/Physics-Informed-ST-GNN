"""
generate_gis_fusion_figure.py
================================
System diagram of the three-service GIS fusion building-vectorization
pipeline (TOPO01K + B5000 + EMAP -> confidence-scored building polygons).

House style: matches fig_overview_pistgnn_white.png -- white background,
light-gray zone containers, white component boxes with black borders,
black arrows, bold bilingual labels; colour reserved for small
source-reference tags only.

Produces:
  figures/fig_gis_fusion_pipeline.pdf
  figures/fig_gis_fusion_pipeline.png
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, zone, box, arrow, tag, ACCENTS, ARROW_COLOR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def elbow(ax, *pts, color=ARROW_COLOR, lw=1.6):
    """Axis-aligned (vertical/horizontal only) connector through waypoints
    *pts*; only the final segment gets an arrowhead, and it lands on the
    destination's outer frame -- no diagonal segments, no routing into a
    specific inner box."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=2, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.0)

fig, ax = plt.subplots(figsize=(12.5, 9.6))
ax.set_xlim(0, 12.5)
ax.set_ylim(-1.2, 9.4)
ax.axis("off")

# ── Zone 1: three data sources (title gap 0.5, box h 1.1, bottom pad 0.15) ─
zone(ax, 0.2, 7.35, 12.1, 1.85, "圖資來源", "Data Sources", fontsize=11)
box(ax, 0.4, 7.5, 3.75, 1.1, "TOPO01K (WMTS)\n粉色封閉曲線建物輪廓\n+ 樓高文字（唯一來源）", 8.6)
box(ax, 4.4, 7.5, 3.75, 1.1, "B5000 (WMS)\n黑色細線封閉輪廓\n幾何獨立驗證來源", 8.6)
box(ax, 8.4, 7.5, 3.75, 1.1, "EMAP (WMS)\n淡紫灰色填色區域\n面積獨立驗證來源", 8.6)
tag(ax, 12.0, 8.9, "§3.4.2", ACCENTS[0])

# ── pixel-aligned fetch ─────────────────────────────────────────────────
box(ax, 3.05, 6.05, 6.4, 0.85, "相同 EPSG:3857 邊界框／相同像素尺寸取得柵格圖磚\n像素級對齊，無需重新取樣", 8.3)

elbow(ax, (2.275, 7.35), (2.275, 7.15), (3.55, 7.15), (3.55, 6.9))
elbow(ax, (6.275, 7.35), (6.275, 6.9))
elbow(ax, (10.275, 7.35), (10.275, 7.15), (8.95, 7.15), (8.95, 6.9))

# ── Zone 2: TOPO01K main processing chain (4 stacked boxes) ────────────
zone(ax, 0.2, 0.9, 3.7, 4.9, "TOPO01K 主處理鏈", "Main Chain", fontsize=10)
box(ax, 0.4, 4.0, 3.3, 0.9, "建物範圍重建\npink | ink 聯集遮罩", 8.3)
box(ax, 0.4, 2.95, 3.3, 0.9, "粗細線分離\n形態學開運算去除 hatch", 8.3)
box(ax, 0.4, 1.9, 3.3, 0.9, "子區塊輪廓萃取\ncv2 次像素等高線追蹤", 8.3)
box(ax, 0.4, 1.05, 3.3, 0.75, "多字元群集樓高辨識\n逐區塊 OCR，非單點中心", 8.0)

elbow(ax, (3.05, 6.05), (3.05, 5.8))
arrow(ax, (2.05, 4.0), (2.05, 3.85))
arrow(ax, (2.05, 2.95), (2.05, 2.8))
arrow(ax, (2.05, 1.9), (2.05, 1.8))

# ── Zone 3/4: validation branches ──────────────────────────────────────
zone(ax, 4.15, 3.6, 3.8, 1.75, "B5000 驗證", "Validation", fontsize=9.3)
box(ax, 4.35, 3.75, 3.4, 0.9, "黑線封閉區域\nflood-fill", 8.3)

zone(ax, 8.15, 3.6, 3.8, 1.75, "EMAP 驗證", "Validation", fontsize=9.3)
box(ax, 8.35, 3.75, 3.4, 0.9, "色彩容差比對", 8.3)

elbow(ax, (5.8, 6.05), (5.8, 5.35))
elbow(ax, (9.2, 6.05), (9.2, 5.35))

# ── Output: confidence-scored polygons ─────────────────────────────────
zone(ax, 2.5, -1.05, 8.0, 1.85, "輸出", "Output", fontsize=10.5)
box(ax, 2.7, -0.9, 7.6, 1.1,
    "跨服務信心度評估：像素重疊率比對 → high / medium / low / unverified\n"
    "輸出：floor_text | floor_kind | floor_number | est_height_m | confidence", 8.4)

elbow(ax, (3.0, 0.9), (3.0, 0.8))
elbow(ax, (6.0, 3.6), (6.0, 0.8))
elbow(ax, (9.5, 3.6), (9.5, 0.8))

fig.suptitle("三服務融合建物向量化系統流程  Three-Service GIS Fusion Pipeline",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_gis_fusion_pipeline.pdf"
out_png = FIG_DIR / "fig_gis_fusion_pipeline.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
