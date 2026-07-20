"""
04_training/generate_material_zone_figure.py
════════════════════════════════════════════════════════════════
Figure: OSM land-use/natural/leisure polygons rasterised into physical
surface-material classes (pavement / ground-cover GIS acquisition),
paired with the real OSM building footprints that override them as
"built" (roof) material in the final land_cover_map.

Two panels for one representative real V4 site:
  (a) raw OSM material polygons colour-coded by class (grass / bare_soil /
      concrete / asphalt / water) + building footprints, within the ~120 m
      shadow-context radius used by the scenario builder;
  (b) the resulting rasterised land_cover_map (1 m grid, buildings painted
      last as "built"), i.e. what actually feeds the runner's per-node
      albedo lookup.

Usage:
    python generate_material_zone_figure.py [--sid 0]
"""
from __future__ import annotations

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "sensing_integration"))
sys.path.insert(0, str(ROOT / "01_data_generation" / "scripts"))

SITES = ROOT / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"
PBF = ROOT / "data" / "osm" / "taiwan-latest.osm.pbf"
CONTEXT_R_M = 120.0
ZONE_RADIUS_M = 300.0

MATERIAL_COLOR = {
    "grass":     "#7fb069",
    "bare_soil": "#c99a5b",
    "concrete":  "#a9a9a9",
    "asphalt":   "#4a4a4a",
    "water":     "#5b8fd6",
}
MATERIAL_LABEL = {
    "grass": "草地／植被", "bare_soil": "裸土", "concrete": "鋪面／混凝土",
    "asphalt": "瀝青／工業", "water": "水域",
}
LCCODE_COLOR = {0: "#8a5a2b", 1: "#4a4a4a", 2: "#c99a5b", 3: "#5b8fd6", 4: "#7fb069"}
LCCODE_LABEL = {0: "建物（built）", 1: "瀝青", 2: "裸土", 3: "水域", 4: "草地"}


def main(sid):
    import json
    from osm_pbf_extract import RegionOSM

    sites = json.loads(SITES.read_text(encoding="utf-8"))["sites"]
    scen = pickle.load(open(SCEN, "rb"))
    by_id = {s["scenario_id"]: s for s in scen}

    if sid is None:
        # pick a scene with the richest material diversity in its land_cover_map
        def n_classes(sc):
            return len(set(sc.get("land_cover_map", {}).values())) if sc.get("land_cover_map") else 0
        sid = max(by_id, key=lambda k: n_classes(by_id[k]))
    sc = by_id[sid]
    site = next(s for s in sites if abs(s["lat"] - sc["site_lat"]) < 1e-6 and abs(s["lon"] - sc["site_lon"]) < 1e-6)

    bbox = (23.95, 120.55, 25.10, 121.30)
    region = RegionOSM(str(PBF), bbox).load()
    mat_zones = region.materials_local(site["lat"], site["lon"], radius_m=ZONE_RADIUS_M)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.4))
    half = CONTEXT_R_M

    # ── Panel (a): raw OSM material polygons + buildings ────────────
    used = set()
    for material, polys in mat_zones.items():
        color = MATERIAL_COLOR.get(material, "#dddddd")
        for ring in polys:
            xy = np.array(ring)
            if xy.shape[0] < 3:
                continue
            axA.add_patch(MplPoly(xy, closed=True, facecolor=color, edgecolor="none", alpha=0.75, zorder=2))
            used.add(material)
    for b in sc["buildings"]:
        fp = b["footprint"]
        xy = list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)
        axA.add_patch(MplPoly(xy, closed=True, facecolor="#e8dcc8", edgecolor="#5a3d00", lw=0.6, zorder=3))
    axA.add_patch(plt.Rectangle((-40, -40), 80, 80, fill=False, edgecolor="#333", lw=1.6, ls="--", zorder=4))
    axA.set_xlim(-half, half); axA.set_ylim(-half, half); axA.set_aspect("equal")
    axA.set_xticks([]); axA.set_yticks([])
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=MATERIAL_COLOR[m],
                      markersize=10, label=MATERIAL_LABEL[m]) for m in MATERIAL_COLOR if m in used]
    handles.append(Line2D([0], [0], marker="s", color="none", markerfacecolor="#e8dcc8",
                          markeredgecolor="#5a3d00", markersize=10, label="建物足跡"))
    axA.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    axA.set_title(f"(a) 場景 #{sid} 之真實 OSM 地表材質多邊形\n（landuse／natural／leisure 標籤分類）", fontsize=10)

    # ── Panel (b): rasterised land_cover_map ─────────────────────────
    lcm = sc.get("land_cover_map") or {}
    if lcm:
        xs = np.array([k[0] for k in lcm]); ys = np.array([k[1] for k in lcm])
        cs = np.array([LCCODE_COLOR.get(v, "#ffffff") for v in lcm.values()])
        axB.scatter(xs, ys, c=cs, s=3, marker="s", zorder=2)
    axB.add_patch(plt.Rectangle((-40, -40), 80, 80, fill=False, edgecolor="#333", lw=1.6, ls="--", zorder=4))
    axB.set_xlim(-half, half); axB.set_ylim(-half, half); axB.set_aspect("equal")
    axB.set_xticks([]); axB.set_yticks([])
    used_codes = sorted(set(lcm.values())) if lcm else []
    handles2 = [Line2D([0], [0], marker="s", color="none", markerfacecolor=LCCODE_COLOR[c],
                       markersize=10, label=LCCODE_LABEL[c]) for c in used_codes]
    axB.legend(handles=handles2, loc="upper right", fontsize=8, framealpha=0.9)
    axB.set_title("(b) 柵格化 land\\_cover\\_map（1 m 網格）\n建物覆蓋於材質圖層之上，供逐節點反射率查表", fontsize=10)

    fig.suptitle("OSM 地表鋪面與材質圖資擷取流程", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = HERE / "figures" / "fig_material_zones.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out} (scene #{sid}, materials={sorted(used)}, lc codes={used_codes})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, default=None)
    args = ap.parse_args()
    main(args.sid)
