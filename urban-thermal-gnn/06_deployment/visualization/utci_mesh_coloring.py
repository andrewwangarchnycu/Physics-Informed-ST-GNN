"""
06_deployment/visualization/utci_mesh_coloring.py
════════════════════════════════════════════════════════════════
UTCI [REMOVED_ZH:9]
[REMOVED_ZH:1] UTCIPredictor.ghpy [REMOVED_ZH:1] Grasshopper [REMOVED_ZH:3]。

[REMOVED_ZH:2]：RhinoCommon (Rhino.Geometry, System.Drawing)
      ── [REMOVED_ZH:2] Rhino 8 / GhPython [REMOVED_ZH:5] ──

[REMOVED_ZH:4]（matplotlib）[REMOVED_ZH:3] plot_utci_standalone()。
"""
from __future__ import annotations
import math
from typing import List, Tuple

# ── UTCI Thermal Stress[REMOVED_ZH:4]（Fiala 2012）──────────────────────────
UTCI_THRESHOLDS = [-40, 9, 26, 32, 38, 46, 200]
UTCI_COLORS_RGB = [
    (70,  130, 180),   # [REMOVED_ZH:3] <9°C    — [REMOVED_ZH:2]
    (144, 238, 144),   # [REMOVED_ZH:2]   9–26°C  — [REMOVED_ZH:2]
    (255, 255, 102),   # [REMOVED_ZH:2]   26–32°C — [REMOVED_ZH:1]
    (255, 165,   0),   # [REMOVED_ZH:2]   32–38°C — [REMOVED_ZH:1]
    (220,  60,  60),   # [REMOVED_ZH:4] 38–46°C — [REMOVED_ZH:1]
    (139,   0,   0),   # [REMOVED_ZH:2]   >46°C   — [REMOVED_ZH:2]
]
UTCI_LABELS = [
    "No stress (<9°C)",
    "Slight (9–26°C)",
    "Moderate (26–32°C)",
    "Strong (32–38°C)",
    "Very strong (38–46°C)",
    "Extreme (>46°C)",
]

# [REMOVED_ZH:6]（[REMOVED_ZH:8]）
_GRADIENT = [
    (9,   (144, 238, 144)),   # [REMOVED_ZH:2]
    (26,  (144, 238, 144)),
    (32,  (255, 255, 102)),   # [REMOVED_ZH:1]
    (38,  (255, 165,   0)),   # [REMOVED_ZH:1]
    (46,  (220,  60,  60)),   # [REMOVED_ZH:1]
]


def utci_to_color_rgb(utci_val: float) -> Tuple[int, int, int]:
    """
    [REMOVED_ZH:3] UTCI Values（°C）[REMOVED_ZH:2] (R, G, B) [REMOVED_ZH:2]，
    [REMOVED_ZH:10]（[REMOVED_ZH:5]）。
    """
    if utci_val <= 9.0:
        return (70, 130, 180)   # [REMOVED_ZH:1]

    for i in range(len(_GRADIENT) - 1):
        t0, c0 = _GRADIENT[i]
        t1, c1 = _GRADIENT[i + 1]
        if utci_val <= t1:
            t = (utci_val - t0) / (t1 - t0 + 1e-9)
            t = max(0.0, min(1.0, t))
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)

    return (139, 0, 0)   # [REMOVED_ZH:2]（>46°C）


def utci_to_class(utci_val: float) -> int:
    for i, (lo, hi) in enumerate(zip(UTCI_THRESHOLDS[:-1],
                                      UTCI_THRESHOLDS[1:])):
        if lo <= utci_val < hi:
            return i
    return len(UTCI_THRESHOLDS) - 2


# ════════════════════════════════════════════════════════════════
# Grasshopper Mesh [REMOVED_ZH:2]（[REMOVED_ZH:2] Rhino [REMOVED_ZH:3]Run）
# ════════════════════════════════════════════════════════════════

def create_utci_mesh_gh(sensor_pts_2d, utci_values,
                         cell_size: float = 1.8,
                         z_offset: float = 0.05):
    """
    [REMOVED_ZH:1] Rhino 8 GhPython [REMOVED_ZH:1]Build UTCI [REMOVED_ZH:2] Mesh。

    Parameters
    ----------
    sensor_pts_2d : list of [x, y] or [[x,y],...]
    utci_values   : list of float (N,) — UTCI [REMOVED_ZH:5] (°C)
    cell_size     : [REMOVED_ZH:13] (m)
    z_offset      : [REMOVED_ZH:7] (m)

    Returns
    -------
    Rhino.Geometry.Mesh (vertex-colored)
    """
    import Rhino.Geometry as rg
    import System.Drawing as sd

    mesh = rg.Mesh()
    half = cell_size / 2.0

    for (pt, val) in zip(sensor_pts_2d, utci_values):
        x, y  = float(pt[0]), float(pt[1])
        z     = z_offset
        r, g, b = utci_to_color_rgb(float(val))
        color   = sd.Color.FromArgb(255, r, g, b)

        i0 = mesh.Vertices.Count
        mesh.Vertices.Add(x - half, y - half, z)
        mesh.Vertices.Add(x + half, y - half, z)
        mesh.Vertices.Add(x + half, y + half, z)
        mesh.Vertices.Add(x - half, y + half, z)

        for _ in range(4):
            mesh.VertexColors.Add(color)

        mesh.Faces.AddFace(i0, i0+1, i0+2, i0+3)

    mesh.Normals.ComputeNormals()
    mesh.Compact()
    return mesh


def create_legend_geometry_gh(x_origin: float = 0.0,
                                y_origin: float = 0.0,
                                z_origin: float = 0.0,
                                bar_w: float = 3.0,
                                bar_h: float = 2.0):
    """
    Build 6 [REMOVED_ZH:5] Mesh [REMOVED_ZH:6]，
    [REMOVED_ZH:6]（[REMOVED_ZH:2] meshes list [REMOVED_ZH:1] texts list）。
    """
    import Rhino.Geometry as rg
    import System.Drawing as sd

    meshes = []
    labels = []

    for i, (rgb, label) in enumerate(zip(UTCI_COLORS_RGB, UTCI_LABELS)):
        y = y_origin + i * (bar_h + 0.2)
        m = rg.Mesh()
        r, g, b = rgb
        color   = sd.Color.FromArgb(255, r, g, b)
        m.Vertices.Add(x_origin,         y,         z_origin)
        m.Vertices.Add(x_origin + bar_w, y,         z_origin)
        m.Vertices.Add(x_origin + bar_w, y + bar_h, z_origin)
        m.Vertices.Add(x_origin,         y + bar_h, z_origin)
        for _ in range(4):
            m.VertexColors.Add(color)
        m.Faces.AddFace(0, 1, 2, 3)
        m.Normals.ComputeNormals()
        meshes.append(m)
        labels.append((x_origin + bar_w + 0.5, y + bar_h/2, z_origin, label))

    return meshes, labels


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:5]（matplotlib，[REMOVED_ZH:3] Rhino）
# ════════════════════════════════════════════════════════════════

def plot_utci_standalone(sensor_pts, utci_values,
                          title: str = "UTCI Distribution",
                          out_path: str = None):
    """
    [REMOVED_ZH:2] matplotlib [REMOVED_ZH:2] UTCI [REMOVED_ZH:2]（[REMOVED_ZH:1] Rhino [REMOVED_ZH:8]）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts   = [[float(p[0]), float(p[1])] for p in sensor_pts]
    xs    = [p[0] for p in pts]
    ys    = [p[1] for p in pts]
    vals  = [float(v) for v in utci_values]
    colors = [utci_to_color_rgb(v) for v in vals]
    rgb01  = [(r/255, g/255, b/255) for r,g,b in colors]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, c=rgb01, s=30, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # [REMOVED_ZH:2]
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=[c/255 for c in rgb], label=lbl)
        for rgb, lbl in zip(UTCI_COLORS_RGB, UTCI_LABELS)
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
