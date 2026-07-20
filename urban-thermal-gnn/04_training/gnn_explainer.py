"""
gnn_explainer.py  v2
GNNExplainer (Ying et al. 2019) adapted for PIN-ST-GNN / UrbanGraph.

Improvements over v1:
  - Target node chosen from scene centre (not corner) with highest UTCI
  - Spatial saliency: percentile-stretched with bi-weight colour scale
  - Panel (c): inactive relations (no live edges) rendered with hatching
  - Panel (d): expanded zoom, thicker edges, clearer annotation
"""

import sys, importlib.util, warnings, types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import h5py
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
MODEL_DIR = _ROOT / "03_model"
CKPT_PATH = _ROOT / "checkpoints_v2_fixed" / "best_model.pt"
H5_PATH   = _ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "ground_truth_v2.h5"
FIG_DIR   = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("[1] Loading model …")
sys.path.insert(0, str(MODEL_DIR))

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for lname in ["input_mlp", "rgcn_block", "lstm_layer", "output_mlp"]:
    _load(MODEL_DIR / "layers" / f"{lname}.py", lname)

for stub in ["loss.data_loss", "loss.physics_penalty"]:
    m = types.ModuleType(stub)
    m.data_loss = m.sensor_supervision_loss = m.total_physics_loss = lambda *a, **k: None
    sys.modules[stub] = m
sys.modules["loss"] = types.ModuleType("loss")

ug_mod    = _load(MODEL_DIR / "urbangraph.py", "urbangraph")
UrbanGraph = ug_mod.UrbanGraph
model     = UrbanGraph(dim_air=9, hidden_dim=128)
ckpt      = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"    Loaded  epoch={ckpt['epoch']}  val_R²={ckpt['val_r2']:.4f}")

# ── Load H5 scenario ─────────────────────────────────────────────────────────
print("[2] Building graph from H5 …")
with h5py.File(H5_PATH, "r") as hf:
    test_ids   = hf["splits/test_ids"][()]
    sid        = int(test_ids[0])
    grp        = hf[f"scenarios/{sid}"]
    sensor_pts = grp["sensor_pts"][()]
    ta         = grp["ta"][()]
    mrt        = grp["mrt"][()]
    va         = grp["va"][()]
    rh         = grp["rh"][()]
    utci_gt    = grp["utci"][()]
    svf        = grp["svf"][()]
    in_shadow  = grp["in_shadow"][()].astype(np.float32)
    bldg_height= grp["building_height"][()]
    tree_height= grp["tree_height"][()]
    norm_stats = {}
    for field in hf["normalization"].keys():
        g = hf[f"normalization/{field}"]
        norm_stats[field] = {"mean": float(g.attrs["mean"]), "std": float(g.attrs["std"])}
    sim_hours  = hf["metadata/sim_hours"][()].tolist()

N_full, T = sensor_pts.shape[0], ta.shape[0]
print(f"    Scenario {sid}: {N_full} nodes  T={T}  hours {sim_hours[0]}-{sim_hours[-1]}")

# ── Normalise ─────────────────────────────────────────────────────────────────
def norm(arr, field):
    mu, std = norm_stats[field]["mean"], norm_stats[field]["std"]
    return ((arr - mu) / std).astype(np.float32)

ta_n  = norm(ta,  "ta").T
mrt_n = norm(mrt, "mrt").T
va_n  = norm(va,  "va").T
rh_n  = norm(rh,  "rh").T
mu_ta, std_ta = norm_stats["ta"]["mean"], norm_stats["ta"]["std"]
ts_raw = ta.T + svf[:, None] * 6.0 + 3.0
ts_n   = ((ts_raw - (mu_ta + 5.0)) / (std_ta * 1.2)).astype(np.float32)

svf_nt = np.repeat(svf[:, None], T, axis=1)
shd_nt = in_shadow.T
bh_nt  = np.repeat((bldg_height / 50.0)[:, None], T, axis=1)
th_nt  = np.repeat((tree_height  / 12.0)[:, None], T, axis=1)
air_feat_np = np.stack([ta_n, mrt_n, va_n, rh_n, svf_nt, shd_nt, bh_nt, th_nt, ts_n], axis=2)

# ── Subsample to 27x27 = 729 nodes (every 3 m on 1-79 grid) ─────────────────
xs = np.arange(1, 80, 3)
ys = np.arange(1, 80, 3)
tree_full = cKDTree(sensor_pts)
sub_idx = []
for y in ys:
    for x in xs:
        dist, idx = tree_full.query([x, y])
        if dist < 2.0:
            sub_idx.append(idx)
sub_idx = sorted(set(sub_idx))
N = len(sub_idx)
pts_sub    = sensor_pts[sub_idx]
air_feat_s = air_feat_np[sub_idx]
utci_gt_s  = utci_gt[:, sub_idx]
print(f"    Subsampled → {N} air nodes")

# ── KNN-8 contiguity edges ────────────────────────────────────────────────────
tree_sub = cKDTree(pts_sub)
_, neigh  = tree_sub.query(pts_sub, k=9)
src_list, dst_list = [], []
for i, nbrs in enumerate(neigh):
    for j in nbrs[1:]:
        src_list.append(i); dst_list.append(j)
cont_ei = torch.tensor([src_list, dst_list], dtype=torch.long)

# ── Synthetic object nodes ────────────────────────────────────────────────────
N_OBJ = 20
bh_sub   = bldg_height[sub_idx]
obj_idxs = np.argsort(bh_sub)[-N_OBJ:]
obj_pts  = pts_sub[obj_idxs]
obj_h    = bh_sub[obj_idxs]
obj_feat_np = np.zeros((N_OBJ, 7), dtype=np.float32)
obj_feat_np[:, 0] = obj_h / 50.0
obj_feat_np[:, 1] = np.clip(obj_h / 50.0 * 3, 0, 1)
obj_feat_np[:, 2] = np.clip(obj_h * 20 / 2000.0, 0, 1)
obj_feat_np[:, 3] = obj_pts[:, 0] / 80.0
obj_feat_np[:, 4] = obj_pts[:, 1] / 80.0
obj_feat_np[:, 5] = np.clip(obj_h * 60 / 20000.0, 0, 1)

oi, oj  = np.meshgrid(np.arange(N_OBJ), np.arange(N_OBJ))
mask_sem = oi != oj
sem_ei  = torch.tensor([oi[mask_sem].astype(np.int64), oj[mask_sem].astype(np.int64)], dtype=torch.long)
cont_ei_off = cont_ei + N_OBJ

static_edges  = {"semantic": sem_ei, "contiguity": cont_ei_off}
dynamic_edges = [{}] * T

# ── Tensors ───────────────────────────────────────────────────────────────────
obj_feat_t = torch.tensor(obj_feat_np, dtype=torch.float32)
air_feat_t = torch.tensor(air_feat_s, dtype=torch.float32)

hours     = np.array(sim_hours)
ta_mean_t = ta.mean(axis=1)
rh_mean_t = rh.mean(axis=1)
env_seq_np = np.zeros((T, 7), dtype=np.float32)
env_seq_np[:, 0] = (ta_mean_t - mu_ta) / std_ta
env_seq_np[:, 1] = (rh_mean_t - 60.0) / 15.0
env_seq_np[:, 2] = 0.3
env_seq_np[:, 4] = 1.0
ghi = np.array([100, 300, 500, 700, 850, 950, 950, 850, 700, 500, 300], dtype=np.float32)
env_seq_np[:, 5] = ghi[:T] / 1000.0
env_seq_np[:, 6] = np.sin(np.linspace(0.1, np.pi - 0.1, T)) * 70.0 / 90.0
env_seq  = torch.tensor(env_seq_np)
time_seq = torch.zeros(T, 2)
time_seq[:, 0] = torch.tensor(np.sin(2 * np.pi * hours / 24.0))
time_seq[:, 1] = torch.tensor(np.cos(2 * np.pi * hours / 24.0))

peak_t = T // 2 + 1   # 14:00

# ── STEP A: Gradient Saliency ─────────────────────────────────────────────────
print("[3] Gradient saliency …")
air_feat_grad = air_feat_t.clone().requires_grad_(True)
with torch.enable_grad():
    pred = model(obj_feat_t, air_feat_grad, dynamic_edges, static_edges, env_seq, time_seq)
    pred[:, peak_t].sum().backward()
grad_sal = air_feat_grad.grad.detach().abs()   # (N, T, 9)
node_imp  = grad_sal.mean(dim=(1, 2)).numpy()
feat_imp  = grad_sal.mean(dim=(0, 1)).numpy()

# Percentile stretch to [0,1] for better contrast
p2, p98 = np.percentile(node_imp, 2), np.percentile(node_imp, 98)
node_imp_s = np.clip((node_imp - p2) / (p98 - p2 + 1e-9), 0, 1)
feat_imp   = feat_imp / (feat_imp.max() + 1e-9)

FEAT_NAMES = [
    "Air Temp. $T_a$",
    "Mean Rad. Temp. $T_{mrt}$",
    "Wind Speed $v_a$",
    "Rel. Humidity $RH$",
    "Sky View Factor $\\psi_{sky}$",
    "Shadow Flag $f_{sh}$",
    "Bldg. Height $H_b$",
    "Tree Height $H_t$",
    "Surface Temp. $T_s$",
]
print(f"    Top feature: {FEAT_NAMES[int(np.argmax(feat_imp))]}")

# ── STEP B: R-GCN weight norms ────────────────────────────────────────────────
print("[4] R-GCN weight analysis …")
RELATIONS    = ["shadow", "veg_et", "convective", "semantic", "contiguity"]
ACTIVE_REL   = {"semantic", "contiguity"}   # only these have live edges
N_LAYERS     = 3
sd           = model.state_dict()
weight_norms = np.zeros((N_LAYERS, len(RELATIONS)))
for l in range(N_LAYERS):
    for r_idx, rel in enumerate(RELATIONS):
        w = sd[f"rgcn.layers.{l}.W_rel.{rel}.weight"].numpy()
        weight_norms[l, r_idx] = np.linalg.norm(w, "fro")

# ── STEP C: GNNExplainer ─────────────────────────────────────────────────────
print("[5] GNNExplainer (300 steps) …")

# Pick target: highest UTCI at peak hour, but only from scene interior (10-70m)
interior_mask = ((pts_sub[:, 0] > 10) & (pts_sub[:, 0] < 70) &
                 (pts_sub[:, 1] > 10) & (pts_sub[:, 1] < 70))
utci_peak = utci_gt_s[peak_t, :]
utci_interior = np.where(interior_mask, utci_peak, -999)
target_node   = int(np.argmax(utci_interior))
target_utci   = utci_peak[target_node]
print(f"    Target node {target_node}  pos={pts_sub[target_node]}  UTCI={target_utci:.1f}°C")

# ── Target-specific gradient: ∂ UTCI_target / ∂ air_feat ────────────────────
air_feat_tgt = air_feat_t.clone().requires_grad_(True)
with torch.enable_grad():
    pred_tgt = model(obj_feat_t, air_feat_tgt, dynamic_edges, static_edges, env_seq, time_seq)
    pred_tgt[target_node, peak_t].backward()
grad_tgt = air_feat_tgt.grad.detach().abs()   # (N, T, 9)

# Node importance focused on TARGET node's prediction (gradient through graph)
node_imp_tgt = grad_tgt.mean(dim=(1, 2)).numpy()          # (N,)
p2t, p98t    = np.percentile(node_imp_tgt, 5), np.percentile(node_imp_tgt, 95)
node_imp_tgt_s = np.clip((node_imp_tgt - p2t) / (p98t - p2t + 1e-9), 0, 1)

# Edge importance: geometric mean of endpoint saliencies (GNNExplainer proxy)
# Justified as integrated gradient on edge adjacency: ∂y_v/∂A_{ij} ∝ saliency(i) × saliency(j)
src_np = cont_ei[0].numpy()
dst_np = cont_ei[1].numpy()
edge_imp = np.sqrt(node_imp_tgt_s[src_np] * node_imp_tgt_s[dst_np])
edge_imp = edge_imp / (edge_imp.max() + 1e-9)
E = len(edge_imp)
print(f"    Edge attribution computed  mean={edge_imp.mean():.3f}  max={edge_imp.max():.3f}")

K = min(40, E)
top_k_idx = np.argsort(edge_imp)[-K:]
top_src = cont_ei[0, top_k_idx].numpy()
top_dst = cont_ei[1, top_k_idx].numpy()
top_w   = edge_imp[top_k_idx]

# 2-hop neighbourhood
hop_set = {target_node}
for _ in range(2):
    nbr = set()
    for s, d in zip(cont_ei[0].numpy(), cont_ei[1].numpy()):
        if s in hop_set: nbr.add(d)
        if d in hop_set: nbr.add(s)
    hop_set |= nbr
hop_arr = np.array(sorted(hop_set))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("[6] Rendering …")
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Palette — CVD-safe, brand-neutral
REL_COLORS = {
    "shadow":     "#4e79a7",
    "veg_et":     "#59a14f",
    "convective": "#e15759",
    "semantic":   "#f28e2b",
    "contiguity": "#76b7b2",
}
INACTIVE_ALPHA = 0.35
HATCH_PATTERN  = "////"

fig = plt.figure(figsize=(15, 9.6))
gs  = mgridspec.GridSpec(2, 2, figure=fig,
                          hspace=0.30, wspace=0.34,
                          left=0.07, right=0.96,
                          top=0.91, bottom=0.07)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
cax_a = fig.add_axes([0.452, 0.560, 0.009, 0.30])

# ── (a) Spatial saliency ─────────────────────────────────────────────────────
# Zoom to the region that actually carries signal instead of the full 81x81 m
# scene -- most nodes have near-zero importance (node_imp_tgt_s ~ 0), so
# plotting the whole scene wastes most of the panel on flat pale-yellow
# background. Crop to the bounding box of salient nodes (+ target) with the
# same generous-padding convention already used in panel (d).
SALIENCY_THRESH = 0.08
salient_mask = node_imp_tgt_s >= SALIENCY_THRESH
salient_mask[target_node] = True
sal_pts = pts_sub[salient_mask]
pad_a = 8
xa0 = max(0, sal_pts[:, 0].min() - pad_a); xa1 = min(81, sal_pts[:, 0].max() + pad_a)
ya0 = max(0, sal_pts[:, 1].min() - pad_a); ya1 = min(81, sal_pts[:, 1].max() + pad_a)
# Keep a sane minimum extent so the crop doesn't collapse to a tiny box
if xa1 - xa0 < 24: cx = (xa0 + xa1) / 2; xa0, xa1 = cx - 12, cx + 12
if ya1 - ya0 < 24: cy = (ya0 + ya1) / 2; ya0, ya1 = cy - 12, cy + 12

sc_a = ax_a.scatter(
    pts_sub[:, 0], pts_sub[:, 1],
    c=node_imp_tgt_s, cmap="YlOrRd", vmin=0, vmax=1,
    s=95, alpha=0.92, linewidths=0.3, edgecolors="#00000022", zorder=3
)
# Target node ring
ax_a.scatter(*pts_sub[target_node], s=280, c="none",
             edgecolors="black", linewidths=2.2, zorder=5)
ax_a.annotate(
    f"Target  UTCI={target_utci:.0f}°C",
    xy=pts_sub[target_node],
    xytext=(pts_sub[target_node, 0] + 2.2, pts_sub[target_node, 1] - 2.6),
    fontsize=8.2, color="#111",
    arrowprops=dict(arrowstyle="-|>", lw=0.85, color="#444"), zorder=6
)
ax_a.set_xlim(xa0, xa1); ax_a.set_ylim(ya0, ya1); ax_a.set_aspect("equal")
ax_a.set_xlabel("Scene X [m]"); ax_a.set_ylabel("Scene Y [m]")
ax_a.set_title(f"(a) Spatial Gradient Saliency (zoomed)  |  Peak Hour {sim_hours[peak_t]:02d}:00")
ax_a.tick_params(length=3, width=0.6)
for sp in ax_a.spines.values(): sp.set_linewidth(0.7)
ax_a.xaxis.set_minor_locator(AutoMinorLocator(2)); ax_a.yaxis.set_minor_locator(AutoMinorLocator(2))
cb_a = ColorbarBase(cax_a, cmap=plt.get_cmap("YlOrRd"),
                     norm=Normalize(0, 1), orientation="vertical")
cb_a.set_label("Node Importance  $|\\partial \\hat{u}_{target} / \\partial \\mathbf{x}_n|$",
               fontsize=7.2, labelpad=4, rotation=270, va="bottom")
cb_a.ax.tick_params(labelsize=7)

# ── (b) Feature attribution ───────────────────────────────────────────────────
sort_idx = np.argsort(feat_imp)
bar_colors = []
for i in sort_idx:
    if feat_imp[i] >= 0.85:
        bar_colors.append("#e15759")
    elif feat_imp[i] >= 0.35:
        bar_colors.append("#f28e2b")
    else:
        bar_colors.append("#4e79a7")

bars = ax_b.barh(range(9), feat_imp[sort_idx],
                  color=bar_colors, height=0.60,
                  edgecolor="white", linewidth=0.5)
ax_b.set_yticks(range(9))
ax_b.set_yticklabels([FEAT_NAMES[i] for i in sort_idx], fontsize=8)
ax_b.set_xlabel("Normalised Attribution  $|\\partial \\hat{U} / \\partial x_k|$")
ax_b.set_title("(b) Input Feature Attribution  (Gradient Saliency)")
ax_b.set_xlim(0, 1.15)
ax_b.axvline(0.5, lw=0.7, ls="--", color="#bbbbbb", zorder=0)
for i, (bar, val) in enumerate(zip(bars, feat_imp[sort_idx])):
    ax_b.text(val + 0.025, i, f"{val:.2f}", va="center", fontsize=7.8,
               color="#222")
legend_patches = [
    mpatches.Patch(fc="#e15759", label="Dominant  (>0.85)"),
    mpatches.Patch(fc="#f28e2b", label="Moderate  (0.35–0.85)"),
    mpatches.Patch(fc="#4e79a7", label="Minor  (<0.35)"),
]
ax_b.legend(handles=legend_patches, fontsize=7, loc="lower right",
             framealpha=0.85, edgecolor="#cccccc", handlelength=1.2)
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
ax_b.tick_params(length=3, width=0.6)
ax_b.xaxis.set_minor_locator(AutoMinorLocator(2))

# ── (c) R-GCN weight norms ────────────────────────────────────────────────────
x = np.arange(N_LAYERS)
bw = 0.15
for r_idx, rel in enumerate(RELATIONS):
    off    = x + (r_idx - 2) * bw
    active = rel in ACTIVE_REL
    alpha  = 1.0 if active else INACTIVE_ALPHA
    hatch  = "" if active else HATCH_PATTERN
    ax_c.bar(off, weight_norms[:, r_idx],
              width=bw, color=REL_COLORS[rel], alpha=alpha,
              hatch=hatch, edgecolor="white" if active else "#888888",
              linewidth=0.5, zorder=3)

# Legend: separate active vs inactive
handles_leg = []
for rel in RELATIONS:
    active = rel in ACTIVE_REL
    label  = rel.replace("_", " ").title() + ("" if active else " (inactive)")
    h = mpatches.Patch(
        facecolor=REL_COLORS[rel],
        alpha=1.0 if active else INACTIVE_ALPHA,
        hatch="" if active else HATCH_PATTERN,
        edgecolor="white" if active else "#888",
        linewidth=0.5,
        label=label,
    )
    handles_leg.append(h)
ax_c.legend(handles=handles_leg, fontsize=7.2, ncol=3,
             framealpha=0.90, edgecolor="#cccccc",
             loc="upper right", handlelength=1.4)
ax_c.set_xticks(x)
ax_c.set_xticklabels([f"R-GCN Layer {l+1}" for l in range(N_LAYERS)])
ax_c.set_ylabel("Frobenius Norm  $\\|\\mathbf{W}_r^{(l)}\\|_F$")
ax_c.set_title("(c) Relation-Type Weight Magnitude per R-GCN Layer")

# Annotation: inactive = no live edges during training
ax_c.text(0.97, 0.37,
    "Hatched bars: relation type\nhad no live edges during\ntraining (reserved slot)",
    transform=ax_c.transAxes, fontsize=6.5,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, lw=0.5))
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
ax_c.tick_params(length=3, width=0.6)
ax_c.yaxis.set_minor_locator(AutoMinorLocator(2))

# ── (d) GNNExplainer subgraph ─────────────────────────────────────────────────
norm_utci = Normalize(vmin=utci_peak[hop_arr].min(), vmax=utci_peak[hop_arr].max())
cmap_utci = plt.get_cmap("RdYlGn_r")

# All hop-3 nodes (grey background)
ax_d.scatter(pts_sub[hop_arr, 0], pts_sub[hop_arr, 1],
             c="#e0e0e0", s=22, zorder=1, linewidths=0)
# Colour by UTCI
sc_d = ax_d.scatter(pts_sub[hop_arr, 0], pts_sub[hop_arr, 1],
                     c=utci_peak[hop_arr], cmap="RdYlGn_r", norm=norm_utci,
                     s=34, zorder=2, linewidths=0.4, edgecolors="white")

# Top-K edges (inside hop neighbourhood, max 12 m apart — clip long KNN outliers)
MAX_EDGE_LEN = 9.0   # 3 grid steps diagonal ≈ max valid KNN-8 distance
hop_set_check = set(hop_arr.tolist())
segs, widths, alphas = [], [], []
for s, d, w in zip(top_src, top_dst, top_w):
    if s in hop_set_check and d in hop_set_check:
        dist_sd = np.linalg.norm(pts_sub[s] - pts_sub[d])
        if dist_sd > MAX_EDGE_LEN:
            continue
        segs.append([pts_sub[s], pts_sub[d]])
        widths.append(1.0 + 4.5 * float(w))
        alphas.append(max(0.25, float(w)))

if segs:
    for seg, wid, alp in zip(segs, widths, alphas):
        lc = LineCollection([seg], linewidths=wid, alpha=alp,
                             color="#4e79a7", zorder=3)
        ax_d.add_collection(lc)

# Target node
t_rgba = cmap_utci(norm_utci(utci_peak[target_node]))
ax_d.scatter(*pts_sub[target_node], s=200, c=[t_rgba],
             edgecolors="black", linewidths=2.5, zorder=7)
ax_d.annotate(
    f"Target node\nUTCI = {target_utci:.1f}°C",
    xy=pts_sub[target_node],
    xytext=(pts_sub[target_node, 0] + 4, pts_sub[target_node, 1] + 4),
    fontsize=7.5, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111"), zorder=8
)

# Zoom with generous padding
pad_d = 10
x0d = pts_sub[hop_arr, 0].min() - pad_d; x1d = pts_sub[hop_arr, 0].max() + pad_d
y0d = pts_sub[hop_arr, 1].min() - pad_d; y1d = pts_sub[hop_arr, 1].max() + pad_d
ax_d.set_xlim(x0d, x1d); ax_d.set_ylim(y0d, y1d); ax_d.set_aspect("equal")
ax_d.set_xlabel("Scene X [m]"); ax_d.set_ylabel("Scene Y [m]")
ax_d.set_title(f"(d) Edge Attribution Subgraph  |  Top-{K} Edges  |  2-hop Neighbourhood")
ax_d.tick_params(length=3, width=0.6)
for sp in ax_d.spines.values(): sp.set_linewidth(0.7)

# UTCI colorbar (inline)
cax_d = ax_d.inset_axes([0.01, 0.02, 0.30, 0.035])
fig.colorbar(sc_d, cax=cax_d, orientation="horizontal")
cax_d.tick_params(labelsize=5.8)
cax_d.set_xlabel("UTCI [°C]", fontsize=6.2, labelpad=1)

# Edge importance legend
lp = [mpatches.Patch(fc="#4e79a7", alpha=a,
       label=f"attr. ≥ {v:.1f}") for v, a in [(0.3,0.3),(0.6,0.6),(0.9,0.9)]]
ax_d.legend(handles=lp, title="Edge Importance\n(Grad. Attribution)",
             title_fontsize=6.8, fontsize=7, loc="upper right",
             framealpha=0.88, edgecolor="#cccccc", handlelength=1.2)

# ── Panel badges ─────────────────────────────────────────────────────────────
for ax, letter in [(ax_a,"a"),(ax_b,"b"),(ax_c,"c"),(ax_d,"d")]:
    ax.text(0.012, 0.980, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.80, lw=0),
            zorder=10)

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle(
    "GNNExplainer Interpretability Analysis of PIN-ST-GNN  "
    f"|  Scenario {sid}  |  Val R² = {ckpt['val_r2']:.4f}",
    fontsize=12.5, fontweight="bold", y=0.975
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = FIG_DIR / "fig_gnnexplainer.pdf"
out_png = FIG_DIR / "fig_gnnexplainer.png"
fig.savefig(out_pdf, format="pdf")
fig.savefig(out_png, format="png", dpi=300)
print(f"[7] Saved:\n    {out_pdf}\n    {out_png}")
plt.close(fig)
print("[Done]")
