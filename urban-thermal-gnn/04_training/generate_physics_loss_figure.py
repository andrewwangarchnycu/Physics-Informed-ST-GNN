"""
generate_physics_loss_figure.py
=================================
Schematic figure for the three physics-informed loss terms (L_rad, L_temp,
L_wind) used in PIN-ST-GNN training. Each panel shows the real functional
form (ReLU-hinge penalty) plotted over its actual domain and threshold
values as defined in urban-thermal-gnn's loss/physics_penalty.py /
Thesis_GIA chapter 3 equations, not a generic mock-up.

House style: matches fig_overview_pistgnn_white.png -- white background,
bordered panels, bold black bilingual titles; colour retained per-curve
(a legitimate encoding of the three distinct loss terms, akin to the
reference style's own coloured relation tags) but muted to the same
desaturated palette used by the other redrawn schematic figures.

Produces:
  figures/fig_physics_loss.pdf
  figures/fig_physics_loss.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9,
    "axes.unicode_minus": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

# --- Panel 1: L_rad --- (matches physics_penalty.py radiation_penalty:
# relu(u_shade - u_sun + margin)^2 -- SQUARED, and margin is ADDED, so the
# constraint is "sun must exceed shade by >= margin", not merely "shade
# must not exceed sun". Trigger point is therefore at diff = -margin.
ax = axes[0]
m_rad = 0.5
diff = np.linspace(-2.5, 2, 400)   # diff = u_shade - u_sun
penalty = np.maximum(0, diff + m_rad) ** 2
ax.plot(diff, penalty, color="#8a3a1f", linewidth=2)
ax.axvspan(-m_rad, 2, color="#8a3a1f", alpha=0.12)
ax.axvline(-m_rad, color="#555555", linestyle="--", linewidth=1)
ax.text(-m_rad + 0.05, ax.get_ylim()[1]*0.85, r"$-m_{rad}=-0.5$", fontsize=8, color="#000000")
ax.set_xlabel(r"$\bar{u}_{shade} - \bar{u}_{sun}$ (normalized UTCI)")
ax.set_ylabel(r"$\mathcal{L}_{rad}$ penalty")
ax.set_title("輻射空間一致性 Radiation Consistency\n（最小差距約束，平方鉸鏈損失）", color="#000000")

# --- Panel 2: L_temp --- (matches temporal_smoothness_penalty:
# relu(|delta| - max_delta), LINEAR not squared)
ax = axes[1]
delta_temp = 0.625
d_utci = np.linspace(-2, 2, 400)
penalty2 = np.maximum(0, np.abs(d_utci) - delta_temp)
ax.plot(d_utci, penalty2, color="#4a6a8a", linewidth=2)
ax.axvspan(delta_temp, 2, color="#4a6a8a", alpha=0.12)
ax.axvspan(-2, -delta_temp, color="#4a6a8a", alpha=0.12)
ax.axvline(delta_temp, color="#555555", linestyle="--", linewidth=1)
ax.axvline(-delta_temp, color="#555555", linestyle="--", linewidth=1)
ax.text(delta_temp + 0.05, ax.get_ylim()[1]*0.85, r"$\delta_{temp}=0.625$", fontsize=8, color="#000000")
ax.set_xlabel(r"$\hat{u}_{i,t+1} - \hat{u}_{i,t}$ (normalized UTCI)")
ax.set_ylabel(r"$\mathcal{L}_{temp}$ penalty")
ax.set_title("都市熱慣性時序平滑 Temporal Smoothness\n（線性鉸鏈損失，上限 5°C/hr）", color="#000000")

# --- Panel 3: L_wind --- (matches wind_obstruction_penalty:
# relu(u - upper_bound), LINEAR not squared)
ax = axes[2]
z = np.linspace(-3, 4, 400)  # (u_hat - mu) / sigma equivalent axis
thresh = 1.5
penalty3 = np.maximum(0, z - thresh)
ax.plot(z, penalty3, color="#4f7a55", linewidth=2)
ax.axvspan(thresh, 4, color="#4f7a55", alpha=0.12)
ax.axvline(thresh, color="#555555", linestyle="--", linewidth=1)
ax.text(thresh + 0.05, ax.get_ylim()[1]*0.85, r"$1.5\sigma_{\hat{u}}$", fontsize=8, color="#000000")
ax.set_xlabel(r"$(\hat{u}_i - \mu_{\hat{u}}) / \sigma_{\hat{u}}$, $i \in \mathcal{B}_{tall}$")
ax.set_ylabel(r"$\mathcal{L}_{wind}$ penalty")
ax.set_title("背風渦流風場遮蔽 Leeward Wake Blockage\n（線性鉸鏈損失，熱滯留死區）", color="#000000")

for ax in axes:
    ax.grid(alpha=0.25, color="#bfbfbf")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.2); spine.set_color("black")

fig.suptitle("物理引導損失函數之懲罰項　Physics-Informed Loss Penalty Functions",
             fontsize=12.5, fontweight="bold", y=1.04, color="#000000")
fig.tight_layout()

out_pdf = FIG_DIR / "fig_physics_loss.pdf"
out_png = FIG_DIR / "fig_physics_loss.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
