"""
generate_physics_loss_figure.py
=================================
Schematic figure for the three physics-informed loss terms (L_rad, L_temp,
L_wind) used in PIN-ST-GNN training. Each panel shows the real functional
form (ReLU-hinge penalty) plotted over its actual domain and threshold
values as defined in urban-thermal-gnn's loss/physics_penalty.py /
Thesis_GIA chapter 3 equations, not a generic mock-up.

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
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

# --- Panel 1: L_rad --- (matches physics_penalty.py radiation_penalty:
# relu(u_shade - u_sun + margin)^2 -- SQUARED, and margin is ADDED, so the
# constraint is "sun must exceed shade by >= margin", not merely "shade
# must not exceed sun". Trigger point is therefore at diff = -margin.
ax = axes[0]
m_rad = 0.5
diff = np.linspace(-2.5, 2, 400)   # diff = u_shade - u_sun
penalty = np.maximum(0, diff + m_rad) ** 2
ax.plot(diff, penalty, color="#c9463d", linewidth=2)
ax.axvspan(-m_rad, 2, color="#c9463d", alpha=0.12)
ax.axvline(-m_rad, color="gray", linestyle="--", linewidth=1)
ax.text(-m_rad + 0.05, ax.get_ylim()[1]*0.85, r"$-m_{rad}=-0.5$", fontsize=8)
ax.set_xlabel(r"$\bar{u}_{shade} - \bar{u}_{sun}$ (normalized UTCI)")
ax.set_ylabel(r"$\mathcal{L}_{rad}$ penalty")
ax.set_title("Radiation spatial consistency\n(min. gap enforced, squared hinge)")

# --- Panel 2: L_temp --- (matches temporal_smoothness_penalty:
# relu(|delta| - max_delta), LINEAR not squared)
ax = axes[1]
delta_temp = 0.625
d_utci = np.linspace(-2, 2, 400)
penalty2 = np.maximum(0, np.abs(d_utci) - delta_temp)
ax.plot(d_utci, penalty2, color="#2f6690", linewidth=2)
ax.axvspan(delta_temp, 2, color="#2f6690", alpha=0.12)
ax.axvspan(-2, -delta_temp, color="#2f6690", alpha=0.12)
ax.axvline(delta_temp, color="gray", linestyle="--", linewidth=1)
ax.axvline(-delta_temp, color="gray", linestyle="--", linewidth=1)
ax.text(delta_temp + 0.05, ax.get_ylim()[1]*0.85, r"$\delta_{temp}=0.625$", fontsize=8)
ax.set_xlabel(r"$\hat{u}_{i,t+1} - \hat{u}_{i,t}$ (normalized UTCI)")
ax.set_ylabel(r"$\mathcal{L}_{temp}$ penalty")
ax.set_title("Urban thermal-inertia temporal\nsmoothness (linear hinge, max 5°C/hr)")

# --- Panel 3: L_wind --- (matches wind_obstruction_penalty:
# relu(u - upper_bound), LINEAR not squared)
ax = axes[2]
z = np.linspace(-3, 4, 400)  # (u_hat - mu) / sigma equivalent axis
thresh = 1.5
penalty3 = np.maximum(0, z - thresh)
ax.plot(z, penalty3, color="#3c8a5a", linewidth=2)
ax.axvspan(thresh, 4, color="#3c8a5a", alpha=0.12)
ax.axvline(thresh, color="gray", linestyle="--", linewidth=1)
ax.text(thresh + 0.05, ax.get_ylim()[1]*0.85, r"$1.5\sigma_{\hat{u}}$", fontsize=8)
ax.set_xlabel(r"$(\hat{u}_i - \mu_{\hat{u}}) / \sigma_{\hat{u}}$, $i \in \mathcal{B}_{tall}$")
ax.set_ylabel(r"$\mathcal{L}_{wind}$ penalty")
ax.set_title("Leeward-wake wind blockage\n(linear hinge, thermal dead zone)")

for ax in axes:
    ax.grid(alpha=0.25)

fig.suptitle("Physics-Informed Loss Penalty Functions (active-region shaded)",
             fontsize=11.5, fontweight="bold", y=1.04)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_physics_loss.pdf"
out_png = FIG_DIR / "fig_physics_loss.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
