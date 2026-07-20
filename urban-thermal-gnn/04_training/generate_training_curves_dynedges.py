"""
generate_training_curves_dynedges.py
======================================
Regenerates the Ch4 training-convergence figure from the REAL training
history of checkpoints_v2_dynedges (the corrected pipeline with real
shadow/veg_et/convective edges, see dataset.py + 11_build_dynamic_edge_cache.py),
replacing the pre-fix checkpoints_v2_fixed run's figure.

Produces:
  viz_output/training/figA_training_curves_dynedges.png/pdf
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
CKPT_DIR = _HERE.parent / "checkpoints_v2_dynedges"
OUT_DIR = _HERE / "viz_output" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

with open(CKPT_DIR / "training_history.json", encoding="utf-8") as f:
    h = json.load(f)

train_loss = np.array(h["train_loss"])
val_loss   = np.array(h["val_loss"])
val_r2     = np.array(h["val_r2"])
lr         = np.array(h["lr"])
epochs     = np.arange(1, len(train_loss) + 1)

# Find LR decay points (where lr drops from previous epoch)
decay_epochs = epochs[1:][lr[1:] < lr[:-1]]

best_epoch = int(np.argmax(val_r2)) + 1
best_r2 = val_r2[best_epoch - 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax1.plot(epochs, train_loss, color="#3a6ea5", linewidth=1.3, label="訓練損失")
ax1.plot(epochs, val_loss, color="#d98c3d", linewidth=1.3, label="驗證損失")
for de in decay_epochs:
    ax1.axvline(de, color="#999999", linestyle="--", linewidth=0.7, alpha=0.7)
ax1.set_ylabel("MSE（正規化空間）")
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax1.set_title(f"模型訓練收斂曲線（共 {len(epochs)} Epoch，修正後管線 checkpoints_v2_dynedges）",
              fontsize=11, fontweight="bold")
ax1.grid(alpha=0.25)

ax2.plot(epochs, val_r2, color="#5b8c5a", linewidth=1.3)
ax2.axhline(best_r2, color="#c9463d", linestyle=":", linewidth=0.9)
ax2.scatter([best_epoch], [best_r2], color="#c9463d", s=30, zorder=5)
ax2.annotate(f"最佳 R²={best_r2:.4f}\n(Epoch {best_epoch})",
             xy=(best_epoch, best_r2), xytext=(best_epoch + 5, best_r2 - 0.015),
             fontsize=8.3, color="#c9463d")
for de in decay_epochs:
    ax2.axvline(de, color="#999999", linestyle="--", linewidth=0.7, alpha=0.7)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("驗證 $R^2$")
ax2.grid(alpha=0.25)

fig.tight_layout()

out_png = OUT_DIR / "figA_training_curves_dynedges.png"
out_pdf = OUT_DIR / "figA_training_curves_dynedges.pdf"
fig.savefig(out_png)
fig.savefig(out_pdf)
print(f"Saved: {out_png}\n       {out_pdf}")
print(f"n_epochs={len(epochs)}, best_epoch={best_epoch}, best_val_r2={best_r2:.4f}, "
      f"decay_epochs={decay_epochs.tolist()}")
