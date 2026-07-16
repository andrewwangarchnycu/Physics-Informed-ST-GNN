"""
04_training/viz_ablation_progress.py
Live visualization of ablation study progress.
Reads training_history.json from each variant subfolder under ablation_ckpts/.

Usage:
    python viz_ablation_progress.py
    python viz_ablation_progress.py --ablation_dir ablation_ckpts --out ablation_ckpts/ablation_progress.png
"""
import argparse, json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "V0": "V0: MLP baseline\n(no graph, no LSTM)",
    "V3": "V3: Full arch\n(no physics loss)",
    "V4": "V4: + L_rad",
    "V5": "V5: + L_rad + L_temp",
    "V6": "V6: Full model\n(proposed)",
}
COLORS = {
    "V0": "#ff7b72",
    "V3": "#58a6ff",
    "V4": "#ffa657",
    "V5": "#56d364",
    "V6": "#bc8cff",
}
ORDER = ["V0", "V3", "V4", "V5", "V6"]


def load_variant(vdir: Path, max_epochs: int = 200) -> dict:
    hist_p = vdir / "training_history.json"
    ckpt_p = vdir / "best_model.pt"
    if not hist_p.exists():
        return {"id": vdir.name, "status": "pending"}
    with open(hist_p) as f:
        h = json.load(f)
    n        = len(h["train_loss"])
    best_r2  = max(h["val_r2"])
    best_ep  = int(np.argmax(h["val_r2"])) + 1
    best_loss = min(h["val_loss"])
    # Heuristic: done if we hit max_epochs or early-stopping fired (no improvement for 20 ep)
    early_stopped = n > 20 and (n - best_ep) >= 20
    done = (n >= max_epochs) or early_stopped
    return {
        "id":        vdir.name,
        "status":    "done" if done else "running",
        "n":         n,
        "best_r2":   best_r2,
        "best_ep":   best_ep,
        "best_loss": best_loss,
        "history":   h,
    }


def make_plot(ablation_dir: Path, out_path: Path, max_epochs: int = 200):
    summaries = [
        load_variant(ablation_dir / vid, max_epochs)
        for vid in ORDER
        if (ablation_dir / vid).exists() or True   # always include all for status table
    ]
    # Fill in pending placeholders for variants not yet started
    existing_ids = {s["id"] for s in summaries}
    for vid in ORDER:
        if vid not in existing_ids:
            summaries.append({"id": vid, "status": "pending"})
    summaries.sort(key=lambda s: ORDER.index(s["id"]))

    bkg  = "#161b22"
    txt  = "#c9d1d9"
    grid = "#21262d"

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                          left=0.06, right=0.98, top=0.90, bottom=0.08)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_r2   = fig.add_subplot(gs[0, 1])
    ax_bar  = fig.add_subplot(gs[0, 2])
    ax_stat = fig.add_subplot(gs[1, :])

    for ax in [ax_loss, ax_r2, ax_bar, ax_stat]:
        ax.set_facecolor(bkg)
        ax.tick_params(colors=txt, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    # ── Loss and R² curves ─────────────────────────────────────
    for s in summaries:
        if "history" not in s:
            continue
        h  = s["history"]
        c  = COLORS.get(s["id"], "#ffffff")
        ep = list(range(1, s["n"] + 1))
        lbl = s["id"] + (" [done]" if s["status"] == "done" else " [running]")
        ax_loss.semilogy(ep, h["val_loss"], color=c, lw=1.8, label=lbl, alpha=0.9)
        ax_r2.plot(ep,   h["val_r2"],   color=c, lw=1.8, label=lbl, alpha=0.9)
        # Mark best epoch
        ax_r2.scatter([s["best_ep"]], [s["best_r2"]], color=c, s=50, zorder=5)

    ax_loss.set_xlabel("Epoch", color=txt, fontsize=9)
    ax_loss.set_ylabel("Val Loss (log scale)", color=txt, fontsize=9)
    ax_loss.set_title("Validation Loss per Variant", color=txt, fontsize=10, fontweight="bold")
    ax_loss.grid(True, alpha=0.3, color=grid, which="both")
    ax_loss.legend(fontsize=8, facecolor=bkg, edgecolor="#30363d", labelcolor=txt, loc="upper right")

    ax_r2.axhline(0.990, color="#ffa657", ls="--", lw=1.2, alpha=0.85, label="R²=0.99 target")
    ax_r2.set_xlabel("Epoch", color=txt, fontsize=9)
    ax_r2.set_ylabel("Val R²", color=txt, fontsize=9)
    ax_r2.set_title("Validation R² per Variant", color=txt, fontsize=10, fontweight="bold")
    ax_r2.set_ylim(0.85, 1.003)
    ax_r2.grid(True, alpha=0.3, color=grid)
    ax_r2.legend(fontsize=8, facecolor=bkg, edgecolor="#30363d", labelcolor=txt, loc="lower right")

    # ── Bar chart of best R² (variants with data only) ─────────
    active = [s for s in summaries if "best_r2" in s]
    if active:
        xs   = np.arange(len(active))
        bars = ax_bar.bar(xs, [s["best_r2"] for s in active],
                          color=[COLORS.get(s["id"], "#aaa") for s in active],
                          alpha=0.85, width=0.55, zorder=3)
        ax_bar.axhline(0.990, color="#ffa657", ls="--", lw=1.2, alpha=0.85, zorder=2)
        ax_bar.set_ylim(max(0.96, min(s["best_r2"] for s in active) - 0.005), 1.003)
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels([s["id"] for s in active], color=txt, fontsize=9)
        ax_bar.set_ylabel("Best Val R²", color=txt, fontsize=9)
        ax_bar.set_title("Best Val R² by Variant", color=txt, fontsize=10, fontweight="bold")
        ax_bar.grid(True, alpha=0.3, color=grid, axis="y", zorder=0)
        for bar, s in zip(bars, active):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0001,
                        f"{s['best_r2']:.4f}", ha="center", va="bottom",
                        color=txt, fontsize=8, fontweight="bold")
    else:
        ax_bar.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    color=txt, fontsize=12, transform=ax_bar.transAxes)
        ax_bar.set_title("Best Val R² by Variant", color=txt, fontsize=10, fontweight="bold")

    # ── Status table ───────────────────────────────────────────
    ax_stat.axis("off")
    col_labels = ["Variant", "Description", "Status", "Epochs Run",
                  "Best Epoch", "Best Val R²", "Best Val Loss"]
    rows = []
    for s in summaries:
        desc = VARIANT_LABELS.get(s["id"], "").replace("\n", " ")
        if s["status"] == "pending":
            rows.append([s["id"], desc, "PENDING", "—", "—", "—", "—"])
        else:
            icon = "[DONE]" if s["status"] == "done" else "[RUNNING]"
            rows.append([s["id"], desc, icon, str(s["n"]),
                         str(s["best_ep"]),
                         f"{s['best_r2']:.4f}", f"{s['best_loss']:.5f}"])

    tbl = ax_stat.table(cellText=rows, colLabels=col_labels,
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=txt, fontweight="bold")
        else:
            status = rows[r - 1][2]
            if   status == "[DONE]":    cell.set_facecolor("#162716")
            elif status == "[RUNNING]": cell.set_facecolor("#0e2233")
            else:                       cell.set_facecolor(bkg)
            cell.set_text_props(color=txt)

    # ── Title ──────────────────────────────────────────────────
    n_done    = sum(1 for s in summaries if s["status"] == "done")
    n_running = sum(1 for s in summaries if s["status"] == "running")
    n_pending = sum(1 for s in summaries if s["status"] == "pending")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Ablation Study Progress  |  "
        f"{n_done} done  /  {n_running} running  /  {n_pending} pending  |  {ts}",
        fontsize=12, color=txt, fontweight="bold", y=0.96,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # CLI summary
    print()
    for s in summaries:
        if "best_r2" in s:
            print(f"  {s['id']}  ep={s['n']:3d}  "
                  f"bestR2={s['best_r2']:.4f}  "
                  f"bestLoss={s['best_loss']:.5f}  "
                  f"[{s['status'].upper()}]")
        else:
            print(f"  {s['id']}  [PENDING]")


def main():
    ap = argparse.ArgumentParser(description="Visualize ablation study progress")
    ap.add_argument("--ablation_dir",
        default=r"c:\Users\user\Desktop\UTIC GNN\Physics-Informed ST-GNN\urban-thermal-gnn\04_training\ablation_ckpts")
    ap.add_argument("--out",
        default=r"c:\Users\user\Desktop\UTIC GNN\Physics-Informed ST-GNN\urban-thermal-gnn\04_training\ablation_ckpts\ablation_progress.png")
    ap.add_argument("--max_epochs", type=int, default=200)
    args = ap.parse_args()
    make_plot(Path(args.ablation_dir), Path(args.out), args.max_epochs)


if __name__ == "__main__":
    main()
