"""
04_training/live_loss_plot.py
================================================================================
PIN-ST-GNN  —  Real-time training loss curve visualizer

Features:
  - Updates the plot window after every epoch, non-blocking
  - Three sub-plots: Loss curve (log scale) / Validation R2 / Learning Rate
  - Dark theme; highlights best val_loss point and R2 milestones
  - Auto-saves a PNG snapshot every N epochs; saves final high-res image at end
  - Falls back to silent file-save mode when no display is available (headless)

Standalone usage:
  plotter = LiveLossPlotter(max_epochs=250, save_dir="checkpoints_v2")
  for epoch in range(1, max_epochs + 1):
      ...
      plotter.update(epoch, tr_loss, va_loss, va_r2, lr)
  plotter.save()
  plotter.close()
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# Try interactive backends; fall back to Agg (headless / file-save only).
# ---------------------------------------------------------------------------
import matplotlib
_TRIED_BACKENDS = ["TkAgg", "Qt5Agg", "QtAgg", "WXAgg"]


def _setup_backend() -> bool:
    """Try to set an interactive backend. Returns True if successful."""
    current = matplotlib.get_backend()
    if current.lower() not in ("agg", ""):
        return True                          # already interactive

    for backend in _TRIED_BACKENDS:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as _plt
            _plt.figure()
            _plt.close("all")
            return True
        except Exception:
            continue

    matplotlib.use("Agg")                   # headless fallback
    return False


_INTERACTIVE = _setup_backend()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Color theme
# ---------------------------------------------------------------------------
_BG_DARK   = "#0d1117"   # figure background
_BG_AX     = "#161b22"   # axes background
_FG        = "#c9d1d9"   # text / tick color
_GRID_COL  = "#21262d"   # grid lines
_SPINE_COL = "#30363d"   # axes border
_C_TRAIN   = "#58a6ff"   # Train loss line
_C_VAL     = "#ff7b72"   # Val loss line
_C_R2      = "#56d364"   # R2 line
_C_LR      = "#bc8cff"   # LR line
_C_BEST    = "#ffa657"   # best-point marker


class LiveLossPlotter:
    """
    Real-time loss / R2 / LR curve visualizer during training.

    Parameters
    ----------
    max_epochs : int
        Total training epochs (sets the x-axis range).
    save_dir : str | Path
        Directory where snapshot PNGs are saved (usually the checkpoints folder).
    title : str
        Window title shown in the figure header.
    autosave_every : int
        Save a snapshot every N epochs (0 = disabled).
    """

    def __init__(
        self,
        max_epochs:     int = 200,
        save_dir:       str = ".",
        title:          str = "PIN-ST-GNN  Live Training Monitor",
        autosave_every: int = 10,
    ):
        self.max_epochs     = max_epochs
        self.save_dir       = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.autosave_every = autosave_every
        self.interactive    = _INTERACTIVE

        # Data buffers
        self.epochs:     list[int]   = []
        self.train_loss: list[float] = []
        self.val_loss:   list[float] = []
        self.val_r2:     list[float] = []
        self.lr_hist:    list[float] = []

        self._build_figure(title)

        if self.interactive:
            plt.show(block=False)
            plt.pause(0.15)
        else:
            print("[LiveLossPlotter] No interactive backend found — "
                  "running in silent file-save mode.")

    # -----------------------------------------------------------------------
    # Figure construction
    # -----------------------------------------------------------------------
    def _build_figure(self, title: str):
        self.fig = plt.figure(figsize=(14, 5), facecolor=_BG_DARK)
        if self.interactive:
            try:
                self.fig.canvas.manager.set_window_title(title)
            except Exception:
                pass

        gs = gridspec.GridSpec(
            1, 3, figure=self.fig,
            left=0.07, right=0.97, top=0.85, bottom=0.13,
            wspace=0.38,
        )
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_r2   = self.fig.add_subplot(gs[0, 1])
        self.ax_lr   = self.fig.add_subplot(gs[0, 2])

        for ax in (self.ax_loss, self.ax_r2, self.ax_lr):
            self._apply_dark_style(ax)

        self.title_obj = self.fig.text(
            0.5, 0.95, title,
            ha="center", va="top",
            fontsize=12, color=_FG, fontweight="bold",
            transform=self.fig.transFigure,
        )

    def _apply_dark_style(self, ax):
        ax.set_facecolor(_BG_AX)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE_COL)
        ax.grid(True, color=_GRID_COL, linewidth=0.6, linestyle="--", alpha=0.8)

    # -----------------------------------------------------------------------
    # Main update call
    # -----------------------------------------------------------------------
    def update(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        val_r2:     float,
        lr:         float,
    ):
        """
        Call at the end of every epoch to append new data and redraw.

        Parameters
        ----------
        epoch      : current epoch number (1-based)
        train_loss : mean training loss for this epoch
        val_loss   : mean validation loss for this epoch
        val_r2     : validation R2 score for this epoch
        lr         : current learning rate
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_r2.append(val_r2)
        self.lr_hist.append(lr)

        self._draw_loss()
        self._draw_r2()
        self._draw_lr()
        self._update_title(epoch, val_loss, val_r2)

        if self.interactive:
            self.fig.canvas.draw_idle()
            plt.pause(0.05)

        if self.autosave_every > 0 and epoch % self.autosave_every == 0:
            self._autosave(epoch)

    # -----------------------------------------------------------------------
    # Sub-plot drawing
    # -----------------------------------------------------------------------
    def _draw_loss(self):
        ax = self.ax_loss
        ax.cla()
        self._apply_dark_style(ax)

        ep = self.epochs
        ax.plot(ep, self.train_loss, color=_C_TRAIN, lw=1.8,
                label="Train", zorder=3)
        ax.plot(ep, self.val_loss,   color=_C_VAL,   lw=1.8,
                label="Val",   zorder=3)

        # Highlight the best val_loss point
        best_i = int(np.argmin(self.val_loss))
        ax.scatter(
            [ep[best_i]], [self.val_loss[best_i]],
            color=_C_BEST, s=70, zorder=5,
            label=f"Best = {self.val_loss[best_i]:.5f}",
        )

        if len(ep) > 3 and min(self.train_loss) > 0:
            ax.set_yscale("log")

        ax.set_xlim(1, max(self.max_epochs, ep[-1]))
        ax.set_xlabel("Epoch",       color=_FG, fontsize=9)
        ax.set_ylabel("MSE Loss",    color=_FG, fontsize=9)
        ax.set_title("Loss Curve",   color=_FG, fontsize=10)
        ax.legend(
            fontsize=7, facecolor=_BG_AX,
            labelcolor=_FG, framealpha=0.8,
            edgecolor=_SPINE_COL,
        )

    def _draw_r2(self):
        ax = self.ax_r2
        ax.cla()
        self._apply_dark_style(ax)

        ep     = self.epochs
        r2_arr = np.array(self.val_r2)
        ax.plot(ep, r2_arr, color=_C_R2, lw=1.8, zorder=3)

        # Milestone reference lines
        ax.axhline(0.90, color="#ffa657", lw=1.0, ls="--",
                   label="Deploy threshold 0.90", alpha=0.8)
        ax.axhline(0.99, color="#f0e040", lw=1.0, ls="--",
                   label="Excellent 0.99",         alpha=0.8)

        # Shade above / below deploy threshold
        ax.fill_between(ep, r2_arr, 0.90,
                        where=(r2_arr >= 0.90),
                        color=_C_R2,  alpha=0.12, zorder=2)
        ax.fill_between(ep, r2_arr, 0.90,
                        where=(r2_arr < 0.90),
                        color=_C_VAL, alpha=0.12, zorder=2)

        best_r2 = float(np.max(r2_arr))
        ax.set_ylim(max(0.0, float(np.min(r2_arr)) - 0.03), 1.005)
        ax.set_xlim(1, max(self.max_epochs, ep[-1]))
        ax.set_xlabel("Epoch", color=_FG, fontsize=9)
        ax.set_ylabel("R²",    color=_FG, fontsize=9)
        ax.set_title(f"Validation R²  (best {best_r2:.4f})",
                     color=_FG, fontsize=10)
        ax.legend(
            fontsize=7, facecolor=_BG_AX,
            labelcolor=_FG, framealpha=0.8,
            edgecolor=_SPINE_COL,
        )

    def _draw_lr(self):
        ax = self.ax_lr
        ax.cla()
        self._apply_dark_style(ax)

        ep = self.epochs
        ax.semilogy(ep, self.lr_hist, color=_C_LR, lw=1.8)

        # Mark each LR decay step with a vertical dotted line
        if len(self.lr_hist) > 1:
            for i in range(1, len(self.lr_hist)):
                if self.lr_hist[i] < self.lr_hist[i - 1] * 0.6:
                    ax.axvline(ep[i], color=_C_BEST, lw=0.8,
                               ls=":", alpha=0.7)

        ax.set_xlim(1, max(self.max_epochs, ep[-1]))
        ax.set_xlabel("Epoch",         color=_FG, fontsize=9)
        ax.set_ylabel("Learning Rate", color=_FG, fontsize=9)
        ax.set_title("Learning Rate",  color=_FG, fontsize=10)

    def _update_title(self, epoch: int, val_loss: float, val_r2: float):
        progress = epoch / self.max_epochs * 100
        self.title_obj.set_text(
            f"PIN-ST-GNN  Live Training Monitor  ·  "
            f"Epoch {epoch}/{self.max_epochs}  ({progress:.1f}%)  "
            f"|  val_loss = {val_loss:.5f}  ·  R² = {val_r2:.4f}"
        )

    # -----------------------------------------------------------------------
    # Save / close
    # -----------------------------------------------------------------------
    def _autosave(self, epoch: int):
        snap_dir = self.save_dir / "live_plots"
        snap_dir.mkdir(exist_ok=True)
        path = snap_dir / f"live_ep{epoch:04d}.png"
        self.fig.savefig(path, dpi=120, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())

    def save(self, filename: str = "training_curve_final.png"):
        """Save a high-resolution final PNG at the end of training."""
        path = self.save_dir / filename
        self.fig.savefig(path, dpi=180, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"[LiveLossPlotter] Final curve saved -> {path}")
        return path

    def close(self):
        """Close the figure window and release resources."""
        plt.close(self.fig)

    # -----------------------------------------------------------------------
    # Pre-populate from saved history (resume training)
    # -----------------------------------------------------------------------
    def load_history(self, history: dict):
        """
        Pre-fill curve with data from a training_history.json dict.
        Useful when resuming training from a checkpoint.

        Parameters
        ----------
        history : dict with keys train_loss, val_loss, val_r2, lr
        """
        tl = history.get("train_loss", [])
        vl = history.get("val_loss",   [])
        vr = history.get("val_r2",     [])
        lr = history.get("lr",         [])
        n  = min(len(tl), len(vl), len(vr), len(lr))
        if n == 0:
            return
        self.epochs     = list(range(1, n + 1))
        self.train_loss = list(tl[:n])
        self.val_loss   = list(vl[:n])
        self.val_r2     = list(vr[:n])
        self.lr_hist    = list(lr[:n])

        self._draw_loss()
        self._draw_r2()
        self._draw_lr()
        self._update_title(n, self.val_loss[-1], self.val_r2[-1])
        if self.interactive:
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
        print(f"[LiveLossPlotter] Loaded {n} epochs of history.")
