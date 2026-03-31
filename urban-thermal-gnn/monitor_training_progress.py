#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_training_progress.py
Background training progress monitor - generates loss/R² plots every N epochs

Features:
  - Runs in background (non-blocking)
  - Reads training_history.json periodically
  - Generates 4 visualizations:
    1. Loss convergence curve (train & val)
    2. Validation R² progression
    3. Learning rate schedule
    4. Combined 12-panel detailed analysis
  - Saves progress images to checkpoints_v2/monitoring/
  - Shows epoch progress and ETA
  - Detects when training completes (Epoch 250 or early stopping)
"""

import json
import time
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TrainingMonitor:
    def __init__(self,
                 history_path: str = "checkpoints_v2/training_history.json",
                 output_dir: str = "checkpoints_v2/monitoring",
                 max_epochs: int = 250,
                 check_interval: int = 30):
        """
        Initialize monitor

        Parameters
        ----------
        history_path : Path to training_history.json
        output_dir : Directory to save monitoring plots
        max_epochs : Expected total epochs
        check_interval : Seconds between checks
        """
        self.history_path = Path(history_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = max_epochs
        self.check_interval = check_interval
        self.last_epoch_count = 0
        self.start_time = datetime.now()

        print(f"\n{'='*70}")
        print(f"Training Progress Monitor Started")
        print(f"{'='*70}")
        print(f"History file: {self.history_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Check interval: {check_interval} seconds")
        print(f"{'='*70}\n")

    def load_history(self):
        """Load training history from JSON"""
        if not self.history_path.exists():
            return None

        try:
            with open(self.history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            return None

    def estimate_time_remaining(self, current_epoch, epochs_data):
        """Estimate remaining training time"""
        if current_epoch < 2:
            return None

        # Calculate average time per epoch
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_time_per_epoch = elapsed / current_epoch

        remaining_epochs = self.max_epochs - current_epoch
        eta_seconds = remaining_epochs * avg_time_per_epoch
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)

        return {
            "avg_time_per_epoch": avg_time_per_epoch,
            "remaining_epochs": remaining_epochs,
            "eta_seconds": eta_seconds,
            "eta_time": eta_time
        }

    def generate_simple_plot(self, history):
        """Generate 3-panel simple plot"""
        n_epochs = len(history["train_loss"])
        epochs = list(range(1, n_epochs + 1))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor("#0d1117")

        # Loss curve
        ax = axes[0]
        ax.set_facecolor("#161b22")
        ax.plot(epochs, history["train_loss"], label="Train Loss",
               color="#58a6ff", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val Loss",
               color="#ff7b72", linewidth=2)
        best_idx = np.argmin(history["val_loss"])
        ax.scatter([epochs[best_idx]], [history["val_loss"][best_idx]],
                  color="#ffa657", s=100, zorder=5,
                  label=f"Best: {history['val_loss'][best_idx]:.5f}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("Loss (log scale)", color="#c9d1d9", fontsize=10)
        ax.set_title("Loss Convergence", color="#c9d1d9", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, color="#21262d")
        ax.legend(fontsize=9, loc="upper right", facecolor="#161b22",
                 edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.tick_params(colors="#c9d1d9", labelsize=9)

        # Validation R²
        ax = axes[1]
        ax.set_facecolor("#161b22")
        ax.plot(epochs, history["val_r2"], color="#56d364", linewidth=2)
        ax.axhline(0.990, color="#ffa657", linestyle="--", linewidth=1.5,
                  alpha=0.8, label="Target R²=0.990")
        best_r2 = max(history["val_r2"])
        best_r2_epoch = history["val_r2"].index(best_r2) + 1
        ax.scatter([best_r2_epoch], [best_r2], color="#ffa657", s=100,
                  zorder=5, label=f"Best: {best_r2:.6f}")
        ax.fill_between(epochs, history["val_r2"], 0.990,
                        where=(np.array(history["val_r2"]) >= 0.990),
                        color="#56d364", alpha=0.15)
        ax.set_ylim(min(0.95, min(history["val_r2"]) - 0.01), 1.001)
        ax.set_xlabel("Epoch", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("R²", color="#c9d1d9", fontsize=10)
        ax.set_title(f"Validation R² (Current: {history['val_r2'][-1]:.6f})",
                    color="#c9d1d9", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, color="#21262d")
        ax.legend(fontsize=9, loc="lower right", facecolor="#161b22",
                 edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.tick_params(colors="#c9d1d9", labelsize=9)

        # Learning rate
        ax = axes[2]
        ax.set_facecolor("#161b22")
        ax.semilogy(epochs, history["lr"], color="#bc8cff", linewidth=2)
        ax.set_xlabel("Epoch", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("Learning Rate (log scale)", color="#c9d1d9", fontsize=10)
        ax.set_title("Learning Rate Schedule", color="#c9d1d9", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, color="#21262d", which="both")
        ax.tick_params(colors="#c9d1d9", labelsize=9)

        # Mark LR decay points
        if len(history["lr"]) > 1:
            for i in range(1, len(history["lr"])):
                if history["lr"][i] < history["lr"][i-1] * 0.6:
                    ax.axvline(epochs[i], color="#ffa657", linestyle=":",
                              linewidth=1, alpha=0.7)

        fig.suptitle(f"Phase 5 Training Progress - Epoch {n_epochs}/{self.max_epochs}",
                    fontsize=12, color="#c9d1d9", fontweight="bold", y=1.02)
        fig.tight_layout()

        return fig

    def generate_detailed_plot(self, history):
        """Generate 12-panel detailed analysis"""
        n_epochs = len(history["train_loss"])
        epochs = list(range(1, n_epochs + 1))

        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        bkg_color = "#161b22"
        text_color = "#c9d1d9"
        grid_color = "#21262d"

        # 1. Loss (linear)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor(bkg_color)
        ax.plot(epochs, history["train_loss"], label="Train", color="#58a6ff", lw=1.8)
        ax.plot(epochs, history["val_loss"], label="Val", color="#ff7b72", lw=1.8)
        ax.set_ylabel("Loss", color=text_color, fontsize=9)
        ax.set_title("Loss (Linear)", color=text_color, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color)
        ax.legend(fontsize=8, facecolor=bkg_color, labelcolor=text_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # 2. Loss (log)
        ax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor(bkg_color)
        ax.semilogy(epochs, history["train_loss"], label="Train", color="#58a6ff", lw=1.8)
        ax.semilogy(epochs, history["val_loss"], label="Val", color="#ff7b72", lw=1.8)
        ax.set_ylabel("Loss (log)", color=text_color, fontsize=9)
        ax.set_title("Loss (Log Scale)", color=text_color, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color, which="both")
        ax.legend(fontsize=8, facecolor=bkg_color, labelcolor=text_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # 3. R²
        ax = fig.add_subplot(gs[0, 2])
        ax.set_facecolor(bkg_color)
        ax.plot(epochs, history["val_r2"], color="#56d364", lw=2)
        ax.axhline(0.990, color="#ffa657", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_ylabel("R²", color=text_color, fontsize=9)
        ax.set_title(f"Validation R² (Current: {history['val_r2'][-1]:.6f})",
                    color=text_color, fontsize=10, fontweight="bold")
        ax.set_ylim([min(0.95, min(history["val_r2"])-0.01), 1.001])
        ax.grid(True, alpha=0.3, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # 4. Learning rate
        ax = fig.add_subplot(gs[0, 3])
        ax.set_facecolor(bkg_color)
        ax.semilogy(epochs, history["lr"], color="#bc8cff", lw=2)
        ax.set_ylabel("LR (log)", color=text_color, fontsize=9)
        ax.set_title("Learning Rate Schedule", color=text_color, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color, which="both")
        ax.tick_params(colors=text_color, labelsize=8)

        # 5-8. Metrics trends (last 50 epochs)
        n_recent = min(50, len(epochs))
        recent_epochs = epochs[-n_recent:]
        recent_train_loss = history["train_loss"][-n_recent:]
        recent_val_loss = history["val_loss"][-n_recent:]
        recent_r2 = history["val_r2"][-n_recent:]
        recent_lr = history["lr"][-n_recent:]

        # Train loss trend
        ax = fig.add_subplot(gs[1, 0])
        ax.set_facecolor(bkg_color)
        ax.plot(recent_epochs, recent_train_loss, color="#58a6ff", lw=2, marker="o", ms=3)
        ax.set_ylabel("Train Loss", color=text_color, fontsize=9)
        ax.set_title(f"Train Loss (Last {n_recent} epochs)", color=text_color,
                    fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # Val loss trend
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor(bkg_color)
        ax.plot(recent_epochs, recent_val_loss, color="#ff7b72", lw=2, marker="o", ms=3)
        ax.set_ylabel("Val Loss", color=text_color, fontsize=9)
        ax.set_title(f"Val Loss (Last {n_recent} epochs)", color=text_color,
                    fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # R² trend
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor(bkg_color)
        ax.plot(recent_epochs, recent_r2, color="#56d364", lw=2, marker="o", ms=3)
        ax.axhline(0.990, color="#ffa657", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_ylabel("R²", color=text_color, fontsize=9)
        ax.set_title(f"R² Trend (Last {n_recent} epochs)", color=text_color,
                    fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8)

        # LR trend
        ax = fig.add_subplot(gs[1, 3])
        ax.set_facecolor(bkg_color)
        ax.semilogy(recent_epochs, recent_lr, color="#bc8cff", lw=2, marker="o", ms=3)
        ax.set_ylabel("LR", color=text_color, fontsize=9)
        ax.set_title(f"LR Trend (Last {n_recent} epochs)", color=text_color,
                    fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, color=grid_color, which="both")
        ax.tick_params(colors=text_color, labelsize=8)

        # 9-12. Statistics text panels
        # Overall stats
        ax = fig.add_subplot(gs[2, 0])
        ax.axis("off")
        stats_text = f"""OVERALL STATISTICS
Epochs Completed: {n_epochs}/250
Progress: {(n_epochs/250)*100:.1f}%

Best Val Loss: {min(history['val_loss']):.6f}
  @ Epoch {history['val_loss'].index(min(history['val_loss']))+1}

Best Val R²: {max(history['val_r2']):.6f}
  @ Epoch {history['val_r2'].index(max(history['val_r2']))+1}

Current LR: {history['lr'][-1]:.2e}
"""
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               color=text_color, bbox=dict(boxstyle="round",
               facecolor=bkg_color, edgecolor="#30363d"))

        # Latest epoch stats
        ax = fig.add_subplot(gs[2, 1])
        ax.axis("off")
        latest_text = f"""LATEST EPOCH ({n_epochs})
Train Loss: {history['train_loss'][-1]:.6f}
Val Loss: {history['val_loss'][-1]:.6f}
Val R²: {history['val_r2'][-1]:.6f}
LR: {history['lr'][-1]:.2e}

Loss Improvement (vs Epoch 1):
  Train: {((history['train_loss'][0] - history['train_loss'][-1])/history['train_loss'][0]*100):.1f}%
  Val: {((history['val_loss'][0] - history['val_loss'][-1])/history['val_loss'][0]*100):.1f}%
"""
        ax.text(0.05, 0.95, latest_text, transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               color=text_color, bbox=dict(boxstyle="round",
               facecolor=bkg_color, edgecolor="#30363d"))

        # Improvement trend (last 10 epochs)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        last_10 = min(10, len(epochs))
        improvement_text = f"""RECENT IMPROVEMENT (Last {last_10} epochs)
Epoch Range: {epochs[-last_10]} - {epochs[-1]}

Loss Change:
  Train: {((history['train_loss'][-last_10] - history['train_loss'][-1])/history['train_loss'][-last_10]*100):.2f}%
  Val: {((history['val_loss'][-last_10] - history['val_loss'][-1])/history['val_loss'][-last_10]*100):.2f}%

R² Change: {((history['val_r2'][-1] - history['val_r2'][-last_10])/abs(history['val_r2'][-last_10])*100):.4f}%

Avg Epoch Time:
  {(datetime.now() - self.start_time).total_seconds()/n_epochs:.1f} sec
"""
        ax.text(0.05, 0.95, improvement_text, transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               color=text_color, bbox=dict(boxstyle="round",
               facecolor=bkg_color, edgecolor="#30363d"))

        # Convergence analysis
        ax = fig.add_subplot(gs[2, 3])
        ax.axis("off")
        convergence_text = f"""CONVERGENCE STATUS
Target R²: 0.990
Current R²: {history['val_r2'][-1]:.6f}
Gap: {(0.990 - history['val_r2'][-1])*1e6:.2f} ppm

Above Target: {sum(1 for r in history['val_r2'] if r >= 0.990)}/{n_epochs} epochs

Early Stopping Check:
  Patience: 25 epochs
  (No improvement counter)

Est. Completion:
  Best case: Epoch 250
  (Subject to early stopping)
"""
        ax.text(0.05, 0.95, convergence_text, transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               color=text_color, bbox=dict(boxstyle="round",
               facecolor=bkg_color, edgecolor="#30363d"))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f"Phase 5 Detailed Training Analysis - {timestamp}",
                    fontsize=13, color=text_color, fontweight="bold", y=0.995)

        return fig

    def monitor_loop(self):
        """Main monitoring loop"""
        iteration = 0

        try:
            while True:
                iteration += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                history = self.load_history()
                if history is None:
                    print(f"[{timestamp}] Waiting for training to start...")
                    time.sleep(self.check_interval)
                    continue

                n_epochs = len(history["train_loss"])

                # Estimate time remaining
                eta_info = self.estimate_time_remaining(n_epochs, history)

                if eta_info:
                    hours = eta_info["eta_seconds"] / 3600
                    eta_str = eta_info["eta_time"].strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Epoch {n_epochs}/250 | "
                          f"R²={history['val_r2'][-1]:.6f} | "
                          f"Loss={history['val_loss'][-1]:.6f} | "
                          f"ETA: {hours:.1f}h ({eta_str})")
                else:
                    print(f"[{timestamp}] Epoch {n_epochs}/250 | "
                          f"R²={history['val_r2'][-1]:.6f} | "
                          f"Loss={history['val_loss'][-1]:.6f}")

                # Generate plots every 10 iterations or when epoch increases
                if iteration % 10 == 0 or n_epochs != self.last_epoch_count:
                    try:
                        # Simple 3-panel plot
                        fig_simple = self.generate_simple_plot(history)
                        path_simple = self.output_dir / f"training_progress_ep{n_epochs:04d}.png"
                        fig_simple.savefig(path_simple, dpi=120, bbox_inches="tight",
                                         facecolor=fig_simple.get_facecolor())
                        plt.close(fig_simple)

                        # Detailed 12-panel plot (every 5 iterations)
                        if iteration % 5 == 0:
                            fig_detailed = self.generate_detailed_plot(history)
                            path_detailed = self.output_dir / f"training_detailed_ep{n_epochs:04d}.png"
                            fig_detailed.savefig(path_detailed, dpi=120, bbox_inches="tight",
                                               facecolor=fig_detailed.get_facecolor())
                            plt.close(fig_detailed)

                        self.last_epoch_count = n_epochs
                    except Exception as e:
                        print(f"  Error generating plots: {e}")

                # Check if training completed
                if n_epochs >= self.max_epochs:
                    print(f"\n{'='*70}")
                    print(f"Training Completed at Epoch {n_epochs}")
                    print(f"Final Val R²: {history['val_r2'][-1]:.6f}")
                    print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
                    print(f"{'='*70}\n")
                    break

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitor error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run monitoring in foreground"""
    monitor = TrainingMonitor(
        history_path="checkpoints_v2/training_history.json",
        output_dir="checkpoints_v2/monitoring",
        max_epochs=250,
        check_interval=30  # Check every 30 seconds
    )
    monitor.monitor_loop()


if __name__ == "__main__":
    main()
