#!/usr/bin/env python3
"""
monitor_training_v2.py
Continuously monitor training progress and trigger Phase 6 when complete
"""

import torch
import time
from pathlib import Path
import json

def get_training_status():
    """Get current training status from checkpoint"""
    checkpoint_path = Path("checkpoints_v2/best_model.pt")
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'val_r2': checkpoint.get('val_r2', 0),
            'timestamp': time.time()
        }
    except:
        return None

def main():
    print("\n" + "="*70)
    print("  TRAINING PROGRESS MONITOR")
    print("  Monitoring checkpoints_v2/best_model.pt")
    print("="*70 + "\n")

    last_epoch = 0
    max_epochs = 250

    while True:
        status = get_training_status()

        if status is None:
            print("[WAITING] No checkpoint found yet...")
            time.sleep(5)
            continue

        current_epoch = status['epoch']
        progress = (current_epoch / max_epochs) * 100

        # Print progress update if epoch changed
        if current_epoch > last_epoch:
            print(f"[EPOCH {current_epoch:3d}/{max_epochs}] "
                  f"Progress: {progress:5.1f}% | "
                  f"Val Loss: {status['val_loss']:.6f} | "
                  f"Val R²: {status['val_r2']:.6f}")
            last_epoch = current_epoch

        # Check if training is complete
        if current_epoch >= max_epochs:
            print("\n" + "="*70)
            print("  TRAINING COMPLETE!")
            print(f"  Final Epoch: {current_epoch}")
            print(f"  Best Val Loss: {status['val_loss']:.6f}")
            print(f"  Best Val R-squared: {status['val_r2']:.6f}")
            print("="*70)
            print("\n  Ready for Phase 6: Evaluation & Visualization")
            break

        # Check if training is taking too long (more than 10 hours)
        if time.time() - status['timestamp'] > 36000:
            print("\n[WARNING] Training taking longer than expected")

        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()
