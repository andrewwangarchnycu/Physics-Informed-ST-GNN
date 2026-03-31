#!/usr/bin/env python3
"""
Quick status check for training progress
Usage: python quick_check.py
"""

import torch
from pathlib import Path

try:
    checkpoint = torch.load("checkpoints_v2/best_model.pt", weights_only=False)
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    val_r2 = checkpoint['val_r2']

    progress = (epoch / 250) * 100
    remaining = 250 - epoch

    print(f"\nTraining Status (Epoch {epoch}/250):")
    print(f"  Progress: {progress:.1f}%")
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"  Val R²: {val_r2:.6f}")
    print(f"  Remaining: {remaining} epochs")
    print(f"  Est. Time: ~{remaining*1.5:.0f} min\n")

except Exception as e:
    print(f"Error: {e}")
