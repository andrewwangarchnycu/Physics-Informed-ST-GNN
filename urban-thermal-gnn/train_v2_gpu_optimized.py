#!/usr/bin/env python3
"""
train_v2_gpu_optimized.py
Optimized training script with proper GPU management and logging
"""

import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).parent / "03_model"))
sys.path.insert(0, str(Path(__file__).parent / "04_training"))

import torch
import time

print("\n" + "="*70)
print("  PHASE 5 TRAINING - GPU OPTIMIZED")
print("="*70)

# Show GPU info
print(f"\nGPU Configuration:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Import training
from train import main as train_main

data_dir = Path(__file__).parent / "01_data_generation" / "outputs" / "raw_simulations"

print(f"\nStarting Training...")
print(f"  Data: {data_dir / 'ground_truth_v2.h5'}")
print(f"  Checkpoint: checkpoints_v2/best_model.pt")
print(f"  Device: cuda")
print("="*70 + "\n")

try:
    train_main(
        cfg_path=str(Path(__file__).parent / "00_config" / "urbangraph_params.yaml"),
        h5_path=str(data_dir / "ground_truth_v2.h5"),
        scenario_pkl=str(data_dir / "scenarios.pkl"),
        epw_pkl=str(data_dir / "epw_data.pkl"),
        out_dir="checkpoints_v2",
        device_str="cuda",
        live_plot=False,
    )
    print("\n[OK] Training completed successfully")

except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()

    # Try to show checkpoint status
    try:
        checkpoint = torch.load("checkpoints_v2/best_model.pt", weights_only=False)
        print(f"\n[INFO] Last checkpoint: Epoch {checkpoint.get('epoch', 'N/A')}")
    except:
        pass

    sys.exit(1)
