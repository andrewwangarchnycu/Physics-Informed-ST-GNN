#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_training_v2.py
Simple wrapper to run training with v2 data, avoiding Unicode issues
"""

import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "02_graph_construction"))
sys.path.insert(0, str(root_dir / "03_model"))
sys.path.insert(0, str(root_dir / "04_training"))

print("\n" + "=" * 70)
print("  Phase 5: Physics-Informed ST-GNN Training (v2 High-Resolution)")
print("=" * 70)

# Import and run training
from train import main as train_main

data_dir = root_dir / "01_data_generation" / "outputs" / "raw_simulations"

try:
    train_main(
        cfg_path=str(root_dir / "00_config" / "urbangraph_params.yaml"),
        h5_path=str(data_dir / "ground_truth_v2.h5"),
        scenario_pkl=str(data_dir / "scenarios.pkl"),
        epw_pkl=str(data_dir / "epw_data.pkl"),
        out_dir="checkpoints_v2",
        device_str="cuda",
    )
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
