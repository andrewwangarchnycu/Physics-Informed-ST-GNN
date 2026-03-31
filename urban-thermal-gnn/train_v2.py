#!/usr/bin/env python3
"""
train_v2.py
════════════════════════════════════════════════════════════════
Phase 5: Train Physics-Informed ST-GNN with v2 high-resolution data

Features:
  - 2x spatial resolution (1.0m grid = ~6,241 nodes per scenario)
  - Real weather calibration (CWB + MOENV IoT)
  - Real-time progress monitoring
  - GPU-accelerated training

Usage:
  cd 04_training
  python ../train_v2.py [--device cuda --epochs 250 --batch_size 1]
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "02_graph_construction"))
sys.path.insert(0, str(root_dir / "03_model"))
sys.path.insert(0, str(root_dir / "04_training"))

import torch
from train import main as train_main


def main():
    """Train on v2 high-resolution data"""
    parser = argparse.ArgumentParser(
        description="Train on v2 high-resolution thermal data (1.0m resolution)"
    )

    # V2 data paths
    data_dir = root_dir / "01_data_generation" / "outputs" / "raw_simulations"
    parser.add_argument(
        "--h5",
        default=str(data_dir / "ground_truth_v2.h5"),
        help="Path to v2 ground truth HDF5 file"
    )
    parser.add_argument(
        "--scenarios",
        default=str(data_dir / "scenarios.pkl"),
        help="Path to scenarios pickle file"
    )
    parser.add_argument(
        "--epw",
        default=str(data_dir / "epw_data.pkl"),
        help="Path to EPW data pickle file"
    )

    # Training configuration
    parser.add_argument(
        "--config",
        default=str(root_dir / "00_config" / "urbangraph_params.yaml"),
        help="Path to model configuration YAML"
    )
    parser.add_argument(
        "--out",
        default="checkpoints_v2",
        help="Output checkpoint directory"
    )

    # Training hyperparameters (v2 optimized)
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (graphs are too large to batch, keep at 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (5e-4 for v2)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--live-plot",
        dest="live_plot", action="store_true", default=True,
        help="Show real-time loss curve window (default: on)"
    )
    parser.add_argument(
        "--no-live-plot",
        dest="live_plot", action="store_false",
        help="Disable live curve window (headless / SSH mode)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Phase 5: Physics-Informed ST-GNN Training (v2 High-Resolution)")
    print("=" * 70)
    print(f"  Data source: {Path(args.h5).name}")
    print(f"  Resolution: 1.0m grid (~6,241 nodes per scenario)")
    print(f"  Device: {args.device}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 70 + "\n")

    # Run training with v2 parameters
    train_main(
        cfg_path=args.config,
        h5_path=args.h5,
        scenario_pkl=args.scenarios,
        epw_pkl=args.epw,
        out_dir=args.out,
        device_str=args.device,
        live_plot=args.live_plot,
    )


if __name__ == "__main__":
    main()
