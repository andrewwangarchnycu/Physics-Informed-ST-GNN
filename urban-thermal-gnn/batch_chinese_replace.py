#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch replace ALL Chinese characters with English across Python files"""
import re
from pathlib import Path

# Comprehensive translation dictionary
translations = {
    # ===== TRAINING & VISUALIZATION =====
    "Model Training Verification Visualization": "Model Training Verification Visualization",
    "Run": "Run",
    "Training Convergence Curve": "Training Convergence Curve",
    "Loss Convergence": "Loss Convergence",
    "Loss Convergence": "Loss Convergence",
    "Deployment Threshold": "Deployment Threshold",
    "Best Epoch": "Best Epoch",
    "Best Epoch=": "Best Epoch=",
    "R² Convergence": "R² Convergence",
    "Best=": "Best=",
    "Predicted vs Ground Truth Scatter": "Predicted vs Ground Truth Scatter",
    "Prediction vs Ground Truth": "Prediction vs Ground Truth",
    "Temperature Distribution": "Temperature Distribution",
    "UTCI Values": "UTCI Values",
    "Wind Speed": "Wind Speed",
    "Solar Radiation": "Solar Radiation",
    "Relative Humidity": "Relative Humidity",
    "Shadow": "Shadow",
    "Thermal Stress": "Thermal Stress",
    "Thermal Stress Class Confusion Matrix": "Thermal Stress Class Confusion Matrix",
    "Graph Structure Feature Analysis": "Graph Structure Feature Analysis",
    "Model Training Convergence Analysis": "Model Training Convergence Analysis",
    "Dataset Statistics": "Dataset Statistics",
    "Build Global Climate Sequence": "Build Global Climate Sequence",
    "Build Global Environment & Time Feature Sequence": "Build Global Environment & Time Feature Sequence",

    # ===== DATA PROCESSING =====
    "Global Environment & Time Feature Construction": "Global Environment & Time Feature Construction",
    "Batch Collate Function": "Batch Collate Function",
    "Manual Training Loop": "Manual Training Loop",
    "Main Program": "Main Program",
    "Global Environment Sequence": "Global Environment Sequence",
    "Training History": "Training History",
    "Real-time Loss Curve": "Real-time Loss Curve",
    "Static Edges": "Static Edges",
    "Dynamic Edges": "Dynamic Edges",
    "Solar Altitude Angle Sequence": "Solar Altitude Angle Sequence",
    "R² Calculation": "R² Calculation",
    "Default Hyperparameters": "Default Hyperparameters",
    "Dataset": "Dataset",

    # ===== ERRORS & WARNINGS =====
    "lightning not installed, using manual training loop": "lightning not installed, using manual training loop",
    "epochs without improvement, stopping training": "epochs without improvement, stopping training",
    "Training completed  best val_loss=": "Training completed  best val_loss=",
    "Training history saved": "Training history saved",
    "Device": "Device",
    "Model Parameters": "Model Parameters",
    "Starting training": "Starting training",
    "Best model saved": "Best model saved",
    "Try PyTorch Lightning": "Try PyTorch Lightning",

    # ===== DOCUMENTATION =====
    "Perform forward propagation on single HeteroData and compute loss": "Perform forward propagation on single HeteroData and compute loss",
    "Process each HeteroData independently (no batching; graphs differ in size)": "Process each HeteroData independently (no batching; graphs differ in size)",
    "read from npz fields": "read from npz fields",
    "static": "static",
    "simplified: fixed": "simplified: fixed",
    "denormalize": "denormalize",
    "Build": "Build",
    "Sequence": "Sequence",
    "and": "and",
    "compute": "compute",

    # ===== MODEL ARCHITECTURE =====
    "Forward Propagation": "Forward Propagation",
    "Object Node Initial Embedding": "Object Node Initial Embedding",
    "Merge Dynamic and Static Edges": "Merge Dynamic and Static Edges",
    "Solar Radiation Node": "Solar Radiation Node",
    "Shadow Node": "Shadow Node",
    "Constraint Penalty": "Constraint Penalty",
    "Physics Constraints": "Physics Constraints",
    "Sensor Fusion": "Sensor Fusion",

    # ===== OPTIMIZATION =====
    "Non-dominated Sorting": "Non-dominated Sorting",
    "Crowding Distance": "Crowding Distance",
    "Crossover": "Crossover",
    "Mutation": "Mutation",
    "Fitness": "Fitness",

    # ===== DATA GENERATION =====
    "Batch Simulation": "Batch Simulation",
    "Physical Correction": "Physical Correction",
    "Shadow Determination": "Shadow Determination",
    "Ground Temperature Offset": "Ground Temperature Offset",
    "Spatial Filtering": "Spatial Filtering",
    "Hsinchu Area": "Hsinchu Area",
    "Sensor Grid": "Sensor Grid",
}

files_to_skip = {".git", "__pycache__", ".pytest_cache", "checkpoints_v2", "outputs", "tmp"}

def should_skip(path: Path) -> bool:
    """Check if path should be skipped"""
    path_str = str(path).lower()
    for skip in files_to_skip:
        if skip in path_str:
            return True
    return False

# Get all Python files
py_files = list(Path(".").rglob("*.py"))
py_files = [f for f in py_files if not should_skip(f)]

print(f"Found {len(py_files)} Python files to process")
print("=" * 70)

modified_count = 0
errors = []

for py_file in sorted(py_files):
    try:
        content = py_file.read_text(encoding="utf-8")
        original = content

        # Sort by length (longest first) to avoid partial replacements
        for zh, en in sorted(translations.items(), key=lambda x: len(x[0]), reverse=True):
            if zh in content:
                content = content.replace(zh, en)

        if content != original:
            py_file.write_text(content, encoding="utf-8")
            modified_count += 1
            print(f"[OK]  {py_file}")
        else:
            print(f"[SK]  {py_file}")

    except UnicodeDecodeError:
        errors.append((str(py_file), "UnicodeDecodeError"))
        print(f"[ERR] {py_file} - UnicodeDecodeError")
    except Exception as e:
        errors.append((str(py_file), str(e)))
        print(f"[ERR] {py_file} - {type(e).__name__}")

print("=" * 70)
print(f"\nSummary:")
print(f"  Total files processed: {len(py_files)}")
print(f"  Modified: {modified_count}")
print(f"  Errors: {len(errors)}")

if errors:
    print(f"\nErrors encountered:")
    for fpath, err in errors:
        print(f"  - {fpath}: {err}")
