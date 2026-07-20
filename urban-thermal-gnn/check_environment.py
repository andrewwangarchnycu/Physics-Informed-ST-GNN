"""
check_environment.py
=====================
Cross-machine environment readiness check for urban-thermal-gnn ML training.
Run this after following ../ENVIRONMENT_SETUP.md, before starting any training
job on a new machine. Checks three independent things and prints a PASS/FAIL
report for each: (1) Python version, (2) required packages + GPU visibility,
(3) presence of the large data/checkpoint files that git does NOT track.

Usage:
    python check_environment.py
    python check_environment.py --h5 ground_truth_v3.h5 --ckpt checkpoints_v3
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # urban-thermal-gnn/
REPO_ROOT = ROOT.parent

PASS, FAIL, WARN = "[ PASS ]", "[ FAIL ]", "[ WARN ]"
results = []  # (status, message)


def check(label, ok, detail="", warn_only=False):
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((status, f"{label}{'  -- ' + detail if detail else ''}"))


# ── 1. Python version ────────────────────────────────────────────────────
py_ok = sys.version_info[:2] == (3, 11)
check("Python 3.11.x", py_ok,
      detail=f"found {sys.version.split()[0]}" + ("" if py_ok else " (recommended: 3.11.x)"),
      warn_only=True)

# ── 2. Required packages ─────────────────────────────────────────────────
CORE = ["numpy", "pandas", "scipy", "sklearn", "h5py", "matplotlib", "yaml", "torch"]
GIS = ["shapely", "pyproj", "rasterio", "contextily", "geopandas", "overpy"]
COMFORT = ["pythermalcomfort"]
DEPLOY = ["fastapi", "uvicorn", "requests"]
OPTIONAL = ["optuna", "lightning"]

for group_name, group, required in [
    ("core", CORE, True),
    ("GIS/geospatial", GIS, False),
    ("thermal comfort", COMFORT, False),
    ("deployment server", DEPLOY, False),
    ("optional", OPTIONAL, False),
]:
    for pkg in group:
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "?")
            check(f"import {pkg}", True, detail=f"v{v}")
        except Exception as e:
            check(f"import {pkg}", False, detail=f"{group_name} dep, {e.__class__.__name__}",
                  warn_only=not required)

# ── 2b. GPU visibility ────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    detail = f"torch {torch.__version__}, cuda={torch.version.cuda}"
    if cuda_ok:
        detail += f", device={torch.cuda.get_device_name(0)}"
    else:
        detail += " -- CPU only: training/inference will be much slower (see ENVIRONMENT_SETUP.md section 3.1)"
    check("torch.cuda.is_available()", cuda_ok, detail=detail, warn_only=True)
except ImportError:
    check("torch.cuda.is_available()", False, detail="torch not installed")

# ── 3. Data / checkpoint files (git does NOT track these) ────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--h5", default="ground_truth_v2.h5",
                     help="which ground-truth H5 to check for (default: v2, matches the thesis checkpoint)")
parser.add_argument("--ckpt", default="checkpoints_v2_fixed",
                     help="which checkpoint directory to check for (default: checkpoints_v2_fixed)")
args = parser.parse_args()

RAW_SIM = ROOT / "01_data_generation" / "outputs" / "raw_simulations"
data_files = {
    args.h5: RAW_SIM / args.h5,
    "epw_data.pkl": RAW_SIM / "epw_data.pkl",
    "scenarios.pkl": RAW_SIM / "scenarios.pkl",
}
for name, path in data_files.items():
    ok = path.exists()
    size = f"{path.stat().st_size / 1e6:.1f} MB" if ok else "MISSING -- see ENVIRONMENT_SETUP.md section 4"
    check(f"data file: {path.relative_to(REPO_ROOT)}", ok, detail=size)

ckpt_path = ROOT / args.ckpt / "best_model.pt"
ok = ckpt_path.exists()
detail = f"{ckpt_path.stat().st_size / 1e6:.1f} MB" if ok else "MISSING -- see ENVIRONMENT_SETUP.md section 4"
check(f"checkpoint: {ckpt_path.relative_to(REPO_ROOT)}", ok, detail=detail)

# ── report ─────────────────────────────────────────────────────────────
print("=" * 78)
print("urban-thermal-gnn environment check")
print("=" * 78)
for status, msg in results:
    print(f"{status}  {msg}")
print("=" * 78)

n_fail = sum(1 for s, _ in results if s == FAIL)
n_warn = sum(1 for s, _ in results if s == WARN)
if n_fail == 0:
    print(f"All required checks passed ({n_warn} warning(s)). Ready to run training.")
    sys.exit(0)
else:
    print(f"{n_fail} required check(s) FAILED, {n_warn} warning(s). "
          f"Fix FAILs before starting training -- see ENVIRONMENT_SETUP.md.")
    sys.exit(1)
