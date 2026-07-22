"""
run_real_nsga2.py
==================
Standalone script that actually executes the real NSGA-II optimizer
(nsga2_engine.NSGA2Optimizer + fitness.FitnessEvaluator) against the
trained PI-ST-GNN checkpoint (V5-300 by default; falls back to V4/V2 if
unavailable), mirroring exactly how 06_deployment/app.py wires these
objects together for the live Grasshopper WebSocket endpoint.

Purpose: produce a REAL convergence log (generation -> best_utci,
best_green, n_feasible, pareto_count) instead of a fabricated/schematic
curve, for use in the thesis's NSGA-II convergence figure.

NOTE ON OBJECTIVE COUNT: as implemented in fitness.py, FitnessEvaluator
returns only 2 objectives (mean_utci, -green_ratio). The thesis's third
objective (walkway thermal exposure, f2) is mathematically specified but
not yet wired into this evaluator -- this script honestly reflects the
current 2-objective implementation, it does not fabricate a third axis.

Produces:
  outputs/nsga2_real_run.json   (per-generation progress log + final Pareto set)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import h5py
import pickle

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "06_deployment"))
sys.path.insert(0, str(_ROOT / "03_model"))
sys.path.insert(0, str(_ROOT / "02_graph_construction"))

OUT_DIR = _HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

_CKPT_V5    = _ROOT / "04_training" / "checkpoints_v5_300" / "best_model.pt"
_CKPT_V4    = _ROOT / "04_training" / "checkpoints_v4" / "best_model.pt"
_CKPT_V2    = _ROOT / "checkpoints_v2_fixed" / "best_model.pt"
CKPT_PATH   = _CKPT_V5 if _CKPT_V5.exists() else (_CKPT_V4 if _CKPT_V4.exists() else _CKPT_V2)
_H5_V5      = _ROOT / "01_data_generation/outputs/real_simulations_v5/ground_truth_v5.h5"
_H5_V4      = _ROOT / "01_data_generation/outputs/real_simulations_v4/ground_truth_v4.h5"
_H5_LEGACY  = _ROOT / "01_data_generation/outputs/raw_simulations/ground_truth.h5"
H5_PATH     = _H5_V5 if _H5_V5.exists() else (_H5_V4 if _H5_V4.exists() else _H5_LEGACY)
# Real CWB weather-forcing sequence for the LSTM's env/time embedding -- this
# is version-independent (same physical 2025 weather data used across
# V1-V4), so it is NOT swapped when the checkpoint/H5 version changes.
EPW_PKL_PATH= _ROOT / "01_data_generation/outputs/raw_simulations/epw_data.pkl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[1] device={DEVICE}, checkpoint={CKPT_PATH}")

with h5py.File(H5_PATH, "r") as hf:
    norm_stats = {}
    for field in hf["normalization"].keys():
        grp = hf[f"normalization/{field}"]
        norm_stats[field] = {"mean": float(grp.attrs["mean"]), "std": float(grp.attrs["std"])}
print(f"    norm_stats: {list(norm_stats.keys())}")

import __main__
from shared import HourlyClimate as _HC, EPWData as _ED
__main__.HourlyClimate = _HC
__main__.EPWData = _ED
with open(EPW_PKL_PATH, "rb") as f:
    epw_data = pickle.load(f)
print("    EPW loaded")

from urbangraph import build_model
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
state = ckpt["model_state"]
dim_air = state["air_encoder.net.0.weight"].shape[1] if "air_encoder.net.0.weight" in state else 9
model_cfg = {"model": {"out_timesteps": 11, "dim_air": dim_air}}
model = build_model(model_cfg).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"    model loaded, epoch={ckpt.get('epoch','?')}, val_R2={ckpt.get('val_r2','?')}, dim_air={dim_air}")

from chromosome import ChromosomeConfig
from constraints import ConstraintChecker
from fitness import FitnessEvaluator
from nsga2_engine import NSGA2Optimizer

site_pts = [[0, 0], [80, 0], [80, 80], [0, 80]]
cfg = ChromosomeConfig(site_bbox=(0.0, 0.0, 80.0, 80.0), floor_height=4.5)
checker = ConstraintChecker(site_pts, setback=3.0, far_max=3.0, bcr_max=0.6, floor_h=4.5)
evaluator = FitnessEvaluator(
    model=model, norm_stats=norm_stats, epw_data=epw_data,
    site_pts=site_pts, cfg=cfg, checker=checker, device=DEVICE,
    dim_air=dim_air,
)

POP_SIZE = 40
N_GEN = 50
optimizer = NSGA2Optimizer(evaluator=evaluator, cfg=cfg, pop_size=POP_SIZE, n_gen=N_GEN, seed=42)

print(f"[2] Running real NSGA-II: pop_size={POP_SIZE}, n_gen={N_GEN} ...")
history = []
t_start = time.perf_counter()

def cb(info):
    history.append(info)
    if info["generation"] % 5 == 0 or info["generation"] == 1:
        print(f"    gen {info['generation']:3d}/{info['n_gen']}  "
              f"n_feasible={info['n_feasible']:3d}  "
              f"best_utci={info['best_utci']}  best_green={info['best_green']}  "
              f"pareto={info['pareto_count']}  ({info['elapsed_s']}s/gen)")

result = optimizer.run_sync(progress_callback=cb)
t_total = time.perf_counter() - t_start
print(f"[3] Done in {t_total:.1f}s. status={result['status']}, "
      f"generations_run={result['generations_run']}, "
      f"final Pareto set size={len(result['pareto_designs'])}")

out = {
    "pop_size": POP_SIZE,
    "n_gen": N_GEN,
    "seed": 42,
    "checkpoint": str(CKPT_PATH),
    "checkpoint_epoch": ckpt.get("epoch", None),
    "checkpoint_val_r2": ckpt.get("val_r2", None),
    "total_wall_time_s": round(t_total, 2),
    "history": history,
    "final_pareto": result["pareto_designs"],
    "status": result["status"],
}
out_path = OUT_DIR / "nsga2_real_run.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print(f"[4] Saved: {out_path}")
