"""
run_abc_walkway_nsga2.py
==========================
Three-objective NSGA-II runs (mean_utci, walkway_exposure, -green_ratio)
comparing three fixed pedestrian route definitions (A/B/C) through the
SAME 80x80 m site and building-design search space, using the trained
PI-ST-GNN V4 checkpoint.

This is the "A/B/C comparison experiment" requested to address the
committee's complaint that the thesis lacked a return to building-form and
walking-path discussion: rather than reframing the single existing
2-objective run (`run_real_nsga2.py`), this script actually executes THREE
independent NSGA-II optimizations, one per route, each using the walkway
exposure metric wired into `fitness.py` (`walkway_exposure()`, a simplified
single-route implementation of the thesis appendix's derived
$\\bar{\\Phi}_{walkway}$ formula -- see appx:walkway_derivation).

Route definitions (site-local coordinates, matching site_pts bbox
[[0,0],[80,0],[80,80],[0,80]]):
  A -- street frontage:   a path along the site's near-boundary edge,
                          representing a pedestrian walking along the
                          building street frontage.
  B -- open plaza/courtyard: a path through the site's horizontal midline,
                          representing a route through an open/central
                          courtyard space.
  C -- diagonal crossing: a path cutting diagonally across the site,
                          representing a route that traverses the full
                          building-massing search space.

Honest scope note: each route is evaluated as a fixed input (not itself
optimized) while NSGA-II searches building/tree massing; this validates the
already-derived walkway penalty formula and three-objective formulation
across genuinely different route geometries, but is not the full
A*-based multi-route walkway graph described in the appendix derivation.

Usage:
    python run_abc_walkway_nsga2.py --route A
    python run_abc_walkway_nsga2.py --route B
    python run_abc_walkway_nsga2.py --route C

Produces:
  outputs/nsga2_route_A.json / _B.json / _C.json
"""
import sys, json, time, argparse
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

_CKPT_V4    = _ROOT / "04_training" / "checkpoints_v4" / "best_model.pt"
_CKPT_V2    = _ROOT / "checkpoints_v2_fixed" / "best_model.pt"
CKPT_PATH   = _CKPT_V4 if _CKPT_V4.exists() else _CKPT_V2
_H5_V4      = _ROOT / "01_data_generation/outputs/real_simulations_v4/ground_truth_v4.h5"
_H5_LEGACY  = _ROOT / "01_data_generation/outputs/raw_simulations/ground_truth.h5"
H5_PATH     = _H5_V4 if _H5_V4.exists() else _H5_LEGACY
EPW_PKL_PATH= _ROOT / "01_data_generation/outputs/raw_simulations/epw_data.pkl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed pedestrian routes (site-local [x,y], 80x80 m site)
ROUTES = {
    "A": {"name": "街道臨街面路線 (street frontage)",
          "waypoints": [[5.0, 5.0], [75.0, 5.0]]},
    "B": {"name": "開放廣場/中庭路線 (open plaza/courtyard)",
          "waypoints": [[10.0, 40.0], [70.0, 40.0]]},
    "C": {"name": "對角線橫越路線 (diagonal crossing)",
          "waypoints": [[5.0, 5.0], [75.0, 75.0]]},
}


def main(route_key: str, pop_size: int, n_gen: int, seed: int):
    route = ROUTES[route_key]
    print(f"[0] Route {route_key}: {route['name']}  waypoints={route['waypoints']}")
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
        dim_air=dim_air, walkway_route=route["waypoints"],
    )
    assert evaluator.n_obj == 3, "walkway_route did not activate 3-objective mode"

    optimizer = NSGA2Optimizer(evaluator=evaluator, cfg=cfg, pop_size=pop_size, n_gen=n_gen, seed=seed)

    print(f"[2] Running route-{route_key} NSGA-II: pop_size={pop_size}, n_gen={n_gen}, n_obj=3 ...")
    history = []
    t_start = time.perf_counter()

    def cb(info):
        history.append(info)
        if info["generation"] % 5 == 0 or info["generation"] == 1:
            print(f"    gen {info['generation']:3d}/{info['n_gen']}  "
                  f"n_feasible={info['n_feasible']:3d}  "
                  f"best_utci={info['best_utci']}  "
                  f"best_walk={info.get('best_walkway_exposure')}  "
                  f"best_green={info['best_green']}  "
                  f"pareto={info['pareto_count']}  ({info['elapsed_s']}s/gen)")

    result = optimizer.run_sync(progress_callback=cb)
    t_total = time.perf_counter() - t_start
    print(f"[3] Done in {t_total:.1f}s. status={result['status']}, "
          f"generations_run={result['generations_run']}, "
          f"final Pareto set size={len(result['pareto_designs'])}")

    out = {
        "route_key": route_key,
        "route_name": route["name"],
        "route_waypoints": route["waypoints"],
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed": seed,
        "n_objectives": 3,
        "checkpoint": str(CKPT_PATH),
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_val_r2": ckpt.get("val_r2", None),
        "total_wall_time_s": round(t_total, 2),
        "history": history,
        "final_pareto": result["pareto_designs"],
        "status": result["status"],
    }
    out_path = OUT_DIR / f"nsga2_route_{route_key}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"[4] Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["A", "B", "C"], required=True)
    ap.add_argument("--pop_size", type=int, default=40)
    ap.add_argument("--n_gen", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.route, args.pop_size, args.n_gen, args.seed)
