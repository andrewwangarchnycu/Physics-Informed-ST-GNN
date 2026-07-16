"""
04_training/benchmark_timing.py
════════════════════════════════════════════════════════════════
Measure GNN surrogate inference latency vs. reference simulation.

Outputs benchmark_results.json:
  mean_inference_ms    — GNN surrogate, averaged over N repetitions
  std_inference_ms
  lbt_simulation_ms    — manually supplied reference time
  speedup_factor
  hardware_spec

Usage:
    python benchmark_timing.py --ckpt ../checkpoints_v2/best_model.pt --reps 100
    python benchmark_timing.py --ckpt ../checkpoints_v2/best_model.pt --reps 100 --lbt 18.5
"""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import torch


# ── Hardware info ─────────────────────────────────────────────
def get_hardware_info() -> dict:
    info: dict = {"cpu": platform.processor(), "os": platform.system()}
    if torch.cuda.is_available():
        info["gpu"]     = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


# ── Timed inference ───────────────────────────────────────────
def measure_inference(model, dataset, epw, device: str, n_reps: int = 100) -> dict:
    """Time a single-scenario forward pass n_reps times; return stats in ms."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))
    from train import build_env_time_seq

    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours, month=7)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)

    data     = dataset.get(0)
    obj_feat = data["object"].x.to(device)
    air_feat = data["air"].x.to(device)

    static_edges: dict = {}
    for rel in ["semantic", "contiguity"]:
        key = ("object", rel, "object") if rel == "semantic" else ("air", rel, "air")
        if hasattr(data[key], "edge_index"):
            static_edges[rel] = data[key].edge_index.to(device)
    dyn = getattr(data, "dynamic_edges", [{}] * air_feat.shape[1])

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(obj_feat, air_feat, dyn, static_edges, env_seq, time_seq)

    if device == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = model(obj_feat, air_feat, dyn, static_edges, env_seq, time_seq)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms":      float(np.mean(times)),
        "std_ms":       float(np.std(times)),
        "min_ms":       float(np.min(times)),
        "max_ms":       float(np.max(times)),
        "n_reps":       n_reps,
        "n_air_nodes":  int(air_feat.shape[0]),
        "n_timesteps":  int(air_feat.shape[1]),
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Benchmark GNN inference latency.")
    ap.add_argument("--ckpt",
        default="../checkpoints_v2/best_model.pt")
    ap.add_argument("--h5",
        default="../../01_data_generation/outputs/raw_simulations/ground_truth_v2.h5")
    ap.add_argument("--scenarios",
        default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",
        default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--reps",   type=int,   default=100)
    ap.add_argument("--lbt",    type=float, default=None,
        help="LBT simulation time in minutes (measured separately on same machine)")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",    default="benchmark_results.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

    import pickle, __main__
    from shared     import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED

    from dataset    import UTCIGraphDataset
    from urbangraph import build_model

    with open(args.epw, "rb") as f:
        epw = pickle.load(f)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    # Auto-detect dim_air from the checkpoint's first encoder weight shape
    enc_w = ckpt["model_state"].get("air_encoder.net.0.weight")
    dim_air = int(enc_w.shape[1]) if enc_w is not None else 9
    print(f"  Auto-detected dim_air={dim_air} from checkpoint")

    test_ds = UTCIGraphDataset(
        args.h5, args.scenarios, args.epw, split="test", dim_air=dim_air)

    cfg   = {"model": {"out_timesteps": len(test_ds.sim_hours), "dim_air": dim_air}}
    model = build_model(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    print(f"\n[Benchmark] device={args.device}  reps={args.reps}")
    timing = measure_inference(model, test_ds, epw, args.device, args.reps)
    hw     = get_hardware_info()

    result = {**timing, "hardware": hw}
    if args.lbt is not None:
        lbt_ms = args.lbt * 60 * 1000
        result["lbt_simulation_ms"] = lbt_ms
        result["speedup_factor"]    = round(lbt_ms / timing["mean_ms"], 0)

    print(f"\n{'-'*50}")
    print(f"  GNN inference:  {timing['mean_ms']:.1f} ± {timing['std_ms']:.1f} ms")
    print(f"  Hardware:       {hw.get('gpu', hw['cpu'])}")
    if args.lbt is not None:
        print(f"  LBT reference:  {args.lbt:.1f} min")
        print(f"  Speed-up:       {result['speedup_factor']:.0f}×")
    print(f"{'-'*50}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
