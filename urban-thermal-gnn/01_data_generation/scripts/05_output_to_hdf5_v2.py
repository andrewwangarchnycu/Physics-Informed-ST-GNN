"""
05_output_to_hdf5_v2.py
════════════════════════════════════════════════════════════════
Phase 3: Aggregate v2 simulations (sim_*_v2.npz) into ground_truth_v2.h5
Features:
  - Reads 2x resolution sensor data (~6,241 nodes per scenario)
  - Real weather calibration (CWB + MOENV IoT)
  - Computes normalization statistics for 4x denser datasets
  - Generates stratified train/val/test split (70/15/15)
  - Saves comprehensive dataset summary
"""

from __future__ import annotations
import sys, json, warnings, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import h5py

try:
    from sklearn.cluster import KMeans
    _SK = True
except ImportError:
    _SK = False
    warnings.warn("sklearn not installed, using random split.")


def scan_npz_v2(sim_dir: str) -> list:
    """Scan for v2 simulation files (sim_*_v2.npz)"""
    files = sorted(Path(sim_dir).glob("sim_*_v2.npz"))
    if not files:
        raise FileNotFoundError(f"No sim_*_v2.npz files found in {sim_dir}")
    print(f"  Found {len(files)} v2 .npz files")

    records = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=False)
            records.append({
                "path":            fp,
                "scenario_id":     int(d["scenario_id"]),
                "ta":              d["ta"],
                "mrt":             d["mrt"],
                "va":              d["va"],
                "rh":              d["rh"],
                "utci":            d["utci"],
                "svf":             d["svf"],
                "in_shadow":       d["in_shadow"],
                "building_height": d["building_height"],
                "tree_height":     d["tree_height"],
                "sensor_pts":      d["sensor_pts"],
                "far":             float(d.get("far", 0.0)),
                "bcr":             float(d.get("bcr", 0.0)),
            })
        except Exception as e:
            warnings.warn(f"Skipped {fp.name}: {e}")

    print(f"  Successfully loaded {len(records)} scenarios")
    return records


def compute_norm_stats(records: list) -> dict:
    """Compute normalization statistics for all thermal fields"""
    fields = ["ta", "mrt", "va", "rh", "utci"]
    stats = {}

    print("  Computing normalization statistics:")
    for f in fields:
        vals = np.concatenate([r[f].ravel() for r in records])
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        if s < 1e-6:
            s = 1.0
        stats[f] = {"mean": round(m, 4), "std": round(s, 4)}
        print(f"    {f:5s}  mean={m:7.3f}  std={s:6.3f}")

    return stats


def stratified_split(records: list, ratios=(0.70, 0.15, 0.15),
                      n_clusters=10, seed=42):
    """Generate stratified train/val/test split using KMeans clustering"""
    ids = [r["scenario_id"] for r in records]
    feats = np.array([[r["far"], r["bcr"], len(r["sensor_pts"])]
                       for r in records], dtype=np.float32)

    if _SK and len(records) >= n_clusters:
        labels = KMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=10).fit_predict(feats)
    else:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, n_clusters, size=len(records))

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []

    for c in range(n_clusters):
        idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(n * ratios[0]))
        n_va = max(1, int(n * ratios[1]))
        tr += [ids[i] for i in idx[:n_tr]]
        va += [ids[i] for i in idx[n_tr:n_tr+n_va]]
        te += [ids[i] for i in idx[n_tr+n_va:]]

    print(f"  Split: train={len(tr)}  val={len(va)}  test={len(te)}")
    return tr, va, te


def write_hdf5_v2(records: list, norm_stats: dict, split: tuple,
                   out_path: str, compress: int = 4):
    """Write aggregated v2 data to HDF5"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    opts = dict(compression="gzip", compression_opts=compress)

    # Get sim_hours from first record
    sim_hours = records[0]["ta"].shape[0] if records else 11
    if isinstance(sim_hours, (list, np.ndarray)):
        sim_hours = list(sim_hours) if not isinstance(sim_hours, list) else sim_hours
    else:
        # Default: 11 hourly timestamps (8am-6pm)
        sim_hours = list(range(8, 19))

    with h5py.File(out, "w") as hf:
        # Metadata
        mg = hf.create_group("metadata")
        mg.attrs["data_source"] = "real_cwb_iot_canopy"
        mg.attrs["resolution_m"] = 1.0
        mg.attrs["n_scenarios"] = len(records)
        mg.attrs["description"] = "2x spatial resolution (1.0m grid) with real weather calibration"

        # Add sim_hours as dataset (required by dataset.py)
        mg.create_dataset("sim_hours", data=np.array(sim_hours, dtype=np.int32))

        # Normalization statistics
        ng = hf.create_group("normalization")
        for field, sv in norm_stats.items():
            fg = ng.create_group(field)
            fg.attrs["mean"] = sv["mean"]
            fg.attrs["std"] = sv["std"]

        # Train/Val/Test splits
        sg = hf.create_group("splits")
        sg.create_dataset("train_ids", data=np.array(split[0], np.int32))
        sg.create_dataset("val_ids",   data=np.array(split[1], np.int32))
        sg.create_dataset("test_ids",  data=np.array(split[2], np.int32))

        # Scenario data
        sc_grp = hf.create_group("scenarios")
        for i, rec in enumerate(records):
            sid = rec["scenario_id"]
            g = sc_grp.create_group(str(sid))

            # Datasets
            for key in ["sensor_pts", "ta", "mrt", "va", "rh", "utci",
                        "svf", "in_shadow", "building_height", "tree_height"]:
                if key in rec and rec[key] is not None:
                    g.create_dataset(key, data=rec[key], **opts)

            # Attributes (urban morphology & geometry)
            g.attrs["far"] = rec["far"]
            g.attrs["bcr"] = rec["bcr"]
            g.attrs["n_nodes"] = rec["sensor_pts"].shape[0]
            g.attrs["n_timesteps"] = rec["ta"].shape[0]

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(records)} scenarios written")

    size_mb = out.stat().st_size / 1e6
    print(f"  OK: HDF5 complete: {size_mb:.1f} MB -> {out}")


def save_summary_v2(records, norm_stats, split, out_path):
    """Save dataset summary as JSON"""
    node_counts = [r["sensor_pts"].shape[0] for r in records]

    summary = {
        "pipeline_version": "v2_high_resolution",
        "n_scenarios": len(records),
        "split": {
            "train": len(split[0]),
            "val": len(split[1]),
            "test": len(split[2])
        },
        "spatial_resolution": {
            "grid_spacing_m": 1.0,
            "scaling_factor": 2.0
        },
        "data_sources": [
            "Central Weather Bureau (CWB)",
            "MOENV IoT Sensor Network (824 stations)",
            "Meta Global Canopy Height"
        ],
        "nodes_per_scenario": {
            "min": int(min(node_counts)),
            "max": int(max(node_counts)),
            "mean": float(np.mean(node_counts))
        },
        "n_timesteps": len(records[0]["ta"]) if records else 0,
        "normalization": norm_stats,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  OK: Summary saved: {out_path}")


def main(sim_dir: str = "../outputs/raw_simulations",
          out_h5: str = "../outputs/raw_simulations/ground_truth_v2.h5",
          split_ratio=(0.70, 0.15, 0.15),
          compress: int = 4):
    """Main aggregation pipeline"""
    print("\n[05_output_to_hdf5_v2] Phase 3: Aggregate v2 Ground Truth Data")
    print("=" * 60)

    records = scan_npz_v2(sim_dir)
    norm_stats = compute_norm_stats(records)
    split = stratified_split(records, ratios=split_ratio)
    write_hdf5_v2(records, norm_stats, split, out_h5, compress)

    summary_path = str(Path(out_h5).parent / "dataset_summary_v2.json")
    save_summary_v2(records, norm_stats, split, summary_path)

    print("[05_output_to_hdf5_v2] Complete\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 3: Aggregate v2 simulations to HDF5")
    ap.add_argument("--sim_dir", default="../outputs/raw_simulations",
                    help="Directory containing sim_*_v2.npz files")
    ap.add_argument("--out", default="../outputs/raw_simulations/ground_truth_v2.h5",
                    help="Output HDF5 file path")
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    help="Train/Val/Test ratio")
    ap.add_argument("--compress", type=int, default=4,
                    help="HDF5 compression level (0-9)")

    args = ap.parse_args()
    main(args.sim_dir, args.out, tuple(args.split), args.compress)
