"""
01_data_generation/scripts/10_output_to_hdf5_v4.py
════════════════════════════════════════════════════════════════
Aggregate the V4 real-scene simulations (real_simulations_v4/sim_*.npz)
into ground_truth_v4.h5 for training.

Differences vs the V2/V3 aggregator:
  * Reads the V4 npz which additionally carry ``sensor_utci`` / ``sensor_mask``
    (real-IoT-derived direct-supervision signal) and ``assigned_month``.
  * Writes ``sensor_utci`` into each scenario group so dataset.py's
    (previously dead) real-sensor supervision path is actually fed.
  * Stratifies the train/val/test split by assigned month so every split
    spans June / July / August.
"""
from __future__ import annotations

import sys
import json
import warnings
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import h5py


def scan_npz_v4(sim_dir: str) -> list:
    files = sorted(Path(sim_dir).glob("sim_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sim_*.npz in {sim_dir}")
    print(f"  found {len(files)} V4 .npz files")
    # UTCI is only defined on an operational assessment scale (~ -40..+46 C,
    # top category "extreme heat stress"). At very low wind + extreme MRT the
    # polynomial extrapolates to physically meaningless values (>50 C); clip
    # the labels to a sane operational range so training targets stay valid.
    UTCI_LO, UTCI_HI = -30.0, 50.0
    recs = []
    n_clip = 0
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=False)
            rec = {k: d[k] for k in d.files}
            u = np.asarray(rec["utci"], dtype=np.float32)
            n_clip += int(((u < UTCI_LO) | (u > UTCI_HI)).sum())
            rec["utci"] = np.clip(u, UTCI_LO, UTCI_HI)
            if "sensor_utci" in rec:
                su = np.asarray(rec["sensor_utci"], dtype=np.float32)
                rec["sensor_utci"] = np.where(np.isfinite(su),
                                              np.clip(su, UTCI_LO, UTCI_HI), su)
            rec["path"] = fp
            recs.append(rec)
        except Exception as e:
            warnings.warn(f"skip {fp.name}: {e}")
    print(f"  clipped {n_clip} UTCI values to [{UTCI_LO}, {UTCI_HI}] C")
    print(f"  loaded {len(recs)} scenarios")
    return recs


def compute_norm_stats(records: list) -> dict:
    stats = {}
    for f in ["ta", "mrt", "va", "rh", "utci"]:
        vals = np.concatenate([np.asarray(r[f]).ravel() for r in records])
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        stats[f] = {"mean": round(m, 4), "std": round(max(s, 1e-6), 4)}
        print(f"    {f:5s} mean={m:8.3f} std={s:7.3f}")
    return stats


def stratified_split_by_month(records, ratios=(0.70, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    by_month = {}
    for r in records:
        m = int(r["assigned_month"]) if "assigned_month" in r else 0
        by_month.setdefault(m, []).append(int(r["scenario_id"]))
    tr, va, te = [], [], []
    for m, ids in by_month.items():
        ids = list(ids); rng.shuffle(ids)
        n = len(ids); n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
        tr += ids[:n_tr]; va += ids[n_tr:n_tr+n_va]; te += ids[n_tr+n_va:]
    print(f"  split (month-stratified): train={len(tr)} val={len(va)} test={len(te)}")
    return sorted(tr), sorted(va), sorted(te)


def write_hdf5_v4(records, norm_stats, split, out_path, compress=4):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(compression="gzip", compression_opts=compress)
    sim_hours = list(range(8, 19))
    if records and "sim_hours" in records[0]:
        sim_hours = [int(x) for x in np.asarray(records[0]["sim_hours"]).ravel()]

    n_real_supervised = 0
    with h5py.File(out, "w") as hf:
        mg = hf.create_group("metadata")
        mg.attrs["data_source"] = "real_osm_eth_cwb_iot_v4"
        mg.attrs["resolution_m"] = 2.0
        mg.attrs["n_scenarios"] = len(records)
        mg.attrs["description"] = ("300 real-geometry scenes (OSM buildings + "
                                   "ETH canopy) forced by real MOENV IoT air "
                                   "temperature/humidity + CWB wind + clear-sky solar")
        mg.create_dataset("sim_hours", data=np.array(sim_hours, dtype=np.int32))

        ng = hf.create_group("normalization")
        for field, sv in norm_stats.items():
            fg = ng.create_group(field)
            fg.attrs["mean"] = sv["mean"]; fg.attrs["std"] = sv["std"]

        sg = hf.create_group("splits")
        sg.create_dataset("train_ids", data=np.array(split[0], np.int32))
        sg.create_dataset("val_ids",   data=np.array(split[1], np.int32))
        sg.create_dataset("test_ids",  data=np.array(split[2], np.int32))

        sc_grp = hf.create_group("scenarios")
        for i, rec in enumerate(records):
            sid = int(rec["scenario_id"])
            g = sc_grp.create_group(str(sid))
            for key in ["sensor_pts", "ta", "mrt", "va", "rh", "utci",
                        "svf", "in_shadow", "building_height", "tree_height"]:
                if key in rec:
                    g.create_dataset(key, data=rec[key], **opts)
            # V4: real-IoT-derived direct-supervision signal.
            if "sensor_utci" in rec and "sensor_mask" in rec:
                su = np.asarray(rec["sensor_utci"], dtype=np.float32)
                sm = np.asarray(rec["sensor_mask"]).astype(bool)
                # store NaN where not a real observation (dataset.py masks on NaN)
                su = np.where(sm, su, np.nan).astype(np.float32)
                g.create_dataset("sensor_utci", data=su, **opts)
                if sm.any():
                    n_real_supervised += 1
            g.attrs["far"] = float(rec.get("far", 0.0))
            g.attrs["bcr"] = float(rec.get("bcr", 0.0))
            g.attrs["assigned_month"] = int(rec.get("assigned_month", 0))
            g.attrs["n_nodes"] = int(np.asarray(rec["sensor_pts"]).shape[0])
            g.attrs["n_timesteps"] = int(np.asarray(rec["ta"]).shape[0])
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(records)} written")

    print(f"  scenes with real sensor supervision: {n_real_supervised}/{len(records)}")
    print(f"  OK: {out.stat().st_size/1e6:.1f} MB -> {out}")
    return n_real_supervised


def main(sim_dir, out_h5, ratios, compress):
    print("\n[10_output_to_hdf5_v4] aggregate V4 real ground truth")
    print("=" * 60)
    records = scan_npz_v4(sim_dir)
    norm_stats = compute_norm_stats(records)
    split = stratified_split_by_month(records, ratios=ratios)
    n_sup = write_hdf5_v4(records, norm_stats, split, out_h5, compress)

    node_counts = [int(np.asarray(r["sensor_pts"]).shape[0]) for r in records]
    summary = {
        "pipeline_version": "v4_real_geometry_real_iot",
        "n_scenarios": len(records),
        "split": {"train": len(split[0]), "val": len(split[1]), "test": len(split[2])},
        "n_scenes_real_supervised": n_sup,
        "data_sources": ["OSM buildings (footprints+heights)",
                         "ETH GlobalCanopyHeight 10m",
                         "MOENV IoT temperature + humidity (real 2025-06..08)",
                         "CWB Hsinchu station C0D660 wind",
                         "clear-sky solar model (CWB has no irradiance)"],
        "nodes_per_scenario": {"min": int(min(node_counts)),
                               "max": int(max(node_counts)),
                               "mean": float(np.mean(node_counts))},
        "normalization": norm_stats,
    }
    with open(Path(out_h5).parent / "dataset_summary_v4.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[10_output_to_hdf5_v4] complete\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default=str(Path(__file__).resolve().parents[1] / "outputs" / "real_simulations_v4"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "real_simulations_v4" / "ground_truth_v4.h5"))
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--compress", type=int, default=4)
    args = ap.parse_args()
    main(args.sim_dir, args.out, tuple(args.split), args.compress)
