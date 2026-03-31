"""
05_output_to_hdf5.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/
[REMOVED_ZH:2] : [REMOVED_ZH:1] sim_XXXX.npz [REMOVED_ZH:3] ground_truth.h5，
       compute[REMOVED_ZH:1]Dataset[REMOVED_ZH:5]，[REMOVED_ZH:2] stratified 70/15/15 split。
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
    warnings.warn("sklearn [REMOVED_ZH:3]，[REMOVED_ZH:4] split。")


def scan_npz(sim_dir: str) -> list:
    files = sorted(Path(sim_dir).glob("sim_????.npz"))
    if not files:
        raise FileNotFoundError(f"[REMOVED_ZH:3] sim_XXXX.npz in {sim_dir}")
    print(f"  [REMOVED_ZH:2] {len(files)} [REMOVED_ZH:1] .npz")
    records = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=False)
            records.append({
                "path":        fp,
                "scenario_id": int(d["scenario_id"]),
                "ta":          d["ta"],   "mrt": d["mrt"],
                "va":          d["va"],   "rh":  d["rh"],
                "utci":        d["utci"], "svf": d["svf"],
                "in_shadow":   d["in_shadow"],
                "building_height": d["building_height"],
                "tree_height":     d["tree_height"],
                "land_cover":      d["land_cover"],
                "sensor_pts":  d["sensor_pts"],
                "sim_hours":   d["sim_hours"].tolist(),
                "far":  float(d["far"]),
                "bcr":  float(d["bcr"]),
                "n_buildings": int(d["n_buildings"]),
            })
        except Exception as e:
            warnings.warn(f"[REMOVED_ZH:2] {fp.name}: {e}")
    print(f"  [REMOVED_ZH:4] {len(records)} [REMOVED_ZH:1]")
    return records


def compute_norm_stats(records: list) -> dict:
    fields = ["ta", "mrt", "va", "rh", "utci"]
    stats  = {}
    for f in fields:
        vals = np.concatenate([r[f].ravel() for r in records])
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        if s < 1e-6: s = 1.0
        stats[f] = {"mean": round(m, 4), "std": round(s, 4)}
        print(f"    {f:5s}  μ={m:7.3f}  σ={s:6.3f}")
    return stats


def stratified_split(records: list, ratios=(0.70, 0.15, 0.15),
                      n_clusters=10, seed=42):
    ids   = [r["scenario_id"] for r in records]
    feats = np.array([[r["far"], r["bcr"], r["n_buildings"]]
                       for r in records], dtype=np.float32)
    if _SK and len(records) >= n_clusters:
        labels = KMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=10).fit_predict(feats)
    else:
        rng    = np.random.default_rng(seed)
        labels = rng.integers(0, n_clusters, size=len(records))

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(n_clusters):
        idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(idx)
        n   = len(idx)
        n_tr = max(1, int(n * ratios[0]))
        n_va = max(1, int(n * ratios[1]))
        tr += [ids[i] for i in idx[:n_tr]]
        va += [ids[i] for i in idx[n_tr:n_tr+n_va]]
        te += [ids[i] for i in idx[n_tr+n_va:]]
    print(f"  Split: train={len(tr)}  val={len(va)}  test={len(te)}")
    return tr, va, te


def write_hdf5(records: list, norm_stats: dict, split: tuple,
                out_path: str, epw_pkl: str = None, compress: int = 4):
    city = lat = lon = ""
    if epw_pkl and Path(epw_pkl).exists():
        import pickle
        with open(epw_pkl, "rb") as f:
            epw = pickle.load(f)
        city, lat, lon = epw.city, epw.latitude, epw.longitude

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sim_hours = records[0]["sim_hours"] if records else []
    opts = dict(compression="gzip", compression_opts=compress)

    with h5py.File(out, "w") as hf:
        mg = hf.create_group("metadata")
        mg.attrs["city"]        = city or ""
        mg.attrs["latitude"]    = lat or 0.0
        mg.attrs["longitude"]   = lon or 0.0
        mg.attrs["n_scenarios"] = len(records)
        mg.create_dataset("sim_hours",
                           data=np.array(sim_hours, dtype=np.int32))

        ng = hf.create_group("normalization")
        for field, sv in norm_stats.items():
            fg = ng.create_group(field)
            fg.attrs["mean"] = sv["mean"]
            fg.attrs["std"]  = sv["std"]

        sg = hf.create_group("splits")
        sg.create_dataset("train_ids", data=np.array(split[0], np.int32))
        sg.create_dataset("val_ids",   data=np.array(split[1], np.int32))
        sg.create_dataset("test_ids",  data=np.array(split[2], np.int32))

        sc_grp = hf.create_group("scenarios")
        for i, rec in enumerate(records):
            sid = rec["scenario_id"]
            g   = sc_grp.create_group(str(sid))
            for key in ["sensor_pts","ta","mrt","va","rh","utci",
                         "svf","in_shadow","building_height","tree_height","land_cover"]:
                g.create_dataset(key, data=rec[key], **opts)
            g.attrs["far"]         = rec["far"]
            g.attrs["bcr"]         = rec["bcr"]
            g.attrs["n_buildings"] = rec["n_buildings"]
            g.attrs["n_nodes"]     = rec["sensor_pts"].shape[0]
            g.attrs["n_timesteps"] = rec["ta"].shape[0]
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(records)} [REMOVED_ZH:3]")

    size = out.stat().st_size / 1e6
    print(f"  ✓ HDF5 [REMOVED_ZH:2]  {size:.1f} MB  {out}")


def save_summary(records, norm_stats, split, out_path):
    far_arr  = [r["far"] for r in records]
    bcr_arr  = [r["bcr"] for r in records]
    nn_arr   = [r["sensor_pts"].shape[0] for r in records]
    summary  = {
        "n_scenarios":  len(records),
        "split":        {"train": len(split[0]), "val": len(split[1]), "test": len(split[2])},
        "far":          {"min": min(far_arr), "max": max(far_arr), "mean": float(np.mean(far_arr))},
        "bcr":          {"min": min(bcr_arr), "max": max(bcr_arr), "mean": float(np.mean(bcr_arr))},
        "n_air_nodes":  {"min": min(nn_arr), "max": max(nn_arr), "mean": float(np.mean(nn_arr))},
        "n_timesteps":  len(records[0]["sim_hours"]) if records else 0,
        "normalization": norm_stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ [REMOVED_ZH:2]: {out_path}")


def main(sim_dir:  str = "../outputs/raw_simulations",
          out_h5:   str = "../outputs/raw_simulations/ground_truth.h5",
          epw_pkl:  str = "../outputs/raw_simulations/epw_data.pkl",
          split_ratio=(0.70, 0.15, 0.15), compress: int = 4):
    print("\n[05_output_to_hdf5] ── [REMOVED_ZH:2] Ground Truth ──")
    records    = scan_npz(sim_dir)
    norm_stats = compute_norm_stats(records)
    split      = stratified_split(records, ratios=split_ratio)
    write_hdf5(records, norm_stats, split, out_h5, epw_pkl, compress)
    summary_path = str(Path(out_h5).parent / "dataset_summary.json")
    save_summary(records, norm_stats, split, summary_path)
    print("[05_output_to_hdf5] [REMOVED_ZH:2]。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default="../outputs/raw_simulations")
    ap.add_argument("--out",     default="../outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--epw",     default="../outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--split",   nargs=3, type=float, default=[0.70,0.15,0.15])
    ap.add_argument("--compress",type=int, default=4)
    args = ap.parse_args()
    main(args.sim_dir, args.out, args.epw, tuple(args.split), args.compress)