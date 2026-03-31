"""
04_training/evaluate.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:6]
compute: R², RMSE, MAE, UTCI Thermal Stress[REMOVED_ZH:7]

Run:
  python evaluate.py --ckpt checkpoints/best_model.pt
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import numpy as np
import torch

from dataset   import UTCIGraphDataset
from urbangraph import UrbanGraph, build_model
from train     import build_env_time_seq


# ════════════════════════════════════════════════════════════════
# UTCI Thermal Stress[REMOVED_ZH:2]（Fiala 2012）
# ════════════════════════════════════════════════════════════════
UTCI_THRESHOLDS = [-40, 9, 26, 32, 38, 46, 200]
UTCI_LABELS     = ["[REMOVED_ZH:3] (<9)", "[REMOVED_ZH:2] (9–26)", "[REMOVED_ZH:2] (26–32)",
                    "[REMOVED_ZH:2] (32–38)", "[REMOVED_ZH:4] (38–46)", "[REMOVED_ZH:2] (>46)"]

def utci_to_class(utci_vals: np.ndarray) -> np.ndarray:
    classes = np.zeros_like(utci_vals, dtype=int)
    for i, (lo, hi) in enumerate(zip(UTCI_THRESHOLDS[:-1], UTCI_THRESHOLDS[1:])):
        classes[(utci_vals >= lo) & (utci_vals < hi)] = i
    return classes


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:2]compute
# ════════════════════════════════════════════════════════════════
def compute_metrics(pred_norm: np.ndarray,
                     tgt_norm:  np.ndarray,
                     norm_stats: dict) -> dict:
    """
    pred_norm, tgt_norm: [REMOVED_ZH:5] UTCI，shape (N_all,)
    denormalize[REMOVED_ZH:1]compute[REMOVED_ZH:4]。
    """
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]
    pred = pred_norm * std + mean
    tgt  = tgt_norm  * std + mean

    ss_res = np.sum((pred - tgt)**2)
    ss_tot = np.sum((tgt - tgt.mean())**2) + 1e-9
    r2     = float(1 - ss_res / ss_tot)
    rmse   = float(np.sqrt(np.mean((pred - tgt)**2)))
    mae    = float(np.mean(np.abs(pred - tgt)))

    # UTCI [REMOVED_ZH:7]
    pred_cls = utci_to_class(pred)
    tgt_cls  = utci_to_class(tgt)
    acc      = float((pred_cls == tgt_cls).mean())

    # [REMOVED_ZH:3] accuracy
    per_cls_acc = {}
    for i, label in enumerate(UTCI_LABELS):
        mask = tgt_cls == i
        if mask.sum() > 0:
            per_cls_acc[label] = float((pred_cls[mask] == i).mean())

    return {
        "R2":   r2,
        "RMSE": rmse,
        "MAE":  mae,
        "category_accuracy":    acc,
        "per_category_accuracy": per_cls_acc,
        "n_samples": len(pred),
    }


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:5]
# ════════════════════════════════════════════════════════════════
def evaluate(model:      UrbanGraph,
              dataset:   UTCIGraphDataset,
              epw,
              device:    str = "cpu") -> dict:

    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours, month=7)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)

    all_pred, all_tgt = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset.get(idx)
            obj_feat = data["object"].x.to(device)
            air_feat = data["air"].x.to(device)
            target   = data["air"].y.to(device)

            static_edges = {}
            for rel in ["semantic", "contiguity"]:
                key = ("object",rel,"object") if rel=="semantic" else ("air",rel,"air")
                if hasattr(data[key], "edge_index"):
                    static_edges[rel] = data[key].edge_index.to(device)

            dynamic_edges = getattr(data, "dynamic_edges", [{}]*air_feat.shape[1])

            pred = model(obj_feat, air_feat, dynamic_edges,
                          static_edges, env_seq, time_seq)

            all_pred.append(pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())

    pred_all = np.concatenate([p.ravel() for p in all_pred])
    tgt_all  = np.concatenate([t.ravel() for t in all_tgt])

    metrics = compute_metrics(pred_all, tgt_all, dataset.norm_stats)
    return metrics


def main(ckpt_path:    str = "checkpoints/best_model.pt",
          h5_path:     str = "../../01_data_generation/outputs/raw_simulations/ground_truth.h5",
          scenario_pkl:str = "../../01_data_generation/outputs/raw_simulations/scenarios.pkl",
          epw_pkl:     str = "../../01_data_generation/outputs/raw_simulations/epw_data.pkl",
          split:       str = "test",
          out_json:    str = "eval_results.json"):

    import pickle, __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED
    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device)

    # [REMOVED_ZH:4]
    ds  = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split=split)
    cfg = {"model": {"out_timesteps": len(ds.sim_hours)}}
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    print(f"\n[Evaluate] split={split}  {len(ds)} [REMOVED_ZH:2]  "
          f"[REMOVED_ZH:2] epoch={ckpt.get('epoch','?')}")

    metrics = evaluate(model, ds, epw, device)

    print(f"\n{'─'*50}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.3f} °C")
    print(f"  MAE  = {metrics['MAE']:.3f} °C")
    print(f"  [REMOVED_ZH:5] = {metrics['category_accuracy']*100:.1f}%")
    print(f"\n  [REMOVED_ZH:6]:")
    for label, acc in metrics["per_category_accuracy"].items():
        print(f"    {label}: {acc*100:.1f}%")
    print(f"{'─'*50}\n")

    if metrics["R2"] >= 0.90:
        print("  ✓ R² ≥ 0.90，[REMOVED_ZH:3]Deployment Threshold。")
    else:
        print(f"  ✗ R² = {metrics['R2']:.4f} < 0.90，[REMOVED_ZH:11]。")

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [REMOVED_ZH:4]: {out_json}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     default="checkpoints/best_model.pt")
    ap.add_argument("--h5",       default="../../01_data_generation/outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--scenarios",default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",      default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--split",    default="test", choices=["train","val","test"])
    ap.add_argument("--out",      default="eval_results.json")
    args = ap.parse_args()
    main(args.ckpt, args.h5, args.scenarios, args.epw, args.split, args.out)