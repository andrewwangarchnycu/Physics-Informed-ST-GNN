"""
01_data_generation/scripts/11_select_v5_subset.py
════════════════════════════════════════════════════════════════
V5 scenario subsampler.

Real Radiance + EnergyPlus (via the official `utci_comfort_map` Honeybee
recipe) replace V4's custom SVF/shadow/MRT approximation, but at real cost
per scenario -- so V5 runs on a subset of the 300 V4 scenarios rather than
all of them. This script picks that subset, reusing V4's real OSM building
geometry, real ETH canopy trees and real MOENV IoT-derived scenario
provenance (scenarios_v4.pkl / manifest_v4.json) -- nothing here is
resimulated or re-fetched, it is pure selection.

Selection criteria:
  * Stratified across the three summer months (June/July/August) so V5
    keeps V4's seasonal spread.
  * Prefer scenarios where V4's real-IoT air temperature was usable
    (ta_source == "iot" in manifest_v4.json) -- these are scenarios with a
    genuine, validated real-time station signal, matching V4's own
    sensor_utci supervision logic in 09_run_real_sim_v4.py.
  * Within each month, spread across building density (far_actual) so the
    subset is not just N near-identical low-rise sites.

Usage:
    python 11_select_v5_subset.py --n 40
"""
from __future__ import annotations

import json
import pickle
import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v4"


def main(n_target: int, out_path: str):
    with open(V4_DIR / "scenarios_v4.pkl", "rb") as f:
        scenarios = pickle.load(f)
    manifest = json.loads((V4_DIR / "manifest_v4.json").read_text(encoding="utf-8"))
    prov = {s["scenario_id"]: s for s in manifest["scenes"]}

    by_sid = {sc["scenario_id"]: sc for sc in scenarios}
    months = sorted({sc["assigned_month"] for sc in scenarios})
    per_month = max(1, n_target // len(months))

    rng = np.random.default_rng(42)
    selected = []
    for m in months:
        cand = [sc for sc in scenarios if sc["assigned_month"] == m
                and sc["scenario_id"] in prov]
        # prefer real-IoT-supervised scenarios (ta_source == "iot")
        iot_ok = [sc for sc in cand if prov[sc["scenario_id"]]["ta_source"] == "iot"]
        pool = iot_ok if len(iot_ok) >= per_month else cand
        # spread across building density (far_actual) via quantile bucketing
        pool_sorted = sorted(pool, key=lambda sc: sc["far_actual"])
        k = min(per_month, len(pool_sorted))
        if k > 0:
            idxs = np.linspace(0, len(pool_sorted) - 1, k).round().astype(int)
            idxs = sorted(set(idxs.tolist()))
            # top up if quantile de-dup left us short
            chosen = [pool_sorted[i] for i in idxs]
            j = 0
            while len(chosen) < k and j < len(pool_sorted):
                if pool_sorted[j] not in chosen:
                    chosen.append(pool_sorted[j])
                j += 1
            selected.extend(chosen[:k])

    selected_ids = sorted(sc["scenario_id"] for sc in selected)
    print(f"[select_v5] selected {len(selected_ids)} / {n_target} target "
          f"from {len(scenarios)} V4 scenarios")
    for m in months:
        n_m = sum(1 for sc in selected if sc["assigned_month"] == m)
        n_iot = sum(1 for sc in selected if sc["assigned_month"] == m
                    and prov[sc["scenario_id"]]["ta_source"] == "iot")
        print(f"  month {m}: {n_m} scenes ({n_iot} real-IoT ta)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset = [by_sid[sid] for sid in selected_ids]
    with open(out, "wb") as f:
        pickle.dump(subset, f)
    print(f"  saved: {out}  ({len(subset)} scenarios)")

    summary = {
        "n_selected": len(subset),
        "scenario_ids": selected_ids,
        "months": {int(m): sum(1 for sc in subset if sc["assigned_month"] == m)
                   for m in months},
        "source": "subset of V4 real_osm_eth_v4 scenarios (scenarios_v4.pkl), "
                   "selected for V5 real-Radiance/EnergyPlus resimulation",
    }
    (out.parent / "v5_subset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(_ROOT / "01_data_generation" / "outputs" /
                                          "real_simulations_v5" / "scenarios_v5_subset.pkl"))
    args = ap.parse_args()
    main(args.n, args.out)
