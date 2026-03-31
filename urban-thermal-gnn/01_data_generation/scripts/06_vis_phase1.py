"""
viz_phase1.py  — Phase 1 [REMOVED_ZH:7]
════════════════════════════════════════
Run: python viz_phase1.py --sim_dir ../outputs/raw_simulations --out_dir viz_output/phase1
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import pickle

# UTCI Thermal Stress[REMOVED_ZH:4]（[REMOVED_ZH:1] Fiala 2012）
UTCI_CMAP = mcolors.LinearSegmentedColormap.from_list("utci", [
    (0.0,  "#313695"),  # <9°C  [REMOVED_ZH:3] ([REMOVED_ZH:1])
    (0.22, "#4575b4"),
    (0.35, "#74add1"),
    (0.46, "#abd9e9"),  # 9–26 [REMOVED_ZH:2]
    (0.55, "#ffffbf"),  # 26–32 [REMOVED_ZH:2]
    (0.65, "#fdae61"),  # 32–38 [REMOVED_ZH:2]
    (0.78, "#f46d43"),  # 38–46 [REMOVED_ZH:4]
    (0.90, "#d73027"),
    (1.0,  "#a50026"),  # >46 [REMOVED_ZH:2]
], N=256)
UTCI_NORM = mcolors.Normalize(vmin=18, vmax=50)


def load_sim(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def fig1_epw_forcing(forcing_csv: Path, out_dir: Path):
    """Fig 1: EPW [REMOVED_ZH:11]。"""
    import csv
    rows = []
    with open(forcing_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    hours  = [r["hour"] for r in rows]
    ta     = [r["ta"]   for r in rows]
    rh     = [r["rh"]   for r in rows]
    ghi    = [r["ghi"]  for r in rows]
    ws     = [r["wind_speed"] for r in rows]
    sol_alt= [r["solar_alt"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Phase 1 — EPW [REMOVED_ZH:15]", fontsize=13)

    axes[0,0].plot(hours, ta,  "r-o", ms=4); axes[0,0].set_ylabel("Ta [°C]");  axes[0,0].set_title("[REMOVED_ZH:4]")
    axes[0,1].plot(hours, rh,  "b-o", ms=4); axes[0,1].set_ylabel("RH [%]");   axes[0,1].set_title("Relative Humidity")
    axes[0,2].plot(hours, ghi, "y-o", ms=4); axes[0,2].set_ylabel("GHI [W/m²]"); axes[0,2].set_title("[REMOVED_ZH:5]")
    axes[1,0].plot(hours, ws,  "g-o", ms=4); axes[1,0].set_ylabel("WS [m/s]"); axes[1,0].set_title("Wind Speed")
    axes[1,1].plot(hours, sol_alt, "k-o", ms=4); axes[1,1].set_ylabel("Alt [°]"); axes[1,1].set_title("[REMOVED_ZH:5]")
    axes[1,2].axis("off")
    stats_txt = (f"Ta: {min(ta):.1f}–{max(ta):.1f} °C\n"
                  f"RH: {min(rh):.0f}–{max(rh):.0f} %\n"
                  f"GHI peak: {max(ghi):.0f} W/m²\n"
                  f"WS: {min(ws):.1f}–{max(ws):.1f} m/s")
    axes[1,2].text(0.1, 0.5, stats_txt, transform=axes[1,2].transAxes,
                    fontsize=12, va="center", family="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow"))

    for ax in axes.ravel():
        if ax.get_title():
            ax.set_xlabel("Hour"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "fig1_epw_forcing.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


def fig2_scenario_samples(scenarios: list, out_dir: Path, n_show: int = 6):
    """Fig 2: [REMOVED_ZH:8] + [REMOVED_ZH:4]。"""
    import random
    rng = random.Random(0)
    sample = rng.sample(scenarios, min(n_show, len(scenarios)))

    cols = min(n_show, 3)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in row]
    fig.suptitle("Phase 1 — [REMOVED_ZH:10]", fontsize=13)

    for idx, sc in enumerate(sample):
        ax  = axes[idx]
        site = sc["site_polygon"]
        sx, sy = zip(*list(site.exterior.coords))
        ax.fill(sx, sy, fc="#e8f4e8", ec="k", lw=1.5, label="[REMOVED_ZH:2]")

        for b in sc["buildings"]:
            fp = b["footprint"]
            bx, by = zip(*list(fp.exterior.coords))
            ax.fill(bx, by, fc="#7f9fbf", ec="#2c5f8a", lw=1, alpha=0.8)

        for t in sc.get("trees", []):
            circ = plt.Circle(t["pos"], t.get("radius", 3.0),
                               color="#2e7d32", alpha=0.7, zorder=3)
            ax.add_patch(circ)

        ax.set_xlim(-2, sc["site_polygon"].bounds[2]+2)
        ax.set_ylim(-2, sc["site_polygon"].bounds[3]+2)
        ax.set_aspect("equal")
        n_tr = len(sc.get("trees", []))
        ax.set_title(f"Scene {sc['scenario_id']:03d}\n"
                      f"FAR={sc['far_actual']:.2f}  BCR={sc['bcr_actual']:.2f}  "
                      f"[REMOVED_ZH:1]={n_tr}",
                      fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.tight_layout()
    p = out_dir / "fig2_scenario_samples.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


def fig3_utci_heatmap(sim_dir: Path, out_dir: Path, n_show: int = 3):
    """Fig 3: [REMOVED_ZH:4] UTCI [REMOVED_ZH:6]（[REMOVED_ZH:3]）。"""
    npz_files = sorted(Path(sim_dir).glob("sim_????.npz"))[:n_show]
    if not npz_files:
        print("  [fig3] [REMOVED_ZH:1] sim_XXXX.npz，[REMOVED_ZH:2]")
        return

    hours_to_show = [0, 4, 8]   # 8h, 12h, 16h（[REMOVED_ZH:2]）

    for fp in npz_files:
        d   = load_sim(fp)
        sid = int(d["scenario_id"])
        pts = d["sensor_pts"]   # (N, 2)
        utci= d["utci"]         # (T, N)
        T   = utci.shape[0]
        sim_hours = d["sim_hours"].tolist()

        n_col = len(hours_to_show)
        fig, axes = plt.subplots(1, n_col, figsize=(n_col * 5, 5))
        fig.suptitle(f"Scene {sid:03d} — UTCI [REMOVED_ZH:4]（[REMOVED_ZH:2] EPW [REMOVED_ZH:5]）",
                      fontsize=12)

        for i, t_idx in enumerate(hours_to_show):
            if t_idx >= T:
                axes[i].axis("off")
                continue
            ax   = axes[i]
            vals = utci[t_idx]
            sc   = ax.scatter(pts[:, 0], pts[:, 1], c=vals,
                               cmap=UTCI_CMAP, norm=UTCI_NORM,
                               s=20, marker="s")
            plt.colorbar(sc, ax=ax, label="UTCI [°C]", shrink=0.8)
            hr  = sim_hours[t_idx] if t_idx < len(sim_hours) else t_idx + 8
            ax.set_title(f"{hr:02d}:00  μ={vals.mean():.1f}°C", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

        fig.tight_layout()
        p = out_dir / f"fig3_utci_heatmap_s{sid:03d}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {p}")


def fig4_shadow_timeseries(sim_dir: Path, out_dir: Path):
    """Fig 4: Diurnal variation curve of shadow coverage (verifying solar geometric calculations)"""
    files = sorted(Path(sim_dir).glob("sim_????.npz"))[:10]
    if not files:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Phase 1 — Shadow[REMOVED_ZH:6]（[REMOVED_ZH:6] + [REMOVED_ZH:4]）", fontsize=12)

    for fp in files:
        d   = load_sim(fp)
        shd = d["in_shadow"].astype(float)   # (T, N)
        frac = shd.mean(axis=1)               # (T,)
        sim_hours = d["sim_hours"].tolist()
        ax.plot(sim_hours, frac * 100,
                 alpha=0.5, lw=1.2, label=f"s{int(d['scenario_id']):03d}")

    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Shadow Coverage [%]")
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "fig4_shadow_timeseries.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


def fig5_stats_distribution(sim_dir: Path, out_dir: Path):
    """Fig 5: UTCI / MRT / Ta [REMOVED_ZH:4]（[REMOVED_ZH:6]）。"""
    files  = sorted(Path(sim_dir).glob("sim_????.npz"))
    if not files:
        return
    utci_all, mrt_all, ta_all, va_all = [], [], [], []
    for fp in files[:50]:          # [REMOVED_ZH:2] 50 [REMOVED_ZH:1]
        d = load_sim(fp)
        utci_all.append(d["utci"].ravel())
        mrt_all.append(d["mrt"].ravel())
        ta_all.append(d["ta"].ravel())
        va_all.append(d["va"].ravel())
    utci_all = np.concatenate(utci_all)
    mrt_all  = np.concatenate(mrt_all)
    ta_all   = np.concatenate(ta_all)
    va_all   = np.concatenate(va_all)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Phase 1 — [REMOVED_ZH:6]（[REMOVED_ZH:1] 50 [REMOVED_ZH:2]）", fontsize=12)

    for ax, data, label, color in zip(
        axes,
        [utci_all, mrt_all, ta_all, va_all],
        ["UTCI [°C]", "MRT [°C]", "Ta [°C]", "WS [m/s]"],
        ["coral", "orange", "steelblue", "seagreen"]
    ):
        ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(np.mean(data), color="k", ls="--", lw=1.5,
                    label=f"μ={np.mean(data):.1f}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "fig5_stats_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


def main(sim_dir: str, out_dir: str, scenarios_pkl: str = None,
          forcing_csv: str = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[viz_phase1] ── Phase 1 [REMOVED_ZH:5] ──")

    if forcing_csv and Path(forcing_csv).exists():
        fig1_epw_forcing(Path(forcing_csv), out)

    if scenarios_pkl and Path(scenarios_pkl).exists():
        with open(scenarios_pkl, "rb") as f:
            scenarios = pickle.load(f)
        fig2_scenario_samples(scenarios, out)

    fig3_utci_heatmap(Path(sim_dir), out)
    fig4_shadow_timeseries(Path(sim_dir), out)
    fig5_stats_distribution(Path(sim_dir), out)

    print(f"\n[viz_phase1] [REMOVED_ZH:2]，[REMOVED_ZH:5] {out}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir",  default="../outputs/raw_simulations")
    ap.add_argument("--out_dir",  default="viz_output/phase1")
    ap.add_argument("--scenarios",default="../outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--forcing_csv",
                    default="../outputs/raw_simulations/forcing_M07.csv")
    args = ap.parse_args()
    main(args.sim_dir, args.out_dir, args.scenarios, args.forcing_csv)