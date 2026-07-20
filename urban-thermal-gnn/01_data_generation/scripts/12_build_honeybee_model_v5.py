"""
01_data_generation/scripts/12_build_honeybee_model_v5.py
════════════════════════════════════════════════════════════════
V5 Honeybee model builder.

Converts each V5 scenario (real OSM building footprints/heights + real ETH
canopy trees, from scenarios_v4.pkl via 11_select_v5_subset.py) into a
Honeybee model:

  * Buildings -> Rooms, extruded from the real footprint to the real height,
    with a generic lightweight program/construction + ideal-air system so
    EnergyPlus can produce plausible envelope surface temperatures (these
    feed the recipe's longwave MRT map for nearby sensors). Indoor comfort
    itself is not of interest here.
  * Trees -> horizontal disc Shades at the tree's real canopy height and
    radius, given a semi-transparent ("Trans") Radiance modifier to
    approximate canopy porosity rather than treating trees as fully opaque.
  * A SensorGrid at z=1.5 m built with the exact same grid-generation logic
    (build_sensor_grid) V4 used, so V5 sensor points are directly comparable
    to V4's per-scenario sensor points.

Output: one HBJSON per scenario in real_simulations_v5/hbjson/.
"""
from __future__ import annotations

import sys
import json
import pickle
import argparse
import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.polyface import Polyface3D
from honeybee.room import Room
from honeybee.shade import Shade
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.modifier.material import Trans
from honeybee_energy.lib.programtypes import program_type_by_identifier
from honeybee_energy.lib.constructionsets import construction_set_by_identifier

# ── reuse V4's real geometry helpers (pruning + sensor grid) ─────
_spec = importlib.util.spec_from_file_location(
    "lbt_runner", str(_SCRIPT_DIR / "03_lbt_batch_runner.py"))
_lbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbt)
build_sensor_grid = _lbt.build_sensor_grid

_spec2 = importlib.util.spec_from_file_location(
    "v4_runner", str(_SCRIPT_DIR / "09_run_real_sim_v4.py"))
_v4 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_v4)
prune_scene = _v4.prune_scene

GRID_SPACING = 4.0      # same as V4's real-scene runner
SENSOR_Z = 1.5
TREE_TRANSMITTANCE = 0.4   # canopy porosity approximation
TREE_MODIFIER = Trans(
    "canopy_trans", r_reflectance=0.15, g_reflectance=0.2, b_reflectance=0.15,
    transmitted_diff=TREE_TRANSMITTANCE, transmitted_spec=0.0)

_PROGRAM = program_type_by_identifier("2019::MidriseApartment::Apartment")
_CONSTR = construction_set_by_identifier("2019::ClimateZone1::SteelFramed")


def _disc_points(cx, cy, r, n=10):
    return [(cx + r * np.cos(2 * np.pi * i / n), cy + r * np.sin(2 * np.pi * i / n))
            for i in range(n)]


def build_model(scenario: dict) -> Model:
    sid = scenario["scenario_id"]
    scp = prune_scene(scenario)

    rooms = []
    for i, b in enumerate(scp["buildings"]):
        fp = b["footprint"]
        if fp.is_empty or fp.area < 4.0:
            continue
        coords = list(fp.exterior.coords)[:-1]
        if len(coords) < 3:
            continue
        pts = [Point3D(x, y, 0) for x, y in coords]
        face = Face3D(pts)
        if face.normal.z < 0:  # ensure outward-up winding for offset
            face = face.flip()
        height = max(3.0, float(b["height"]))
        polyface = Polyface3D.from_offset_face(face, height)
        room = Room.from_polyface3d(f"bldg_{sid}_{i}", polyface)
        room.properties.energy.program_type = _PROGRAM
        room.properties.energy.construction_set = _CONSTR
        room.properties.energy.add_default_ideal_air()
        rooms.append(room)

    shades = []
    for j, t in enumerate(scp["trees"]):
        cx, cy = t["pos"]
        r = float(t.get("radius", t["height"] * 0.4))
        h = float(t["height"])
        pts = [Point3D(x, y, h) for x, y in _disc_points(cx, cy, r)]
        shd = Shade(f"tree_{sid}_{j}", Face3D(pts))
        shd.properties.radiance.modifier = TREE_MODIFIER
        shades.append(shd)

    if not rooms:
        raise ValueError(f"scenario {sid}: no valid buildings after pruning")

    sensor_pts = build_sensor_grid(scp, GRID_SPACING)
    if len(sensor_pts) < 4:
        raise ValueError(f"scenario {sid}: only {len(sensor_pts)} sensor points")
    positions = [(float(x), float(y), SENSOR_Z) for x, y in sensor_pts]
    directions = [(0, 0, 1)] * len(positions)
    grid = SensorGrid.from_position_and_direction(f"grid_{sid}", positions, directions)

    model = Model(f"scene_{sid:04d}", rooms, orphaned_shades=shades)
    model.properties.radiance.sensor_grids = [grid]
    return model


def main(scenarios_pkl: str, out_dir: str):
    with open(scenarios_pkl, "rb") as f:
        scenarios = pickle.load(f)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        try:
            model = build_model(sc)
        except Exception as e:
            print(f"  [skip] scene {sid}: {e}")
            continue
        model.to_hbjson(name=f"scene_{sid:04d}", folder=str(out))
        n_rooms = len(model.rooms)
        n_shades = len(model.orphaned_shades)
        n_sensors = len(model.properties.radiance.sensor_grids[0].sensors)
        manifest.append({"scenario_id": sid, "n_rooms": n_rooms,
                          "n_shades": n_shades, "n_sensors": n_sensors})
        print(f"  scene {sid:04d}: {n_rooms} bldgs, {n_shades} trees, "
              f"{n_sensors} sensors")

    (out / "hbjson_manifest_v5.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build_honeybee_v5] built {len(manifest)}/{len(scenarios)} models -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"))
    ap.add_argument("--out", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "hbjson"))
    args = ap.parse_args()
    main(args.scenarios, args.out)
