"""
shared/surface_materials.py
Surface material properties and ground surface temperature estimation.

Provides:
  - SurfaceMaterial dataclass with thermal/radiative properties
  - MATERIALS database: grass, concrete, asphalt, permeable_pavement, water, bare_soil
  - compute_surface_temperature()  — energy-balance surface temperature
  - compute_surface_temp_scalar_batch()  — vectorised version (N sensors)
  - assign_materials_to_sensors()  — spatial zone-to-material mapping
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


@dataclass
class SurfaceMaterial:
    """Thermal and radiative properties of ground surfaces."""
    name: str
    albedo: float                   # Solar reflectance (0-1)
    emissivity: float               # Thermal emissivity (0-1)
    thermal_conductivity: float     # W/(m*K)
    evapotranspiration_coeff: float # ET coefficient (0-1)
    roughness_length: float         # Aerodynamic roughness (m)


MATERIALS: Dict[str, SurfaceMaterial] = {
    "grass": SurfaceMaterial(
        name="Grass",
        albedo=0.25,
        emissivity=0.95,
        thermal_conductivity=0.5,
        evapotranspiration_coeff=0.8,
        roughness_length=0.03,
    ),
    "permeable_pavement": SurfaceMaterial(
        name="Permeable Pavement",
        albedo=0.30,
        emissivity=0.90,
        thermal_conductivity=1.0,
        evapotranspiration_coeff=0.4,
        roughness_length=0.01,
    ),
    "concrete": SurfaceMaterial(
        name="Concrete",
        albedo=0.35,
        emissivity=0.88,
        thermal_conductivity=1.7,
        evapotranspiration_coeff=0.1,
        roughness_length=0.005,
    ),
    "asphalt": SurfaceMaterial(
        name="Asphalt",
        albedo=0.08,
        emissivity=0.95,
        thermal_conductivity=0.75,
        evapotranspiration_coeff=0.0,
        roughness_length=0.002,
    ),
    "water": SurfaceMaterial(
        name="Water Body",
        albedo=0.10,
        emissivity=0.96,
        thermal_conductivity=0.6,
        evapotranspiration_coeff=1.0,
        roughness_length=0.0002,
    ),
    "bare_soil": SurfaceMaterial(
        name="Bare Soil",
        albedo=0.20,
        emissivity=0.92,
        thermal_conductivity=1.2,
        evapotranspiration_coeff=0.3,
        roughness_length=0.02,
    ),
}


# ────────────────────────────────────────────────────────────────
# Surface Temperature
# ────────────────────────────────────────────────────────────────

def compute_surface_temperature(
    material_name: str,
    air_temp: float,
    ghi: float,
    wind_speed: float,
    relative_humidity: float = 0.5,
) -> float:
    """
    Estimate ground surface temperature using simplified energy balance.

    Model:
      Q_solar = GHI * (1 - albedo)
      h_conv  = 5.7 + 3.8 * wind_speed          (ASHRAE)
      h_rad   = 4 * sigma * eps * T_mean^3       (linearised Stefan-Boltzmann)
      Q_evap  = ET_coeff * (1 - RH) * 0.00005 * L_v   (VPD-driven)
      dT      = (Q_solar - Q_evap) / (h_conv + h_rad)
      T_surface = T_air + dT

    Parameters
    ----------
    material_name : str   — key in MATERIALS dict
    air_temp      : float — ambient air temperature [C]
    ghi           : float — global horizontal irradiance [W/m2]
    wind_speed    : float — wind speed [m/s]
    relative_humidity : float — RH in [0, 1], default 0.5

    Returns
    -------
    float — estimated surface temperature [C]
    """
    mat = MATERIALS.get(material_name, MATERIALS["concrete"])

    # Solar absorption
    Q_solar = ghi * (1.0 - mat.albedo)

    # Convective heat transfer coefficient
    h_conv = 5.7 + 3.8 * wind_speed

    # Linearised radiative heat transfer coefficient
    sigma = 5.67e-8
    T_mean_K = (air_temp + 273.15) + 20.0
    h_rad = 4.0 * sigma * mat.emissivity * (T_mean_K ** 3.0)

    # Evapotranspiration cooling (VPD approach)
    latent_heat = 2.45e6  # J/kg
    vpd_factor = max(0.0, 1.0 - relative_humidity)
    evap_rate = mat.evapotranspiration_coeff * vpd_factor * 0.00005
    Q_evap = evap_rate * latent_heat

    # Energy balance
    delta_T = (Q_solar - Q_evap) / (h_conv + h_rad + 1e-8)
    T_surface = air_temp + delta_T

    # Physical bounds
    T_surface = max(T_surface, air_temp - 10.0)
    if ghi > 500:
        T_surface = min(T_surface, air_temp + 50.0)

    return float(T_surface)


def compute_surface_temp_scalar_batch(
    material_name: str,
    air_temps: np.ndarray,
    ghi: float,
    wind_speeds: np.ndarray,
    rh: float,
) -> np.ndarray:
    """
    Vectorised surface temperature for N sensors, single timestep, single material.

    Parameters
    ----------
    material_name : str      — material key
    air_temps     : (N,)     — air temperatures [C]
    ghi           : float    — GHI [W/m2]
    wind_speeds   : (N,)     — wind speeds [m/s]
    rh            : float    — relative humidity [0, 1]

    Returns
    -------
    (N,) surface temperature [C]
    """
    mat = MATERIALS.get(material_name, MATERIALS["concrete"])

    Q_solar = ghi * (1.0 - mat.albedo)
    h_conv = 5.7 + 3.8 * wind_speeds

    sigma = 5.67e-8
    T_mean_K = (air_temps + 273.15) + 20.0
    h_rad = 4.0 * sigma * mat.emissivity * (T_mean_K ** 3.0)

    vpd_factor = max(0.0, 1.0 - rh)
    evap_rate = mat.evapotranspiration_coeff * vpd_factor * 0.00005
    Q_evap = evap_rate * 2.45e6

    delta_T = (Q_solar - Q_evap) / (h_conv + h_rad + 1e-8)
    T_surface = air_temps + delta_T

    T_surface = np.clip(T_surface, air_temps - 10.0, air_temps + 50.0)
    return T_surface.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Spatial Zone Assignment
# ────────────────────────────────────────────────────────────────

def assign_materials_to_sensors(
    sensor_pts: list,
    material_zones: Dict[str, list],
) -> Dict[Tuple[float, float], str]:
    """
    Assign surface materials to sensor points based on spatial zones.

    Parameters
    ----------
    sensor_pts     : [[x1, y1], [x2, y2], ...]
    material_zones : {"grass": [[[x,y], ...]], "concrete": [...]}

    Returns
    -------
    {(x, y): "material_name"}  — defaults to "concrete" if outside all zones.
    """
    sensor_materials = {}

    try:
        from shapely.geometry import Point, Polygon

        for pt in sensor_pts:
            x, y = float(pt[0]), float(pt[1])
            point = Point(x, y)
            assigned = False

            for material, zone_list in material_zones.items():
                for zone_pts in zone_list:
                    if len(zone_pts) >= 3:
                        poly = Polygon(zone_pts)
                        if poly.contains(point):
                            sensor_materials[(x, y)] = material
                            assigned = True
                            break
                if assigned:
                    break

            if not assigned:
                sensor_materials[(x, y)] = "concrete"

    except ImportError:
        # Fallback: bounding-box test
        for pt in sensor_pts:
            x, y = float(pt[0]), float(pt[1])
            assigned = False

            for material, zone_list in material_zones.items():
                for zone_pts in zone_list:
                    if len(zone_pts) >= 3:
                        xs = [p[0] for p in zone_pts]
                        ys = [p[1] for p in zone_pts]
                        if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                            sensor_materials[(x, y)] = material
                            assigned = True
                            break
                if assigned:
                    break

            if not assigned:
                sensor_materials[(x, y)] = "concrete"

    return sensor_materials


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def get_material(material_name: str) -> SurfaceMaterial:
    """Get material by name. Falls back to concrete."""
    return MATERIALS.get(material_name, MATERIALS["concrete"])


def list_materials() -> List[str]:
    """Return available material names."""
    return list(MATERIALS.keys())
