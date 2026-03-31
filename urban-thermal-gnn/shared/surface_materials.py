"""
shared/surface_materials.py
════════════════════════════════════════════════════════════════
Material database and surface temperature calculation for urban thermal modeling.

Provides:
  - SurfaceMaterial: Thermal and radiative properties of ground surfaces
  - MATERIALS: Database of common urban materials (grass, concrete, asphalt, etc.)
  - compute_surface_temperature(): Energy balance-based surface temperature estimation
  - assign_materials_to_sensors(): Spatial zone-to-material mapping

Author: Urban Thermal GNN Team
Date: 2024
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


@dataclass
class SurfaceMaterial:
    """
    Thermal and radiative properties of ground surfaces.

    Attributes:
      name : str
        Material name (e.g., "Grass", "Concrete")
      albedo : float
        Solar reflectance [0, 1]. Higher values reflect more sunlight.
        Examples: grass=0.25, asphalt=0.08
      emissivity : float
        Thermal emissivity [0, 1]. Ability to emit thermal radiation.
        Most surfaces ≈ 0.88-0.95
      thermal_conductivity : float
        Thermal conductivity [W/(m·K)]. Heat diffusion into ground.
        Examples: grass=0.5, asphalt=0.75
      evapotranspiration_coeff : float
        Evapotranspiration coefficient [0, 1]. Cooling due to water evaporation.
        Higher values = stronger cooling effect.
        Examples: grass=0.8 (high ET), asphalt=0.0 (no water)
      roughness_length : float
        Aerodynamic roughness length [m]. Affects wind profile.
        Examples: grass=0.03m, concrete=0.005m
    """
    name: str
    albedo: float
    emissivity: float
    thermal_conductivity: float
    evapotranspiration_coeff: float
    roughness_length: float


# ════════════════════════════════════════════════════════════════
# Material Database
# ════════════════════════════════════════════════════════════════

MATERIALS: Dict[str, SurfaceMaterial] = {
    "grass": SurfaceMaterial(
        name="Grass / Vegetation",
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
        albedo=0.08,  # Very low albedo → high solar absorption
        emissivity=0.95,
        thermal_conductivity=0.75,
        evapotranspiration_coeff=0.0,  # No water on asphalt
        roughness_length=0.002,
    ),
    "water": SurfaceMaterial(
        name="Water Body / Pond",
        albedo=0.10,
        emissivity=0.96,
        thermal_conductivity=0.6,
        evapotranspiration_coeff=1.0,  # Maximum evaporation
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


# ════════════════════════════════════════════════════════════════
# Surface Temperature Computation
# ════════════════════════════════════════════════════════════════

def compute_surface_temperature(
    material_name: str,
    air_temp: float,
    ghi: float,
    wind_speed: float,
    relative_humidity: float = 0.5,
) -> float:
    """
    Estimate ground surface temperature using simplified energy balance.

    Physical Model:
    ───────────────
    Energy balance at surface:
      Q_absorbed = Q_emitted + Q_convection + Q_evaporation

    Where:
      Q_absorbed    = GHI × (1 - albedo)          [W/m²]
      Q_emitted     = σ × ε × (T_surface)^4       [W/m²] (Stefan-Boltzmann)
      Q_convection  = h_conv × (T_surface - T_air) [W/m²]
      Q_evaporation = ET_coeff × RH × L_v        [W/m²]

    Rearranging to solve for T_surface:
      ΔT = (Q_solar - Q_evap) / (h_conv + h_rad)
      T_surface = T_air + ΔT

    Parameters
    ──────────
    material_name : str
      Material type key: "grass", "concrete", "asphalt", "water", etc.
    air_temp : float
      Ambient air temperature [°C]
    ghi : float
      Global Horizontal Irradiance (solar radiation) [W/m²]
      Typical range: 0-1000 W/m² (0-100 W/m² at dawn/dusk, ~700-900 at noon)
    wind_speed : float
      Wind speed at measurement height [m/s]
    relative_humidity : float, optional
      Relative humidity [0, 1]. Default: 0.5 (50%)
      Affects evapotranspiration cooling effect.

    Returns
    ───────
    T_surface : float
      Estimated surface temperature [°C]

    Notes
    ─────
    - Simplified model (steady-state, ignores ground heat conduction)
    - Suitable for hourly timesteps
    - More sophisticated models would solve full heat balance with ground diffusion

    References
    ──────────
    - Karsisto, P. et al. (2016). Seasonal surface urban heat island
      characteristics in a Northern European city (Finnish). Clim. Res.
    """
    # Get material properties, default to concrete if unknown
    mat = MATERIALS.get(material_name, MATERIALS["concrete"])

    # ── 1. Solar radiation absorbed at surface ────────────────────────
    Q_solar = ghi * (1.0 - mat.albedo)  # [W/m²]

    # ── 2. Convective heat transfer coefficient (from empirical models) ──
    # Simple model: h = h_0 + c × wind_speed
    # References: ASHRAE (h0 ≈ 5.7 W/m²K, c ≈ 3.8 W/m²Ks/m)
    h_conv = 5.7 + 3.8 * wind_speed  # [W/m²K]

    # ── 3. Radiative heat transfer coefficient (linearized) ────────────
    # For T around 20-40°C, radiative h_rad ≈ 4-5 W/m²K (small contribution)
    sigma = 5.67e-8  # Stefan-Boltzmann constant [W/m²K⁴]
    T_mean_K = (air_temp + 273.15) + 20.0  # assume ΔT ~ 20K between air and surface
    h_rad = 4.0 * sigma * mat.emissivity * (T_mean_K ** 3.0)  # [W/m²K]

    # ── 4. Evapotranspiration cooling effect ──────────────────────────
    # Uses vapour pressure deficit (VPD) approach:
    #   ET ∝ ET_coeff × (1 - RH)   (evaporation driven by saturation deficit)
    # L_v = latent heat of vaporization ≈ 2.45 MJ/kg
    latent_heat = 2.45e6  # [J/kg]
    vpd_factor = max(0.0, 1.0 - relative_humidity)  # [0, 1] saturation deficit
    evap_rate = mat.evapotranspiration_coeff * vpd_factor * 0.00005  # [kg/(m²·s)]
    Q_evap = evap_rate * latent_heat  # [W/m²]

    # ── 5. Solve energy balance for temperature rise ────────────────────
    delta_T = (Q_solar - Q_evap) / (h_conv + h_rad + 1e-8)  # [K]
    T_surface = air_temp + delta_T

    # ── 6. Physically plausible bounds ────────────────────────────────
    # Surface temperature shouldn't exceed solar temperature or be arbitrarily cold
    T_surface = max(T_surface, air_temp - 10.0)  # Can't be >10°C colder than air
    if ghi > 500:  # Strong solar radiation
        T_surface = min(T_surface, air_temp + 50.0)  # Cap at +50K above air (typical max ~70°C)

    return float(T_surface)


# ════════════════════════════════════════════════════════════════
# Spatial Zone Assignment
# ════════════════════════════════════════════════════════════════

def assign_materials_to_sensors(
    sensor_pts: List[List[float]],
    material_zones: Dict[str, List[List[List[float]]]],
) -> Dict[Tuple[float, float], str]:
    """
    Assign surface materials to sensor points based on spatial zones.

    Implements point-in-polygon test for each sensor against material zone boundaries.
    If a sensor is outside all zones, defaults to "concrete".

    Parameters
    ──────────
    sensor_pts : List[List[float]]
      Sensor locations [[x1, y1], [x2, y2], ...]
    material_zones : Dict[str, List[List[List[float]]]]
      Material type → list of polygon boundaries
      Example:
        {
          "grass": [[[0,0],[10,0],[10,10],[0,10]]],
          "concrete": [[[10,0],[20,0],[20,20],[10,20]]]
        }

    Returns
    ───────
    sensor_materials : Dict[Tuple[float, float], str]
      Mapping: (x, y) → material name

    Notes
    ─────
    - Uses shapely.geometry for point-in-polygon testing
    - Falls back to bbox check if shapely not available
    - Sensor assigned to first matching zone (order matters)
    """
    sensor_materials = {}

    try:
        # Try to use shapely for accurate point-in-polygon tests
        from shapely.geometry import Point, Polygon

        for pt in sensor_pts:
            x, y = float(pt[0]), float(pt[1])
            point = Point(x, y)
            assigned = False

            # Check each material type
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

            # Default to concrete if not in any zone
            if not assigned:
                sensor_materials[(x, y)] = "concrete"

    except ImportError:
        # Fallback: simple bounding box check if shapely unavailable
        for pt in sensor_pts:
            x, y = float(pt[0]), float(pt[1])
            assigned = False

            for material, zone_list in material_zones.items():
                for zone_pts in zone_list:
                    if len(zone_pts) >= 3:
                        # Bounding box test (simple, less accurate)
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


# ════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════

def get_material(material_name: str) -> SurfaceMaterial:
    """
    Get material properties by name.

    Parameters
    ──────────
    material_name : str
      Material key or name

    Returns
    ───────
    SurfaceMaterial
      Material properties. Returns concrete as fallback if not found.
    """
    return MATERIALS.get(material_name, MATERIALS["concrete"])


def list_materials() -> List[str]:
    """Return list of available material names."""
    return list(MATERIALS.keys())


def compute_surface_temps_array(
    material_types: List[str],
    air_temps: np.ndarray,
    ghi_array: np.ndarray,
    wind_speeds: np.ndarray,
    relative_humidity: np.ndarray,
) -> np.ndarray:
    """
    Vectorized surface temperature computation for multiple sensors and timesteps.

    Parameters
    ──────────
    material_types : List[str]
      Material for each sensor (length N)
    air_temps : np.ndarray
      Shape (N, T) - air temperature for each sensor and timestep [°C]
    ghi_array : np.ndarray
      Shape (N, T) - global horizontal irradiance [W/m²]
    wind_speeds : np.ndarray
      Shape (N, T) - wind speed [m/s]
    relative_humidity : np.ndarray
      Shape (N, T) - relative humidity [0, 1]

    Returns
    ───────
    T_surface : np.ndarray
      Shape (N, T) - surface temperature for each sensor and timestep [°C]
    """
    N, T = air_temps.shape
    T_surface = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        mat_name = material_types[i] if i < len(material_types) else "concrete"
        for t in range(T):
            T_surface[i, t] = compute_surface_temperature(
                mat_name,
                float(air_temps[i, t]),
                float(ghi_array[i, t]),
                float(wind_speeds[i, t]),
                float(relative_humidity[i, t]),
            )

    return T_surface


def compute_surface_temp_scalar_batch(
    material_name: str,
    air_temps: np.ndarray,
    ghi: float,
    wind_speeds: np.ndarray,
    rh: float,
) -> np.ndarray:
    """
    Fast surface temperature for N sensors, single timestep, single material.

    Parameters
    ──────────
    material_name : str — material key (e.g. "concrete")
    air_temps : (N,) air temperatures [°C]
    ghi : scalar GHI [W/m²]
    wind_speeds : (N,) wind speeds [m/s]
    rh : scalar relative humidity [0, 1]

    Returns
    ───────
    T_surface : (N,) surface temperature [°C]
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


if __name__ == "__main__":
    """Example usage and testing"""
    print("=== Urban Surface Materials Database ===\n")
    print("Available materials:")
    for name in list_materials():
        mat = get_material(name)
        print(f"  {name:20s}: albedo={mat.albedo:.2f}, ET_coeff={mat.evapotranspiration_coeff:.2f}")

    print("\n=== Surface Temperature Example ===")
    print("Conditions: Air=25°C, GHI=700 W/m², Wind=2 m/s, RH=0.5")
    for material_name in ["grass", "concrete", "asphalt", "water"]:
        T_surf = compute_surface_temperature(material_name, 25.0, 700.0, 2.0, 0.5)
        print(f"  {material_name:15s} → T_surface = {T_surf:6.2f}°C (ΔT = {T_surf-25:+6.2f}K)")
