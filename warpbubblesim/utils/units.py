"""
Unit conversion utilities for WarpBubbleSim.

Default internal units: G = c = 1 (geometric units)
All lengths are measured in units of some reference length L_ref.
Times are measured in units of L_ref/c = L_ref (since c=1).

For typical warp drive simulations, L_ref might be 1 meter or 1 km.
The simulator is scale-independent; units only matter for interpretation.
"""

import numpy as np
from warpbubblesim.config import CONSTANTS


def geometric_to_si(
    value: float,
    quantity: str,
    length_scale: float = 1.0,
    mass_scale: float | None = None
) -> float:
    """
    Convert from geometric units (G=c=1) to SI units.

    Parameters
    ----------
    value : float
        Value in geometric units.
    quantity : str
        Type of quantity: 'length', 'time', 'mass', 'energy_density', 'velocity'.
    length_scale : float
        Reference length scale in meters (L_ref).
    mass_scale : float, optional
        Reference mass scale in kg. If None, derived from length_scale.

    Returns
    -------
    float
        Value in SI units.
    """
    c = CONSTANTS["c"]
    G = CONSTANTS["G"]

    # If mass_scale not given, use L_ref * c^2 / G
    if mass_scale is None:
        mass_scale = length_scale * c**2 / G

    if quantity == "length":
        return value * length_scale
    elif quantity == "time":
        return value * length_scale / c
    elif quantity == "mass":
        return value * mass_scale
    elif quantity == "velocity":
        return value * c
    elif quantity == "energy_density":
        # ρ in geometric has units 1/L^2, convert to J/m^3
        return value * c**4 / (G * length_scale**2)
    elif quantity == "curvature":
        # Curvature has units 1/L^2
        return value / length_scale**2
    else:
        raise ValueError(f"Unknown quantity type: {quantity}")


def si_to_geometric(
    value: float,
    quantity: str,
    length_scale: float = 1.0,
    mass_scale: float | None = None
) -> float:
    """
    Convert from SI units to geometric units (G=c=1).

    Parameters
    ----------
    value : float
        Value in SI units.
    quantity : str
        Type of quantity: 'length', 'time', 'mass', 'energy_density', 'velocity'.
    length_scale : float
        Reference length scale in meters (L_ref).
    mass_scale : float, optional
        Reference mass scale in kg. If None, derived from length_scale.

    Returns
    -------
    float
        Value in geometric units.
    """
    c = CONSTANTS["c"]
    G = CONSTANTS["G"]

    if mass_scale is None:
        mass_scale = length_scale * c**2 / G

    if quantity == "length":
        return value / length_scale
    elif quantity == "time":
        return value * c / length_scale
    elif quantity == "mass":
        return value / mass_scale
    elif quantity == "velocity":
        return value / c
    elif quantity == "energy_density":
        return value * G * length_scale**2 / c**4
    elif quantity == "curvature":
        return value * length_scale**2
    else:
        raise ValueError(f"Unknown quantity type: {quantity}")


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert length between different units.

    Supported units: m, km, ly (light-year), pc (parsec), au
    """
    # Convert to meters first
    to_m = {
        "m": 1.0,
        "km": 1000.0,
        "ly": CONSTANTS["ly"],
        "pc": CONSTANTS["pc"],
        "au": 1.495978707e11,
    }

    if from_unit not in to_m or to_unit not in to_m:
        raise ValueError(f"Unknown unit. Supported: {list(to_m.keys())}")

    value_m = value * to_m[from_unit]
    return value_m / to_m[to_unit]


def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert time between different units.

    Supported units: s, min, hr, day, yr
    """
    to_s = {
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
        "yr": 365.25 * 86400.0,
    }

    if from_unit not in to_s or to_unit not in to_s:
        raise ValueError(f"Unknown unit. Supported: {list(to_s.keys())}")

    value_s = value * to_s[from_unit]
    return value_s / to_s[to_unit]


def convert_mass(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert mass between different units.

    Supported units: kg, g, M_sun (solar masses)
    """
    to_kg = {
        "kg": 1.0,
        "g": 1e-3,
        "M_sun": CONSTANTS["M_sun"],
    }

    if from_unit not in to_kg or to_unit not in to_kg:
        raise ValueError(f"Unknown unit. Supported: {list(to_kg.keys())}")

    value_kg = value * to_kg[from_unit]
    return value_kg / to_kg[to_unit]


def convert_energy_density(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert energy density between units.

    Supported units: J/m^3, erg/cm^3, geometric (1/L^2 in G=c=1)
    For geometric, assumes L_ref = 1 m.
    """
    c = CONSTANTS["c"]
    G = CONSTANTS["G"]

    # Convert to J/m^3 first
    to_jm3 = {
        "J/m^3": 1.0,
        "erg/cm^3": 0.1,  # 1 erg/cm^3 = 0.1 J/m^3
        "geometric": c**4 / G,  # assuming L_ref = 1 m
    }

    if from_unit not in to_jm3 or to_unit not in to_jm3:
        raise ValueError(f"Unknown unit. Supported: {list(to_jm3.keys())}")

    value_jm3 = value * to_jm3[from_unit]
    return value_jm3 / to_jm3[to_unit]


def schwarzschild_radius(mass_kg: float) -> float:
    """
    Compute the Schwarzschild radius for a given mass.

    Parameters
    ----------
    mass_kg : float
        Mass in kilograms.

    Returns
    -------
    float
        Schwarzschild radius in meters.
    """
    return 2 * CONSTANTS["G"] * mass_kg / CONSTANTS["c"]**2


def planck_units() -> dict[str, float]:
    """
    Return Planck units in SI.

    Returns
    -------
    dict
        Dictionary with Planck length, time, mass, and energy density.
    """
    c = CONSTANTS["c"]
    G = CONSTANTS["G"]
    hbar = 1.054571817e-34  # reduced Planck constant

    l_p = np.sqrt(hbar * G / c**3)
    t_p = l_p / c
    m_p = np.sqrt(hbar * c / G)
    rho_p = m_p * c**2 / l_p**3

    return {
        "length": l_p,
        "time": t_p,
        "mass": m_p,
        "energy_density": rho_p,
    }
