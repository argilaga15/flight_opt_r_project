"""
Aircraft-specific performance profiles.
"""

import numpy as np

AIRCRAFT_PROFILES = {
    "A320": {
        "empty_mass": 42600,
        "payload": 16000,
        "initial_fuel": 15000,
        "cruise_speed": 230.0,
        # Mass is in tonnes; burn_factor scales current mass in kg.
        "mass_curve_t": np.array([50.0, 60.0, 70.0, 80.0]),
        "burn_factor": np.array([0.000072, 0.000076, 0.000080, 0.000085]),
    },
    "B787": {
        "empty_mass": 128000,
        "payload": 22000,
        "initial_fuel": 70000,
        "cruise_speed": 250.0,
        "mass_curve_t": np.array([160.0, 190.0, 220.0, 250.0]),
        "burn_factor": np.array([0.000054, 0.000058, 0.000063, 0.000069]),
    },
}

DEFAULT_AIRCRAFT = "A320"


def get_aircraft_profile(aircraft_type):
    aircraft_key = aircraft_type.upper()

    if aircraft_key not in AIRCRAFT_PROFILES:
        raise ValueError(
            f"Unknown aircraft '{aircraft_type}'. "
            f"Available options: {', '.join(AIRCRAFT_PROFILES)}"
        )

    return AIRCRAFT_PROFILES[aircraft_key]
