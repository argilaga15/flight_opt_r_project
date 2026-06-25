"""
Aircraft fuel burn model.
"""

import numpy as np

from aircraft_profiles import DEFAULT_AIRCRAFT, get_aircraft_profile


def fuel_burn_segment(distance, mass, wind_component, aircraft_type=DEFAULT_AIRCRAFT):
    profile = get_aircraft_profile(aircraft_type)

    gs = max(profile["cruise_speed"] + wind_component, 120)
    time = distance / gs

    mass_tonnes = mass / 1000.0
    burn_factor = np.interp(
        mass_tonnes,
        profile["mass_curve_t"],
        profile["burn_factor"],
    )
    burn_rate = burn_factor * mass

    fuel = burn_rate * time
    return fuel, time
