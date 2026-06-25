"""Route decoding and evaluation."""

from __future__ import annotations

import numpy as np

from aircraft_profiles import DEFAULT_AIRCRAFT, get_aircraft_profile
from fuel_model import fuel_burn_segment
from network_model import GRAPH, NODES, ROUTE_CANDIDATES, ROUTE_DECISION_DIMENSIONS
from weather_model import wind_effect


def decode_particle(x):
    """Map a PSO/NSGA particle onto one of the candidate airway routes."""

    if ROUTE_CANDIDATES:
        bits = min(len(x), ROUTE_DECISION_DIMENSIONS)
        index = 0
        for bit in range(bits):
            if x[bit] >= 0.5:
                index |= 1 << bit
        return ROUTE_CANDIDATES[index % len(ROUTE_CANDIDATES)]

    # Fallback for unexpected cases where the network could not be initialized.
    return ["A", "W1", "W3", "W5", "B"] if ("W1" in NODES and "W5" in NODES) else ["A", "B"]


def evaluate_route(route, weather_strength=1.0, aircraft_type=DEFAULT_AIRCRAFT):
    profile = get_aircraft_profile(aircraft_type)

    fuel = profile["initial_fuel"]
    mass = profile["empty_mass"] + profile["payload"] + fuel

    total_fuel = 0.0
    total_time = 0.0

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        edge = GRAPH[u][v]
        p1 = np.array(NODES[u])
        p2 = np.array(NODES[v])

        wind = wind_effect(p1, p2, weather_strength)
        burn, t = fuel_burn_segment(
            edge["distance"],
            mass,
            wind,
            aircraft_type=aircraft_type,
        )

        fuel -= burn
        mass -= burn
        total_fuel += burn
        total_time += t

    co2 = total_fuel * 3.16
    return total_fuel, total_time, co2


def monte_carlo_cost(route, runs=20, aircraft_type=DEFAULT_AIRCRAFT):
    costs = []
    for _ in range(runs):
        weather = np.random.normal(1.0, 0.2)
        fuel, time, _ = evaluate_route(route, weather, aircraft_type=aircraft_type)
        costs.append(fuel + 500.0 * time)

    return float(np.mean(costs))


def monte_carlo_route_stats(route, runs=20, aircraft_type=DEFAULT_AIRCRAFT):
    """Return mean fuel, mean time, and weather-risk for NSGA-II.

    The risk term is the standard deviation of a combined operational cost
    under randomized weather strength. This keeps the third objective from
    collapsing into a deterministic multiple of fuel.
    """

    fuel_values = []
    time_values = []
    combined_costs = []

    for _ in range(runs):
        weather = np.random.normal(1.0, 0.2)
        fuel, time, _ = evaluate_route(route, weather, aircraft_type=aircraft_type)
        fuel_values.append(fuel)
        time_values.append(time)
        combined_costs.append(fuel + 500.0 * time)

    return (
        float(np.mean(fuel_values)),
        float(np.mean(time_values)),
        float(np.std(combined_costs)),
    )
