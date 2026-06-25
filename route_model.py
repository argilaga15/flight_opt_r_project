"""
Route decoding and evaluation.
"""

import numpy as np

from aircraft_profiles import DEFAULT_AIRCRAFT, get_aircraft_profile
from fuel_model import fuel_burn_segment
from network_model import GRAPH, NODES
from weather_model import wind_effect


def decode_particle(x):
    route = ["A"]

    route.append("W1" if x[0] < 0.5 else "W2")

    if route[-1] == "W1":
        route.append("W3")
    else:
        route.append("W3" if x[1] < 0.5 else "W4")

    if route[-1] == "W3":
        route.append("W5")
    else:
        route.append("W5" if x[2] < 0.5 else "W6")

    route.append("B")
    return route


def evaluate_route(route, weather_strength=1.0, aircraft_type=DEFAULT_AIRCRAFT):
    profile = get_aircraft_profile(aircraft_type)

    fuel = profile["initial_fuel"]
    mass = profile["empty_mass"] + profile["payload"] + fuel

    total_fuel = 0
    total_time = 0

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
        costs.append(fuel + 500 * time)

    return np.mean(costs)
