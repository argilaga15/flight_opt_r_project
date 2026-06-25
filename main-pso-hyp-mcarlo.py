
"""
Flight Optimization Research Project
====================================
Entry point for the modularized flight optimization demo.
"""

import argparse

import optuna

from aircraft_profiles import AIRCRAFT_PROFILES, DEFAULT_AIRCRAFT
from optimization_model import optuna_objective, run_nsga, run_pso
from route_model import decode_particle
from visualization_model import plot_pareto, plot_route


def main():
    parser = argparse.ArgumentParser(
        description="Flight optimization with aircraft-specific performance curves"
    )
    parser.add_argument(
        "--aircraft",
        choices=sorted(AIRCRAFT_PROFILES),
        default=DEFAULT_AIRCRAFT,
        help="Aircraft performance profile to use",
    )
    args = parser.parse_args()

    aircraft_type = args.aircraft

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optuna_objective(trial, aircraft_type=aircraft_type),
        n_trials=20,
    )

    print("Best params")
    print(study.best_params)

    best_cost, best_pos = run_pso(
        w=study.best_params["w"],
        c1=study.best_params["c1"],
        c2=study.best_params["c2"],
        swarm_size=study.best_params["swarm"],
        iters=study.best_params["iters"],
        aircraft_type=aircraft_type,
    )

    route = decode_particle(best_pos)

    print("Best route:", route)
    print("Cost:", best_cost)

    plot_route(route)

    pop = run_nsga(aircraft_type=aircraft_type)
    plot_pareto(pop)


if __name__ == "__main__":
    main()
