"""Flight optimization research demo entry point."""

from __future__ import annotations

import argparse
import os

import optuna

from airway_data import SUPPORTED_SOURCES


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(
        description="Flight optimization with synthetic or real airway networks"
    )
    parser.add_argument(
        "--mode",
        choices=["pso", "optuna", "monte-carlo", "pareto"],
        default="pso",
        help="Workflow to run",
    )
    parser.add_argument(
        "--aircraft",
        choices=["A320", "B787"],
        default=os.getenv("AIRCRAFT_TYPE", "A320"),
        help="Aircraft performance profile to use",
    )
    parser.add_argument(
        "--airway-source",
        choices=sorted(SUPPORTED_SOURCES),
        default=os.getenv("AIRWAY_DATA_SOURCE", "synthetic"),
        help="Data source used to build the airway graph",
    )
    parser.add_argument(
        "--airway-path",
        default=os.getenv("AIRWAY_DATA_PATH", ""),
        help="Path to a dataset file or directory",
    )
    parser.add_argument(
        "--origin",
        default=os.getenv("AIRWAY_ORIGIN", "A"),
        help="Origin node, airport code, or waypoint identifier",
    )
    parser.add_argument(
        "--destination",
        default=os.getenv("AIRWAY_DESTINATION", "B"),
        help="Destination node, airport code, or waypoint identifier",
    )
    parser.add_argument(
        "--route-count",
        type=int,
        default=int(os.getenv("AIRWAY_ROUTE_COUNT", "8")),
        help="Number of candidate routes to keep from the real graph",
    )
    parser.add_argument(
        "--snap-precision-deg",
        type=float,
        default=float(os.getenv("AIRWAY_SNAP_PRECISION_DEG", "0.05")),
        help="OpenSky/NATS track snapping precision in degrees",
    )
    parser.add_argument(
        "--segment-gap-minutes",
        type=float,
        default=float(os.getenv("AIRWAY_SEGMENT_GAP_MINUTES", "30")),
        help="Gap threshold used to split track data into segments",
    )
    parser.add_argument(
        "--pso-w",
        type=float,
        default=float(os.getenv("PSO_W", "0.7")),
        help="Inertia weight for a single PSO run",
    )
    parser.add_argument(
        "--pso-c1",
        type=float,
        default=float(os.getenv("PSO_C1", "1.5")),
        help="Cognitive coefficient for a single PSO run",
    )
    parser.add_argument(
        "--pso-c2",
        type=float,
        default=float(os.getenv("PSO_C2", "1.5")),
        help="Social coefficient for a single PSO run",
    )
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=int(os.getenv("PSO_SWARM_SIZE", "40")),
        help="Swarm size for a single PSO run",
    )
    parser.add_argument(
        "--pso-iters",
        type=int,
        default=int(os.getenv("PSO_ITERS", "80")),
        help="Iteration count for a single PSO run",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=int(os.getenv("OPTUNA_TRIALS", "20")),
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=int(os.getenv("MC_RUNS", "50")),
        help="Monte Carlo repetitions for uncertainty estimation",
    )
    parser.add_argument(
        "--route-index",
        type=int,
        default=0,
        help="Candidate route index used for Monte Carlo mode",
    )
    parser.add_argument(
        "--pareto-population",
        type=int,
        default=int(os.getenv("PARETO_POPULATION", "100")),
        help="Population size for Pareto mode",
    )
    parser.add_argument(
        "--pareto-generations",
        type=int,
        default=int(os.getenv("PARETO_GENERATIONS", "50")),
        help="Generation count for Pareto mode",
    )
    args = parser.parse_args()

    _set_env("AIRCRAFT_TYPE", args.aircraft)
    _set_env("AIRWAY_DATA_SOURCE", args.airway_source)
    _set_env("AIRWAY_DATA_PATH", args.airway_path)
    _set_env("AIRWAY_ORIGIN", args.origin)
    _set_env("AIRWAY_DESTINATION", args.destination)
    _set_env("AIRWAY_ROUTE_COUNT", str(args.route_count))
    _set_env("AIRWAY_SNAP_PRECISION_DEG", str(args.snap_precision_deg))
    _set_env("AIRWAY_SEGMENT_GAP_MINUTES", str(args.segment_gap_minutes))

    # Import the network-dependent modules only after the configuration is set.
    from network_model import NETWORK_METADATA, NETWORK_SOURCE
    from optimization_model import optuna_objective, run_nsga, run_pso
    from route_model import decode_particle, evaluate_route, monte_carlo_route_stats
    from network_model import ROUTE_CANDIDATES
    from visualization_model import plot_pareto, plot_route

    print("Airway source:", NETWORK_SOURCE)
    if NETWORK_METADATA.get("fallback_reason"):
        print("Fallback:", NETWORK_METADATA["fallback_reason"])
    if NETWORK_METADATA.get("requested_origin") or NETWORK_METADATA.get("requested_destination"):
        print(
            "Requested corridor:",
            NETWORK_METADATA.get("requested_origin"),
            "->",
            NETWORK_METADATA.get("requested_destination"),
        )

    if args.mode == "pso":
        best_cost, best_pos = run_pso(
            w=args.pso_w,
            c1=args.pso_c1,
            c2=args.pso_c2,
            swarm_size=args.swarm_size,
            iters=args.pso_iters,
            aircraft_type=args.aircraft,
        )
        route = decode_particle(best_pos)
        print("PSO route:", route)
        print("PSO cost:", best_cost)
        plot_route(route)
        return

    if args.mode == "optuna":
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: optuna_objective(trial, aircraft_type=args.aircraft),
            n_trials=args.optuna_trials,
        )
        print("Best params")
        print(study.best_params)

        best_cost, best_pos = run_pso(
            w=study.best_params["w"],
            c1=study.best_params["c1"],
            c2=study.best_params["c2"],
            swarm_size=study.best_params["swarm"],
            iters=study.best_params["iters"],
            aircraft_type=args.aircraft,
        )
        route = decode_particle(best_pos)
        print("Best route:", route)
        print("Best cost:", best_cost)
        plot_route(route)
        return

    if args.mode == "monte-carlo":
        if not ROUTE_CANDIDATES:
            raise RuntimeError("No candidate routes are available for Monte Carlo mode")
        route = ROUTE_CANDIDATES[args.route_index % len(ROUTE_CANDIDATES)]
        fuel_mean, time_mean, risk = monte_carlo_route_stats(
            route,
            runs=args.mc_runs,
            aircraft_type=args.aircraft,
        )
        fuel_single, time_single, co2_single = evaluate_route(
            route,
            aircraft_type=args.aircraft,
        )
        print("Selected route:", route)
        print("Single-run fuel:", fuel_single)
        print("Single-run time:", time_single)
        print("Single-run CO2:", co2_single)
        print("Monte Carlo mean fuel:", fuel_mean)
        print("Monte Carlo mean time:", time_mean)
        print("Monte Carlo risk:", risk)
        plot_route(route)
        return

    if args.mode == "pareto":
        pop = run_nsga(
            aircraft_type=args.aircraft,
            population_size=args.pareto_population,
            generations=args.pareto_generations,
            mc_runs=args.mc_runs,
        )
        plot_pareto(pop)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
