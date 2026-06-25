"""
PSO, Optuna, and NSGA-II wrappers.
"""

import random

import numpy as np
import optuna
from deap import base, creator, tools, algorithms
from pyswarms.single import GlobalBestPSO

from aircraft_profiles import DEFAULT_AIRCRAFT
from route_model import decode_particle, evaluate_route, monte_carlo_cost


def swarm_objective(X, aircraft_type=DEFAULT_AIRCRAFT):
    vals = []

    for particle in X:
        route = decode_particle(particle)
        vals.append(monte_carlo_cost(route, aircraft_type=aircraft_type))

    return np.array(vals)


def run_pso(w, c1, c2, swarm_size, iters, aircraft_type=DEFAULT_AIRCRAFT):
    optimizer = GlobalBestPSO(
        n_particles=swarm_size,
        dimensions=3,
        options={
            "c1": c1,
            "c2": c2,
            "w": w,
        },
        bounds=(
            np.zeros(3),
            np.ones(3),
        ),
    )

    cost, pos = optimizer.optimize(
        lambda X: swarm_objective(X, aircraft_type=aircraft_type),
        iters=iters,
        verbose=False,
    )

    return cost, pos


def optuna_objective(trial, aircraft_type=DEFAULT_AIRCRAFT):
    cost, _ = run_pso(
        w=trial.suggest_float("w", 0.3, 0.95),
        c1=trial.suggest_float("c1", 0.5, 3),
        c2=trial.suggest_float("c2", 0.5, 3),
        swarm_size=trial.suggest_int("swarm", 20, 100),
        iters=trial.suggest_int("iters", 50, 150),
        aircraft_type=aircraft_type,
    )

    return cost


def build_nsga_toolbox(aircraft_type=DEFAULT_AIRCRAFT):
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=3,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def nsga_eval(ind):
        route = decode_particle(ind)
        return evaluate_route(route, aircraft_type=aircraft_type)

    toolbox.register("evaluate", nsga_eval)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def run_nsga(aircraft_type=DEFAULT_AIRCRAFT, population_size=100, generations=50):
    toolbox = build_nsga_toolbox(aircraft_type=aircraft_type)
    pop = toolbox.population(n=population_size)

    algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=population_size,
        lambda_=population_size,
        cxpb=0.7,
        mutpb=0.3,
        ngen=generations,
        verbose=False,
    )

    return pop
