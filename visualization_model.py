"""
Plotting helpers for routes and Pareto fronts.
"""

import numpy as np
import matplotlib.pyplot as plt
from deap import tools

from network_model import AIRWAYS, NODES


def plot_route(route):
    plt.figure(figsize=(8, 5))

    for u, v in AIRWAYS:
        p1 = NODES[u]
        p2 = NODES[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "gray")

    coords = np.array([NODES[r] for r in route])
    plt.plot(coords[:, 0], coords[:, 1], linewidth=3)

    for node, pos in NODES.items():
        plt.text(pos[0], pos[1], node)

    plt.title("Optimized Route")
    plt.show()


def plot_pareto(pop):
    if not pop:
        return

    fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
    front = fronts[0] if fronts else []
    front_vals = np.array([ind.fitness.values for ind in front], dtype=float)
    all_vals = np.array([ind.fitness.values for ind in pop], dtype=float)

    plt.figure(figsize=(7, 5))

    if all_vals.size:
        unique_all = np.unique(np.round(all_vals, 8), axis=0)
        plt.scatter(
            unique_all[:, 0],
            unique_all[:, 1],
            c="lightgray",
            alpha=0.35,
            s=22,
            label="Population",
        )

    if front_vals.size:
        unique_front = np.unique(np.round(front_vals, 8), axis=0)
        scatter = plt.scatter(
            unique_front[:, 0],
            unique_front[:, 1],
            c=unique_front[:, 2] if unique_front.shape[1] > 2 else None,
            cmap="viridis",
            s=80,
            edgecolors="black",
            linewidths=0.5,
            label="Pareto front",
        )
        if unique_front.shape[1] > 2:
            plt.colorbar(scatter, label="Weather-risk std dev")

    plt.xlabel("Mean fuel")
    plt.ylabel("Mean time")
    plt.title("Pareto Front")
    plt.legend(loc="best")
    plt.show()
