"""
Plotting helpers for routes and Pareto fronts.
"""

import numpy as np
import matplotlib.pyplot as plt

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
    vals = np.array([ind.fitness.values for ind in pop])

    plt.figure(figsize=(7, 5))
    plt.scatter(vals[:, 0], vals[:, 1])
    plt.xlabel("Fuel")
    plt.ylabel("Time")
    plt.title("Pareto Front")
    plt.show()
