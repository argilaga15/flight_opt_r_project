"""
Wind model used by the route evaluator.
"""

import numpy as np


def wind_vector(x, y, strength=1.0):
    vx = strength * 30 * np.sin(x / 250)
    vy = strength * 15 * np.cos(y / 150)
    return np.array([vx, vy])


def wind_effect(p1, p2, strength=1.0):
    midpoint = (p1 + p2) / 2
    wind = wind_vector(midpoint[0], midpoint[1], strength)
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)
    return np.dot(wind, direction)
