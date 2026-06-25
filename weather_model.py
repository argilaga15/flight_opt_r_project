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
    norm = np.linalg.norm(direction)

    # Zero-length connector segments can appear at corridor endpoints when the
    # loader snaps data or attaches explicit origin/destination nodes.
    if norm == 0 or not np.isfinite(norm):
        return 0.0

    direction = direction / norm
    return float(np.dot(wind, direction))
