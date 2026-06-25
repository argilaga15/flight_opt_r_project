"""
Airway network definition.
"""

import networkx as nx
import numpy as np

NODES = {
    "A": (0, 0),
    "W1": (150, 100),
    "W2": (250, -50),
    "W3": (400, 150),
    "W4": (550, 0),
    "W5": (700, 100),
    "W6": (850, -50),
    "B": (1000, 0),
}

AIRWAYS = [
    ("A", "W1"),
    ("A", "W2"),
    ("W1", "W3"),
    ("W2", "W3"),
    ("W2", "W4"),
    ("W3", "W5"),
    ("W4", "W5"),
    ("W4", "W6"),
    ("W5", "B"),
    ("W6", "B"),
]


def build_graph():
    graph = nx.Graph()

    for node, pos in NODES.items():
        graph.add_node(node, pos=pos)

    for u, v in AIRWAYS:
        p1 = np.array(NODES[u])
        p2 = np.array(NODES[v])
        graph.add_edge(u, v, distance=np.linalg.norm(p2 - p1))

    return graph


GRAPH = build_graph()
