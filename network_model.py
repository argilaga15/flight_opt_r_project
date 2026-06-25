"""Airway network definition and runtime configuration."""

from __future__ import annotations

import os

from airway_data import build_airway_network

DEFAULT_SOURCE = os.getenv("AIRWAY_DATA_SOURCE", "synthetic")
DEFAULT_DATA_PATH = os.getenv("AIRWAY_DATA_PATH", "")
DEFAULT_ORIGIN = os.getenv("AIRWAY_ORIGIN", "A")
DEFAULT_DESTINATION = os.getenv("AIRWAY_DESTINATION", "B")
DEFAULT_ROUTE_COUNT = int(os.getenv("AIRWAY_ROUTE_COUNT", "8"))
DEFAULT_SNAP_PRECISION_DEG = float(os.getenv("AIRWAY_SNAP_PRECISION_DEG", "0.05"))
DEFAULT_SEGMENT_GAP_MINUTES = float(os.getenv("AIRWAY_SEGMENT_GAP_MINUTES", "30"))

NETWORK = build_airway_network(
    source=DEFAULT_SOURCE,
    path=DEFAULT_DATA_PATH,
    origin=DEFAULT_ORIGIN,
    destination=DEFAULT_DESTINATION,
    max_routes=DEFAULT_ROUTE_COUNT,
    snap_precision_deg=DEFAULT_SNAP_PRECISION_DEG,
    segment_gap_minutes=DEFAULT_SEGMENT_GAP_MINUTES,
)

GRAPH = NETWORK.graph
NODES = NETWORK.nodes
AIRWAYS = NETWORK.airways
ROUTE_CANDIDATES = NETWORK.route_catalog
ROUTE_DECISION_DIMENSIONS = NETWORK.decision_dimensions
NETWORK_METADATA = NETWORK.metadata
NETWORK_SOURCE = NETWORK.source
