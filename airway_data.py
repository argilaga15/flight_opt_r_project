"""Airway-network loading and normalization helpers.

The project originally used a synthetic waypoint graph. This module adds a
real-data path that can ingest:

- normalized node/edge tables exported from FAA CIFP, EUROCONTROL EAD, or
  NATS-style route datasets;
- OpenSky trajectory tables, which are converted into a corridor graph by
  snapping track points onto a small grid; and
- AIXM-like XML exports from EUROCONTROL.

The rest of the project consumes a uniform ``networkx.Graph`` plus a list of
k-shortest candidate routes.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

EARTH_RADIUS_KM = 6371.0088

SUPPORTED_SOURCES = {
    "synthetic",
    "normalized",
    "opensky",
    "eurocontrol",
    "faa",
    "nats",
}

DEFAULT_SYNTHETIC_NODES = {
    "A": (0.0, 0.0),
    "W1": (150.0, 100.0),
    "W2": (250.0, -50.0),
    "W3": (400.0, 150.0),
    "W4": (550.0, 0.0),
    "W5": (700.0, 100.0),
    "W6": (850.0, -50.0),
    "B": (1000.0, 0.0),
}

DEFAULT_SYNTHETIC_AIRWAYS = [
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

TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".tab",
    ".dat",
    ".json",
    ".jsonl",
    ".ndjson",
    ".geojson",
}

XML_EXTENSIONS = {".xml", ".aixm"}


@dataclass
class AirwayNetwork:
    """Normalized airway graph plus metadata for the optimizer."""

    graph: nx.Graph
    origin: str
    destination: str
    source: str
    route_catalog: list[list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> dict[str, tuple[float, float]]:
        return {
            node: tuple(data.get("pos", (0.0, 0.0)))
            for node, data in self.graph.nodes(data=True)
        }

    @property
    def airways(self) -> list[tuple[str, str]]:
        return list(self.graph.edges())

    @property
    def decision_dimensions(self) -> int:
        route_count = max(len(self.route_catalog), 1)
        return max(3, math.ceil(math.log2(max(route_count, 2))))


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _clean_identifier(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", "", text).lstrip("#")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)) and not np.isnan(value):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "bidirectional", "both"}


def _project_latlon(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    mean_lat = math.radians((lat + ref_lat) / 2.0)
    x = (lon - ref_lon) * math.cos(mean_lat) * 111.32
    y = (lat - ref_lat) * 110.574
    return x, y


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        re.sub(r"[\s\-.]+", "_", str(col).strip().lower())
        for col in normalized.columns
    ]
    return normalized


def _read_text_source(name: str, text: str) -> pd.DataFrame | None:
    stripped = text.lstrip()
    if not stripped:
        return None

    suffix = Path(name).suffix.lower()

    if suffix in XML_EXTENSIONS or stripped.startswith("<"):
        return None

    if suffix in {".json", ".jsonl", ".ndjson", ".geojson"} or stripped[:1] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                return _normalize_frame(pd.read_json(io.StringIO(text), lines=True))
            except Exception:
                return None

        if isinstance(data, list):
            return _normalize_frame(pd.DataFrame(data))
        if isinstance(data, dict):
            if "features" in data and isinstance(data["features"], list):
                rows: list[dict[str, Any]] = []
                for feature in data["features"]:
                    geometry = feature.get("geometry", {}) if isinstance(feature, dict) else {}
                    properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
                    row = dict(properties)
                    if geometry.get("type") == "Point":
                        coords = geometry.get("coordinates", [])
                        if len(coords) >= 2:
                            row.setdefault("lon", coords[0])
                            row.setdefault("lat", coords[1])
                    rows.append(row)
                return _normalize_frame(pd.DataFrame(rows))
            return _normalize_frame(pd.json_normalize(data))
        return None

    if suffix in TABULAR_EXTENSIONS or suffix == "":
        try:
            frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if frame.shape[1] <= 1:
                return None
            return _normalize_frame(frame)
        except Exception:
            try:
                frame = pd.read_csv(io.StringIO(text))
                if frame.shape[1] <= 1:
                    return None
                return _normalize_frame(frame)
            except Exception:
                return None

    return None


def _iter_text_sources(root: Path) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    if root.is_file():
        if root.suffix.lower() == ".zip":
            with zipfile.ZipFile(root) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    suffix = Path(info.filename).suffix.lower()
                    if suffix not in TABULAR_EXTENSIONS and suffix not in XML_EXTENSIONS:
                        continue
                    text = zf.read(info).decode("utf-8", errors="ignore")
                    sources.append((info.filename, text))
        else:
            sources.append((root.name, root.read_text(encoding="utf-8", errors="ignore")))
        return sources

    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        if suffix not in TABULAR_EXTENSIONS and suffix not in XML_EXTENSIONS:
            continue
        sources.append((str(file), file.read_text(encoding="utf-8", errors="ignore")))
    return sources


def _collect_tables(root: Path) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    for name, text in _iter_text_sources(root):
        table = _read_text_source(name, text)
        if table is not None and not table.empty:
            tables.append((name, table))
    return tables


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _node_identifier_from_row(row: pd.Series, node_col: str | None, lat_col: str | None, lon_col: str | None) -> str:
    if node_col:
        ident = _clean_identifier(row.get(node_col))
        if ident:
            return ident
    lat = _safe_float(row.get(lat_col)) if lat_col else None
    lon = _safe_float(row.get(lon_col)) if lon_col else None
    if lat is not None and lon is not None:
        return f"N_{lat:.5f}_{lon:.5f}"
    for candidate in ("designator", "name", "identifier", "id", "fix", "point"):
        ident = _clean_identifier(row.get(candidate))
        if ident:
            return ident
    return ""


def _extract_table_nodes(frame: pd.DataFrame, source: str) -> dict[str, dict[str, Any]]:
    node_col = _find_column(
        frame,
        [
            "node_id",
            "point_id",
            "fix_id",
            "waypoint_id",
            "ident",
            "identifier",
            "id",
            "designator",
            "name",
        ],
    )
    lat_col = _find_column(frame, ["lat", "latitude", "y"])
    lon_col = _find_column(frame, ["lon", "longitude", "x"])
    kind_col = _find_column(frame, ["kind", "type", "feature_type", "node_type", "category"])
    name_col = _find_column(frame, ["name", "label", "designator", "description"])

    nodes: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        node_id = _node_identifier_from_row(row, node_col, lat_col, lon_col)
        if not node_id:
            continue

        lat = _safe_float(row.get(lat_col)) if lat_col else None
        lon = _safe_float(row.get(lon_col)) if lon_col else None
        if lat is None or lon is None:
            continue

        kind = _clean_identifier(row.get(kind_col)).lower() if kind_col else ""
        if not kind:
            kind = "waypoint"
        if any(token in node_id for token in ("APT", "AIRPORT")):
            kind = "airport"

        nodes[node_id] = {
            "lat": lat,
            "lon": lon,
            "kind": kind,
            "name": str(row.get(name_col, node_id)) if name_col else node_id,
            "source": source,
            "raw": row.to_dict(),
        }
        if node_id not in nodes:
            nodes[node_id] = {"lat": lat, "lon": lon, "kind": kind, "name": node_id, "source": source}
    return nodes


def _extract_table_edges(frame: pd.DataFrame, nodes: dict[str, dict[str, Any]], source: str) -> list[dict[str, Any]]:
    u_col = _find_column(
        frame,
        ["from", "source", "start", "u", "origin", "from_node", "from_id", "begin"],
    )
    v_col = _find_column(
        frame,
        ["to", "target", "end", "v", "destination", "to_node", "to_id", "finish"],
    )
    airway_col = _find_column(
        frame,
        ["airway", "route", "route_id", "segment", "awy", "path", "name"],
    )
    distance_col = _find_column(
        frame,
        ["distance_km", "distance", "length_km", "length", "segment_length"],
    )
    direction_col = _find_column(
        frame,
        ["bidirectional", "both_directions", "two_way", "reverse"],
    )

    edge_records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        u = _clean_identifier(row.get(u_col)) if u_col else ""
        v = _clean_identifier(row.get(v_col)) if v_col else ""
        if not u or not v or u == v:
            continue
        if u not in nodes or v not in nodes:
            continue

        distance = _safe_float(row.get(distance_col)) if distance_col else None
        if distance is None:
            distance = _haversine_km(
                nodes[u]["lat"],
                nodes[u]["lon"],
                nodes[v]["lat"],
                nodes[v]["lon"],
            )

        edge_records.append(
            {
                "u": u,
                "v": v,
                "airway": _clean_identifier(row.get(airway_col)) if airway_col else "",
                "distance": float(distance),
                "distance_km": float(distance),
                "bidirectional": _truthy(row.get(direction_col)) if direction_col else True,
                "source": source,
                "raw": row.to_dict(),
            }
        )
    return edge_records


def _build_graph(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    source: str,
) -> nx.Graph:
    graph = nx.Graph()

    for node_id, attrs in nodes.items():
        graph.add_node(node_id, **attrs)

    for edge in edges:
        u = edge["u"]
        v = edge["v"]
        if u not in graph or v not in graph:
            continue
        attrs = dict(edge)
        attrs.setdefault("count", 1)
        attrs.setdefault("kind", "airway")
        attrs["distance"] = float(attrs.get("distance_km", attrs.get("distance", 0.0)))
        attrs["distance_km"] = float(attrs["distance"])
        if graph.has_edge(u, v):
            existing = graph[u][v]
            existing["count"] = int(existing.get("count", 1)) + 1
            if not existing.get("airway") and attrs.get("airway"):
                existing["airway"] = attrs["airway"]
            if attrs["distance"] < existing.get("distance", attrs["distance"]):
                existing["distance"] = attrs["distance"]
                existing["distance_km"] = attrs["distance_km"]
        else:
            graph.add_edge(u, v, **attrs)

    for _, data in graph.nodes(data=True):
        data.setdefault("source", source)

    return graph


def _assign_positions(graph: nx.Graph) -> None:
    coords = [
        (float(data["lat"]), float(data["lon"]))
        for _, data in graph.nodes(data=True)
        if data.get("lat") is not None and data.get("lon") is not None
    ]
    if not coords:
        for _, data in graph.nodes(data=True):
            data.setdefault("pos", (0.0, 0.0))
        return

    ref_lat = float(np.mean([lat for lat, _ in coords]))
    ref_lon = float(np.mean([lon for _, lon in coords]))
    for _, data in graph.nodes(data=True):
        if data.get("pos") is not None:
            continue
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            data["pos"] = (0.0, 0.0)
            continue
        data["pos"] = _project_latlon(float(lat), float(lon), ref_lat, ref_lon)


def _node_distance_km(graph: nx.Graph, a: str, b: str) -> float:
    na = graph.nodes[a]
    nb = graph.nodes[b]
    if na.get("lat") is not None and nb.get("lat") is not None:
        return _haversine_km(
            float(na["lat"]),
            float(na["lon"]),
            float(nb["lat"]),
            float(nb["lon"]),
        )
    pa = na.get("pos", (0.0, 0.0))
    pb = nb.get("pos", (0.0, 0.0))
    return float(np.linalg.norm(np.array(pb) - np.array(pa)))


def _resolve_node(graph: nx.Graph, label: str | None) -> str | None:
    if not label:
        return None
    target = _clean_identifier(label)
    if not target:
        return None

    for candidate in (label, label.strip(), label.upper(), target):
        if candidate in graph:
            return candidate

    for node, data in graph.nodes(data=True):
        attrs = [
            node,
            data.get("name"),
            data.get("designator"),
            data.get("identifier"),
            data.get("icao"),
            data.get("iata"),
        ]
        cleaned = {_clean_identifier(value) for value in attrs if value is not None}
        if target in cleaned:
            return node
    return None


def _unique_node_id(graph: nx.Graph, base: str) -> str:
    if base not in graph:
        return base
    index = 1
    while f"{base}_{index}" in graph:
        index += 1
    return f"{base}_{index}"


def _clone_node_attrs(graph: nx.Graph, node: str) -> dict[str, Any]:
    attrs = dict(graph.nodes[node])
    attrs.pop("raw", None)
    attrs.pop("source", None)
    return attrs


def _choose_connected_pair(graph: nx.Graph) -> tuple[str, str]:
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("The airway graph needs at least two nodes")

    # Prefer airports when available.
    airport_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if str(data.get("kind", "")).lower() in {"airport", "aerodrome", "origin", "destination"}
    ]
    candidate_nodes = airport_nodes if len(airport_nodes) >= 2 else nodes

    sample = candidate_nodes[: min(len(candidate_nodes), 40)]
    best_pair = None
    best_distance = -1.0
    for a, b in itertools.combinations(sample, 2):
        if a == b or not nx.has_path(graph, a, b):
            continue
        distance = _node_distance_km(graph, a, b)
        if distance > best_distance:
            best_distance = distance
            best_pair = (a, b)

    if best_pair is not None:
        return best_pair

    component = max(nx.connected_components(graph), key=len)
    component_nodes = list(component)
    if len(component_nodes) >= 2:
        return component_nodes[0], component_nodes[-1]

    return nodes[0], nodes[-1]


def _attach_endpoint(
    graph: nx.Graph,
    existing_node: str | None,
    label: str | None,
    anchor_node: str,
    source: str,
    kind: str,
    use_existing: bool = True,
) -> str:
    if use_existing and existing_node is not None:
        return existing_node

    resolved = existing_node
    if resolved is None:
        base_label = _clean_identifier(label) if label else ""
        if not base_label:
            base_label = kind.upper()
        resolved = _unique_node_id(graph, base_label)
        attrs = _clone_node_attrs(graph, anchor_node)
        attrs.update(
            {
                "kind": kind,
                "name": label or resolved,
                "source": source,
                "synthetic_endpoint": True,
            }
        )
        graph.add_node(resolved, **attrs)
        graph.add_edge(
            resolved,
            anchor_node,
            distance=0.0,
            distance_km=0.0,
            airway="endpoint",
            source=source,
            kind="endpoint",
            count=1,
        )
    return resolved


def _candidate_routes(
    graph: nx.Graph,
    origin: str,
    destination: str,
    limit: int,
) -> list[list[str]]:
    if origin not in graph or destination not in graph:
        return []

    try:
        generator = nx.shortest_simple_paths(graph, origin, destination, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    routes: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in generator:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        routes.append(list(path))
        if len(routes) >= limit:
            break
    return routes


def _finalize_network(
    graph: nx.Graph,
    source: str,
    origin: str | None,
    destination: str | None,
    max_routes: int,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    _assign_positions(graph)

    requested_origin = origin
    requested_destination = destination
    resolved_origin = _resolve_node(graph, origin)
    resolved_destination = _resolve_node(graph, destination)

    use_resolved_path = (
        resolved_origin is not None
        and resolved_destination is not None
        and nx.has_path(graph, resolved_origin, resolved_destination)
    )
    if use_resolved_path:
        base_origin, base_destination = resolved_origin, resolved_destination
    else:
        base_origin, base_destination = _choose_connected_pair(graph)

    if requested_origin is None:
        origin_node = base_origin
    else:
        origin_node = _attach_endpoint(
            graph,
            resolved_origin if use_resolved_path else None,
            requested_origin,
            base_origin,
            source,
            "airport",
            use_existing=use_resolved_path,
        )

    if requested_destination is None:
        destination_node = base_destination
    else:
        destination_node = _attach_endpoint(
            graph,
            resolved_destination if use_resolved_path else None,
            requested_destination,
            base_destination,
            source,
            "airport",
            use_existing=use_resolved_path,
        )

    # Recompute positions after endpoint insertion.
    _assign_positions(graph)

    routes = _candidate_routes(graph, origin_node, destination_node, max_routes)
    if not routes and source != "synthetic":
        return load_synthetic_network(
            origin="A",
            destination="B",
            max_routes=max_routes,
            metadata={
                "fallback_from": source,
                "fallback_reason": "Could not build a connected real-airway corridor",
            },
        )

    payload = {
        "source": source,
        "requested_origin": requested_origin,
        "requested_destination": requested_destination,
        "resolved_origin": resolved_origin,
        "resolved_destination": resolved_destination,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }
    if metadata:
        payload.update(metadata)

    return AirwayNetwork(
        graph=graph,
        origin=origin_node,
        destination=destination_node,
        source=source,
        route_catalog=routes or [[origin_node, destination_node]],
        metadata=payload,
    )


def load_synthetic_network(
    origin: str | None = "A",
    destination: str | None = "B",
    max_routes: int = 8,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    graph = nx.Graph()
    for node_id, (x, y) in DEFAULT_SYNTHETIC_NODES.items():
        graph.add_node(
            node_id,
            lat=y,
            lon=x,
            pos=(x, y),
            kind="airport" if node_id in {"A", "B"} else "waypoint",
            name=node_id,
            source="synthetic",
        )

    for u, v in DEFAULT_SYNTHETIC_AIRWAYS:
        p1 = np.array(DEFAULT_SYNTHETIC_NODES[u])
        p2 = np.array(DEFAULT_SYNTHETIC_NODES[v])
        distance = float(np.linalg.norm(p2 - p1))
        graph.add_edge(
            u,
            v,
            distance=distance,
            distance_km=distance,
            airway="synthetic",
            source="synthetic",
            kind="airway",
            count=1,
        )

    routes = _candidate_routes(graph, "A", "B", max_routes)
    payload = {
        "source": "synthetic",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }
    if metadata:
        payload.update(metadata)

    return AirwayNetwork(
        graph=graph,
        origin="A",
        destination="B",
        source="synthetic",
        route_catalog=routes,
        metadata=payload,
    )


def _build_from_tables(
    tables: list[tuple[str, pd.DataFrame]],
    source: str,
    origin: str | None,
    destination: str | None,
    max_routes: int,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for name, frame in tables:
        has_latlon = _find_column(frame, ["lat", "latitude"]) is not None and _find_column(
            frame, ["lon", "longitude"]
        ) is not None
        has_edge_cols = (
            _find_column(frame, ["from", "source", "start", "u", "origin"]) is not None
            and _find_column(frame, ["to", "target", "end", "v", "destination"]) is not None
        )
        if has_edge_cols:
            edge_frames.append(frame)
        elif has_latlon:
            node_frames.append(frame)

    if not node_frames:
        raise ValueError("No node table with latitude/longitude columns was found")

    node_records: dict[str, dict[str, Any]] = {}
    for frame in node_frames:
        node_records.update(_extract_table_nodes(frame, source))
    if not node_records:
        raise ValueError("Could not extract any nodes from the input tables")

    edge_records: list[dict[str, Any]] = []
    for frame in edge_frames:
        edge_records.extend(_extract_table_edges(frame, node_records, source))

    graph = _build_graph(node_records, edge_records, source)
    return _finalize_network(graph, source, origin, destination, max_routes, metadata=metadata)


def _load_eurocontrol_xml(
    root: Path,
    source: str,
    origin: str | None,
    destination: str | None,
    max_routes: int,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    sources = _iter_text_sources(root)
    xml_sources = [(name, text) for name, text in sources if Path(name).suffix.lower() in XML_EXTENSIONS or text.lstrip().startswith("<")]
    if not xml_sources:
        raise ValueError("No XML source was found for the EUROCONTROL loader")

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    node_tags = {
        "designatedpoint",
        "navaid",
        "airportheliport",
        "routepoint",
        "waypoint",
        "point",
        "radioaid",
        "vor",
        "ndb",
        "dme",
    }
    edge_tags = {
        "route",
        "routesegment",
        "airway",
        "airwaysegment",
        "atsroute",
        "segment",
        "routeelement",
    }

    def extract_identifier(elem: ET.Element) -> str:
        for sub in elem.iter():
            text = (sub.text or "").strip()
            if not text:
                continue
            if _local_name(sub.tag) in {"designator", "name", "identifier", "id", "code"}:
                cleaned = _clean_identifier(text)
                if cleaned:
                    return cleaned
        for attr in ("{http://www.opengis.net/gml}id", "gml:id", "id", "name"):
            value = _clean_identifier(elem.attrib.get(attr))
            if value:
                return value
        return ""

    def extract_latlon(elem: ET.Element) -> tuple[float | None, float | None]:
        lat = None
        lon = None
        pos_text = None

        for sub in elem.iter():
            lname = _local_name(sub.tag)
            text = (sub.text or "").strip()
            if not text:
                continue
            if lname in {"pos", "position", "coordinates"}:
                pos_text = text
                break
            if lname in {"latitude", "lat"} and lat is None:
                lat = _safe_float(text)
            if lname in {"longitude", "lon"} and lon is None:
                lon = _safe_float(text)

        if pos_text:
            values = [token for token in re.split(r"[\s,]+", pos_text) if token]
            if len(values) >= 2:
                lat = _safe_float(values[0])
                lon = _safe_float(values[1])

        if lat is not None and lon is not None:
            return lat, lon

        return None, None

    def extract_refs(elem: ET.Element) -> list[str]:
        refs: list[str] = []
        for sub in elem.iter():
            for attr_name, attr_value in sub.attrib.items():
                if attr_name.lower().endswith("href"):
                    cleaned = _clean_identifier(attr_value.split("#")[-1].split("/")[-1])
                    if cleaned:
                        refs.append(cleaned)
            text = (sub.text or "").strip()
            cleaned = _clean_identifier(text)
            if cleaned and len(cleaned) <= 15 and any(ch.isalpha() for ch in cleaned):
                refs.append(cleaned)
        return _dedupe_preserve_order(refs)

    for name, text in xml_sources:
        try:
            root_elem = ET.fromstring(text)
        except ET.ParseError:
            continue

        for elem in root_elem.iter():
            lname = _local_name(elem.tag).lower()
            if lname in node_tags:
                node_id = extract_identifier(elem)
                lat, lon = extract_latlon(elem)
                if not node_id or lat is None or lon is None:
                    continue
                nodes[node_id] = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "kind": lname,
                    "name": node_id,
                    "source": source,
                    "raw": {"source_file": name},
                }
            elif lname in edge_tags:
                refs = extract_refs(elem)
                if len(refs) < 2:
                    continue
                airway = extract_identifier(elem)
                distance_override = None
                for sub in elem.iter():
                    if _local_name(sub.tag).lower() in {"length", "distance"}:
                        distance_override = _safe_float(sub.text)
                        if distance_override is not None:
                            break
                for u, v in zip(refs, refs[1:]):
                    edges.append(
                        {
                            "u": u,
                            "v": v,
                            "airway": airway,
                            "distance_km": distance_override if distance_override is not None else 0.0,
                            "distance": distance_override if distance_override is not None else 0.0,
                            "bidirectional": True,
                            "source": source,
                            "raw": {"source_file": name},
                        }
                    )

    if not nodes:
        raise ValueError("No navigational points could be extracted from the EUROCONTROL XML")

    # Keep only edges whose endpoints were discovered.
    filtered_edges = [edge for edge in edges if edge["u"] in nodes and edge["v"] in nodes]
    if not filtered_edges:
        # Fall back to a route-only graph from the node table.
        filtered_edges = []
        ordered_nodes = list(nodes)
        for u, v in zip(ordered_nodes, ordered_nodes[1:]):
            filtered_edges.append(
                {
                    "u": u,
                    "v": v,
                    "airway": "xml",
                    "distance_km": _haversine_km(
                        nodes[u]["lat"],
                        nodes[u]["lon"],
                        nodes[v]["lat"],
                        nodes[v]["lon"],
                    ),
                    "distance": _haversine_km(
                        nodes[u]["lat"],
                        nodes[u]["lon"],
                        nodes[v]["lat"],
                        nodes[v]["lon"],
                    ),
                    "bidirectional": True,
                    "source": source,
                    "raw": {"source_file": "fallback"},
                }
            )

    graph = _build_graph(nodes, filtered_edges, source)
    return _finalize_network(graph, source, origin, destination, max_routes, metadata=metadata)


def _split_track_segments(frame: pd.DataFrame, segment_gap_seconds: float) -> list[pd.DataFrame]:
    time_col = _find_column(frame, ["time", "timestamp", "lastseen", "last_contact", "date"])
    if time_col is None:
        return [frame]

    times = frame[time_col]
    if np.issubdtype(times.dtype, np.number):
        numeric = pd.to_numeric(times, errors="coerce")
    else:
        numeric = pd.to_datetime(times, errors="coerce").astype("int64") / 1_000_000_000.0

    if numeric.isna().all():
        return [frame]

    frame = frame.copy()
    frame["_track_time"] = numeric
    frame = frame.sort_values("_track_time")
    gaps = frame["_track_time"].diff().fillna(0.0).to_numpy()
    split_points = np.where(gaps > segment_gap_seconds)[0]
    if len(split_points) == 0:
        return [frame.drop(columns=["_track_time"])]

    segments: list[pd.DataFrame] = []
    start = 0
    for split_index in split_points:
        segment = frame.iloc[start:split_index].drop(columns=["_track_time"])
        if not segment.empty:
            segments.append(segment)
        start = split_index
    tail = frame.iloc[start:].drop(columns=["_track_time"])
    if not tail.empty:
        segments.append(tail)
    return segments


def _load_track_network(
    root: Path,
    source: str,
    origin: str | None,
    destination: str | None,
    max_routes: int,
    snap_precision_deg: float = 0.05,
    segment_gap_minutes: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    tables = _collect_tables(root)
    if not tables:
        raise ValueError("No track tables could be read from the input path")

    frames = [frame for _, frame in tables]
    frame = pd.concat(frames, ignore_index=True)
    frame = _normalize_frame(frame)

    lat_col = _find_column(frame, ["lat", "latitude"])
    lon_col = _find_column(frame, ["lon", "longitude"])
    if lat_col is None or lon_col is None:
        raise ValueError("The track data must provide latitude and longitude columns")

    group_candidates = [
        ["flight_id"],
        ["trajectory_id"],
        ["route_id"],
        ["callsign", "icao24"],
        ["callsign"],
        ["icao24"],
    ]
    group_cols: list[str] = []
    for candidate in group_candidates:
        if all(column in frame.columns for column in candidate):
            group_cols = candidate
            break

    if not group_cols:
        frame = frame.assign(_group_key="track")
        group_cols = ["_group_key"]

    departure_cols = [
        "estdepartureairport",
        "departure_airport",
        "origin",
        "dep",
        "from_airport",
    ]
    arrival_cols = [
        "estarrivalairport",
        "arrival_airport",
        "destination",
        "arr",
        "to_airport",
    ]
    dep_col = _find_column(frame, departure_cols)
    arr_col = _find_column(frame, arrival_cols)

    node_stats: dict[str, dict[str, Any]] = {}
    edge_stats: dict[tuple[str, str], dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []

    for group_key, group in frame.groupby(group_cols, dropna=True, sort=False):
        if isinstance(group_key, tuple):
            group_label = "_".join(_clean_identifier(part) or "X" for part in group_key)
        else:
            group_label = _clean_identifier(group_key) or "TRACK"
        working = group.copy()
        working = working.dropna(subset=[lat_col, lon_col])
        if working.empty:
            continue

        time_col = _find_column(working, ["time", "timestamp", "lastseen", "date"])
        if time_col is not None:
            working = working.sort_values(time_col)
        for segment in _split_track_segments(working, segment_gap_minutes * 60.0):
            segment = segment.dropna(subset=[lat_col, lon_col])
            if len(segment) < 2:
                continue

            snapped_cells: list[str] = []
            raw_points: list[tuple[float, float]] = []
            for _, row in segment.iterrows():
                lat = _safe_float(row.get(lat_col))
                lon = _safe_float(row.get(lon_col))
                if lat is None or lon is None:
                    continue
                raw_points.append((lat, lon))
                snap_lat = round(lat / snap_precision_deg) * snap_precision_deg
                snap_lon = round(lon / snap_precision_deg) * snap_precision_deg
                node_id = f"T_{snap_lat:.2f}_{snap_lon:.2f}"
                snapped_cells.append(node_id)
                node_entry = node_stats.setdefault(
                    node_id,
                    {
                        "lat_sum": 0.0,
                        "lon_sum": 0.0,
                        "count": 0,
                        "kind": "track_point",
                        "name": node_id,
                        "source": source,
                        "raw": {"sources": set()},
                    },
                )
                node_entry["lat_sum"] += lat
                node_entry["lon_sum"] += lon
                node_entry["count"] += 1
                node_entry["raw"]["sources"].add(group_label)

            snapped_cells = _dedupe_preserve_order(snapped_cells)
            if len(snapped_cells) < 2:
                continue

            segment_name = group_label
            if dep_col is not None and arr_col is not None:
                departure = _clean_identifier(segment.iloc[0].get(dep_col))
                arrival = _clean_identifier(segment.iloc[0].get(arr_col))
                if departure or arrival:
                    segment_name = f"{departure or group_label}_{arrival or group_label}"

            segment_distance = 0.0
            for (lat1, lon1), (lat2, lon2) in zip(raw_points, raw_points[1:]):
                segment_distance += _haversine_km(lat1, lon1, lat2, lon2)

            segments.append(
                {
                    "name": segment_name,
                    "cells": snapped_cells,
                    "first": raw_points[0],
                    "last": raw_points[-1],
                    "distance_km": segment_distance,
                }
            )

            for u, v in zip(snapped_cells, snapped_cells[1:]):
                if u == v:
                    continue
                edge_key = tuple(sorted((u, v)))
                edge_entry = edge_stats.setdefault(
                    edge_key,
                    {
                        "u": edge_key[0],
                        "v": edge_key[1],
                        "airway": segment_name,
                        "distance_km": 0.0,
                        "count": 0,
                        "kind": "track",
                        "source": source,
                        "raw": {"segments": []},
                    },
                )
                edge_entry["count"] += 1
                edge_entry["distance_km"] += _haversine_km(
                    node_stats[u]["lat_sum"] / node_stats[u]["count"],
                    node_stats[u]["lon_sum"] / node_stats[u]["count"],
                    node_stats[v]["lat_sum"] / node_stats[v]["count"],
                    node_stats[v]["lon_sum"] / node_stats[v]["count"],
                )
                edge_entry["raw"]["segments"].append(segment_name)

    if not segments:
        raise ValueError("The OpenSky/NATS track data did not contain any usable flight segments")

    # Pick the longest segment as the corridor anchor.
    anchor = max(segments, key=lambda item: item["distance_km"])
    origin_label = _clean_identifier(origin) if origin else ""
    destination_label = _clean_identifier(destination) if destination else ""
    if not origin_label:
        origin_label = f"{anchor['name']}_ORIGIN" if anchor["name"] else "ORIGIN"
    if not destination_label:
        destination_label = f"{anchor['name']}_DESTINATION" if anchor["name"] else "DESTINATION"

    graph = nx.Graph()
    for node_id, stats in node_stats.items():
        graph.add_node(
            node_id,
            lat=stats["lat_sum"] / stats["count"],
            lon=stats["lon_sum"] / stats["count"],
            kind=stats["kind"],
            name=stats["name"],
            source=source,
            raw={"sources": sorted(stats["raw"]["sources"])},
        )

    for segment in segments:
        cells = segment["cells"]
        for u, v in zip(cells, cells[1:]):
            edge_key = tuple(sorted((u, v)))
            if edge_key not in edge_stats:
                continue
            entry = edge_stats[edge_key]
            if graph.has_edge(u, v):
                graph[u][v]["count"] = int(graph[u][v].get("count", 1)) + 1
            else:
                graph.add_edge(
                    u,
                    v,
                    distance=float(entry["distance_km"]) if entry["distance_km"] else 0.0,
                    distance_km=float(entry["distance_km"]) if entry["distance_km"] else 0.0,
                    airway=entry["airway"],
                    source=source,
                    kind="track",
                    count=entry["count"],
                    raw=entry["raw"],
                )

    # Add explicit endpoints so the optimizer has a stable origin and destination.
    origin_anchor = anchor["cells"][0]
    destination_anchor = anchor["cells"][-1]
    origin_node = _unique_node_id(graph, origin_label)
    destination_node = _unique_node_id(graph, destination_label)

    origin_attrs = _clone_node_attrs(graph, origin_anchor)
    destination_attrs = _clone_node_attrs(graph, destination_anchor)
    origin_attrs.update({"kind": "airport", "name": origin_label, "source": source, "synthetic_endpoint": True})
    destination_attrs.update(
        {"kind": "airport", "name": destination_label, "source": source, "synthetic_endpoint": True}
    )
    graph.add_node(origin_node, **origin_attrs)
    graph.add_node(destination_node, **destination_attrs)
    graph.add_edge(
        origin_node,
        origin_anchor,
        distance=0.0,
        distance_km=0.0,
        airway="origin",
        source=source,
        kind="endpoint",
        count=1,
    )
    graph.add_edge(
        destination_anchor,
        destination_node,
        distance=0.0,
        distance_km=0.0,
        airway="destination",
        source=source,
        kind="endpoint",
        count=1,
    )

    _assign_positions(graph)
    routes = _candidate_routes(graph, origin_node, destination_node, max_routes)
    if not routes:
        return load_synthetic_network(
            metadata={
                "fallback_from": source,
                "fallback_reason": "Could not build a connected track corridor",
            }
        )

    payload = {
        "source": source,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "snap_precision_deg": snap_precision_deg,
        "segment_gap_minutes": segment_gap_minutes,
    }
    if metadata:
        payload.update(metadata)

    return AirwayNetwork(
        graph=graph,
        origin=origin_node,
        destination=destination_node,
        source=source,
        route_catalog=routes,
        metadata=payload,
    )


def build_airway_network(
    source: str = "synthetic",
    path: str | Path | None = None,
    origin: str | None = None,
    destination: str | None = None,
    max_routes: int = 8,
    *,
    snap_precision_deg: float = 0.05,
    segment_gap_minutes: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> AirwayNetwork:
    """Build an airway network from a local dataset or fall back to synthetic."""

    source_name = (source or "synthetic").lower()
    if source_name not in SUPPORTED_SOURCES:
        raise ValueError(f"Unknown airway source '{source}'. Supported sources: {', '.join(sorted(SUPPORTED_SOURCES))}")

    root = Path(path).expanduser() if path else None

    if source_name == "synthetic" or root is None or not root.exists():
        payload = dict(metadata or {})
        if source_name != "synthetic":
            payload.setdefault("fallback_from", source_name)
            payload.setdefault(
                "fallback_reason",
                f"No usable path was supplied for source '{source_name}', so the synthetic network was used",
            )
        return load_synthetic_network(metadata=payload)

    try:
        if source_name in {"normalized", "faa"}:
            tables = _collect_tables(root)
            if not tables:
                raise ValueError("No tabular node/edge data could be read")
            return _build_from_tables(tables, source_name, origin, destination, max_routes, metadata=metadata)

        if source_name in {"opensky", "nats"}:
            return _load_track_network(
                root,
                source_name,
                origin,
                destination,
                max_routes,
                snap_precision_deg=snap_precision_deg,
                segment_gap_minutes=segment_gap_minutes,
                metadata=metadata,
            )

        if source_name == "eurocontrol":
            try:
                return _load_eurocontrol_xml(root, source_name, origin, destination, max_routes, metadata=metadata)
            except ValueError:
                tables = _collect_tables(root)
                if tables:
                    return _build_from_tables(tables, source_name, origin, destination, max_routes, metadata=metadata)
                raise
    except Exception as exc:
        return load_synthetic_network(
            metadata={
                "fallback_from": source_name,
                "fallback_reason": str(exc),
                **(metadata or {}),
            }
        )

    return load_synthetic_network(
        metadata={
            "fallback_from": source_name,
            "fallback_reason": f"Source '{source_name}' is not implemented",
            **(metadata or {}),
        }
    )
