"""Synthetic and CSV trajectory adapters plus contact-window utilities.

The simulator consumes ordered :class:`ContactWindow` intervals. A CSV trace
contains trajectory samples, so consecutive samples associated with the same
RSU are converted to one interval. Position and speed are retained in
``TrajectoryPoint`` for provenance; the current contact abstraction does not
assume RSU coordinates or a coverage radius.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .models import ContactWindow


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp: float
    vehicle_id: str
    rsu_id: str
    x: float
    y: float
    speed: float

    def __post_init__(self) -> None:
        values = (self.timestamp, self.x, self.y, self.speed)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("trajectory values must be finite")
        if self.timestamp < 0:
            raise ValueError("trajectory timestamp must be non-negative")
        if self.speed < 0:
            raise ValueError("trajectory speed must be non-negative")
        if not str(self.vehicle_id):
            raise ValueError("vehicle_id must not be empty")
        if not str(self.rsu_id):
            raise ValueError("rsu_id must not be empty")


@dataclass(frozen=True)
class RsuSite:
    """RSU location used to associate position samples with coverage windows."""

    rsu_id: str
    x: float
    y: float
    coverage_radius_m: float

    def __post_init__(self) -> None:
        if not str(self.rsu_id):
            raise ValueError("rsu_id must not be empty")
        if not all(math.isfinite(float(v)) for v in (self.x, self.y, self.coverage_radius_m)):
            raise ValueError("RSU coordinates and radius must be finite")
        if self.coverage_radius_m <= 0:
            raise ValueError("coverage_radius_m must be positive")


@dataclass(frozen=True)
class TraceProvenance:
    """Reproducible description of a trajectory input and its adapter settings.

    The simulator stores this record alongside results so an external trace is
    not accidentally reported as a synthetic or unverified real-world trace.
    ``source_sha256`` is omitted for live TraCI streams.
    """

    source_kind: str
    parser: str
    source_path: str | None = None
    source_sha256: str | None = None
    parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not str(self.source_kind):
            raise ValueError("source_kind must not be empty")
        if not str(self.parser):
            raise ValueError("parser must not be empty")

    @property
    def is_external(self) -> bool:
        return self.source_kind not in {"synthetic_line", "synthetic_markov", "synthetic"}

    def as_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "parser": self.parser,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "parameters": dict(self.parameters or {}),
            "is_external": self.is_external,
        }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def synthetic_line_trace(
    vehicles: int,
    rsus: int,
    contact_duration_s: float,
    inter_contact_s: float,
    rate_bps: float,
) -> Iterator[ContactWindow]:
    """Generate a deterministic ordered line of RSU contacts."""
    vehicles = _positive_int(vehicles, "vehicles")
    rsus = _positive_int(rsus, "rsus")
    if not math.isfinite(float(contact_duration_s)) or contact_duration_s <= 0:
        raise ValueError("contact_duration_s must be positive and finite")
    if not math.isfinite(float(inter_contact_s)) or inter_contact_s < 0:
        raise ValueError("inter_contact_s must be non-negative and finite")
    if not math.isfinite(float(rate_bps)) or rate_bps <= 0:
        raise ValueError("rate_bps must be positive and finite")
    for vid in range(vehicles):
        for rid in range(rsus):
            start = rid * (contact_duration_s + inter_contact_s) + vid * 0.01
            yield ContactWindow(f"v{vid}", f"r{rid}", start, start + contact_duration_s, rate_bps)


def synthetic_markov_trace(
    vehicles: int,
    rsus: int,
    contact_duration_s: float,
    rate_bps: float,
    seed: int = 0,
) -> Iterator[ContactWindow]:
    """Generate a reproducible controlled trace with random sojourn times."""
    vehicles = _positive_int(vehicles, "vehicles")
    rsus = _positive_int(rsus, "rsus")
    if not math.isfinite(float(contact_duration_s)) or contact_duration_s <= 0:
        raise ValueError("contact_duration_s must be positive and finite")
    if not math.isfinite(float(rate_bps)) or rate_bps <= 0:
        raise ValueError("rate_bps must be positive and finite")
    rng = random.Random(seed)
    for vid in range(vehicles):
        t = 0.0
        for rid in range(rsus):
            duration = contact_duration_s * (0.75 + 0.5 * rng.random())
            yield ContactWindow(f"v{vid}", f"r{rid}", t, t + duration, rate_bps)
            t += duration + 0.5 + rng.random() * 0.5


def _read_csv_points(path: str | Path) -> list[TrajectoryPoint]:
    required = {"timestamp", "vehicle_id", "rsu_id", "x", "y", "speed"}
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"CSV trace missing columns: {sorted(missing)}")
        points: list[TrajectoryPoint] = []
        for row_no, row in enumerate(reader, start=2):
            try:
                points.append(
                    TrajectoryPoint(
                        timestamp=float(row["timestamp"]),
                        vehicle_id=str(row["vehicle_id"]),
                        rsu_id=str(row["rsu_id"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        speed=float(row["speed"]),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid trajectory row {row_no}") from exc
    return points


def contact_windows_from_points(
    points: Iterable[TrajectoryPoint],
    *,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
) -> list[ContactWindow]:
    """Convert timestamped trajectory samples into contact windows."""
    if not math.isfinite(float(sample_interval_s)) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive and finite")
    grouped: dict[str, list[TrajectoryPoint]] = {}
    for point in points:
        if not isinstance(point, TrajectoryPoint):
            raise TypeError("points must contain TrajectoryPoint values")
        grouped.setdefault(point.vehicle_id, []).append(point)

    result: list[ContactWindow] = []
    for vehicle_id, vehicle_points in grouped.items():
        ordered = sorted(vehicle_points, key=lambda p: p.timestamp)
        for idx, point in enumerate(ordered):
            if idx + 1 < len(ordered):
                next_time = ordered[idx + 1].timestamp
                if next_time <= point.timestamp:
                    raise ValueError(f"timestamps for {vehicle_id} must be strictly increasing")
                end = min(next_time, point.timestamp + sample_interval_s)
            else:
                end = point.timestamp + sample_interval_s
            result.append(ContactWindow(vehicle_id, point.rsu_id, point.timestamp, end, 1.0))

    if not merge_consecutive:
        return sorted(result, key=lambda w: (w.vehicle_id, w.start, w.rsu_id))

    merged: list[ContactWindow] = []
    for window in sorted(result, key=lambda w: (w.vehicle_id, w.start, w.rsu_id)):
        if merged and (
            merged[-1].vehicle_id == window.vehicle_id
            and merged[-1].rsu_id == window.rsu_id
            and window.start <= merged[-1].end + 1e-12
        ):
            previous = merged[-1]
            merged[-1] = ContactWindow(
                previous.vehicle_id,
                previous.rsu_id,
                previous.start,
                max(previous.end, window.end),
                previous.rate_bps,
            )
        else:
            merged.append(window)
    return merged


def csv_trace(
    path: str | Path,
    default_rate_bps: float = 1_000_000,
    *,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
) -> list[ContactWindow]:
    """Load the required six-column CSV trajectory format."""
    if not math.isfinite(float(default_rate_bps)) or default_rate_bps <= 0:
        raise ValueError("default_rate_bps must be positive and finite")
    windows = contact_windows_from_points(
        _read_csv_points(path),
        sample_interval_s=sample_interval_s,
        merge_consecutive=merge_consecutive,
    )
    return [ContactWindow(w.vehicle_id, w.rsu_id, w.start, w.end, default_rate_bps) for w in windows]


def file_provenance(
    path: str | Path,
    *,
    source_kind: str,
    parser: str,
    parameters: Mapping[str, object] | None = None,
) -> TraceProvenance:
    """Return a hash-backed provenance record for a trajectory file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trajectory file does not exist: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return TraceProvenance(
        source_kind=source_kind,
        parser=parser,
        source_path=str(source.resolve()),
        source_sha256=digest,
        parameters=dict(parameters or {}),
    )


def csv_trace_with_provenance(
    path: str | Path,
    default_rate_bps: float = 1_000_000,
    *,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
    source_kind: str = "csv_external",
) -> tuple[list[ContactWindow], TraceProvenance]:
    """Load the six-column CSV trace and its exact file provenance."""
    windows = csv_trace(
        path,
        default_rate_bps=default_rate_bps,
        sample_interval_s=sample_interval_s,
        merge_consecutive=merge_consecutive,
    )
    provenance = file_provenance(
        path,
        source_kind=source_kind,
        parser="csv_trajectory_v1",
        parameters={
            "default_rate_bps": float(default_rate_bps),
            "sample_interval_s": float(sample_interval_s),
            "merge_consecutive": bool(merge_consecutive),
        },
    )
    return windows, provenance


def _normalise_rsus(
    rsus: Mapping[str, object] | Sequence[RsuSite],
    coverage_radius_m: float | None,
) -> list[RsuSite]:
    """Normalise compact RSU config forms used by FCD and TraCI adapters."""
    sites: list[RsuSite] = []
    if isinstance(rsus, Mapping):
        entries = rsus.items()
    else:
        entries = ((site.rsu_id, site) for site in rsus)
    for rsu_id, value in entries:
        if isinstance(value, RsuSite):
            site = value
        elif isinstance(value, Mapping):
            try:
                radius = value.get("coverage_radius_m", value.get("radius_m", coverage_radius_m))
                site = RsuSite(str(rsu_id), float(value["x"]), float(value["y"]), float(radius))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid RSU specification for {rsu_id!r}") from exc
        else:
            try:
                x, y = value  # type: ignore[misc]
                if coverage_radius_m is None:
                    raise ValueError("coverage_radius_m is required for coordinate pairs")
                site = RsuSite(str(rsu_id), float(x), float(y), float(coverage_radius_m))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid RSU specification for {rsu_id!r}") from exc
        sites.append(site)
    if not sites:
        raise ValueError("at least one RSU site is required")
    return sites


def _nearest_rsu(
    x: float,
    y: float,
    sites: Sequence[RsuSite],
) -> str | None:
    candidates = []
    for site in sites:
        distance_sq = (x - site.x) ** 2 + (y - site.y) ** 2
        if distance_sq <= site.coverage_radius_m**2 + 1e-12:
            candidates.append((distance_sq, site.rsu_id))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def sumo_fcd_points(
    path: str | Path,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    *,
    coverage_radius_m: float | None = None,
    vehicle_ids: Iterable[str] | None = None,
    allow_vehicle_rsu_attribute: bool = True,
) -> list[TrajectoryPoint]:
    """Parse a SUMO floating-car-data XML file into covered trajectory points.

    SUMO FCD gives positions but not RSU association. Each sample is assigned
    to the nearest configured RSU whose coverage circle contains the sample.
    A ``rsu_id`` vehicle attribute, when present, is accepted as an explicit
    association only if it names one of the configured sites. Samples outside
    all coverage circles are skipped, preserving contact-window semantics.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SUMO FCD file does not exist: {source}")
    sites = _normalise_rsus(rsus, coverage_radius_m)
    known_rsus = {site.rsu_id for site in sites}
    selected = {str(value) for value in vehicle_ids} if vehicle_ids is not None else None
    points: list[TrajectoryPoint] = []
    try:
        context = ET.iterparse(source, events=("start", "end"))
        current_time: float | None = None
        for event_name, element in context:
            if event_name == "start" and element.tag == "timestep":
                try:
                    current_time = float(element.attrib["time"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("SUMO FCD timestep requires a finite time attribute") from exc
                if not math.isfinite(current_time) or current_time < 0:
                    raise ValueError("SUMO FCD timestep time must be finite and non-negative")
            elif event_name == "end" and element.tag == "vehicle":
                if current_time is None:
                    raise ValueError("SUMO FCD vehicle appears outside a timestep")
                vehicle_id = element.attrib.get("id", "")
                if not vehicle_id or (selected is not None and vehicle_id not in selected):
                    element.clear()
                    continue
                try:
                    x = float(element.attrib["x"])
                    y = float(element.attrib["y"])
                    speed = float(element.attrib.get("speed", 0.0))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid SUMO FCD vehicle row for {vehicle_id!r}") from exc
                explicit_rsu = element.attrib.get("rsu_id") if allow_vehicle_rsu_attribute else None
                if explicit_rsu is not None and explicit_rsu not in known_rsus:
                    raise ValueError(f"SUMO FCD vehicle {vehicle_id!r} references unknown rsu_id {explicit_rsu!r}")
                rsu_id = explicit_rsu or _nearest_rsu(x, y, sites)
                if rsu_id is not None:
                    points.append(TrajectoryPoint(current_time, vehicle_id, rsu_id, x, y, speed))
                element.clear()
            elif event_name == "end" and element.tag == "timestep":
                element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"invalid SUMO FCD XML: {source}") from exc
    return points


def sumo_fcd_trace(
    path: str | Path,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    *,
    coverage_radius_m: float | None = None,
    default_rate_bps: float = 1_000_000,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
    vehicle_ids: Iterable[str] | None = None,
) -> list[ContactWindow]:
    """Convert SUMO FCD samples into contact windows.

    SUMO itself is not imported or required. This file-based adapter is usable
    on a machine without SUMO; use :func:`traci_trace` only when a TraCI
    connection is already available.
    """
    if not math.isfinite(float(default_rate_bps)) or default_rate_bps <= 0:
        raise ValueError("default_rate_bps must be positive and finite")
    points = sumo_fcd_points(
        path,
        rsus,
        coverage_radius_m=coverage_radius_m,
        vehicle_ids=vehicle_ids,
    )
    windows = contact_windows_from_points(
        points,
        sample_interval_s=sample_interval_s,
        merge_consecutive=merge_consecutive,
    )
    return [ContactWindow(w.vehicle_id, w.rsu_id, w.start, w.end, float(default_rate_bps)) for w in windows]


def sumo_fcd_trace_with_provenance(
    path: str | Path,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    **kwargs: object,
) -> tuple[list[ContactWindow], TraceProvenance]:
    """Load SUMO FCD windows and return hash-backed parser provenance."""
    effective_kwargs = dict(kwargs)
    if "vehicle_ids" in effective_kwargs and effective_kwargs["vehicle_ids"] is not None:
        effective_kwargs["vehicle_ids"] = [str(value) for value in effective_kwargs["vehicle_ids"]]  # type: ignore[union-attr]
    windows = sumo_fcd_trace(path, rsus, **effective_kwargs)
    effective_radius = effective_kwargs.get("coverage_radius_m")
    sites = _normalise_rsus(rsus, float(effective_radius) if effective_radius is not None else None)
    provenance = file_provenance(
        path,
        source_kind="sumo_fcd",
        parser="sumo_fcd_xml_v1",
        parameters={
            "rsus": [
                {"rsu_id": site.rsu_id, "x": site.x, "y": site.y, "coverage_radius_m": site.coverage_radius_m}
                for site in sites
            ],
            "coverage_radius_m": effective_radius,
            "default_rate_bps": float(effective_kwargs.get("default_rate_bps", 1_000_000)),
            "sample_interval_s": float(effective_kwargs.get("sample_interval_s", 1.0)),
            "merge_consecutive": bool(effective_kwargs.get("merge_consecutive", True)),
            "vehicle_ids": sorted(str(value) for value in effective_kwargs.get("vehicle_ids", []) or []),
        },
    )
    return windows, provenance


def traci_points(
    connection: object,
    *,
    duration_s: float,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    coverage_radius_m: float | None = None,
    sample_interval_s: float = 1.0,
    vehicle_ids: Iterable[str] | None = None,
    start_time: float | None = None,
) -> list[TrajectoryPoint]:
    """Sample an existing TraCI connection without importing SUMO/TraCI.

    The connection must expose the standard ``simulation.getTime()``,
    ``vehicle.getIDList()``, ``vehicle.getPosition(id)``, ``vehicle.getSpeed(id)``
    and ``simulationStep()`` methods. This keeps TraCI optional and makes the
    adapter straightforward to test with a small fake connection.
    """
    if not math.isfinite(float(duration_s)) or duration_s < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if not math.isfinite(float(sample_interval_s)) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive and finite")
    sites = _normalise_rsus(rsus, coverage_radius_m)
    selected = {str(value) for value in vehicle_ids} if vehicle_ids is not None else None
    simulation = getattr(connection, "simulation", None)
    vehicle_api = getattr(connection, "vehicle", None)
    if simulation is None or vehicle_api is None:
        raise TypeError("TraCI connection must expose simulation and vehicle APIs")
    get_time = getattr(simulation, "getTime", None)
    get_ids = getattr(vehicle_api, "getIDList", None)
    get_position = getattr(vehicle_api, "getPosition", None)
    get_speed = getattr(vehicle_api, "getSpeed", None)
    step = getattr(connection, "simulationStep", None)
    if not all(callable(fn) for fn in (get_time, get_ids, get_position, get_speed, step)):
        raise TypeError("TraCI connection lacks one or more required methods")
    first_time = float(get_time()) if start_time is None else float(start_time)
    if not math.isfinite(first_time) or first_time < 0:
        raise ValueError("start_time must be finite and non-negative")
    end_time = first_time + float(duration_s)
    points: list[TrajectoryPoint] = []
    while float(get_time()) <= end_time + 1e-9:
        timestamp = float(get_time())
        for raw_id in get_ids():
            vehicle_id = str(raw_id)
            if selected is not None and vehicle_id not in selected:
                continue
            try:
                x, y = get_position(raw_id)
                speed = get_speed(raw_id)
            except Exception as exc:  # TraCI raises per-vehicle lookup errors
                raise RuntimeError(f"TraCI failed to read vehicle {vehicle_id!r}") from exc
            rsu_id = _nearest_rsu(float(x), float(y), sites)
            if rsu_id is not None:
                points.append(TrajectoryPoint(timestamp, vehicle_id, rsu_id, float(x), float(y), float(speed)))
        if float(get_time()) >= end_time - 1e-9:
            break
        step()
    return points


def traci_trace(
    connection: object,
    *,
    duration_s: float,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    coverage_radius_m: float | None = None,
    default_rate_bps: float = 1_000_000,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
    vehicle_ids: Iterable[str] | None = None,
) -> list[ContactWindow]:
    """Sample an active TraCI connection and return contact windows."""
    if not math.isfinite(float(default_rate_bps)) or default_rate_bps <= 0:
        raise ValueError("default_rate_bps must be positive and finite")
    points = traci_points(
        connection,
        duration_s=duration_s,
        rsus=rsus,
        coverage_radius_m=coverage_radius_m,
        sample_interval_s=sample_interval_s,
        vehicle_ids=vehicle_ids,
    )
    windows = contact_windows_from_points(
        points,
        sample_interval_s=sample_interval_s,
        merge_consecutive=merge_consecutive,
    )
    return [ContactWindow(w.vehicle_id, w.rsu_id, w.start, w.end, float(default_rate_bps)) for w in windows]


def traci_trace_with_provenance(
    connection: object,
    *,
    duration_s: float,
    rsus: Mapping[str, object] | Sequence[RsuSite],
    coverage_radius_m: float | None = None,
    default_rate_bps: float = 1_000_000,
    sample_interval_s: float = 1.0,
    merge_consecutive: bool = True,
    vehicle_ids: Iterable[str] | None = None,
    source_parameters: Mapping[str, object] | None = None,
) -> tuple[list[ContactWindow], TraceProvenance]:
    """Sample TraCI and return a provenance record for the live stream.

    Because a live connection has no single input file, ``source_sha256`` is
    ``None``. Callers should pass hashes of the SUMO network/route/config files
    through ``source_parameters`` when making a reproducible experiment record.
    """
    windows = traci_trace(
        connection,
        duration_s=duration_s,
        rsus=rsus,
        coverage_radius_m=coverage_radius_m,
        default_rate_bps=default_rate_bps,
        sample_interval_s=sample_interval_s,
        merge_consecutive=merge_consecutive,
        vehicle_ids=vehicle_ids,
    )
    sites = _normalise_rsus(rsus, coverage_radius_m)
    parameters = {
        "duration_s": float(duration_s),
        "rsus": [
            {"rsu_id": site.rsu_id, "x": site.x, "y": site.y, "coverage_radius_m": site.coverage_radius_m}
            for site in sites
        ],
        "coverage_radius_m": coverage_radius_m,
        "default_rate_bps": float(default_rate_bps),
        "sample_interval_s": float(sample_interval_s),
        "merge_consecutive": bool(merge_consecutive),
        "vehicle_ids": sorted(str(value) for value in vehicle_ids or []),
    }
    parameters.update(dict(source_parameters or {}))
    return windows, TraceProvenance(source_kind="traci", parser="traci_sampling_v1", parameters=parameters)


# Descriptive aliases kept for callers that use the shorter adapter names.
parse_sumo_fcd = sumo_fcd_points
sumo_trace = sumo_fcd_trace
traci_contact_windows = traci_trace


def trajectory_points(path: str | Path) -> list[TrajectoryPoint]:
    """Public CSV parser for callers that need positions and speed."""
    return _read_csv_points(path)


def contact_windows_for_vehicle(
    windows: Iterable[ContactWindow],
    vehicle_id: str,
) -> list[ContactWindow]:
    """Filter and sort windows into the vehicle's chronological route."""
    return sorted(
        (w for w in windows if w.vehicle_id == vehicle_id),
        key=lambda w: (w.start, w.end, w.rsu_id),
    )


def clip_contact_windows(
    windows: Iterable[ContactWindow],
    start: float,
    end: float,
) -> list[ContactWindow]:
    """Clip intervals to a simulation horizon, dropping empty intersections."""
    if not math.isfinite(float(start)) or not math.isfinite(float(end)) or end < start:
        raise ValueError("require finite clip interval with end >= start")
    clipped: list[ContactWindow] = []
    for window in windows:
        left, right = max(start, window.start), min(end, window.end)
        if right > left:
            clipped.append(ContactWindow(window.vehicle_id, window.rsu_id, left, right, window.rate_bps))
    return sorted(clipped, key=lambda w: (w.vehicle_id, w.start, w.rsu_id))
