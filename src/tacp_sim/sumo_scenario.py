"""Reproducible SUMO scenario generation and execution helpers.

The Phase 4 experiment needs both a controlled SUMO scene and an explicit
fallback for machines where SUMO is not installed.  This module writes the
smallest valid one-way road network and route files, records hashes for every
artifact, and can generate a deterministic FCD file with the same geometry.
The generated FCD is a controlled simulator fixture; it must not be reported
as a real-world mobility trace.

No SUMO/TraCI import is required.  If an executable is available, callers can
request a real SUMO run.  Otherwise ``run_sumo_scenario`` returns a
``controlled_fcd_fallback`` result and leaves the provenance flag explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


GENERATOR_VERSION = "phase4_sumo_scenario_v1"


def _finite(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of an artifact without normalising it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SumoVehicle:
    """One deterministic vehicle route in the controlled straight-road scene."""

    vehicle_id: str
    depart_s: float = 0.0
    initial_x_m: float = 0.0
    speed_mps: float = 10.0

    def __post_init__(self) -> None:
        if not str(self.vehicle_id):
            raise ValueError("vehicle_id must not be empty")
        _finite(self.depart_s, "depart_s", nonnegative=True)
        _finite(self.initial_x_m, "initial_x_m", nonnegative=True)
        _finite(self.speed_mps, "speed_mps", positive=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "vehicle_id": str(self.vehicle_id),
            "depart_s": float(self.depart_s),
            "initial_x_m": float(self.initial_x_m),
            "speed_mps": float(self.speed_mps),
        }


@dataclass(frozen=True)
class SumoRsu:
    """RSU position used by the controlled FCD generator and parser."""

    rsu_id: str
    x_m: float
    y_m: float = 0.0
    coverage_radius_m: float = 10.0

    def __post_init__(self) -> None:
        if not str(self.rsu_id):
            raise ValueError("rsu_id must not be empty")
        _finite(self.x_m, "x_m", nonnegative=True)
        _finite(self.y_m, "y_m")
        _finite(self.coverage_radius_m, "coverage_radius_m", positive=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "rsu_id": str(self.rsu_id),
            "x": float(self.x_m),
            "y": float(self.y_m),
            "coverage_radius_m": float(self.coverage_radius_m),
        }


@dataclass(frozen=True)
class SumoScenario:
    """Frozen parameters for a reproducible one-road SUMO scenario."""

    scenario_id: str
    seed: int
    duration_s: float
    step_s: float
    road_length_m: float
    vehicles: tuple[SumoVehicle, ...]
    rsus: tuple[SumoRsu, ...]
    lane_id: str = "lane0"
    edge_id: str = "edge0"
    route_id: str = "route0"

    def __post_init__(self) -> None:
        if not str(self.scenario_id):
            raise ValueError("scenario_id must not be empty")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        _finite(self.duration_s, "duration_s", positive=True)
        _finite(self.step_s, "step_s", positive=True)
        _finite(self.road_length_m, "road_length_m", positive=True)
        if not self.vehicles:
            raise ValueError("at least one controlled vehicle is required")
        if not self.rsus:
            raise ValueError("at least one RSU is required")
        ids = [v.vehicle_id for v in self.vehicles]
        if len(ids) != len(set(ids)):
            raise ValueError("vehicle IDs must be unique")
        rsu_ids = [r.rsu_id for r in self.rsus]
        if len(rsu_ids) != len(set(rsu_ids)):
            raise ValueError("RSU IDs must be unique")
        for vehicle in self.vehicles:
            if vehicle.initial_x_m > self.road_length_m:
                raise ValueError("vehicle initial_x_m exceeds road length")
        for rsu in self.rsus:
            if rsu.x_m > self.road_length_m:
                raise ValueError("RSU x_m exceeds road length")
        if not all(str(value) for value in (self.lane_id, self.edge_id, self.route_id)):
            raise ValueError("SUMO identifiers must not be empty")

    @property
    def parameters(self) -> dict[str, object]:
        """Canonical JSON-compatible parameters used for the scenario hash."""
        return {
            "generator_version": GENERATOR_VERSION,
            "scenario_id": str(self.scenario_id),
            "seed": int(self.seed),
            "duration_s": float(self.duration_s),
            "step_s": float(self.step_s),
            "road_length_m": float(self.road_length_m),
            "lane_id": str(self.lane_id),
            "edge_id": str(self.edge_id),
            "route_id": str(self.route_id),
            "vehicles": [v.as_dict() for v in self.vehicles],
            "rsus": [r.as_dict() for r in self.rsus],
        }

    @property
    def scenario_hash(self) -> str:
        return _json_hash(self.parameters)

    def as_dict(self) -> dict[str, object]:
        result = dict(self.parameters)
        result["scenario_hash"] = self.scenario_hash
        return result

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SumoScenario":
        """Build a scenario from a YAML/JSON mapping.

        Accepted forms intentionally stay compact: ``vehicles`` is either a
        list of mappings or an integer (which creates ``v0..``), while ``rsus``
        is a list or a mapping keyed by RSU ID.  This parser freezes the
        effective values before any external SUMO process is launched.
        """
        def _vehicles(value: object) -> tuple[SumoVehicle, ...]:
            if isinstance(value, int) and not isinstance(value, bool):
                return tuple(SumoVehicle(f"v{i}") for i in range(value))
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError("vehicles must be a positive integer or list")
            parsed = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError("vehicle entries must be mappings")
                parsed.append(SumoVehicle(
                    str(item.get("id", item.get("vehicle_id", ""))),
                    float(item.get("depart_s", item.get("depart", 0.0))),
                    float(item.get("initial_x_m", item.get("initial_x", 0.0))),
                    float(item.get("speed_mps", item.get("speed", 10.0))),
                ))
            return tuple(parsed)

        def _rsus(value: object) -> tuple[SumoRsu, ...]:
            if isinstance(value, Mapping):
                entries = [(key, item) for key, item in value.items()]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                entries = [(None, item) for item in value]
            else:
                raise ValueError("rsus must be a mapping or list")
            parsed = []
            for key, item in entries:
                if not isinstance(item, Mapping):
                    raise ValueError("RSU entries must be mappings")
                rsu_id = str(item.get("id", item.get("rsu_id", key or "")))
                parsed.append(SumoRsu(
                    rsu_id,
                    float(item.get("x", item.get("x_m", 0.0))),
                    float(item.get("y", item.get("y_m", 0.0))),
                    float(item.get("coverage_radius_m", item.get("radius_m", 10.0))),
                ))
            return tuple(parsed)

        return cls(
            scenario_id=str(config.get("scenario_id", "phase4-sumo-small")),
            seed=int(config.get("seed", 0)),
            duration_s=float(config.get("duration_s", config.get("sim_duration_s", 20.0))),
            step_s=float(config.get("step_s", config.get("step_length_s", 1.0))),
            road_length_m=float(config.get("road_length_m", 200.0)),
            vehicles=_vehicles(config.get("vehicles", 1)),
            rsus=_rsus(config.get("rsus", {})),
            lane_id=str(config.get("lane_id", "lane0")),
            edge_id=str(config.get("edge_id", "edge0")),
            route_id=str(config.get("route_id", "route0")),
        )


@dataclass(frozen=True)
class ScenarioArtifacts:
    """Paths, hashes, and execution metadata written by a scenario run."""

    scenario: SumoScenario
    output_dir: str
    files: Mapping[str, str]
    hashes: Mapping[str, str]
    execution_mode: str = "inputs_only"
    sumo_available: bool = False
    sumo_binary: str | None = None
    sumo_version: str | None = None
    real_sumo_result: bool = False

    @property
    def trace_path(self) -> str:
        return str(self.files.get("fcd", ""))

    @property
    def provenance_hash(self) -> str:
        return _json_hash({
            "scenario_hash": self.scenario.scenario_hash,
            "files": dict(sorted(self.hashes.items())),
            "execution_mode": self.execution_mode,
            "sumo_available": self.sumo_available,
            "sumo_binary": self.sumo_binary,
            "sumo_version": self.sumo_version,
        })

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario.as_dict(),
            "output_dir": self.output_dir,
            "files": dict(self.files),
            "hashes": dict(self.hashes),
            "execution_mode": self.execution_mode,
            "sumo_available": self.sumo_available,
            "sumo_binary": self.sumo_binary,
            "sumo_version": self.sumo_version,
            "real_sumo_result": self.real_sumo_result,
            "controlled_fallback": self.execution_mode == "controlled_fcd_fallback",
            "provenance_hash": self.provenance_hash,
        }


def detect_sumo(binary: str | None = None) -> tuple[str | None, str | None]:
    """Locate a SUMO executable and return ``(path, --version output)``."""
    candidate = binary or os.environ.get("SUMO_BINARY") or shutil.which("sumo") or shutil.which("sumo-gui")
    if not candidate:
        return None, None
    path = shutil.which(candidate) or candidate
    try:
        completed = subprocess.run([path, "--version"], check=False, capture_output=True, text=True, timeout=10)
        version = (completed.stdout or completed.stderr).strip().splitlines()[0] if (completed.stdout or completed.stderr).strip() else None
    except (OSError, subprocess.SubprocessError):
        version = None
    return path, version


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_sumo_inputs(scenario: SumoScenario, output_dir: str | Path) -> dict[str, str]:
    """Write deterministic SUMO network, route, and configuration files."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = scenario.scenario_id
    node_file = target / f"{stem}.nod.xml"
    edge_file = target / f"{stem}.edg.xml"
    route_file = target / f"{stem}.rou.xml"
    cfg_file = target / f"{stem}.sumocfg"
    fcd_file = target / f"{stem}.fcd.xml"
    metadata_file = target / f"{stem}.scenario.json"
    half = scenario.road_length_m
    _write_text(node_file, """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<nodes>\n  <node id=\"n0\" x=\"0\" y=\"0\" type=\"priority\"/>\n  <node id=\"n1\" x=\"{length:.6f}\" y=\"0\" type=\"priority\"/>\n</nodes>\n""".format(length=half))
    _write_text(edge_file, """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<edges>\n  <edge id=\"{edge}\" from=\"n0\" to=\"n1\" priority=\"1\" numLanes=\"1\" speed=\"{speed:.6f}\"/>\n</edges>\n""".format(edge=scenario.edge_id, speed=max(v.speed_mps for v in scenario.vehicles)))
    vehicle_lines = []
    for vehicle in scenario.vehicles:
        vehicle_lines.append(
            "  <vehicle id=\"{id}\" type=\"car\" route=\"{route}\" depart=\"{depart:.6f}\" departPos=\"{pos:.6f}\" departSpeed=\"{speed:.6f}\"/>".format(
                id=vehicle.vehicle_id, route=scenario.route_id, depart=vehicle.depart_s,
                pos=vehicle.initial_x_m, speed=vehicle.speed_mps,
            )
        )
    _write_text(route_file, """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<routes>\n  <vType id=\"car\" accel=\"2.6\" decel=\"4.5\" sigma=\"0\" length=\"5\" minGap=\"2.5\" maxSpeed=\"55\"/>\n  <route id=\"{route}\" edges=\"{edge}\"/>\n{vehicles}\n</routes>\n""".format(route=scenario.route_id, edge=scenario.edge_id, vehicles="\n".join(vehicle_lines)))
    _write_text(cfg_file, """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<configuration>\n  <input>\n    <net-file value=\"{net}\"/>\n    <route-files value=\"{route}\"/>\n  </input>\n  <time>\n    <begin value=\"0\"/>\n    <end value=\"{duration:.6f}\"/>\n    <step-length value=\"{step:.6f}\"/>\n  </time>\n  <seed value=\"{seed}\"/>\n  <output>\n    <fcd-output value=\"{fcd}\"/>\n    <fcd-output.geo value=\"false\"/>\n  </output>\n</configuration>\n""".format(net=node_file.name.replace(".nod.xml", ".net.xml"), route=route_file.name, duration=scenario.duration_s, step=scenario.step_s, seed=scenario.seed, fcd=fcd_file.name))
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "scenario": scenario.as_dict(),
        "rsu_sites": [r.as_dict() for r in scenario.rsus],
        "claims": {
            "controlled_fcd_is_real_world": False,
            "inputs_only_until_sumocheck_or_sumo_run": True,
        },
        "files": {
            "nodes": node_file.name,
            "edges": edge_file.name,
            "routes": route_file.name,
            "config": cfg_file.name,
            "fcd": fcd_file.name,
        },
    }
    _write_text(metadata_file, json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return {
        "nodes": str(node_file),
        "edges": str(edge_file),
        "routes": str(route_file),
        "config": str(cfg_file),
        "fcd": str(fcd_file),
        "metadata": str(metadata_file),
    }


def generate_controlled_fcd(scenario: SumoScenario, path: str | Path) -> str:
    """Generate a deterministic FCD trace from frozen straight-road motion."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("fcd-export", {"scenario_hash": scenario.scenario_hash})
    # SUMO FCD does not require a step at t=0, but including it makes route
    # coverage and parser behavior easy to inspect by hand.
    count = int(math.floor(scenario.duration_s / scenario.step_s + 1e-9))
    for index in range(count + 1):
        timestamp = index * scenario.step_s
        timestep = ET.SubElement(root, "timestep", {"time": f"{timestamp:.6f}"})
        for vehicle in scenario.vehicles:
            if timestamp + 1e-12 < vehicle.depart_s:
                continue
            x = vehicle.initial_x_m + vehicle.speed_mps * (timestamp - vehicle.depart_s)
            if x < -1e-9 or x > scenario.road_length_m + 1e-9:
                continue
            ET.SubElement(timestep, "vehicle", {
                "id": vehicle.vehicle_id,
                "x": f"{max(0.0, min(scenario.road_length_m, x)):.6f}",
                "y": "0.000000",
                "speed": f"{vehicle.speed_mps:.6f}",
            })
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:  # pragma: no cover - Python 3.8 fallback
        pass
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return str(target)


def _build_net(scenario: SumoScenario, output_dir: Path, artifacts: Mapping[str, str]) -> str:
    """Create a valid net.xml without requiring ``netconvert``."""
    path = output_dir / f"{scenario.scenario_id}.net.xml"
    _write_text(path, """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<net version=\"1.9\" junctionCornerDetail=\"5\">\n  <location netOffset=\"0.00,0.00\" convBoundary=\"0.00,0.00,{length:.6f},0.00\" origBoundary=\"0.00,0.00,{length:.6f},0.00\" projParameter=\"!\"/>\n  <junction id=\"n0\" type=\"dead_end\" x=\"0.00\" y=\"0.00\" incLanes=\"\" intLanes=\"\" shape=\"0.00,0.00\"/>\n  <junction id=\"n1\" type=\"dead_end\" x=\"{length:.6f}\" y=\"0.00\" incLanes=\"{lane}\" intLanes=\"\" shape=\"{length:.6f},0.00\"/>\n  <edge id=\"{edge}\" from=\"n0\" to=\"n1\" priority=\"1\" function=\"normal\">\n    <lane id=\"{lane}\" index=\"0\" speed=\"{speed:.6f}\" length=\"{length:.6f}\" shape=\"0.00,0.00 {length:.6f},0.00\"/>\n  </edge>\n</net>\n""".format(length=scenario.road_length_m, edge=scenario.edge_id, lane=scenario.lane_id, speed=max(v.speed_mps for v in scenario.vehicles)))
    return str(path)


def run_sumo_scenario(
    scenario: SumoScenario,
    output_dir: str | Path,
    *,
    run_sumo: bool = False,
    sumo_binary: str | None = None,
    timeout_s: float = 120.0,
) -> ScenarioArtifacts:
    """Materialize a scenario and optionally run a real SUMO process.

    ``run_sumo=False`` still creates a controlled FCD so the rest of the
    simulator can be validated.  This output carries ``controlled_fallback``
    in the returned provenance.  ``run_sumo=True`` raises if the executable is
    unavailable rather than silently substituting a synthetic result.
    """
    target = Path(output_dir)
    files = write_sumo_inputs(scenario, target)
    net_file = _build_net(scenario, target, files)
    files = dict(files)
    files["net"] = net_file
    binary, version = detect_sumo(sumo_binary)
    if run_sumo:
        if binary is None:
            raise RuntimeError("SUMO executable not found; use run_sumo=False for controlled fallback")
        command = [binary, "-c", files["config"], "--seed", str(scenario.seed), "--no-step-log", "true"]
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=float(timeout_s), cwd=str(target)
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"SUMO failed for scenario {scenario.scenario_id}: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SUMO timed out for scenario {scenario.scenario_id}") from exc
        if not Path(files["fcd"]).is_file():
            raise RuntimeError(
                f"SUMO completed without the configured FCD output: {files['fcd']}"
            )
        mode = "sumo"
        real_result = True
    else:
        generate_controlled_fcd(scenario, files["fcd"])
        mode = "controlled_fcd_fallback"
        real_result = False
    hashes = {name: sha256_file(path) for name, path in sorted(files.items()) if Path(path).is_file()}
    # Rewrite metadata after net/FCD execution so it includes all artifact
    # hashes and can itself be checked into a run directory.
    metadata = Path(files["metadata"])
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["execution"] = {
        "mode": mode,
        "sumo_available": binary is not None,
        "sumo_binary": binary,
        "sumo_version": version,
        "real_sumo_result": real_result,
        "artifact_sha256": hashes,
    }
    _write_text(metadata, json.dumps(payload, sort_keys=True, indent=2) + "\n")
    hashes["metadata"] = sha256_file(metadata)
    return ScenarioArtifacts(
        scenario=scenario,
        output_dir=str(target.resolve()),
        files=files,
        hashes=hashes,
        execution_mode=mode,
        sumo_available=binary is not None,
        sumo_binary=binary,
        sumo_version=version,
        real_sumo_result=real_result,
    )


def write_scenario_provenance(path: str | Path, artifacts: ScenarioArtifacts) -> str:
    """Write the full run provenance and return its absolute path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_text(target, json.dumps(artifacts.as_dict(), sort_keys=True, indent=2) + "\n")
    return str(target)
