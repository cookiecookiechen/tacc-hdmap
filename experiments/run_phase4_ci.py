"""Run a small, frozen multi-seed comparison with confidence intervals.

This runner is deliberately separate from ``run_sweep.py``.  It accepts a
small list of explicit seeds, evaluates the existing B1--B7 plans on the same
trace adapter, and writes enough provenance to reproduce the stability gate.
The parameter block is hashed before any episode is run; changing a frozen
parameter therefore produces a different result artifact instead of silently
mixing runs.

The command never expands a parameter grid.  ``full_sweep`` is rejected when
set in the configuration and the number of seeds/episodes is bounded by the
frozen schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tacp_sim.baselines import BASELINES, all_baselines
from tacp_sim.io import load_config
from tacp_sim.mobility import (
    csv_trace_with_provenance,
    sumo_fcd_trace_with_provenance,
    TraceProvenance,
)
from tacp_sim.models import ContactWindow, ObjectVersion
from tacp_sim.simulator import Simulation, _trace_from_config


# These controls describe the validation run, rather than the simulator
# itself.  They are excluded from the frozen-parameter hash so that changing
# the number of repeats or bootstrap draws does not invalidate the model
# parameter provenance.
_RUN_CONTROL_KEYS = {
    "seeds", "episodes_per_seed", "ci_level", "bootstrap_samples", "stability",
    "output", "full_sweep", "baselines",
}
_FROZEN_SCHEMA = "phase4-frozen-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def frozen_parameter_hash(config: Mapping[str, Any]) -> str:
    """Hash simulator parameters while excluding repeat/CI controls."""
    frozen = {str(k): _jsonable(v) for k, v in config.items() if k not in _RUN_CONTROL_KEYS}
    payload = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_config(config: Mapping[str, Any]) -> tuple[list[int], list[str]]:
    """Validate the intentionally small Phase 4 configuration schema."""
    if config.get("frozen_parameters") is not True:
        raise ValueError("Phase 4 requires frozen_parameters: true")
    if str(config.get("frozen_schema", _FROZEN_SCHEMA)) != _FROZEN_SCHEMA:
        raise ValueError(f"frozen_schema must be {_FROZEN_SCHEMA!r}")
    if bool(config.get("full_sweep", False)):
        raise ValueError("full_sweep is forbidden in the Phase 4 CI runner")
    # Keep the CI entry point fail-closed.  A sweep request must go through a
    # separately reviewed sweep runner after the CI gate has passed; silently
    # accepting one here would make it too easy to mix frozen and exploratory
    # parameters in the same artifact.
    for key in ("sweep", "sweep_grid", "parameter_grid", "grid"):
        if config.get(key):
            raise ValueError(f"{key} is forbidden in the Phase 4 CI runner")
    seeds_raw = config.get("seeds")
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
        raise ValueError("seeds must be an explicit list of integer seeds")
    seeds = [int(seed) for seed in seeds_raw]
    if len(seeds) < 2 or len(seeds) > 32 or len(set(seeds)) != len(seeds):
        raise ValueError("Phase 4 requires 2..32 unique seeds")
    episodes = int(config.get("episodes_per_seed", 1))
    if episodes < 1 or episodes > 20:
        raise ValueError("episodes_per_seed must be between 1 and 20 for a small CI run")
    ci_level = float(config.get("ci_level", 0.95))
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0,1)")
    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    if bootstrap_samples < 100 or bootstrap_samples > 100_000:
        raise ValueError("bootstrap_samples must be between 100 and 100000")
    required = ("trace_kind", "object_bytes", "n", "k", "ttl_s", "link_rate_bps")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"frozen config missing required keys: {missing}")
    names = [str(name) for name in config.get("baselines", BASELINES)]
    unknown = sorted(set(names) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown baseline(s): {unknown}")
    if not names:
        raise ValueError("baselines must not be empty")
    stability = config.get("stability", {})
    if not isinstance(stability, Mapping):
        raise ValueError("stability must be a mapping")
    max_range = float(stability.get("max_p_fresh_range", 0.20))
    min_seed_pass_rate = float(stability.get("min_seed_pass_rate", 1.0))
    seed_p_floor = float(stability.get("seed_p_fresh_floor", 1.0))
    if max_range < 0 or not 0.0 <= min_seed_pass_rate <= 1.0 or not 0.0 <= seed_p_floor <= 1.0:
        raise ValueError("invalid stability thresholds")
    return seeds, names


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli success probability."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0,trials]")
    if trials == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - float(confidence)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_mean_interval(
    values: Iterable[float], *, confidence: float = 0.95, samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean (deterministic RNG)."""
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = [sum(rng.choice(data) for _ in data) / len(data) for _ in range(int(samples))]
    alpha = 1.0 - float(confidence)
    return (_quantile(means, alpha / 2.0), _quantile(means, 1.0 - alpha / 2.0))


def _vehicle_metric_interval(
    rows: Sequence[Mapping[str, Any]], key: str, *, confidence: float,
    bootstrap_samples: int, seed: int,
) -> tuple[tuple[float, float], int]:
    """Bootstrap a metric after clustering repeated seeds by vehicle.

    A large CI run repeats each trip for several seeds.  Averaging each
    vehicle first avoids treating those repeated rows as independent traffic
    samples when precision gates are evaluated.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        grouped.setdefault(str(row.get("vehicle_id", "")), []).append(numeric)
    means = [statistics.fmean(values) for values in grouped.values() if values]
    if not means:
        return ((math.nan, math.nan), 0)
    return (bootstrap_mean_interval(
        means, confidence=confidence, samples=bootstrap_samples, seed=seed,
    ), len(means))


def _resolve_config_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve repository-relative trace paths without changing hash inputs."""
    resolved = dict(config)
    for key in ("trace_path", "trace_provenance_path"):
        trace_path = resolved.get(key)
        if trace_path and not Path(str(trace_path)).is_absolute():
            resolved[key] = str((ROOT / str(trace_path)).resolve())
    return resolved


def _load_windows(config: Mapping[str, Any], seed: int) -> tuple[list[ContactWindow], dict[str, Any]]:
    kind = str(config.get("trace_kind", "synthetic_line")).lower()
    rate = float(config.get("link_rate_bps", config.get("rate_bps", 1_000_000)))
    trace_path = config.get("trace_path")
    if kind == "csv":
        if not trace_path:
            raise ValueError("trace_path is required for CSV Phase 4 validation")
        windows, provenance = csv_trace_with_provenance(
            Path(str(trace_path)), default_rate_bps=rate,
            sample_interval_s=float(config.get("sample_interval_s", config.get("trace_sample_interval_s", 1.0))),
            merge_consecutive=bool(config.get("merge_consecutive", config.get("trace_merge_consecutive", True))),
        )
    elif kind in {"sumo", "sumo_fcd", "fcd"}:
        if not trace_path or not isinstance(config.get("rsus"), Mapping):
            raise ValueError("trace_path and rsus are required for SUMO Phase 4 validation")
        windows, provenance = sumo_fcd_trace_with_provenance(
            Path(str(trace_path)), config["rsus"],
            coverage_radius_m=(float(config["coverage_radius_m"]) if config.get("coverage_radius_m") is not None else None),
            default_rate_bps=rate,
            sample_interval_s=float(config.get("sample_interval_s", config.get("trace_sample_interval_s", 1.0))),
            merge_consecutive=bool(config.get("merge_consecutive", config.get("trace_merge_consecutive", True))),
            vehicle_ids=config.get("vehicle_ids"),
        )
    else:
        windows = list(_trace_from_config(config, seed))
        provenance = TraceProvenance(
            source_kind=kind, parser="simulator_trace_v1",
            parameters={"seed": int(seed), "rate_bps": rate},
        )
    provenance_dict = provenance.as_dict()
    # A converted CSV hash proves the exact simulator input, while the
    # adjacent provenance document carries the public dataset, archive hash,
    # and frozen RSU-map hash.  Include both in the CI artifact so a result
    # cannot be detached from its source manifest after the run.
    provenance_path = config.get("trace_provenance_path")
    if provenance_path:
        path = Path(str(provenance_path))
        if not path.is_file():
            raise FileNotFoundError(f"trace provenance does not exist: {path}")
        try:
            external = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trace provenance: {path}") from exc
        if not isinstance(external, Mapping):
            raise ValueError("trace provenance must be a JSON object")
        recorded_output = external.get("output")
        recorded_hash = recorded_output.get("sha256") if isinstance(recorded_output, Mapping) else None
        if recorded_hash != provenance.source_sha256:
            raise ValueError("trace CSV hash differs from trace provenance output hash")
        provenance_dict["dataset_provenance"] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": external.get("source"),
            "input": external.get("input"),
            "output": external.get("output"),
            "source_archive": external.get("source_archive"),
            "rsu_config": external.get("rsu_config"),
            "preprocessing": external.get("preprocessing"),
            "counts": external.get("counts"),
        }
    return windows, provenance_dict


def _trace_vehicle_ids(
    windows: Sequence[ContactWindow], config: Mapping[str, Any],
) -> list[str]:
    """Resolve the frozen vehicle sample used by a comparison.

    CSV adapters intentionally load the complete converted trace so its file
    hash remains the provenance anchor.  The CI config may then select an
    explicit ordered set, ``"all"``, or the first ``trace_vehicle_limit``
    identifiers in lexical order.  Filtering by vehicle before placement is
    important: a placement learned from every taxi would otherwise leak
    future contacts between independent trip episodes.
    """
    available = sorted({str(window.vehicle_id) for window in windows})
    requested = config.get("trace_vehicle_ids")
    if requested is None or requested == "all":
        selected = available
    elif isinstance(requested, str):
        selected = [requested]
    else:
        if not isinstance(requested, Sequence) or isinstance(requested, (bytes, str)):
            raise ValueError("trace_vehicle_ids must be a list of IDs or 'all'")
        selected = [str(value) for value in requested]
        if len(set(selected)) != len(selected):
            raise ValueError("trace_vehicle_ids must not contain duplicates")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"trace_vehicle_ids not present in trace: {unknown[:5]}")
    limit_raw = config.get("trace_vehicle_limit")
    if limit_raw is not None:
        limit = int(limit_raw)
        if limit < 1:
            raise ValueError("trace_vehicle_limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("trace contains no selected vehicle IDs")
    return selected


def _effective_sample_gate(
    rows: Sequence[Mapping[str, Any]], *, baselines: Sequence[str],
    seeds: Sequence[int], trace_vehicle_ids: Sequence[str],
    confidence: float, bootstrap_samples: int, thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Check effective sample size and confidence interval precision.

    Seed repeatability alone can pass with one deterministic trip.  The
    larger-sample gate therefore checks distinct vehicles, total episode rows,
    seed coverage, and optional maximum CI widths.  ``None`` leaves a width
    unconstrained so legacy two-seed unit tests retain their original scope.
    """
    min_vehicles = int(thresholds.get("min_trace_vehicles", 1))
    min_rows = int(thresholds.get("min_total_episodes", 1))
    min_seed_count = int(thresholds.get("min_seed_count", 2))
    if min_vehicles < 1 or min_rows < 1 or min_seed_count < 1:
        raise ValueError("effective sample thresholds must be positive")

    def optional_width(value: Any) -> float | None:
        if value is None:
            return None
        width = float(value)
        if not math.isfinite(width) or width < 0:
            raise ValueError("CI width thresholds must be finite and non-negative")
        return width

    max_p_width = optional_width(thresholds.get("max_p_fresh_ci_width"))
    max_rank_width = optional_width(thresholds.get("max_mean_rank_ci_width"))
    max_decode_width = optional_width(thresholds.get("max_mean_t_decode_ci_width"))
    methods: dict[str, Any] = {}
    passed = True
    for baseline in baselines:
        method_rows = [row for row in rows if row["baseline"] == baseline]
        vehicles = sorted({str(row.get("vehicle_id", "")) for row in method_rows})
        used_seeds = sorted({int(row["seed"]) for row in method_rows})
        summary = _summary(
            method_rows, confidence=confidence, bootstrap_samples=bootstrap_samples,
            seed=used_seeds[0] if used_seeds else 0,
        )
        # Precision gates use vehicle-clustered intervals.  The row-level
        # Wilson/bootstrap intervals remain in the artifact for reporting,
        # while repeated seeds of one vehicle count as one trajectory unit.
        p_ci = summary["P_fresh_vehicle_bootstrap_ci"]
        rank_ci = summary["mean_rank_vehicle_bootstrap_ci"]
        decode_ci = summary["mean_t_decode_vehicle_bootstrap_ci"]

        def width(interval: Sequence[Any]) -> float | None:
            if len(interval) != 2 or interval[0] is None or interval[1] is None:
                return None
            return float(interval[1]) - float(interval[0])

        p_width = width(p_ci)
        rank_width = width(rank_ci)
        decode_width = width(decode_ci)
        checks = {
            "min_trace_vehicles": len(vehicles) >= min_vehicles,
            "min_total_episodes": len(method_rows) >= min_rows,
            "min_seed_count": len(used_seeds) >= min_seed_count,
            "max_p_fresh_ci_width": max_p_width is None or (p_width is not None and p_width <= max_p_width + 1e-12),
            "max_mean_rank_ci_width": max_rank_width is None or (rank_width is not None and rank_width <= max_rank_width + 1e-12),
            "max_mean_t_decode_ci_width": max_decode_width is None or (decode_width is not None and decode_width <= max_decode_width + 1e-12),
        }
        method_passed = all(checks.values())
        passed = passed and method_passed
        methods[baseline] = {
            "selected_vehicles": vehicles,
            "selected_vehicle_count": len(vehicles),
            "available_selected_vehicle_count": len(trace_vehicle_ids),
            "rows": len(method_rows),
            "seeds": used_seeds,
            "summary": summary,
            "ci_widths": {
                "P_fresh_wilson": p_width,
                "mean_rank_bootstrap": rank_width,
                "mean_t_decode_bootstrap": decode_width,
            },
            "checks": checks,
            "passed": method_passed,
        }
    return {
        "passed": passed,
        "thresholds": {
            "min_trace_vehicles": min_vehicles,
            "min_total_episodes": min_rows,
            "min_seed_count": min_seed_count,
            "max_p_fresh_ci_width": max_p_width,
            "max_mean_rank_ci_width": max_rank_width,
            "max_mean_t_decode_ci_width": max_decode_width,
        },
        "methods": methods,
    }


def _object_from_config(config: Mapping[str, Any]) -> ObjectVersion:
    return ObjectVersion(
        str(config.get("object_id", "phase4-object")), int(config.get("version", 1)),
        int(config["object_bytes"]), int(config["n"]), int(config["k"]),
        created_at=float(config.get("created_at", 0.0)), ttl_s=float(config["ttl_s"]),
    )


def _run_one_plan(
    config: Mapping[str, Any], windows: Sequence[ContactWindow], plan: Any,
    *, seed: int, episode: int, vehicle_id: str, object_version: ObjectVersion,
) -> dict[str, Any]:
    simulation_config = dict(config)
    simulation_config["seed"] = int(seed)
    simulation_config["vehicle_id"] = str(vehicle_id)
    if plan.coded:
        placement = plan.placement
    else:
        # Simulation transfers one coded shard per action.  Model an uncoded
        # copy as a one-shard object while retaining plan.storage_bytes.
        simulation_config.update({"object_bytes": object_version.size_bytes, "n": 1, "k": 1})
        placement = {rsu: [0] for rsu, shard_ids in plan.placement.items() if shard_ids}
    result = Simulation(simulation_config, int(seed)).run_episode(
        episode, windows=windows, placement_override=placement,
    )
    return {
        "baseline": plan.name,
        "seed": int(seed),
        "episode": int(episode),
        "vehicle_id": str(vehicle_id),
        "fresh_success": bool(result.fresh_success),
        "t_decode": result.t_decode,
        "rank": int(result.rank),
        "transmissions": len(result.transmissions),
        "storage": int(plan.storage_bytes),
        "backhaul": int(result.backhaul),
        "migration": int(result.migration),
        "repair": int(result.repair),
        "trace_windows": len(windows),
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, confidence: float, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    fresh = [1.0 if bool(row["fresh_success"]) else 0.0 for row in rows]
    rank = [float(row["rank"]) for row in rows]
    decode = [float(row["t_decode"]) for row in rows if row.get("t_decode") is not None]
    successes = int(sum(fresh))
    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    p_ci = wilson_interval(successes, len(rows), confidence)
    rank_ci = bootstrap_mean_interval(
        rank, confidence=confidence, samples=bootstrap_samples, seed=seed + 1,
    )
    decode_ci = bootstrap_mean_interval(
        decode, confidence=confidence, samples=bootstrap_samples, seed=seed + 2,
    )
    vehicle_fresh_ci, vehicle_count = _vehicle_metric_interval(
        rows, "fresh_success", confidence=confidence,
        bootstrap_samples=bootstrap_samples, seed=seed + 3,
    )
    vehicle_rank_ci, _ = _vehicle_metric_interval(
        rows, "rank", confidence=confidence,
        bootstrap_samples=bootstrap_samples, seed=seed + 4,
    )
    vehicle_decode_ci, _ = _vehicle_metric_interval(
        rows, "t_decode", confidence=confidence,
        bootstrap_samples=bootstrap_samples, seed=seed + 5,
    )
    return {
        "episodes": len(rows),
        "fresh_successes": successes,
        "P_fresh": finite_or_none(successes / len(rows) if rows else math.nan),
        "P_fresh_wilson_ci": [finite_or_none(p_ci[0]), finite_or_none(p_ci[1])],
        "P_fresh_bootstrap_ci": bootstrap_mean_interval(
            fresh, confidence=confidence, samples=bootstrap_samples, seed=seed,
        ),
        "P_fresh_vehicle_bootstrap_ci": [finite_or_none(vehicle_fresh_ci[0]), finite_or_none(vehicle_fresh_ci[1])],
        "independent_vehicle_count": vehicle_count,
        "mean_rank": finite_or_none(statistics.fmean(rank) if rank else math.nan),
        "mean_rank_bootstrap_ci": [finite_or_none(rank_ci[0]), finite_or_none(rank_ci[1])],
        "mean_rank_vehicle_bootstrap_ci": [finite_or_none(vehicle_rank_ci[0]), finite_or_none(vehicle_rank_ci[1])],
        "decoded_episodes": len(decode),
        "mean_t_decode": finite_or_none(statistics.fmean(decode) if decode else math.nan),
        "mean_t_decode_bootstrap_ci": [finite_or_none(decode_ci[0]), finite_or_none(decode_ci[1])],
        "mean_t_decode_vehicle_bootstrap_ci": [finite_or_none(vehicle_decode_ci[0]), finite_or_none(vehicle_decode_ci[1])],
    }


def _stability_gate(
    rows: Sequence[Mapping[str, Any]], *, baselines: Sequence[str], seeds: Sequence[int],
    confidence: float, bootstrap_samples: int, thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    max_range = float(thresholds.get("max_p_fresh_range", 0.20))
    max_rank_range = float(thresholds.get("max_mean_rank_range", 1.0))
    min_seed_pass_rate = float(thresholds.get("min_seed_pass_rate", 1.0))
    seed_p_floor = float(thresholds.get("seed_p_fresh_floor", 1.0))
    if not 0.0 <= seed_p_floor <= 1.0:
        raise ValueError("seed_p_fresh_floor must be in [0,1]")
    per_baseline: dict[str, Any] = {}
    all_passed = True
    for baseline in baselines:
        seed_rows: dict[int, list[Mapping[str, Any]]] = {int(seed): [] for seed in seeds}
        for row in rows:
            if row["baseline"] == baseline:
                seed_rows.setdefault(int(row["seed"]), []).append(row)
        summaries = {
            str(seed): _summary(seed_rows[seed], confidence=confidence,
                                bootstrap_samples=bootstrap_samples, seed=seed)
            for seed in seeds
        }
        p_values = [float(item["P_fresh"]) for item in summaries.values()]
        rank_values = [float(item["mean_rank"]) for item in summaries.values()]
        pass_rate = sum(value >= seed_p_floor - 1e-12 for value in p_values) / len(p_values) if p_values else 0.0
        p_range = max(p_values) - min(p_values) if p_values else math.inf
        rank_range = max(rank_values) - min(rank_values) if rank_values else math.inf
        p_std = statistics.pstdev(p_values) if len(p_values) > 1 else 0.0
        rank_std = statistics.pstdev(rank_values) if len(rank_values) > 1 else 0.0
        passed = p_range <= max_range + 1e-12 and rank_range <= max_rank_range + 1e-12 and pass_rate >= min_seed_pass_rate - 1e-12
        all_passed = all_passed and passed
        per_baseline[baseline] = {
            "per_seed": summaries,
            "P_fresh_range": p_range,
            "P_fresh_stddev": p_std,
            "mean_rank_range": rank_range,
            "mean_rank_stddev": rank_std,
            "seed_pass_rate": pass_rate,
            "passed": passed,
        }
    return {
        "passed": all_passed,
        "thresholds": {
            "max_p_fresh_range": max_range,
            "max_mean_rank_range": max_rank_range,
            "min_seed_pass_rate": min_seed_pass_rate,
            "seed_p_fresh_floor": seed_p_floor,
        },
        "methods": per_baseline,
    }


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the frozen, multi-seed small comparison and return JSON data."""
    seeds, baseline_names = validate_frozen_config(config)
    resolved = _resolve_config_paths(config)
    object_version = _object_from_config(resolved)
    confidence = float(resolved.get("ci_level", 0.95))
    bootstrap_samples = int(resolved.get("bootstrap_samples", 2000))
    all_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for seed in seeds:
        windows, provenance = _load_windows(resolved, seed)
        selected_vehicle_ids = _trace_vehicle_ids(windows, resolved)
        trace_records.append({
            "seed": int(seed), "windows": len(windows),
            "selected_vehicle_ids": selected_vehicle_ids,
            "selected_vehicle_count": len(selected_vehicle_ids),
            "provenance": provenance,
        })
        for vehicle_id in selected_vehicle_ids:
            vehicle_windows = [window for window in windows if str(window.vehicle_id) == vehicle_id]
            if not vehicle_windows:
                raise ValueError(f"selected vehicle has no contact windows: {vehicle_id}")
            plans = all_baselines(
                object_version, vehicle_windows,
                capacity_bytes=int(resolved.get("placement_capacity_bytes", object_version.coded_size_bytes)),
            )
            for name in baseline_names:
                plan = plans[name]
                for episode in range(int(resolved.get("episodes_per_seed", 1))):
                    all_rows.append(_run_one_plan(
                        resolved, vehicle_windows, plan, seed=seed, episode=episode,
                        vehicle_id=vehicle_id, object_version=object_version,
                    ))
    summaries: dict[str, Any] = {}
    for index, name in enumerate(baseline_names):
        rows = [row for row in all_rows if row["baseline"] == name]
        summaries[name] = _summary(
            rows, confidence=confidence, bootstrap_samples=bootstrap_samples,
            seed=seeds[index % len(seeds)],
        )
    stability = _stability_gate(
        all_rows, baselines=baseline_names, seeds=seeds,
        confidence=confidence, bootstrap_samples=bootstrap_samples,
        thresholds=resolved.get("stability", {}),
    )
    selected_ids = sorted({str(row["vehicle_id"]) for row in all_rows})
    effective_sample = _effective_sample_gate(
        all_rows, baselines=baseline_names, seeds=seeds,
        trace_vehicle_ids=selected_ids, confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        thresholds=resolved.get("stability", {}),
    )
    # Both dimensions are required: a repeatable but tiny fixture must not
    # unlock a full sweep, and a large sample with unstable seed behavior is
    # equally unsuitable for freezing.
    stability["seed_repeatability_passed"] = bool(stability["passed"])
    stability["effective_sample_passed"] = bool(effective_sample["passed"])
    stability["effective_sample"] = effective_sample
    stability["passed"] = bool(stability["passed"] and effective_sample["passed"])
    return {
        "phase": "phase4_small_ci",
        "sweep_started": False,
        "frozen_parameters": True,
        "frozen_schema": _FROZEN_SCHEMA,
        "frozen_parameter_hash": frozen_parameter_hash(config),
        "seeds": [int(seed) for seed in seeds],
        "episodes_per_seed": int(resolved.get("episodes_per_seed", 1)),
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "trace_records": trace_records,
        "selected_vehicle_ids": selected_ids,
        "selected_vehicle_count": len(selected_ids),
        "baselines": baseline_names,
        "summaries": summaries,
        "seed_repeatability": stability["methods"],
        "stability_gate": stability,
        "runs": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4_frozen.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-gate-failure", action="store_true",
                        help="write the artifact without returning a failing exit code")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config)
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "results" / "processed" / f"phase4_small_ci_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "frozen_parameter_hash": result["frozen_parameter_hash"],
        "stability_gate_passed": result["stability_gate"]["passed"],
        "sweep_started": result["sweep_started"],
    }, sort_keys=True))
    if not result["stability_gate"]["passed"] and not args.allow_gate_failure:
        raise SystemExit("Phase 4 stability gate failed; inspect the JSON artifact")


if __name__ == "__main__":
    main()
