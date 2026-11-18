"""Validate v3 checkpoints and paired physical-entity cluster intervals.

Porto clusters are physical taxi IDs, not trip IDs. Each bootstrap draw keeps
all trips, cells and fixed random seeds of a sampled taxi together and computes
a ratio of sums, retaining the registered trip-weighted estimand. highD uses
recording clusters; its small validation sample has diagnostic intervals only.
No old-version rows enter any v3 comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEGACY_B7 = "B7_proposed_algorithm1_algorithm2"
RUNTIME_B7 = "B7_runtime_rank_aware"
B5 = "B5_freshness_aware_uncoded_caching"
A2 = "A2_no_rank_aware"
FACTORS = ("object_bytes", "n", "k", "ttl_s", "packet_loss",
           "rsu_failure_probability", "placement_capacity_bytes")
NUMERIC_METRICS = ("rank", "rank_deficit_at_expiry", "tx_bytes", "transmissions",
                   "duplicate_dependent", "stale_on_arrival", "backhaul", "migration", "repair")
EQUIVALENCE_FIELDS = ("fresh_success", "rank", "t_decode", "storage", "trace_windows",
                      "effective_k", *NUMERIC_METRICS)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != "tacp-related-work-protocol-v3" or value.get("frozen") is not True:
        raise ValueError("unsupported or unfrozen v3 protocol")
    return value


def current_source_identity(protocol: Mapping[str, Any], dataset: str, *, root: Path = ROOT) -> dict[str, Any]:
    if dataset == "highd" and protocol.get("release", {}).get("highd_inputs") == "requires_licensed_archive":
        raise FileNotFoundError(
            "highD inputs are not included in the public release. Obtain the licensed archive and run "
            "python scripts/prepare_release_inputs.py --highd-archive /path/to/highD-dataset-v1.0.zip"
        )
    registered = protocol["datasets"][dataset]["source_hashes"]
    data = {}
    for relative, expected in registered.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("data source path must be repository relative")
        if not (root / path).is_file():
            if dataset == "highd":
                raise FileNotFoundError(
                    f"Missing highD input: {relative}. Obtain the licensed highD archive and run "
                    "python scripts/prepare_release_inputs.py --highd-archive /path/to/highD-dataset-v1.0.zip"
                )
            raise FileNotFoundError(f"Missing registered input: {relative}")
        actual = file_hash(root/path)
        if actual != expected:
            raise ValueError(f"frozen data source SHA-256 mismatch: {relative}")
        data[relative] = actual
    files = sorted([*root.glob("src/tacp_sim/*.py"), root/"experiments/run_related_work_v3.py",
                    root/"experiments/run_related_work.py", root/"experiments/run_sweep.py",
                    root/"experiments/run_phase4_ci.py", root/"scripts/validate_related_work_v3.py",
                    root/"scripts/prepare_highd_small.py"])
    # Files such as macOS ._*.py are metadata, not importable simulator modules.
    code = {str(path.relative_to(root)): file_hash(path) for path in files
            if not path.name.startswith(".")}
    return {"files": {**data, **code}, "data_sha256": canonical_hash(data), "code_sha256": canonical_hash(code)}


def cluster_id_for(vehicle_id: str, dataset: str) -> str:
    if dataset == "porto":
        match = re.fullmatch(r"taxi_(\d+)_trip_(\d+)", vehicle_id)
        if not match:
            raise ValueError(f"invalid Porto taxi/trip identifier: {vehicle_id}")
        return "taxi_" + match.group(1)
    if dataset == "highd":
        match = re.fullmatch(r"highd_(\d{2})_vehicle_(\d+)", vehicle_id)
        if not match:
            raise ValueError(f"invalid highD recording/vehicle identifier: {vehicle_id}")
        return "highd_" + match.group(1)
    raise ValueError(f"unsupported dataset: {dataset}")


def run_identity(plan: Mapping[str, Any]) -> str:
    return canonical_hash({key: plan[key] for key in
                           ("plan_hash", "protocol_hash", "source_hashes", "mode", "dataset")})


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def cluster_interval(values: Mapping[str, tuple[float, int]], *, samples: int = 2000,
                     seed: int = 20261018, confidence: float = .95) -> list[float | None]:
    """Resample entities, then ratio of all selected observations (not entity means)."""
    data = [value for _, value in sorted(values.items()) if value[1] > 0]
    if not data:
        return [None, None]
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap specification")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [rng.choice(data) for _ in data]
        draws.append(sum(value for value, count in chosen) / sum(count for value, count in chosen))
    alpha = (1-confidence)/2
    return [quantile(draws, alpha), quantile(draws, 1-alpha)]


@dataclass
class Accumulator:
    count: int = 0
    total: float = 0.0
    clusters: dict[str, tuple[float, int]] = field(default_factory=dict)
    trips: set[str] = field(default_factory=set)
    seeds: dict[int, tuple[float, int]] = field(default_factory=dict)
    metrics: dict[str, tuple[float, int]] = field(default_factory=dict)
    decoded_times: list[float] = field(default_factory=list)

    def add(self, row: Mapping[str, Any], value: float | None = None, *, metrics: bool = True) -> None:
        value = float(row["fresh_success"]) if value is None else float(value)
        self.count += 1
        self.total += value
        cluster, seed = str(row["cluster_id"]), int(row["seed"])
        self.trips.add(str(row["vehicle_id"]))
        total, count = self.clusters.get(cluster, (0.0, 0))
        self.clusters[cluster] = (total+value, count+1)
        total, count = self.seeds.get(seed, (0.0, 0))
        self.seeds[seed] = (total+value, count+1)
        if metrics:
            for name in NUMERIC_METRICS:
                if row.get(name) is not None:
                    total, count = self.metrics.get(name, (0.0, 0))
                    self.metrics[name] = (total+float(row[name]), count+1)
            if row.get("t_decode") is not None:
                self.decoded_times.append(float(row["t_decode"]))

    def result(self, bootstrap: Mapping[str, Any], *, metric: str = "P_fresh") -> dict[str, Any]:
        ci = cluster_interval(self.clusters, samples=int(bootstrap["samples"]),
                              seed=int(bootstrap["seed"]), confidence=float(bootstrap["confidence"]))
        by_seed = {str(seed): total/count for seed, (total, count) in sorted(self.seeds.items())}
        values = list(by_seed.values())
        return {"rows": self.count, "trip_count": len(self.trips), "cluster_count": len(self.clusters),
                metric: self.total/self.count, "cluster_bootstrap_95_ci": ci, "ci_width": ci[1]-ci[0],
                "per_seed": by_seed, "seed_range": max(values)-min(values),
                "mean_metrics": {name: total/count for name, (total, count) in self.metrics.items()},
                "mean_decode_time_conditional": statistics.fmean(self.decoded_times) if self.decoded_times else None,
                "median_decode_time_conditional": statistics.median(self.decoded_times) if self.decoded_times else None,
                "decoded_rows": len(self.decoded_times)}


def checkpoint_paths(inputs: Iterable[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    paths, manifests = [], []
    for item in inputs:
        path = Path(item)
        if path.suffix == ".jsonl":
            candidate = path
        else:
            manifest = read_json(path)
            manifests.append(manifest)
            raw = Path(str(manifest.get("checkpoint", str(path)+".jsonl")))
            choices = (raw, path.parent/raw, path.parent/raw.name, Path(str(path)+".jsonl"))
            candidate = next((choice for choice in choices if choice.is_file()), raw)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        candidate = candidate.resolve()
        if candidate in paths:
            raise ValueError(f"same checkpoint supplied twice: {candidate}")
        paths.append(candidate)
    return paths, manifests


def iter_records(paths: Sequence[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"non-object checkpoint at {path}:{line_number}")
                yield path, line_number, record


def row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row["cell_id"]), int(row["seed"]), int(row["episode"]), str(row["vehicle_id"]), str(row["baseline"]))


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _validate_plan(plan: Mapping[str, Any], protocol: Mapping[str, Any], *, root: Path,
                   verify_sources: bool) -> None:
    if plan.get("schema") != "tacp-related-work-plan-v3":
        raise ValueError("unsupported v3 plan schema")
    if plan.get("plan_hash") != canonical_hash({key: value for key, value in plan.items() if key != "plan_hash"}):
        raise ValueError("plan content hash mismatch")
    if plan.get("protocol_hash") != canonical_hash(protocol):
        raise ValueError("plan protocol hash mismatch")
    for name in ("methods", "seeds", "episodes", "vehicle_ids"):
        values = plan.get(name)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"plan {name} must be nonempty and unique")
    dataset = str(plan["dataset"])
    expected_unit = {"porto": "physical_taxi", "highd": "recording"}.get(dataset)
    if not expected_unit or plan.get("cluster_unit") != expected_unit:
        raise ValueError("wrong physical clustering unit")
    expected_clusters = {vehicle: cluster_id_for(vehicle, dataset) for vehicle in plan["vehicle_ids"]}
    if plan.get("cluster_ids") != expected_clusters:
        raise ValueError("plan physical cluster map mismatch")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells or len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("plan cells must be nonempty and unique")
    for cell in cells:
        if canonical_hash(cell["config"])[:16] != cell["cell_id"]:
            raise ValueError("cell configuration hash mismatch")
        if any(key not in cell["config"] for key in FACTORS):
            raise ValueError("cell missing factor")
    expected_rows = len(cells)*math.prod(len(plan[name]) for name in ("methods", "seeds", "episodes", "vehicle_ids"))
    if plan.get("expected_rows") != expected_rows:
        raise ValueError("plan row budget mismatch")
    source = plan.get("source_hashes", {})
    files = source.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("plan missing source file hashes")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()):
        raise ValueError("invalid source file hash")
    if verify_sources:
        for relative, expected in files.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("source path must be repository relative")
            if file_hash(root/relative) != expected:
                raise ValueError(f"source file SHA-256 mismatch: {relative}")
        if source != current_source_identity(protocol, dataset, root=root):
            raise ValueError("source identity omits or changes registered source/code files")
        # The runner owns grid construction; rebuild from the protocol to reject
        # an internally-consistent but smaller/unregistered launch plan.
        from experiments.run_related_work_v3 import build_plan
        expected = build_plan(protocol, dataset=dataset, mode=str(plan["mode"]), root=root)
        if plan != expected:
            raise ValueError("plan differs from registered protocol scope/current sources")


def _validate_cell(record: Mapping[str, Any], cell: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    cid, config = cell["cell_id"], cell["config"]
    rows = record.get("rows")
    if not isinstance(rows, list) or record.get("row_count") != len(rows):
        raise ValueError(f"cell row_count mismatch: {cid}")
    if record.get("cell_config") != config or record.get("subset") != cell["subset"]:
        raise ValueError(f"checkpoint cell config/subset mismatch: {cid}")
    expected = {(cid, seed, episode, vehicle, method) for method, seed, vehicle, episode in itertools.product(
        plan["methods"], plan["seeds"], plan["vehicle_ids"], plan["episodes"])}
    found = {}
    for row in rows:
        required = {"cell_id", "baseline", "seed", "episode", "vehicle_id", "fresh_success", "config_hash",
                    "legacy_cell_id", "dataset", "cluster_id", "random_model", "t_decode", "storage",
                    "trace_windows", "effective_k", *FACTORS, *NUMERIC_METRICS}
        missing = sorted(required-set(row))
        if missing:
            raise ValueError(f"row missing required metrics: {cid}: {missing}")
        key = row_key(row)
        if key in found:
            raise ValueError(f"duplicate row key: {key}")
        if key not in expected:
            raise ValueError(f"row outside registered scope: {key}")
        if any(not isinstance(row[name], int) or isinstance(row[name], bool) for name in ("seed", "episode")):
            raise ValueError(f"invalid seed/episode type: {key}")
        if row["random_model"] != "keyed_v3" or row["dataset"] != plan["dataset"]:
            raise ValueError(f"mixed random model/dataset: {key}")
        if row.get("placement_model", "legacy") != config.get("placement_model", "legacy"):
            raise ValueError(f"mixed placement model: {key}")
        if row["cluster_id"] != plan["cluster_ids"][row["vehicle_id"]]:
            raise ValueError(f"row physical cluster mismatch: {key}")
        if row["config_hash"] != cid or row["legacy_cell_id"] != cell["legacy_cell_id"]:
            raise ValueError(f"row config/legacy cell hash mismatch: {key}")
        if row.get("subset", cell["subset"]) != cell["subset"]:
            raise ValueError(f"row subset mismatch: {key}")
        for name in FACTORS:
            if row[name] != config[name]:
                raise ValueError(f"row factor {name} mismatch: {key}")
        if not isinstance(row["fresh_success"], bool):
            raise ValueError(f"fresh_success is not boolean: {key}")
        for name in (*NUMERIC_METRICS, "storage", "trace_windows"):
            if not _finite_nonnegative(row[name]):
                raise ValueError(f"invalid metric {name}: {key}")
        for name in ("rank", "rank_deficit_at_expiry", "transmissions", "duplicate_dependent", "stale_on_arrival", "trace_windows", "effective_k"):
            if not isinstance(row[name], int) or isinstance(row[name], bool):
                raise ValueError(f"metric {name} must be integral: {key}")
        if row["t_decode"] is not None and not _finite_nonnegative(row["t_decode"]):
            raise ValueError(f"invalid decode time: {key}")
        if row["storage"] > config["placement_capacity_bytes"]:
            raise ValueError(f"storage budget exceeded: {key}")
        effective_k = 1 if row["baseline"].startswith(("B1_", "B5_")) else config["k"]
        if row["effective_k"] != effective_k or not 0 <= row["rank"] <= effective_k or row["rank_deficit_at_expiry"] != max(0, effective_k-row["rank"]):
            raise ValueError(f"rank/deficit inconsistency: {key}")
        if any(row[name] > row["transmissions"] for name in ("duplicate_dependent", "stale_on_arrival")):
            raise ValueError(f"transmission-count inconsistency: {key}")
        if row["fresh_success"] and (row["rank"] < effective_k or row["t_decode"] is None):
            raise ValueError(f"fresh-success/decode inconsistency: {key}")
        found[key] = row
    if set(found) != expected:
        raise ValueError(f"incomplete row Cartesian product in cell {cid}: {len(found)}/{len(expected)}")
    return found


def validate(inputs: Sequence[Path], *, protocol: Mapping[str, Any], plan: Mapping[str, Any] | None = None,
             mode: str | None = None, dataset: str | None = None, root: Path = ROOT,
             verify_sources: bool = True, require_complete: bool = True) -> dict[str, Any]:
    if protocol.get("schema") != "tacp-related-work-protocol-v3" or protocol.get("frozen") is not True:
        raise ValueError("unsupported or unfrozen v3 protocol")
    paths, manifests = checkpoint_paths(inputs)
    if plan is None:
        if not manifests or not isinstance(manifests[0].get("plan"), dict):
            raise ValueError("JSONL-only validation requires an explicit registered plan")
        plan = manifests[0]["plan"]
    _validate_plan(plan, protocol, root=root, verify_sources=verify_sources)
    mode, dataset = mode or str(plan["mode"]), dataset or str(plan["dataset"])
    if (mode, dataset) != (plan["mode"], plan["dataset"]):
        raise ValueError("mode/dataset differs from registered plan")
    for manifest in manifests:
        if manifest.get("plan") != plan:
            raise ValueError("manifests mix plans or code/source versions")
        if manifest.get("run_hash", run_identity(plan)) != run_identity(plan):
            raise ValueError("manifest run hash mismatch")
    selected = {cell["cell_id"]: cell for cell in plan["cells"]}
    aggregate = {"pilot": f"PILOT_all_{len(selected)}", "full": f"FULL_all_{len(selected)}",
                 "dry-run": "DRY_RUN", "replay": "REPLAY", "validation": "VALIDATION"}.get(mode, mode.upper())
    totals: dict[tuple[str, str], Accumulator] = {}
    paired: dict[tuple[str, str, str], Accumulator] = {}
    seen = set()
    actual_rows = equivalence_rows = 0
    for path, line_number, record in iter_records(paths):
        if record.get("schema") != "tacp-related-work-v3":
            raise ValueError("unsupported checkpoint schema; old rows cannot enter v3 comparisons")
        if record.get("record_hash") != canonical_hash({key: value for key, value in record.items() if key != "record_hash"}):
            raise ValueError(f"checkpoint record hash mismatch: {path}:{line_number}")
        for name, expected in (("mode", mode), ("dataset", dataset), ("protocol_hash", plan["protocol_hash"]),
                               ("plan_hash", plan["plan_hash"]), ("run_hash", run_identity(plan)),
                               ("source_hashes", plan["source_hashes"])):
            if record.get(name) != expected:
                raise ValueError(f"checkpoint {name} mismatch: {path}:{line_number}")
        cid = str(record.get("cell_id", ""))
        if cid not in selected:
            raise ValueError(f"unregistered {mode} cell: {cid}")
        if cid in seen:
            raise ValueError(f"duplicate cell across inputs: {cid}")
        cell = selected[cid]
        rows = _validate_cell(record, cell, plan)
        groups = [aggregate] + ([cell["subset"]] if cell["subset"] not in ("outside_prespecified_subsets", aggregate, None, "") else [])
        for key, row in rows.items():
            if row["baseline"] == A2 and LEGACY_B7 in plan["methods"]:
                reference = rows[key[:-1]+(LEGACY_B7,)]
                for name in EQUIVALENCE_FIELDS:
                    if row[name] != reference[name]:
                        raise ValueError(f"A2/legacy B7 mismatch: {key} field {name}")
                equivalence_rows += 1
            for group in groups:
                totals.setdefault((group, row["baseline"]), Accumulator()).add(row)
            for anchor in (LEGACY_B7, RUNTIME_B7, B5):
                anchor_row = rows.get(key[:-1]+(anchor,))
                if anchor_row is None or anchor == row["baseline"]:
                    continue
                delta = float(anchor_row["fresh_success"])-float(row["fresh_success"])
                for group in groups:
                    paired.setdefault((group, anchor, row["baseline"]), Accumulator()).add(row, delta, metrics=False)
        seen.add(cid)
        actual_rows += len(rows)
    if not seen:
        raise ValueError("no completed records")
    missing = sorted(set(selected)-seen)
    if require_complete and missing:
        raise ValueError(f"incomplete {mode}: missing {len(missing)} cells")
    method_results = {f"{group}/{method}": acc.result(protocol["bootstrap"]) for (group, method), acc in sorted(totals.items())}
    paired_results = {f"{group}/{anchor} minus {method}": acc.result(protocol["bootstrap"], metric="delta_P_fresh")
                      for (group, anchor, method), acc in sorted(paired.items())}
    stability_failures = []
    stability_required = dataset == "porto" and mode in ("pilot", "full")
    if stability_required:
        for name, item in method_results.items():
            if item["ci_width"] > protocol["stability"]["max_ci_width"]:
                stability_failures.append(f"{name}: CI width {item['ci_width']:.6f}")
            if item["seed_range"] > protocol["stability"]["max_seed_range"]:
                stability_failures.append(f"{name}: seed range {item['seed_range']:.6f}")
    complete = not missing and actual_rows == plan["expected_rows"]
    technical = complete and verify_sources
    passed = technical and not stability_failures
    unit = str(plan["cluster_unit"])
    explanation = ("All comparisons use only v3 rows matched on trip/vehicle, cell, seed and episode. "
                   f"Bootstrap resamples whole {unit} clusters with all their trips, cells and fixed seeds retained; "
                   "the ratio of sums preserves the trip-weighted estimator. Intervals are conditional on the registered "
                   "seeds, parameter grid and observed cluster sample; they do not estimate variability over new seeds. "
                   "Positive paired delta favors the named anchor. Effect direction is never a launch gate.")
    if dataset == "highd":
        explanation += " Small-sample highD recording-cluster intervals are descriptive diagnostics, not a population precision claim."
    return {"schema": "tacp-related-work-validation-v3", "mode": mode, "dataset": dataset,
            "protocol_hash": plan["protocol_hash"], "plan_hash": plan["plan_hash"], "run_hash": run_identity(plan),
            "source_hashes": plan["source_hashes"], "input_checkpoints": [str(path) for path in paths],
            "input_sha256": {str(path): file_hash(path) for path in paths},
            "complete": complete, "registered_scale": verify_sources, "passed": passed,
            "gate": {"passed": passed, "technical_passed": technical, "stability_passed": not stability_failures,
                     "stability_required": stability_required},
            "gates": {"completeness": complete, "registered_scale": verify_sources,
                      "source_hashes_verified": verify_sources, "record_hashes_verified": True,
                      "A2_legacy_equivalence_checked_rows": equivalence_rows,
                      "old_reference_rows_included": 0},
            "methods": list(plan["methods"]), "seeds": list(plan["seeds"]), "trip_count": len(plan["vehicle_ids"]),
            "cluster_unit": unit, "cluster_count": len(set(plan["cluster_ids"].values())),
            "expected_cells": len(selected), "completed_cells": len(seen), "missing_cells": missing,
            "expected_rows": plan["expected_rows"], "actual_rows": actual_rows,
            "bootstrap": {**dict(protocol["bootstrap"]), "cluster": unit, "estimator": "trip_weighted_ratio_of_sums"},
            "stability_failures": stability_failures, "method_summaries": method_results,
            "paired_summaries": paired_results, "interpretation": explanation}


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Related-work v3 validation", "", f"Dataset: `{report['dataset']}`; mode: `{report['mode']}`; passed: **{report['passed']}**.",
             f"Cells: {report['completed_cells']}/{report['expected_cells']}; rows: {report['actual_rows']:,}/{report['expected_rows']:,}.",
             f"Trips/vehicles: {report['trip_count']}; resampling units: {report['cluster_count']} {report['cluster_unit']} clusters.",
             f"Protocol SHA-256: `{report['protocol_hash']}`", f"Plan SHA-256: `{report['plan_hash']}`", "", report["interpretation"], "",
             "| Subset/method | P_fresh | Cluster-bootstrap 95% CI | CI width | Seed range |",
             "|---|---:|---|---:|---:|"]
    for name, value in report["method_summaries"].items():
        lo, hi = value["cluster_bootstrap_95_ci"]
        lines.append(f"| {name} | {value['P_fresh']:.6f} | [{lo:.6f}, {hi:.6f}] | {value['ci_width']:.6f} | {value['seed_range']:.6f} |")
    lines += ["", "| Matched comparison | Delta P_fresh | Paired cluster-bootstrap 95% CI | CI width |", "|---|---:|---|---:|"]
    for name, value in report["paired_summaries"].items():
        lo, hi = value["cluster_bootstrap_95_ci"]
        lines.append(f"| {name} | {value['delta_P_fresh']:.6f} | [{lo:.6f}, {hi:.6f}] | {value['ci_width']:.6f} |")
    if report["stability_failures"]:
        lines += ["", "Stability gate failures (retain all registered cases):", ""]
        lines += [f"- {failure}" for failure in report["stability_failures"]]
    return "\n".join(lines)+"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/related_work_v3_2.yaml")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--dataset", choices=("porto", "highd"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = validate(args.inputs, protocol=load_protocol(args.protocol), plan=read_json(args.plan) if args.plan else None,
                      mode=args.mode, dataset=args.dataset, require_complete=not args.allow_incomplete)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    md = args.output_md or args.output_json.with_suffix(".md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "dataset", "passed", "complete", "actual_rows", "completed_cells", "cluster_count", "stability_failures")}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
