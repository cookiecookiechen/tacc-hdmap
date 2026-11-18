#!/usr/bin/env python3
"""Freeze a bounded, balanced Porto sample without reading simulation outcomes.

Every source row is checked for duplicate numeric trip IDs. The original rule
stops on a duplicate; an explicit pre-simulation quality amendment may instead
exclude every occurrence of each duplicated ID after a full-source pre-scan.
At most two eligible
rows per taxi are retained. Coverage need not be recomputed for a later trip
that cannot replace an already retained pair; the audit counts distinguish this
optimization from a census of all coverage-eligible trips. The existing Porto
converter produces the final six-column CSV, preserving its geometric and time
semantics. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_porto_trace as legacy

RULE_SCHEMA = "tacp_porto_independent_selection_rules_v1"
SALT = "tacp-porto-independent-taxi-v3.2:"


def _path(value, root):
    value = Path(value)
    return (value if value.is_absolute() else root / value).resolve()


def _label(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _numeric_id(value, field, line):
    value = str(value).strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"invalid {field} at source row {line}")
    return int(value)


def _load_rules(path):
    rules = json.loads(path.read_text(encoding="utf-8"))
    if rules.get("schema") != RULE_SCHEMA or rules.get("frozen") is not True:
        raise ValueError("requires a frozen independent-taxi selection rule")
    fixed = {
        "trips_per_taxi": 2, "max_expansion_rounds": 1,
        "trip_order": "ascending_numeric_TRIP_ID",
        "taxi_order": "ascending_sha256_utf8_salt_plus_canonical_TAXI_ID_then_numeric_TAXI_ID",
        "taxi_hash_salt": SALT, "min_points": 8, "min_covered_points": 4,
        "sample_interval_s": 15.0, "expected_rsu_count": 20,
        "rsu_assignment_rule": "nearest_rsu_within_radius; ties by distance then rsu_id",
        "missing_data_policy": "retain rows whose POLYLINE meets min_points; record MISSING_DATA flag",
        "time_window": None,
        "seeds_unchanged": [101, 137, 173, 211, 257],
        "bootstrap_unchanged": {"samples": 2000, "seed": 20261018, "confidence": 0.95},
        "stability_unchanged": {"max_ci_width": 0.1, "max_seed_range": 0.2},
    }
    for key, expected in fixed.items():
        if rules.get(key) != expected:
            raise ValueError(f"unsupported selection rule: {key}")
    duplicate_policy = rules.get("duplicate_trip_policy", "error")
    if duplicate_policy not in {"error", "exclude_all_occurrences"}:
        raise ValueError("unsupported duplicate_trip_policy")
    if duplicate_policy == "exclude_all_occurrences":
        amendment = rules.get("data_quality_amendment", {})
        required = {"number": 1, "max_amendments": 1,
                    "timing": "before_any_new_simulation_results",
                    "preserve_prior_freeze_and_failure": True,
                    "no_further_rule_revisions": True}
        if any(amendment.get(key) != value for key, value in required.items()):
            raise ValueError("duplicate exclusion requires the single pre-simulation amendment")
    target = rules.get("target_taxis")
    if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 256:
        raise ValueError("target_taxis must be a positive integer no larger than 256")
    for key in ("source_sha256", "rsu_config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(rules.get(key, ""))):
            raise ValueError(f"invalid frozen {key}")
    budget = rules.get("budget", {})
    if (budget.get("trips") != target * 2
            or budget.get("pilot") != {"cells": 54, "seeds": 5, "methods": 8, "rows": 54*target*2*5*8}
            or budget.get("conditional_full") != {"cells": 486, "seeds": 5, "methods": 6, "rows": 486*target*2*5*6}):
        raise ValueError("frozen budget does not match selected taxi/trip counts")
    return rules


def _duplicate_trip_ids(source):
    """Count numeric IDs across the whole source, without materialising rows."""
    trip_counts = Counter()
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or legacy.RAW_REQUIRED - set(fields):
            raise ValueError("source missing required UCI columns")
        if len(fields) != len(set(fields)):
            raise ValueError("source has duplicate column names")
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(f"malformed CSV source row {line}")
            trip = _numeric_id(row["TRIP_ID"], "TRIP_ID", line)
            _numeric_id(row["TAXI_ID"], "TAXI_ID", line)
            trip_counts[trip] += 1
    return {trip: count for trip, count in trip_counts.items() if count > 1}


def _select(source, rules, rsus):
    counts = Counter()
    retained = {}
    seen_trips = set()
    seen_taxis = set()
    duplicate_policy = rules.get("duplicate_trip_policy", "error")
    duplicate_ids = _duplicate_trip_ids(source) if duplicate_policy == "exclude_all_occurrences" else {}
    affected_taxis = set()
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or legacy.RAW_REQUIRED - set(fields):
            raise ValueError("source missing required UCI columns")
        if len(fields) != len(set(fields)):
            raise ValueError("source has duplicate column names")
        for line, row in enumerate(reader, 2):
            counts["source_rows"] += 1
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(f"malformed CSV source row {line}")
            trip = _numeric_id(row["TRIP_ID"], "TRIP_ID", line)
            taxi = _numeric_id(row["TAXI_ID"], "TAXI_ID", line)
            if trip in seen_trips and duplicate_policy == "error":
                raise ValueError(f"duplicate numeric TRIP_ID {trip} at source row {line}")
            seen_trips.add(trip)
            seen_taxis.add(taxi)
            try:
                int(row["TIMESTAMP"])
                points = legacy._parse_polyline(row["POLYLINE"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid UCI source row {line}") from exc
            if str(row.get("MISSING_DATA", "")).strip().lower() in {"true", "1", "yes"}:
                counts["source_missing_data_rows"] += 1
            # Validate every row before exclusion, including later occurrences.
            # Never prefer a first/last record or one with better coverage.
            if trip in duplicate_ids:
                counts["duplicate_rows_excluded"] += 1
                affected_taxis.add(taxi)
                continue
            if len(points) < rules["min_points"]:
                counts["below_min_points_rows"] += 1
                continue
            counts["min_points_eligible_rows"] += 1
            candidates = retained.setdefault(taxi, [])
            if len(candidates) == 2 and trip > candidates[-1][0]:
                counts["coverage_not_needed_later_trip_rows"] += 1
                continue
            counts["coverage_evaluated_rows"] += 1
            covered = sum(legacy._nearest_rsu(lon, lat, rsus) is not None for lon, lat in points)
            if covered < rules["min_covered_points"]:
                counts["below_min_covered_evaluated_rows"] += 1
                continue
            counts["coverage_eligible_evaluated_rows"] += 1
            normalized = dict(row, TRIP_ID=str(trip), TAXI_ID=str(taxi))
            candidates.append((trip, normalized))
            candidates.sort(key=lambda item: item[0])
            del candidates[2:]
    eligible = [taxi for taxi, items in retained.items() if len(items) == 2]
    counts["source_unique_trips"] = len(seen_trips)
    counts["source_unique_taxis"] = len(seen_taxis)
    counts["taxis_with_at_least_one_eligible_trip"] = sum(bool(v) for v in retained.values())
    counts["taxis_with_at_least_two_eligible_trips"] = len(eligible)
    counts["duplicate_trip_ids_excluded"] = len(duplicate_ids)
    counts["duplicate_rows_excluded"] = counts.get("duplicate_rows_excluded", 0)
    counts["duplicate_affected_taxis"] = len(affected_taxis)
    if counts["duplicate_rows_excluded"] != sum(duplicate_ids.values()):
        raise ValueError("source duplicate counts changed between scans")
    # Set zero-valued counters explicitly, making fixtures and runs comparable.
    for key in ("source_missing_data_rows", "below_min_points_rows", "min_points_eligible_rows",
                "coverage_not_needed_later_trip_rows", "coverage_evaluated_rows",
                "below_min_covered_evaluated_rows", "coverage_eligible_evaluated_rows"):
        counts.setdefault(key, 0)
    if len(eligible) < rules["target_taxis"]:
        raise ValueError(f"only {len(eligible)} eligible taxis; frozen target is {rules['target_taxis']}")
    def ranking(taxi):
        return hashlib.sha256((rules["taxi_hash_salt"]+str(taxi)).encode("utf-8")).hexdigest(), taxi
    chosen = sorted(eligible, key=ranking)[:rules["target_taxis"]]
    selections = [{"taxi_id": str(taxi), "selection_sha256": ranking(taxi)[0],
                   "trip_ids": [str(trip) for trip, _ in retained[taxi]],
                   "vehicle_ids": [f"taxi_{taxi}_trip_{trip}" for trip, _ in retained[taxi]]}
                  for taxi in chosen]
    rows = [row for taxi in chosen for _, row in retained[taxi]]
    rows.sort(key=lambda row: int(row["TRIP_ID"]))
    exclusion = {"policy": duplicate_policy,
                 "excluded_trip_ids": [str(trip) for trip in sorted(duplicate_ids)],
                 "excluded_trip_id_count": len(duplicate_ids),
                 "excluded_record_count": counts["duplicate_rows_excluded"],
                 "affected_taxi_count": len(affected_taxis),
                 "affected_taxi_ids": [str(taxi) for taxi in sorted(affected_taxis)]}
    return fields, rows, selections, dict(counts), exclusion


def _publish_new(files):
    """Publish with exclusive creates; roll back only files created by this call."""
    created = []
    try:
        for source, target in files:
            with target.open("xb") as out:
                created.append(target)
                with source.open("rb") as inp:
                    shutil.copyfileobj(inp, out)
                out.flush()
                os.fsync(out.fileno())
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def prepare_independent_porto(source, rules_path, output, provenance, selection_manifest,
                             *, old_provenance=None, root=ROOT):
    root = Path(root).resolve()
    source, rules_path, output, provenance, selection_manifest = [
        _path(p, root) for p in (source, rules_path, output, provenance, selection_manifest)]
    # Read and validate the public frozen rules before opening the source.
    rules = _load_rules(rules_path)
    rsu_path = _path(rules["rsu_config_path"], root)
    old_path = _path(old_provenance, root) if old_provenance is not None else None
    targets = [output, provenance, selection_manifest]
    if len(set(targets)) != len(targets):
        raise ValueError("output, provenance and selection manifest must be distinct")
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite {target}")
    tracked = [source, rules_path, rsu_path, Path(__file__).resolve(), Path(legacy.__file__).resolve()]
    if old_path is not None:
        tracked.append(old_path)
    fingerprints = {path: _fingerprint(path) for path in tracked}
    hashes = {_label(path, root): legacy.sha256_file(path) for path in tracked}
    if hashes[_label(source, root)] != rules["source_sha256"]:
        raise ValueError("source SHA-256 differs from frozen selection rules")
    if hashes[_label(rsu_path, root)] != rules["rsu_config_sha256"]:
        raise ValueError("RSU SHA-256 differs from frozen selection rules")
    rsus = legacy.load_rsus(rsu_path)
    if len(rsus) != rules["expected_rsu_count"]:
        raise ValueError("frozen RSU count changed")
    fields, selected_rows, chosen, counts, exclusion = _select(source, rules, rsus)
    overlap = None
    selected_taxis = {item["taxi_id"] for item in chosen}
    selected_trips = {row["TRIP_ID"] for row in selected_rows}
    if old_path is not None:
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if old.get("schema") != "tacp_real_trace_provenance_v1" or not isinstance(old.get("trips"), list):
            raise ValueError("invalid old Porto provenance for overlap audit")
        old_taxis = {str(t["taxi_id"]) for t in old["trips"]}
        old_trips = {str(t["trip_id"]) for t in old["trips"]}
        overlap = {"old_taxis": len(old_taxis), "old_trips": len(old_trips),
                   "retained_old_taxis": len(selected_taxis & old_taxis),
                   "retained_old_trips": len(selected_trips & old_trips),
                   "shared_taxi_ids": sorted(selected_taxis & old_taxis, key=int),
                   "shared_trip_ids": sorted(selected_trips & old_trips, key=int)}
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="porto-independent-", dir=output.parent) as temporary:
        folder = Path(temporary)
        subset = folder / "selected_source.csv"
        with subset.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected_rows)
        csv_temp, provenance_temp, manifest_temp = [folder/name for name in ("trace.csv", "provenance.json", "selection.json")]
        report = legacy.prepare_porto_csv(subset, csv_temp, provenance_temp, rsus,
            sample_interval_s=rules["sample_interval_s"], min_points=rules["min_points"],
            min_covered_points=rules["min_covered_points"], rsu_config_path=rsu_path)
        expected = rules["target_taxis"] * rules["trips_per_taxi"]
        if report["counts"]["trips_written"] != expected or report["counts"]["trips_discarded"] != 0:
            raise ValueError("legacy converter disagrees with frozen selected trip eligibility")
        for path, before in fingerprints.items():
            if _fingerprint(path) != before:
                raise ValueError(f"source or frozen dependency changed while selecting: {path}")
        counts.update(selected_taxis=len(chosen), selected_trips=expected,
                      selected_output_rows=report["counts"]["rows_written"])
        selection = {
            "schema": "tacp_porto_independent_selection_manifest_v1",
            "revision": rules["revision"], "rules": rules,
            "rules_path": _label(rules_path, root), "source_hashes": hashes,
            "coverage_count_scope": "coverage counts refer only to evaluated candidates after the duplicate policy; exact >=2-eligible-taxi census and exact earliest pair selection",
            "counts": counts, "taxi_ids": [item["taxi_id"] for item in chosen],
            "vehicle_ids": [trip["vehicle_id"] for trip in report["trips"]],
            "selected_trip_ids": [row["TRIP_ID"] for row in selected_rows],
            "selected_taxis": chosen, "old_sample_overlap": overlap,
            "duplicate_exclusion": exclusion,
            "output": {"path": _label(output, root), "sha256": legacy.sha256_file(csv_temp)},
        }
        manifest_temp.write_text(json.dumps(selection, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        report["input"] = {"path": _label(source, root), "sha256": rules["source_sha256"]}
        report["output"] = selection["output"]
        report["rsu_config"] = {"path": _label(rsu_path, root), "sha256": rules["rsu_config_sha256"]}
        report["source_hashes"] = hashes
        report["selection_manifest"] = {"path": _label(selection_manifest, root), "sha256": legacy.sha256_file(manifest_temp)}
        report["selection_rules"] = {"path": _label(rules_path, root), "sha256": hashes[_label(rules_path, root)]}
        report["selection_counts"] = counts
        report["old_sample_overlap"] = overlap
        report["duplicate_exclusion"] = exclusion
        report["counts"].update(source_rows_seen=counts["source_rows"], rows_skipped=counts["below_min_points_rows"])
        report["counts"]["rows_uncovered_scope"] = "selected 512 trips only" if expected == 512 else "selected trips only"
        report["preprocessing"].update(
            trip_selection="first_two_eligible_numeric_trip_ids_per_taxi_then_sha256_taxi_order",
            max_trips=expected, target_taxis=rules["target_taxis"], trips_per_taxi=2,
            selection_target="exact balanced taxi sample after the unchanged point and coverage filters",
            time_window=None, taxi_hash_salt=rules["taxi_hash_salt"],
            duplicate_trip_policy=exclusion["policy"])
        provenance_temp.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        _publish_new([(csv_temp, output), (manifest_temp, selection_manifest), (provenance_temp, provenance)])
    return {"provenance": report, "selection_manifest": selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--old-provenance", type=Path)
    args = parser.parse_args()
    result = prepare_independent_porto(args.source, args.rules, args.output, args.provenance,
        args.selection_manifest, old_provenance=args.old_provenance)
    print(json.dumps({"output": str(args.output), "provenance": str(args.provenance),
                      "selection_manifest": str(args.selection_manifest),
                      "counts": result["selection_manifest"]["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
