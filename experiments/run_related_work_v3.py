"""Versioned paired comparisons: Porto taxi clusters and a small highD supplement.

Old v2 outputs remain immutable. Every method in a v3 comparison is rerun with
the same keyed random-event model; legacy outcome rows are never pooled in.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tacp_sim.baselines import build_baseline
from tacp_sim.simulator import Simulation
from experiments.run_phase4_ci import _load_windows, _object_from_config
from scripts.validate_related_work_v3 import (
    canonical_hash, file_hash, cluster_id_for, current_source_identity,
    load_protocol, run_identity, _validate_cell,
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolved(config, root):
    cfg = dict(config)
    for key in ("trace_path", "trace_provenance_path"):
        if key in cfg:
            cfg[key] = str(root / cfg[key])
    return cfg


def _porto_trace_overrides(data):
    """Opt in to a newly frozen sample without changing the legacy grid.

    This is deliberately not a general config-override facility: only the two
    trace input paths and the registered sample size may change. Existing
    protocols keep their original cells.
    """
    unknown = sorted(key for key in data if "override" in key and key != "trace_override")
    if unknown:
        raise ValueError(f"unsupported Porto override field: {unknown}")
    enabled = data.get("trace_override", False)
    if not isinstance(enabled, bool):
        raise ValueError("trace_override must be a boolean, not a config mapping")
    if not enabled:
        return {}
    overrides = {}
    for source, destination in (("trace_path", "trace_path"),
                                ("provenance_path", "trace_provenance_path")):
        value = data.get(source)
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"{source} override must be repository relative")
        if value not in data.get("source_hashes", {}):
            raise ValueError(f"{source} override is not a registered source")
        overrides[destination] = value
    count = data.get("expected_vehicle_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("expected_vehicle_count must be a positive integer")
    overrides["trace_vehicle_limit"] = count
    return overrides


def load_context(protocol, dataset, *, root=ROOT):
    data = protocol["datasets"][dataset]
    if dataset == "porto":
        overrides = _porto_trace_overrides(data)
        frozen = json.loads((root / data["legacy_plan_path"]).read_text())
        first = frozen["cells"][0]
        if canonical_hash(first["config"])[:16] != first["cell_id"]:
            raise ValueError("corrupt legacy parameter cell")
        cfg = _resolved({**first["config"], **overrides}, root)
        windows, _ = _load_windows(cfg, protocol["seeds"][0])
        vehicles = sorted({str(w.vehicle_id) for w in windows})
        universe = sorted(str(r["rsu_id"]) for r in json.loads((root / data["rsu_map_path"]).read_text())["rsus"])
        by_vehicle = {v: [w for w in windows if str(w.vehicle_id) == v] for v in vehicles}
        universes = {v: universe for v in vehicles}
    elif dataset == "highd":
        from tacp_sim.highd import load_contact_windows_csv
        cfg = json.loads((root / data["config_path"]).read_text())
        folder = root / cfg["input_dir"]
        by_vehicle, universes, windows = {}, {}, []
        for recording in cfg["recordings"]:
            rsus = sorted(json.loads((folder / f"{recording}_rsus.json").read_text()))
            selected = json.loads((folder / f"{recording}_selection.json").read_text())["selected"]
            recording_windows = load_contact_windows_csv(folder / f"{recording}_contacts.csv")
            windows.extend(recording_windows)
            for item in selected:
                vid = item["vehicle_id"]
                by_vehicle[vid] = [w for w in recording_windows if w.vehicle_id == vid]
                universes[vid] = rsus
        vehicles = sorted(by_vehicle)
    else:
        raise ValueError("unsupported dataset")
    if len(vehicles) != data["expected_vehicle_count"]:
        raise ValueError("registered vehicle/trip count changed")
    clusters = {v: cluster_id_for(v, dataset) for v in vehicles}
    if len(set(clusters.values())) != data["expected_cluster_count"]:
        raise ValueError("registered physical cluster count changed")
    if dataset == "porto" and data.get("trips_per_cluster") is not None:
        count = data["trips_per_cluster"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("trips_per_cluster must be a positive integer")
        if any(actual != count for actual in Counter(clusters.values()).values()):
            raise ValueError("registered trips per physical taxi changed")
    if dataset == "porto" and overrides:
        provenance = json.loads((root / data["provenance_path"]).read_text())
        trips = provenance.get("trips")
        selected = [trip.get("vehicle_id") for trip in trips if isinstance(trip, dict)] if isinstance(trips, list) else []
        if (len(selected) != len(vehicles) or not all(isinstance(vehicle, str) for vehicle in selected)
                or set(selected) != set(vehicles)):
            raise ValueError("registered selected trip IDs differ from loaded trace")
        manifest_path = data.get("selection_manifest_path")
        if manifest_path is not None:
            if (not isinstance(manifest_path, str) or not manifest_path or Path(manifest_path).is_absolute()
                    or ".." in Path(manifest_path).parts or manifest_path not in data.get("source_hashes", {})):
                raise ValueError("selection manifest must be a registered repository-relative source")
            manifest = json.loads((root / manifest_path).read_text())
            expected_ids = {
                "vehicle_ids": set(vehicles),
                "taxi_ids": {cluster.removeprefix("taxi_") for cluster in clusters.values()},
            }
            for field, expected in expected_ids.items():
                selected_ids = manifest.get(field) if isinstance(manifest, dict) else None
                if (not isinstance(selected_ids, list) or len(selected_ids) != len(expected)
                        or not all(isinstance(value, str) for value in selected_ids)
                        or set(selected_ids) != expected):
                    raise ValueError("selection manifest IDs differ from loaded trace")
    return vehicles, by_vehicle, windows, universes, clusters


def build_plan(protocol, *, dataset, mode, root=ROOT):
    if protocol.get("schema") != "tacp-related-work-protocol-v3" or protocol.get("frozen") is not True:
        raise ValueError("requires frozen v3 protocol")
    placement_model = protocol.get("placement_model", "legacy")
    if placement_model not in ("legacy", "deadline_capacity_v3_1"):
        raise ValueError("unsupported placement_model")
    if dataset == "highd" and mode != "small":
        raise ValueError("highD is a separately registered small validation only")
    if dataset == "porto" and mode not in ("dry-run", "pilot", "full"):
        raise ValueError("invalid Porto stage")
    source = current_source_identity(protocol, dataset, root=root)
    data = protocol["datasets"][dataset]
    vehicles, by_vehicle, _, _, cluster_ids = load_context(protocol, dataset, root=root)
    cells = []
    if dataset == "porto":
        overrides = _porto_trace_overrides(data)
        frozen = json.loads((root / data["legacy_plan_path"]).read_text())
        if frozen.get("plan_hash", frozen.get("full_plan_hash")) != data["legacy_full_plan_hash"]:
            raise ValueError("legacy grid identity changed")
        groups = {name: [] for name in protocol["subsets"]}
        full = []
        for original in sorted(frozen["cells"], key=lambda c: c["cell_id"]):
            config = dict(original["config"])
            if canonical_hash(config)[:16] != original["cell_id"]:
                raise ValueError("corrupt legacy parameter cell")
            names = [name for name, match in protocol["subsets"].items()
                     if all(float(config[k]) == float(v) for k, v in match.items())]
            if len(names) > 1:
                raise ValueError("overlapping subsets")
            subset = names[0] if names else "outside_prespecified_subsets"
            config.update(overrides)
            config["random_model"] = "keyed_v3"
            if placement_model != "legacy":
                config["placement_model"] = placement_model
            cell = {"cell_id": canonical_hash(config)[:16], "legacy_cell_id": original["cell_id"],
                    "config": config, "subset": subset}
            full.append(cell)
            if names:
                groups[subset].append(cell)
        if len(full) != 486 or any(len(group) != 18 for group in groups.values()):
            raise ValueError("registered 486/54 grid changed")
        if mode == "full":
            cells = full
        elif mode == "pilot":
            cells = [c for group in groups.values() for c in group]
        else:
            cells = [group[0] for group in groups.values()]
            selected = sorted(vehicles, key=lambda v: (-len(by_vehicle[v]), v))[:2]
            if selected != data["dry_run_vehicle_ids"]:
                raise ValueError("registered dry-run sample changed")
            vehicles = selected
    else:
        spec = json.loads((root / data["config_path"]).read_text())
        if spec["methods"] != protocol["methods"][mode] or spec["seeds"] != protocol["seeds"]:
            raise ValueError("highD method/seed protocol mismatch")
        for ttl in spec["ttl_s"]:
            for loss in spec["loss_probability"]:
                config = {"trace_kind": "explicit_contact_windows", "object_id": "highd-v3-small",
                          "created_at": 0.0, "version": 1, "verification": True,
                          "object_bytes": spec["object_bytes"], "n": spec["n"], "k": spec["k"],
                          "placement_capacity_bytes": spec["storage_capacity_bytes"],
                          "ttl_s": ttl, "packet_loss": loss, "link_rate_bps": spec["rate_bps"],
                          "rsu_failure_probability": spec["failure_probability"], "random_model": "keyed_v3"}
                if placement_model != "legacy":
                    config["placement_model"] = placement_model
                cells.append({"cell_id": canonical_hash(config)[:16], "legacy_cell_id": None,
                              "config": config, "subset": "H_short_contact"})
    seeds = protocol["seeds"][:2] if mode == "dry-run" else protocol["seeds"]
    methods = protocol["methods"][mode]
    plan = {"schema": "tacp-related-work-plan-v3", "dataset": dataset, "mode": mode,
            "protocol_hash": canonical_hash(protocol), "source_hashes": source,
            "methods": methods, "seeds": seeds, "episodes": [0], "vehicle_ids": vehicles,
            "cluster_unit": "physical_taxi" if dataset == "porto" else "recording",
            "cluster_ids": {v: cluster_ids[v] for v in vehicles}, "cells": cells,
            "expected_rows": len(cells) * len(vehicles) * len(seeds) * len(methods)}
    budget = data["budget"][mode]
    if len(cells) != budget["cells"] or plan["expected_rows"] != budget["rows"]:
        raise ValueError("registered budget mismatch")
    plan["plan_hash"] = canonical_hash(plan)
    return plan


def check_validation_gate(path, expected_plan, *, require_stability):
    if path is None:
        raise ValueError(f"missing {expected_plan['dataset']}/{expected_plan['mode']} gate")
    gate = json.loads(Path(path).read_text())
    for key in ("dataset", "mode", "plan_hash", "protocol_hash", "source_hashes", "expected_rows"):
        if gate.get(key) != expected_plan[key]:
            raise ValueError(f"gate {key} mismatch")
    checks = gate.get("gate", {})
    if (gate.get("schema") != "tacp-related-work-validation-v3" or gate.get("complete") is not True
            or gate.get("registered_scale") is not True or checks.get("technical_passed") is not True
            or gate.get("passed") is not True or checks.get("passed") is not True
            or gate.get("actual_rows") != expected_plan["expected_rows"]
            or gate.get("completed_cells") != len(expected_plan["cells"])
            or gate.get("expected_cells") != len(expected_plan["cells"])):
        raise ValueError("gate does not certify complete registered data")
    if require_stability and (checks.get("stability_passed") is not True or gate.get("passed") is not True):
        raise ValueError("pilot statistical stability gate failed; full sweep prohibited")
    if not gate.get("input_sha256"):
        raise ValueError("gate missing validated input hashes")
    for input_path, expected in gate["input_sha256"].items():
        if file_hash(Path(input_path)) != expected:
            raise ValueError("validated input changed")


def check_launch_gates(args, protocol, plan):
    if plan["dataset"] == "porto" and plan["mode"] in ("pilot", "full"):
        check_validation_gate(args.dry_gate, build_plan(protocol, dataset="porto", mode="dry-run"), require_stability=False)
    if plan["mode"] == "full":
        check_validation_gate(args.pilot_gate, build_plan(protocol, dataset="porto", mode="pilot"), require_stability=True)
        check_validation_gate(args.highd_gate, build_plan(protocol, dataset="highd", mode="small"), require_stability=False)


def simulate_row(cell, windows, placement, seed, vehicle_id, dataset, cluster_id):
    cfg = _resolved(cell["config"], ROOT)
    cfg["vehicle_id"] = vehicle_id
    obj = _object_from_config(cfg)
    run_obj = obj if placement.coded else replace(obj, n=1, k=1)
    mapping = placement.placement if placement.coded else {rsu: [0] for rsu, ids in placement.placement.items() if ids}
    if not placement.coded:
        cfg.update(n=1, k=1)
    result = Simulation(cfg, seed).run_episode(0, windows=windows, placement_override=mapping,
        object_override=run_obj, symbol_vectors=placement.metadata.get("symbol_vectors"),
        rank_aware_requests=bool(placement.metadata.get("rank_aware_requests", False)))
    tx = result.transmissions
    duplicate = sum(t.failure_reason == "duplicate_or_dependent" for t in tx)
    stale = sum(t.failure_reason == "stale_version" or t.end_time > obj.expires_at for t in tx)
    row = {"cell_id": cell["cell_id"], "config_hash": cell["cell_id"],
           "legacy_cell_id": cell["legacy_cell_id"], "subset": cell["subset"],
           "dataset": dataset, "random_model": "keyed_v3", "cluster_id": cluster_id,
           "baseline": placement.name, "seed": seed, "episode": 0, "vehicle_id": vehicle_id,
           **{key: cell["config"][key] for key in ("object_bytes", "n", "k", "ttl_s", "packet_loss", "placement_capacity_bytes", "rsu_failure_probability")},
           "fresh_success": bool(result.fresh_success), "T_decode": result.t_decode, "t_decode": result.t_decode,
           "rank": result.rank, "effective_k": run_obj.k,
           "rank_deficit_at_expiry": max(0, run_obj.k-result.rank),
           "rank_deficit_fraction": max(0, run_obj.k-result.rank)/run_obj.k,
           "transmissions": len(tx), "tx_bytes": sum(t.bytes for t in tx),
           "duplicate_dependent": duplicate, "duplicate_dependent_ratio": duplicate/len(tx) if tx else 0.0,
           "stale_on_arrival": stale, "stale_on_arrival_ratio": stale/len(tx) if tx else 0.0,
           "storage": placement.storage_bytes, "backhaul": result.backhaul, "migration": result.migration,
           "repair": result.repair, "trace_windows": len(windows),
           "code_seed_hash": placement.metadata.get("code_seed_hash"), "placement_hash": canonical_hash(placement.placement)}
    if "placement_model" in cell["config"]:
        row["placement_model"] = cell["config"]["placement_model"]
    return row


def run(args):
    protocol = load_protocol(args.protocol)
    plan = build_plan(protocol, dataset=args.dataset, mode=args.mode)
    if args.partitions < 1 or not 0 <= args.partition_index < args.partitions:
        raise ValueError("invalid partition")
    selected = [c for i, c in enumerate(plan["cells"]) if i % args.partitions == args.partition_index]
    if not selected:
        raise ValueError("empty partition")
    output = args.output.resolve()
    checkpoint = Path(str(output) + ".jsonl")
    if args.plan_only:
        if output.exists() or checkpoint.exists():
            raise FileExistsError("plan destination already exists")
        atomic_json(output, {"plan_only": True, "plan": plan})
        return
    check_launch_gates(args, protocol, plan)
    if (output.exists() or checkpoint.exists()) and not args.resume:
        raise FileExistsError("use a new path or explicit --resume")
    run_hash = run_identity(plan)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("plan") != plan or previous.get("partition_index") != args.partition_index or previous.get("partitions") != args.partitions:
            raise ValueError("manifest identity mismatch")
    expected = len(plan["vehicle_ids"]) * len(plan["seeds"]) * len(plan["methods"])
    done = {}
    selected_ids = {c["cell_id"] for c in selected}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            record = json.loads(line)
            digest = record.pop("record_hash", None)
            if digest != canonical_hash(record) or record.get("run_hash") != run_hash:
                raise ValueError("checkpoint content or source identity mismatch")
            for key, value in {"schema": "tacp-related-work-v3", "mode": args.mode, "dataset": args.dataset,
                               "protocol_hash": plan["protocol_hash"], "source_hashes": plan["source_hashes"],
                               "plan_hash": plan["plan_hash"]}.items():
                if record.get(key) != value:
                    raise ValueError(f"checkpoint {key} mismatch")
            cid = record["cell_id"]
            if cid in done or cid not in selected_ids or len(record.get("rows", [])) != expected or record.get("row_count") != expected:
                raise ValueError("duplicate/unexpected/incomplete checkpoint cell")
            keys = {(r["vehicle_id"], r["seed"], r["baseline"], r["episode"]) for r in record["rows"]}
            wanted = {(v, s, m, 0) for v in plan["vehicle_ids"] for s in plan["seeds"] for m in plan["methods"]}
            if keys != wanted:
                raise ValueError("checkpoint row coverage mismatch")
            cell = next(c for c in selected if c["cell_id"] == cid)
            _validate_cell(record, cell, plan)
            done[cid] = expected
    output.parent.mkdir(parents=True, exist_ok=True)
    _, by_vehicle, all_windows, universes, _ = load_context(protocol, args.dataset)
    # Cross-fitting uses physical taxi IDs in Porto; highD uses other vehicles
    # from the same recording because coordinate systems cannot be pooled.
    calibration = {}
    for vid in plan["vehicle_ids"]:
        if args.dataset == "porto":
            cluster = cluster_id_for(vid, "porto")
            calibration[vid] = [w for w in all_windows if cluster_id_for(w.vehicle_id, "porto") != cluster]
        else:
            cluster = cluster_id_for(vid, "highd")
            calibration[vid] = [w for w in all_windows if w.vehicle_id != vid and cluster_id_for(w.vehicle_id, "highd") == cluster]
    started = time.monotonic()
    seeded = {"B8_static_rlnc_fountain", "A1_no_trajectory"}
    for cell in selected:
        if cell["cell_id"] in done:
            continue
        rows = []
        obj = _object_from_config(cell["config"])
        capacity = cell["config"]["placement_capacity_bytes"]
        enforce_deadline_capacity = cell["config"].get("placement_model", "legacy") == "deadline_capacity_v3_1"
        for vid in plan["vehicle_ids"]:
            windows = by_vehicle[vid]
            fixed = {method: build_baseline(method, obj, windows, capacity,
                     calibration_windows=calibration[vid], rsu_universe=universes[vid],
                     enforce_deadline_capacity=enforce_deadline_capacity)
                     for method in plan["methods"] if method not in seeded}
            for seed in plan["seeds"]:
                placements = dict(fixed)
                for method in seeded.intersection(plan["methods"]):
                    placements[method] = build_baseline(method, obj, windows, capacity,
                        rsu_universe=universes[vid], experiment_seed=seed,
                        enforce_deadline_capacity=enforce_deadline_capacity)
                for method in plan["methods"]:
                    rows.append(simulate_row(cell, windows, placements[method], seed, vid, args.dataset, plan["cluster_ids"][vid]))
        if len(rows) != expected:
            raise ValueError("row budget mismatch")
        record = {"schema": "tacp-related-work-v3", "mode": args.mode, "dataset": args.dataset,
                  "run_hash": run_hash, "protocol_hash": plan["protocol_hash"], "source_hashes": plan["source_hashes"],
                  "plan_hash": plan["plan_hash"], "cell_id": cell["cell_id"], "cell_config": cell["config"],
                  "legacy_cell_id": cell["legacy_cell_id"], "subset": cell["subset"], "rows": rows, "row_count": len(rows)}
        record["record_hash"] = canonical_hash(record)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done[cell["cell_id"]] = len(rows)
        atomic_json(output, {"schema": "tacp-related-work-result-v3", "plan": plan, "checkpoint": str(checkpoint),
                    "partitions": args.partitions, "partition_index": args.partition_index,
                    "selected_cell_ids": [c["cell_id"] for c in selected], "actual_rows": sum(done.values()),
                    "completed_cells": sorted(done), "complete": len(done) == len(selected),
                    "elapsed_s": time.monotonic()-started})
        print(json.dumps({"dataset": args.dataset, "mode": args.mode, "completed_cells": len(done),
                          "selected_cells": len(selected), "rows": sum(done.values())}), flush=True)
    # A crash after a durable final cell but before its manifest must remain
    # recoverable even when resume has no simulation work left.
    atomic_json(output, {"schema": "tacp-related-work-result-v3", "plan": plan, "checkpoint": str(checkpoint),
                "partitions": args.partitions, "partition_index": args.partition_index,
                "selected_cell_ids": [c["cell_id"] for c in selected], "actual_rows": sum(done.values()),
                "completed_cells": sorted(done), "complete": len(done) == len(selected),
                "elapsed_s": time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/related_work_v3_2.yaml")
    parser.add_argument("--dataset", choices=("porto", "highd"), required=True)
    parser.add_argument("--mode", choices=("dry-run", "pilot", "full", "small"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-gate", type=Path)
    parser.add_argument("--pilot-gate", type=Path)
    parser.add_argument("--highd-gate", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
