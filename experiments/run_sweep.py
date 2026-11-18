"""Deterministic, budgeted Porto full-sweep runner.

``--dry-run`` executes the configured 56-row validation (2 cells x 2 seeds x
2 vehicles x 7 baselines); ``--plan-only`` only expands the grid. Full runs
require a passed Phase 4 CI artifact with matching trace/provenance hashes.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from tacp_sim.baselines import BASELINES, all_baselines
from tacp_sim.io import config_hash, load_config
from experiments.run_phase4_ci import _load_windows, _object_from_config, _run_one_plan, _trace_vehicle_ids
SCHEMA="tacp-full-sweep-v1"
CONTROL={"sweep_schema","frozen_parameters","full_sweep","source_ci_config","source_ci_artifact","fixed","grid","budget","dry_run","seeds","episodes_per_seed","baselines","description","output"}
def canon(x):
    def c(v):
        if isinstance(v,Mapping): return {str(k):c(z) for k,z in sorted(v.items(),key=lambda p:str(p[0]))}
        if isinstance(v,(list,tuple)): return [c(z) for z in v]
        if isinstance(v,Path): return str(v)
        return v
    return json.dumps(c(x),sort_keys=True,separators=(",",":"),ensure_ascii=False)
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def resolve(x,base=ROOT):
    p=Path(str(x))
    if p.is_absolute(): return p
    candidate=(base/p).resolve()
    return candidate if candidate.exists() else (ROOT/p).resolve()
def scalar_values(key,raw):
    if isinstance(raw,(str,bytes)) or not isinstance(raw,Sequence) or not raw or any(isinstance(v,(Mapping,list,tuple)) for v in raw): raise ValueError(f"grid.{key} must be a non-empty scalar list")
    return sorted(list(raw),key=canon)
def base_config(spec,cp):
    if spec.get("source_ci_config"):
        base=dict(load_config(resolve(spec["source_ci_config"],cp.parent)))
    elif isinstance(spec.get("fixed"),Mapping):
        base={}
    else:
        base={str(k):v for k,v in spec.items() if k not in CONTROL}
    for k,v in spec.items():
        if k not in CONTROL: base[str(k)]=v
    if isinstance(spec.get("fixed"),Mapping): base.update(dict(spec["fixed"]))
    for k in ("baselines","stability","seeds","episodes_per_seed","frozen_parameters","frozen_schema","full_sweep"): base.pop(k,None)
    req=("trace_kind","trace_path","object_bytes","n","k","ttl_s","link_rate_bps")
    miss=[k for k in req if k not in base]
    if miss: raise ValueError(f"fixed/source CI config missing required keys: {miss}")
    return base
def normalize(spec,cp):
    if spec.get("sweep_schema")!=SCHEMA: raise ValueError(f"sweep_schema must be {SCHEMA!r}")
    if spec.get("frozen_parameters") is not True or spec.get("full_sweep") is not True: raise ValueError("requires frozen_parameters:true and full_sweep:true")
    gridraw=spec.get("grid")
    if not isinstance(gridraw,Mapping) or not gridraw: raise ValueError("grid must be a non-empty mapping")
    grid={}; variants=[{}]
    for key,raw in sorted(gridraw.items(),key=lambda p:str(p[0])):
        key=str(key)
        if key=="coding":
            if not isinstance(raw,Sequence) or not raw: raise ValueError("grid.coding must be non-empty")
            pairs=[]
            for pair in raw:
                if not isinstance(pair,Mapping) or set(pair)!={"n","k"}: raise ValueError("coding entries require n,k")
                n,k=int(pair["n"]),int(pair["k"])
                if n<1 or k<1 or k>n: raise ValueError(f"invalid coding pair {(n,k)}")
                pairs.append({"n":n,"k":k})
            variants=[{**v,**p} for v in variants for p in sorted(pairs,key=canon)]
        else: grid[key]=scalar_values(key,raw)
    seedsraw=spec.get("seeds")
    if isinstance(seedsraw,(str,bytes)) or not isinstance(seedsraw,Sequence): raise ValueError("seeds must be a list")
    seeds=[int(x) for x in seedsraw]
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("seeds must be unique")
    episodes=int(spec.get("episodes_per_seed",1));
    if episodes<1: raise ValueError("episodes_per_seed must be positive")
    baselines=[str(x) for x in spec.get("baselines",BASELINES)]
    if baselines!=list(BASELINES): raise ValueError("full sweep must evaluate all seven BASELINES in order")
    budget=dict(spec.get("budget",{})); req=("cell_count","row_count","chunk_cells","chunk_count","max_output_bytes")
    miss=[k for k in req if k not in budget]
    if miss: raise ValueError(f"budget missing {miss}")
    return {"base":base_config(spec,cp),"grid":grid,"variants":variants,"seeds":seeds,"episodes":episodes,"baselines":baselines,"budget":budget,"dry":dict(spec.get("dry_run",{}))}
def cells(norm):
    keys=list(norm["grid"]); out=[]
    for vals in itertools.product(*(norm["grid"][k] for k in keys)):
        overlay=dict(zip(keys,vals))
        for var in norm["variants"]:
            c=dict(norm["base"]); c.update(overlay); c.update(var); c["_sweep_cell_id"]=sha(c)[:16]; out.append(c)
    out.sort(key=lambda c:c["_sweep_cell_id"])
    if len({c["_sweep_cell_id"] for c in out})!=len(out): raise ValueError("duplicate cell IDs")
    return out
def stability(spec,cp):
    raw=spec.get("source_ci_artifact")
    if not raw: raise ValueError("source_ci_artifact is required")
    p=resolve(raw)
    if not p.is_file(): raise FileNotFoundError(p)
    a=json.loads(p.read_text()); gate=a.get("stability_gate")
    if not isinstance(gate,Mapping) or gate.get("passed") is not True: raise ValueError("source CI stability gate is not passed")
    if a.get("sweep_started") is not False: raise ValueError("source CI artifact must be a completed Phase 4 validation, not a sweep")
    rec=(a.get("trace_records") or [{}])[0]; prov=rec.get("provenance",{}) if isinstance(rec,Mapping) else {}; ds=prov.get("dataset_provenance",{}) if isinstance(prov,Mapping) else {}
    tp=resolve(spec["trace_path"]); th=hashlib.sha256(tp.read_bytes()).hexdigest(); eh=ds.get("output",{}).get("sha256") if isinstance(ds,Mapping) else None
    if eh and th!=eh: raise ValueError("trace hash differs from source CI artifact")
    pp=None
    if spec.get("trace_provenance_path"):
        pp=resolve(spec["trace_provenance_path"]); ph=hashlib.sha256(pp.read_bytes()).hexdigest(); eph=ds.get("sha256") if isinstance(ds,Mapping) else None
        if eph and ph!=eph: raise ValueError("trace provenance hash differs from source CI artifact")
    return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"passed":True,"trace_sha256":th,"provenance_sha256":ph if pp else None,"frozen_parameter_hash":a.get("frozen_parameter_hash")}
def make_plan(spec, cp, dry=False, limit_cells=None, limit_seeds=None,
              limit_vehicles=None, cell_start=None, cell_end=None):
    n = normalize(spec, cp)
    allc = cells(n)
    full_seeds = list(n["seeds"])
    full_vehicles = int(spec.get("trace_vehicle_limit", 255))
    full_eps = int(n["episodes"])
    full_rows = len(allc) * len(full_seeds) * full_eps * full_vehicles * len(n["baselines"])
    if dry:
        default_cells = int(n["dry"].get("cells", len(allc)))
        seeds = list(n["dry"].get("seeds", full_seeds))
        eps = int(n["dry"].get("episodes_per_seed", full_eps))
        vlim = int(n["dry"].get("trace_vehicle_limit", full_vehicles))
        start = 0 if cell_start is None else int(cell_start)
        end = start + default_cells if cell_end is None else int(cell_end)
    else:
        seeds, eps, vlim = full_seeds, full_eps, full_vehicles
        start = 0 if cell_start is None else int(cell_start)
        end = len(allc) if cell_end is None else int(cell_end)
    if limit_cells is not None:
        if cell_start is not None or cell_end is not None:
            raise ValueError("use cell-start/cell-end or limit-cells, not both")
        end = start + int(limit_cells)
    if limit_seeds is not None:
        seeds = seeds[:int(limit_seeds)]
    if limit_vehicles is not None:
        vlim = min(vlim, int(limit_vehicles))
    if start < 0 or end < start or end > len(allc):
        raise ValueError(f"cell range must satisfy 0 <= start <= end <= {len(allc)}")
    if end == start:
        raise ValueError("cell range must contain at least one cell")
    if not seeds or eps < 1 or vlim < 1:
        raise ValueError("effective sweep limits must be positive")
    selected = allc[start:end]
    rows = len(selected) * len(seeds) * eps * vlim * len(n["baselines"])
    budget = n["budget"]
    if full_rows != int(budget["row_count"]) or len(allc) != int(budget["cell_count"]):
        raise ValueError("frozen full budget mismatch with expanded grid")
    if rows > int(budget["row_count"]) or len(selected) > int(budget["cell_count"]):
        raise ValueError("frozen sweep budget exceeded")
    if int(budget["chunk_cells"]) < 1 or (int(budget["cell_count"]) + int(budget["chunk_cells"]) - 1) // int(budget["chunk_cells"]) != int(budget["chunk_count"]):
        raise ValueError("chunk budget is inconsistent")
    if dry:
        expected = int(n["dry"].get("expected_rows", rows))
        maximum = int(n["dry"].get("max_rows", expected))
        if rows != expected or rows > maximum:
            raise ValueError(f"dry-run budget mismatch: {rows} rows vs {expected}/{maximum}")
    # Preserve the original full-plan identity so checkpoints written by the
    # pre-range runner remain resumable. Range selection is intentionally not
    # part of this payload.
    full_identity = {
        "schema": "tacp_sweep_plan_v1", "full_cell_count": len(allc),
        "cell_count": len(allc), "seeds": full_seeds,
        "episodes_per_seed": full_eps, "vehicle_limit": full_vehicles,
        "baseline_count": len(n["baselines"]), "expected_rows": full_rows,
        "budget": budget,
        "cells": [{"cell_id": c["_sweep_cell_id"], "config": {k: v for k, v in c.items() if not k.startswith("_")}} for c in allc],
        "dry_run": False,
    }
    full_hash = sha(full_identity)
    payload = {
        "schema": "tacp_sweep_plan_v1", "full_plan_hash": full_hash, "plan_hash": full_hash,
        "full_cell_count": len(allc), "full_expected_rows": full_rows,
        "cell_start": start, "cell_end": end, "cell_count": len(selected),
        "seeds": seeds, "episodes_per_seed": eps, "vehicle_limit": vlim,
        "baseline_count": len(n["baselines"]), "expected_rows": rows, "budget": budget,
        "cells": [{"cell_id": c["_sweep_cell_id"], "config": {k: v for k, v in c.items() if not k.startswith("_")}} for c in selected],
        "dry_run": dry,
    }
    return payload, n
def row(raw,cell,cid,ch,baseline,seed,episode,vid,nwin,bounded):
    return {"cell_id":cid,"object_bytes":int(cell["object_bytes"]),"n":int(cell["n"]),"k":int(cell["k"]),"ttl_s":float(cell["ttl_s"]),"packet_loss":float(cell.get("packet_loss",0)),"rsu_failure_probability":float(cell.get("rsu_failure_probability",0)),"placement_capacity_bytes":int(cell.get("placement_capacity_bytes",0)),"seed":int(seed),"episode":int(episode),"vehicle_id":vid,"baseline":baseline,"fresh_success":bool(raw["fresh_success"]),"T_decode":raw["t_decode"],"t_decode":raw["t_decode"],"rank":int(raw["rank"]),"storage":int(raw["storage"]),"backhaul":int(raw["backhaul"]),"migration":int(raw["migration"]),"repair":int(raw["repair"]),"trace_windows":nwin,"config_hash":ch,"oracle_bounded":bounded,"oracle_max_actions":20 if baseline=="B6_oracle" else None}
def execute(spec, cp, output, dry=False, plan_only=False, resume=False,
            limit_cells=None, limit_seeds=None, limit_vehicles=None,
            cell_start=None, cell_end=None):
    gate = stability(spec, cp)
    p, n = make_plan(spec, cp, dry, limit_cells, limit_seeds, limit_vehicles,
                     cell_start, cell_end)
    output = resolve(output)
    if plan_only:
        out = dict(p)
        out.update({"dry_run": False, "plan_only": True, "sweep_started": False,
                    "stability_gate": gate})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        return out
    ck = Path(str(output) + ".jsonl")
    if (output.exists() or ck.exists()) and not resume:
        raise FileExistsError("output/checkpoint exists; use --resume")
    done = {}
    if resume and ck.exists():
        with ck.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    if record.get("plan_hash") != p["plan_hash"]:
                        raise ValueError("checkpoint plan hash mismatch")
                    if not dry:
                        record.pop("rows", None)
                    done[str(record["cell_id"])] = record
    all_cells = cells(n)
    selected_cells = all_cells[p["cell_start"]:p["cell_end"]]
    first = dict(selected_cells[0])
    first["trace_path"] = str(resolve(first["trace_path"]))
    if first.get("trace_provenance_path"):
        first["trace_provenance_path"] = str(resolve(first["trace_provenance_path"]))
    windows, provenance = _load_windows(first, p["seeds"][0])
    vehicle_ids = _trace_vehicle_ids(windows, {**first, "trace_vehicle_limit": p["vehicle_limit"]})
    if len(vehicle_ids) != p["vehicle_limit"]:
        raise ValueError("selected vehicle count differs from frozen limit")
    records = []
    for cell_index, cell in zip(range(p["cell_start"], p["cell_end"]), selected_cells):
        cid = cell["_sweep_cell_id"]
        if cid in done:
            records.append(done[cid])
            continue
        obj = _object_from_config(cell)
        cfg_hash = config_hash({k: v for k, v in cell.items() if not k.startswith("_")})
        cell_rows = []
        for vehicle_id in vehicle_ids:
            vehicle_windows = [w for w in windows if str(w.vehicle_id) == vehicle_id]
            plans = all_baselines(
                obj, vehicle_windows,
                capacity_bytes=int(cell.get("placement_capacity_bytes", obj.coded_size_bytes)),
            )
            for seed in p["seeds"]:
                for episode in range(p["episodes_per_seed"]):
                    for baseline in n["baselines"]:
                        raw = _run_one_plan(
                            cell, vehicle_windows, plans[baseline], seed=seed,
                            episode=episode, vehicle_id=vehicle_id,
                            object_version=obj,
                        )
                        bounded = bool(plans[baseline].metadata.get("bounded", False)) if baseline == "B6_oracle" else None
                        cell_rows.append(row(
                            raw, cell, cid, cfg_hash, baseline, seed, episode,
                            vehicle_id, len(vehicle_windows), bounded,
                        ))
        record = {
            "record_type": "cell", "plan_hash": p["plan_hash"],
            "full_plan_hash": p["full_plan_hash"], "cell_id": cid,
            "cell_index": cell_index,
            "chunk_id": cell_index // int(n["budget"]["chunk_cells"]),
            "config": {k: v for k, v in cell.items() if not k.startswith("_")},
            "rows": cell_rows, "row_count": len(cell_rows),
            "source_trace_sha256": provenance.get("source_sha256"),
        }
        ck.parent.mkdir(parents=True, exist_ok=True)
        with ck.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if ck.stat().st_size > int(n["budget"]["max_output_bytes"]):
            raise ValueError("checkpoint exceeded max_output_bytes")
        records.append(record)
    actual = sum(int(r.get("row_count", len(r.get("rows", [])))) for r in records)
    expected = int(n["dry"].get("expected_rows", p["expected_rows"])) if dry else p["expected_rows"]
    if actual != expected:
        raise ValueError(f"produced {actual} rows, expected {expected}")
    out = {
        "schema": "tacp_sweep_result_v1", "dry_run": dry, "plan_only": False,
        "sweep_started": not dry, "resumed": resume, "stability_gate": gate,
        "plan": p, "checkpoint": str(ck), "completed_cells": len(records),
        "actual_rows": actual, "trace_provenance": provenance,
        "records": records if dry else [
            {k: r[k] for k in ("cell_id", "cell_index", "chunk_id", "row_count")}
            for r in records
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    if tmp.stat().st_size > int(n["budget"]["max_output_bytes"]):
        raise ValueError("result exceeded max_output_bytes")
    tmp.replace(output)
    return out
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/full_sweep.yaml")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-cells", type=int)
    ap.add_argument("--limit-seeds", type=int)
    ap.add_argument("--limit-vehicles", type=int)
    ap.add_argument("--cell-start", type=int, help="inclusive full-grid cell index")
    ap.add_argument("--cell-end", type=int, help="exclusive full-grid cell index")
    args = ap.parse_args()
    if (args.cell_start is None) != (args.cell_end is None):
        ap.error("--cell-start and --cell-end must be provided together")
    if args.limit_cells is not None and args.cell_start is not None:
        ap.error("use --cell-start/--cell-end or --limit-cells, not both")
    config_path = args.config.resolve()
    spec = load_config(config_path)
    name = "full_sweep_dry_run.json" if args.dry_run or args.plan_only else "full_sweep.json"
    output = args.output or ROOT / "results/processed" / name
    result = execute(
        spec, config_path, output, args.dry_run, args.plan_only, args.resume,
        args.limit_cells, args.limit_seeds, args.limit_vehicles,
        args.cell_start, args.cell_end,
    )
    plan_data = result.get("plan", result)
    print(json.dumps({
        "output": str(resolve(output)), "dry_run": result.get("dry_run", False),
        "plan_only": result.get("plan_only", False),
        "full_plan_hash": plan_data.get("full_plan_hash"),
        "cell_start": plan_data["cell_start"], "cell_end": plan_data["cell_end"],
        "cell_count": plan_data["cell_count"], "expected_rows": plan_data["expected_rows"],
        "actual_rows": result.get("actual_rows", 0),
        "sweep_started": result["sweep_started"],
        "stability_gate_passed": result["stability_gate"]["passed"],
    }, sort_keys=True))
if __name__=="__main__": main()
