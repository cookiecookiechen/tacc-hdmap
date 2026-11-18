#!/usr/bin/env python3
"""Prepare a small, reproducible Porto taxi trajectory for the simulator.

The UCI Porto Taxi dataset stores each trip as a JSON encoded ``POLYLINE``
sampled every 15 seconds.  The simulator deliberately consumes a much smaller
contract (``timestamp,vehicle_id,rsu_id,x,y,speed``), so this script performs
only deterministic preprocessing:

* parse and validate the UCI nine-column rows;
* project WGS84 longitude/latitude to a local tangent-plane coordinate system;
* associate each fix with the nearest configured RSU whose coverage circle
  contains it; and
* write a hash-backed provenance record next to the derived CSV.

The checked-in input is a one-trip, publicly documented slice.  It is useful
for parser and small-scenario checks, but is not large enough for paper
statistics.  For a real experiment, download the complete UCI file and run
this same script with ``--input`` and a frozen RSU configuration.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SOURCE_URL = "https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015"
SOURCE_DOI = "10.24432/C55W25"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_CITATION = (
    "Moreira-Matias et al., Taxi Service Trajectory - Prediction Challenge, "
    "ECML PKDD 2015, UCI Machine Learning Repository, DOI:10.24432/C55W25"
)
# UCI's ``Download`` button currently resolves to this static archive.  Keep
# the URL in one place so a remote runner can fetch the public training archive
# without relying on a browser session.
SOURCE_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/339/"
    "taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015.zip"
)
RAW_REQUIRED = {
    "TRIP_ID",
    "TAXI_ID",
    "TIMESTAMP",
    "POLYLINE",
}
OUTPUT_FIELDS = ["timestamp", "vehicle_id", "rsu_id", "x", "y", "speed"]


@dataclass(frozen=True)
class Rsu:
    rsu_id: str
    lon: float
    lat: float
    coverage_radius_m: float

    def __post_init__(self) -> None:
        if not self.rsu_id:
            raise ValueError("RSU id must not be empty")
        values = (self.lon, self.lat, self.coverage_radius_m)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("RSU coordinates and radius must be finite")
        if not -180 <= self.lon <= 180 or not -90 <= self.lat <= 90:
            raise ValueError("RSU coordinates must be WGS84 longitude/latitude")
        if self.coverage_radius_m <= 0:
            raise ValueError("coverage_radius_m must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "rsu_id": self.rsu_id,
            "lon": self.lon,
            "lat": self.lat,
            "coverage_radius_m": self.coverage_radius_m,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_rsus(value: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Rsu]:
    if isinstance(value, Mapping):
        entries = []
        for key, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"invalid RSU entry for {key!r}")
            item = dict(raw)
            item.setdefault("rsu_id", key)
            entries.append(item)
    else:
        entries = list(value)
    rsus: list[Rsu] = []
    for raw in entries:
        try:
            rsus.append(
                Rsu(
                    rsu_id=str(raw["rsu_id"]),
                    lon=float(raw["lon"]),
                    lat=float(raw["lat"]),
                    coverage_radius_m=float(raw["coverage_radius_m"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid RSU specification: {raw!r}") from exc
    if not rsus:
        raise ValueError("at least one RSU is required")
    return rsus


def load_rsus(path: str | Path) -> list[Rsu]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read RSU JSON: {source}") from exc
    if isinstance(value, Mapping) and "rsus" in value:
        value = value["rsus"]
    if not isinstance(value, (Mapping, list, tuple)):
        raise ValueError("RSU JSON must be an object or list")
    return _as_rsus(value)


def load_download_manifest(path: str | Path) -> dict[str, object]:
    """Load and validate the hash manifest emitted by the downloader."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read download manifest: {source}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != "tacp_porto_download_manifest_v1":
        raise ValueError("download manifest has an unsupported schema")
    archive = value.get("archive")
    if not isinstance(archive, Mapping) or not archive.get("sha256"):
        raise ValueError("download manifest is missing archive SHA-256")
    digest = str(archive["sha256"]).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("download manifest archive SHA-256 is invalid")
    return dict(value)


def _project(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    """Approximate local ENU projection (metres) for the small Porto slice."""
    radius = 6_371_000.0
    lat0 = math.radians(origin_lat)
    x = math.radians(lon - origin_lon) * radius * math.cos(lat0)
    y = math.radians(lat - origin_lat) * radius
    return x, y


def _distance_m(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    x, y = _project(lon_b, lat_b, lon_a, lat_a)
    return math.hypot(x, y)


def _nearest_rsu(lon: float, lat: float, rsus: Iterable[Rsu]) -> str | None:
    candidates = []
    for rsu in rsus:
        distance = _distance_m(lon, lat, rsu.lon, rsu.lat)
        if distance <= rsu.coverage_radius_m + 1e-9:
            candidates.append((distance, rsu.rsu_id))
    return min(candidates, key=lambda item: (item[0], item[1]))[1] if candidates else None


def _trip_sort_key(trip_id: str) -> tuple[int, object, str]:
    """Return a stable ordering key for UCI trip identifiers.

    The Porto release uses decimal trip identifiers.  The string fallback
    keeps the selection rule deterministic for small fixtures or future UCI
    exports that contain non-numeric identifiers.
    """
    value = str(trip_id).strip()
    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value, value)


def _select_rows(
    source: Path,
    *,
    max_trips: int | None,
    trip_selection: str,
    min_points: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Read source rows and deterministically select eligible trips.

    For ``lexicographic_trip_id`` only the smallest ``max_trips`` identifiers
    are retained while streaming the input, so a 509 MB archive does not need
    to be materialised in memory.  ``source_order`` preserves the historical
    fixture behaviour and stops after the requested number of eligible rows.
    """
    if trip_selection not in {"source_order", "lexicographic_trip_id"}:
        raise ValueError("trip_selection must be 'source_order' or 'lexicographic_trip_id'")
    if max_trips is not None and (isinstance(max_trips, bool) or int(max_trips) < 1):
        raise ValueError("max_trips must be positive when provided")
    rows_seen = 0
    rows_skipped = 0
    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = RAW_REQUIRED - fields
        if missing:
            raise ValueError(f"input missing UCI columns: {sorted(missing)}")
        if trip_selection == "source_order" or max_trips is None:
            selected: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                rows_seen += 1
                try:
                    trip_id = str(row["TRIP_ID"]).strip()
                    int(row["TIMESTAMP"])
                    points = _parse_polyline(row["POLYLINE"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"invalid UCI row {row_number}") from exc
                if len(points) < min_points:
                    rows_skipped += 1
                    continue
                selected.append(dict(row))
                if max_trips is not None and len(selected) >= max_trips:
                    break
            return selected, rows_seen, rows_skipped

        # A max-heap keyed by the largest trip identifier keeps only the K
        # smallest identifiers.  UCI IDs are numeric; non-numeric fixture IDs
        # are collected separately and sorted after the streaming pass.
        selected_heap: list[tuple[int, str, dict[str, str]]] = []
        fallback_rows: list[tuple[tuple[int, object, str], dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            rows_seen += 1
            try:
                trip_id = str(row["TRIP_ID"]).strip()
                int(row["TIMESTAMP"])
                points = _parse_polyline(row["POLYLINE"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid UCI row {row_number}") from exc
            if len(points) < min_points:
                rows_skipped += 1
                continue
            key = _trip_sort_key(trip_id)
            if key[0] == 0:
                # The smallest negative value is the largest numeric ID and
                # therefore remains at heap root for replacement.
                priority = -int(key[1])
                if len(selected_heap) < max_trips:
                    heapq.heappush(selected_heap, (priority, trip_id, dict(row)))
                else:
                    root_priority, root_trip_id, _ = selected_heap[0]
                    if (priority, trip_id) > (root_priority, root_trip_id):
                        heapq.heapreplace(selected_heap, (priority, trip_id, dict(row)))
            else:
                fallback_rows.append((key, dict(row)))
        numeric_rows = [(trip_id, row) for _, trip_id, row in selected_heap]
        numeric_rows.sort(key=lambda item: _trip_sort_key(item[0]))
        fallback_rows.sort(key=lambda item: item[0])
        selected = [row for _, row in numeric_rows]
        selected.extend(row for _, row in fallback_rows)
        selected.sort(key=lambda row: _trip_sort_key(str(row["TRIP_ID"])))
        return selected[:max_trips], rows_seen, rows_skipped


def _parse_polyline(raw: str) -> list[tuple[float, float]]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("POLYLINE is not valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("POLYLINE must be a JSON list")
    points: list[tuple[float, float]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("POLYLINE points must be [longitude, latitude]")
        lon, lat = float(pair[0]), float(pair[1])
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise ValueError("POLYLINE coordinates must be finite")
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("POLYLINE coordinates must be WGS84 longitude/latitude")
        points.append((lon, lat))
    return points


def prepare_porto_csv(
    input_path: str | Path,
    output_path: str | Path,
    provenance_path: str | Path,
    rsus: Iterable[Rsu],
    *,
    sample_interval_s: float = 15.0,
    max_trips: int | None = None,
    min_points: int = 2,
    min_covered_points: int = 1,
    trip_selection: str = "source_order",
    source_manifest_path: str | Path | None = None,
    rsu_config_path: str | Path | None = None,
) -> dict[str, object]:
    """Convert UCI Porto rows into the simulator six-column CSV contract."""
    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive and finite")
    if min_points < 1:
        raise ValueError("min_points must be positive")
    if min_covered_points < 1:
        raise ValueError("min_covered_points must be positive")
    if trip_selection == "lexicographic_trip_id" and max_trips is None:
        raise ValueError("max_trips is required for lexicographic_trip_id selection")
    rsus = list(rsus)
    if not rsus:
        raise ValueError("at least one RSU is required")
    source = Path(input_path)
    target = Path(output_path)
    provenance_target = Path(provenance_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance_target.parent.mkdir(parents=True, exist_ok=True)
    download_manifest = load_download_manifest(source_manifest_path) if source_manifest_path else None

    selected_rows, rows_seen, rows_skipped = _select_rows(
        source,
        max_trips=max_trips,
        trip_selection=trip_selection,
        min_points=min_points,
    )
    trips_written = 0
    rows_uncovered = 0
    trips_discarded = 0
    missing_data_trips = 0
    trip_records: list[dict[str, object]] = []
    out_rows: list[dict[str, object]] = []
    origin_lon: float | None = None
    origin_lat: float | None = None

    for row_number, row in enumerate(selected_rows, start=2):
        try:
            trip_id = str(row["TRIP_ID"]).strip()
            taxi_id = str(row["TAXI_ID"]).strip()
            int(row["TIMESTAMP"])
            points = _parse_polyline(row["POLYLINE"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid UCI row {row_number}") from exc
        if origin_lon is None:
            origin_lon, origin_lat = points[0]
        assert origin_lat is not None and origin_lon is not None
        vehicle_id = f"taxi_{taxi_id}_trip_{trip_id}"
        missing_data = str(row.get("MISSING_DATA", "")).strip().lower() in {"true", "1", "yes"}
        if missing_data:
            missing_data_trips += 1
        covered_for_trip = 0
        trip_out_rows: list[dict[str, object]] = []
        for idx, (lon, lat) in enumerate(points):
            rsu_id = _nearest_rsu(lon, lat, rsus)
            if rsu_id is None:
                rows_uncovered += 1
                continue
            x, y = _project(lon, lat, origin_lon, origin_lat)
            if idx == 0:
                speed = 0.0
            else:
                prev_lon, prev_lat = points[idx - 1]
                speed = _distance_m(prev_lon, prev_lat, lon, lat) / sample_interval_s
            trip_out_rows.append(
                {
                    "timestamp": round(idx * sample_interval_s, 6),
                    "vehicle_id": vehicle_id,
                    "rsu_id": rsu_id,
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "speed": round(speed, 6),
                }
            )
            covered_for_trip += 1
        if covered_for_trip < min_covered_points:
            trips_discarded += 1
            continue
        out_rows.extend(trip_out_rows)
        trips_written += 1
        trip_records.append(
            {
                "trip_id": trip_id,
                "taxi_id": taxi_id,
                "vehicle_id": vehicle_id,
                "source_timestamp": int(row["TIMESTAMP"]),
                "points_in_source": len(points),
                "points_written": covered_for_trip,
                "missing_data": missing_data,
            }
        )

    if not out_rows:
        raise ValueError("no covered trajectory points were produced")
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    provenance = {
        "schema": "tacp_real_trace_provenance_v1",
        "source": {
            "dataset": "Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015",
            "url": SOURCE_URL,
            "archive_url": SOURCE_ARCHIVE_URL,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
            "citation": SOURCE_CITATION,
        },
        "input": {"path": str(source.resolve()), "sha256": sha256_file(source)},
        "output": {"path": str(target.resolve()), "sha256": sha256_file(target)},
        "source_archive": (
            {
                "manifest_path": str(Path(source_manifest_path).resolve()),
                "manifest_sha256": sha256_file(source_manifest_path),
                "archive": dict(download_manifest["archive"]),
            }
            if download_manifest is not None and source_manifest_path is not None
            else None
        ),
        "rsu_config": (
            {"path": str(Path(rsu_config_path).resolve()), "sha256": sha256_file(rsu_config_path)}
            if rsu_config_path is not None
            else None
        ),
        "preprocessing": {
            "parser": "porto_polyline_to_tacp_csv_v1",
            "sample_interval_s": sample_interval_s,
            "coordinate_system": "local_equirectangular_ENU_metres",
            "origin_lon": origin_lon,
            "origin_lat": origin_lat,
            "rsus": [rsu.as_dict() for rsu in rsus],
            "min_points": min_points,
            "min_covered_points": min_covered_points,
            "rsu_assignment_rule": "nearest_rsu_within_radius; ties by distance then rsu_id",
            "trip_selection": trip_selection,
            "max_trips": max_trips,
            "selection_target": "max_trips eligible source rows before coverage filtering",
            "missing_data_policy": "retain rows whose POLYLINE meets min_points; record MISSING_DATA flag",
        },
        "counts": {
            "source_rows_seen": rows_seen,
            "trips_written": trips_written,
            "rows_written": len(out_rows),
            "rows_uncovered": rows_uncovered,
            "rows_uncovered_scope": "includes selected trips discarded by min_covered_points",
            "rows_skipped": rows_skipped,
            "trips_discarded": trips_discarded,
            "missing_data_trips": missing_data_trips,
        },
        "trips": trip_records,
    }
    provenance_target.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UCI nine-column CSV")
    parser.add_argument("--output", required=True, help="derived six-column CSV")
    parser.add_argument("--provenance", required=True, help="provenance JSON")
    parser.add_argument("--rsus", required=True, help="JSON RSU configuration")
    parser.add_argument(
        "--download-manifest",
        help="optional manifest emitted by download_porto_dataset.py",
    )
    parser.add_argument("--sample-interval-s", type=float, default=15.0)
    parser.add_argument("--max-trips", type=int)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--min-covered-points", type=int, default=1)
    parser.add_argument(
        "--trip-selection",
        choices=("source_order", "lexicographic_trip_id"),
        default="source_order",
        help="deterministic trip selection rule when --max-trips is set",
    )
    args = parser.parse_args()
    report = prepare_porto_csv(
        args.input,
        args.output,
        args.provenance,
        load_rsus(args.rsus),
        sample_interval_s=args.sample_interval_s,
        max_trips=args.max_trips,
        min_points=args.min_points,
        min_covered_points=args.min_covered_points,
        trip_selection=args.trip_selection,
        source_manifest_path=args.download_manifest,
        rsu_config_path=args.rsus,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
