#!/usr/bin/env python3
"""Prepare the deterministic highD supplement directly from a local ZIP archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tacp_sim.highd import highd_contact_windows, point_from_track_row, vehicle_key, write_contact_windows_csv
from tacp_sim.mobility import RsuSite


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_tracks(rows: list[dict[str, str]], *, frame_rate: float, recording_frames: int,
                  count: int, min_duration_s: float) -> tuple[list[dict[str, str]], dict[str, int]]:
    eligible = []
    reasons: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: int(item["id"])):
        initial, final, number = (int(row[key]) for key in ("initialFrame", "finalFrame", "numFrames"))
        if row["class"] != "Car":
            reasons["not_car"] += 1
        elif initial <= 1 or final >= recording_frames:
            reasons["recording_boundary"] += 1
        elif number != final - initial + 1:
            reasons["noncontiguous_metadata"] += 1
        elif (final - initial) / frame_rate < min_duration_s:
            reasons["shorter_than_min_duration"] += 1
        else:
            eligible.append(row)
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible cars, require {count}")
    reasons["eligible_not_selected_after_first_n"] = len(eligible) - count
    return eligible[:count], dict(reasons)


def prepare(archive_path: Path, config_path: Path, *, output_dir: Path, raw_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if max(config["ttl_s"]) > config["selection"]["minimum_duration_s"]:
        raise ValueError("every frozen TTL must fit within the minimum selected observation duration")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty selected-raw directory: {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_digest = sha256_file(archive_path)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for rec_value in config["recordings"]:
            rec = f"{int(rec_value):02d}"
            names = {suffix: f"data/{rec}_{suffix}.csv" for suffix in ("recordingMeta", "tracksMeta", "tracks")}
            recording_bytes, track_meta_bytes = (archive.read(names[key]) for key in ("recordingMeta", "tracksMeta"))
            recording_rows = list(csv.DictReader(io.StringIO(recording_bytes.decode("utf-8-sig"))))
            if len(recording_rows) != 1:
                raise ValueError("expected one recording metadata row")
            metadata = recording_rows[0]
            frame_rate = float(metadata["frameRate"])
            recording_frames = round(float(metadata["duration"]) * frame_rate)
            track_metadata = list(csv.DictReader(io.StringIO(track_meta_bytes.decode("utf-8-sig"))))
            selected_meta, excluded = select_tracks(track_metadata, frame_rate=frame_rate,
                recording_frames=recording_frames, count=config["cars_per_recording"],
                min_duration_s=config["selection"]["minimum_duration_s"])
            selected_ids = {int(row["id"]) for row in selected_meta}
            points = {track_id: [] for track_id in selected_ids}
            source_rows = 0
            x_min, x_max = float("inf"), float("-inf")
            raw_target = raw_dir / f"{rec}_selected_tracks.csv"
            with archive.open(names["tracks"]) as source, raw_target.open("w", newline="", encoding="utf-8") as target:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    source_rows += 1
                    point = point_from_track_row(row)
                    x_min, x_max = min(x_min, point.x), max(x_max, point.x)
                    track_id = int(row["id"])
                    if track_id in selected_ids:
                        points[track_id].append(point)
                        writer.writerow(row)
            x_span = x_max - x_min
            markings = [float(value) for key in ("upperLaneMarkings", "lowerLaneMarkings") for value in metadata[key].split(";")]
            road_mid_y = (min(markings) + max(markings)) / 2
            radius = x_span * config["rsu_rule"]["radius_fraction_of_x_span"]
            sites = [RsuSite(f"highd_{rec}_r{index}", x_min + fraction * x_span, road_mid_y, radius)
                     for index, fraction in enumerate(config["rsu_rule"]["x_fractions"])]
            site_json = {site.rsu_id: {"x": site.x, "y": site.y, "coverage_radius_m": site.coverage_radius_m} for site in sites}
            all_windows = []
            vehicle_records = []
            for track in selected_meta:
                track_id = int(track["id"])
                ordered = sorted(points[track_id], key=lambda point: point.frame)
                frames = [point.frame for point in ordered]
                expected_frames = list(range(int(track["initialFrame"]), int(track["finalFrame"]) + 1))
                if frames != expected_frames or len(frames) != int(track["numFrames"]):
                    raise ValueError(f"recording {rec} car {track_id}: measured frames disagree with metadata")
                key = vehicle_key(rec, track_id)
                windows = highd_contact_windows(ordered, sites, vehicle_id=key, frame_rate=frame_rate,
                    rate_bps=config["rate_bps"])
                all_windows.extend(windows)
                vehicle_records.append({"vehicle_id": key, "physical_vehicle_id": key,
                    "cluster_id": f"highd_{rec}", "recording_cluster_id": f"highd_{rec}", "recording_id": rec,
                    "track_id": track_id, "driving_direction": int(track["drivingDirection"]),
                    "source_initial_frame": frames[0], "source_final_frame": frames[-1],
                    "source_duration_s": (frames[-1] - frames[0]) / frame_rate,
                    "source_points": len(frames), "contact_count": len(windows),
                    "distinct_rsus": len({window.rsu_id for window in windows}),
                    "first_contact_s": min((window.start for window in windows), default=None),
                    "last_contact_s": max((window.end for window in windows), default=None)})
            directions = Counter(row["driving_direction"] for row in vehicle_records)
            if set(directions) != {1, 2}:
                raise ValueError(f"recording {rec}: frozen sample does not include both directions")
            contact_path = output_dir / f"{rec}_contacts.csv"
            site_path = output_dir / f"{rec}_rsus.json"
            selection_path = output_dir / f"{rec}_selection.json"
            write_contact_windows_csv(contact_path, all_windows)
            write_json(site_path, site_json)
            write_json(selection_path, {"schema": "highd_selected_tracks_v1", "recording_id": rec,
                "selection_rule": config["selection"], "selected": vehicle_records, "excluded_counts": excluded})
            (raw_dir / f"{rec}_recordingMeta.csv").write_bytes(recording_bytes)
            (raw_dir / f"{rec}_tracksMeta.csv").write_bytes(track_meta_bytes)
            records.append({"recording_id": rec, "location_id": int(metadata["locationId"]),
                "frame_rate": frame_rate, "recording_metadata": metadata,
                "source_track_member": {"name": names["tracks"], "uncompressed_bytes": archive.getinfo(names["tracks"]).file_size,
                    "crc32": f"{archive.getinfo(names['tracks']).CRC:08x}", "rows_seen": source_rows},
                "selected_raw": {"path": raw_target.as_posix(), "sha256": sha256_file(raw_target)},
                "recording_metadata_sha256": hashlib.sha256(recording_bytes).hexdigest(),
                "track_metadata_sha256": hashlib.sha256(track_meta_bytes).hexdigest(),
                "geometry": {"all_tracks_center_x_min_m": x_min, "all_tracks_center_x_max_m": x_max,
                    "road_mid_y_m": road_mid_y, "radius_m": radius, "rsus": site_json},
                "contacts": {"path": contact_path.as_posix(), "sha256": sha256_file(contact_path)},
                "rsu_config": {"path": site_path.as_posix(), "sha256": sha256_file(site_path)},
                "selection": {"path": selection_path.as_posix(), "sha256": sha256_file(selection_path)},
                "counts": {"source_tracks": len(track_metadata), "selected_cars": len(vehicle_records),
                    "selected_points": sum(len(values) for values in points.values()), "contact_windows": len(all_windows),
                    "directions": dict(directions), "distinct_rsus_histogram": dict(Counter(row["distinct_rsus"] for row in vehicle_records)),
                    "excluded": excluded}})
    provenance = {"schema": "highd_small_v3_provenance", "adapter": "highd_linear_adjacent_frame_disk_v1",
        "archive": {"path": str(archive_path.resolve()), "sha256": archive_digest, "bytes": archive_path.stat().st_size},
        "protocol": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "preparation_code": {"script_sha256": sha256_file(Path(__file__)),
            "adapter_sha256": sha256_file(ROOT / "src/tacp_sim/highd.py")},
        "source_format_reference": "https://www.highd-dataset.com/format", "records": records,
        "preprocessing": {"time_origin": "each selected car's first measured frame", "time_stretch": False,
            "repeat_trajectories": False, "position": "bounding_box_center_metres",
            "interpolation": "linear only between adjacent integer frames; no extrapolation",
            "contact_end": "clipped to last measured sample; no added sample bin",
            "recording_coordinate_systems": "never combined", "source_license": "user-provided highD licensed archive; no public redistribution"},
        "total_selected_vehicles": sum(record["counts"]["selected_cars"] for record in records),
        "total_contact_windows": sum(record["counts"]["contact_windows"] for record in records)}
    write_json(output_dir / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/highd_small_v3.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/highd_small_v3")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/highd_small")
    args = parser.parse_args()
    result = prepare(args.archive, args.config, output_dir=args.output_dir, raw_dir=args.raw_dir)
    print(json.dumps({"vehicles": result["total_selected_vehicles"], "contacts": result["total_contact_windows"],
        "archive_sha256": result["archive"]["sha256"], "records": [{"id": record["recording_id"],
            "counts": record["counts"], "geometry": record["geometry"]} for record in result["records"]]}, indent=2))


if __name__ == "__main__":
    main()
