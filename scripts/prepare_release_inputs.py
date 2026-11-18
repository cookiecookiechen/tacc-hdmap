#!/usr/bin/env python3
"""Verify the included Porto inputs and prepare highD from a licensed archive.

The public protocol identifies this source release. Historical experiment
identities in results/reference and the published raw records are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL_PATH = Path("configs/related_work_v3_2.yaml")
HIGHD_DIR = Path("data/processed/highd_small_v3")
HIGHD_CONFIG = Path("configs/highd_small_v3.yaml")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_porto(protocol: dict, *, root: Path = ROOT) -> None:
    for relative, expected in protocol["datasets"]["porto"]["source_hashes"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing included Porto input: {relative}")
        if file_hash(path) != expected:
            raise ValueError(f"Porto input checksum mismatch: {relative}")


def highd_expected_files(reference: dict) -> dict[str, str]:
    expected = {
        Path(relative).name: digest
        for relative, digest in reference["datasets"]["highd"]["source_hashes"].items()
        if Path(relative).parent == HIGHD_DIR and Path(relative).name != "provenance.json"
    }
    required = {f"{recording}_{suffix}" for recording in ("01", "02", "03")
                for suffix in ("contacts.csv", "rsus.json", "selection.json")}
    if set(expected) != required:
        raise ValueError("Historical highD reference must register all nine contact, RSU and selection files")
    return expected


def verify_highd_files(folder: Path, expected: dict[str, str]) -> None:
    for name, digest in sorted(expected.items()):
        path = folder / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Prepared highD input differs from the published experiment: {name}")


def portable_provenance(provenance: dict) -> dict:
    """Use repository-relative source labels instead of workstation paths."""
    result = json.loads(json.dumps(provenance))
    result["archive"]["path"] = "data/external/highD-dataset-v1.0.zip"
    result["protocol"]["path"] = HIGHD_CONFIG.as_posix()
    for record in result["records"]:
        rec = record["recording_id"]
        record["selected_raw"]["path"] = f"data/raw/highd_small/{rec}_selected_tracks.csv"
        for key, suffix in (("contacts", "contacts.csv"), ("rsu_config", "rsus.json"),
                            ("selection", "selection.json")):
            record[key]["path"] = (HIGHD_DIR / f"{rec}_{suffix}").as_posix()
    result["release_preparation"] = {
        "source_labels": "Repository-relative labels; the licensed source archive remains external.",
        "selected_raw_retained": False,
        "identity": "Contact, RSU and selection bytes match the historical input checksums; provenance is regenerated."
    }
    return result


def prepare_highd(archive: Path, *, root: Path = ROOT) -> dict:
    protocol_path = root / PROTOCOL_PATH
    protocol = read_json(protocol_path)
    verify_porto(protocol, root=root)
    release = protocol["release"]
    reference_path = root / release["historical_protocol_path"]
    if file_hash(reference_path) != release["historical_protocol_sha256"]:
        raise ValueError("Historical reference protocol checksum mismatch")
    reference = read_json(reference_path)
    expected = highd_expected_files(reference)
    config_path = root / HIGHD_CONFIG
    expected_config_hash = reference["datasets"]["highd"]["source_hashes"][HIGHD_CONFIG.as_posix()]
    if file_hash(config_path) != expected_config_hash:
        raise ValueError("highD selection or simulation configuration differs from the historical reference")
    destination = root / HIGHD_DIR
    if destination.exists() and any(destination.iterdir()):
        verify_highd_files(destination, expected)
        provenance_path = destination / "provenance.json"
        if not provenance_path.is_file():
            raise ValueError("Existing highD inputs have no provenance; use an empty working copy")
        provenance = read_json(provenance_path)
        if file_hash(archive) != provenance["archive"]["sha256"]:
            raise ValueError("The supplied highD archive differs from the prepared archive")
        sanitized = portable_provenance(provenance)
        if sanitized != provenance:
            raise ValueError("Existing highD provenance is not in the public release format")
    else:
        from scripts.prepare_highd_small import prepare

        with tempfile.TemporaryDirectory(prefix="trajectory-highd-") as temporary:
            stage = Path(temporary) / "processed"
            provenance = prepare(archive, config_path, output_dir=stage,
                                 raw_dir=Path(temporary) / "selected_raw")
            verify_highd_files(stage, expected)
            provenance = portable_provenance(provenance)
            write_json(stage / "provenance.json", provenance)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, destination, dirs_exist_ok=True)
    registered = {HIGHD_CONFIG.as_posix(): file_hash(config_path)}
    registered.update({(HIGHD_DIR / name).as_posix(): file_hash(destination / name)
                       for name in sorted([*expected, "provenance.json"])})
    protocol["datasets"]["highd"]["source_hashes"] = registered
    protocol["release"]["highd_inputs"] = "prepared_from_licensed_archive"
    temporary_protocol = protocol_path.with_suffix(".yaml.tmp")
    write_json(temporary_protocol, protocol)
    temporary_protocol.replace(protocol_path)
    return {"porto": "verified", "highd": "prepared", "vehicles": provenance["total_selected_vehicles"],
            "contact_windows": provenance["total_contact_windows"], "historical_input_files_matched": len(expected),
            "protocol": PROTOCOL_PATH.as_posix(),
            "note": "Public reproduction identity updated. Run dry-run and validation gates with this protocol."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--highd-archive", type=Path,
                        help="Licensed highD v1.0 ZIP; never copied into the release")
    args = parser.parse_args()
    if args.highd_archive is None:
        protocol = read_json(ROOT / PROTOCOL_PATH)
        verify_porto(protocol)
        result = {"porto": "verified", "highd": protocol["release"]["highd_inputs"],
                  "note": "Use --highd-archive with a licensed ZIP to prepare the highway supplement."}
    else:
        result = prepare_highd(args.highd_archive)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
