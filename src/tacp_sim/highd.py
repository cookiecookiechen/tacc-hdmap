"""highD local-metre trajectories and exact linear-segment contact geometry.

Each recording has its own coordinates.  This adapter never extrapolates a
track, fills a missing frame, repeats a trip, or rescales its time axis.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .mobility import RsuSite
from .models import ContactWindow


@dataclass(frozen=True)
class HighDPoint:
    frame: int
    x: float
    y: float

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or int(self.frame) != self.frame or self.frame < 1:
            raise ValueError("highD frame must be a positive integer")
        if not all(math.isfinite(float(value)) for value in (self.x, self.y)):
            raise ValueError("highD coordinates must be finite")


def vehicle_key(recording: str | int, track_id: str | int) -> str:
    if int(recording) < 1 or int(track_id) < 1:
        raise ValueError("recording and track ID must be positive")
    return f"highd_{int(recording):02d}_vehicle_{int(track_id)}"


def point_from_track_row(row: dict[str, str]) -> HighDPoint:
    """Use the bounding-box centre; highD x/y describe its upper-left corner."""
    width, height = float(row["width"]), float(row["height"])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("highD bounding-box dimensions must be positive and finite")
    return HighDPoint(int(row["frame"]), float(row["x"]) + width / 2, float(row["y"]) + height / 2)


def segment_circle_interval(p0: HighDPoint, p1: HighDPoint, site: RsuSite) -> tuple[float, float] | None:
    """Return the positive-length fraction of a straight segment inside a disk.

    A tangential point has no transmission duration and produces no interval.
    The interval includes disk boundaries and is clipped to the measured segment.
    """
    dx, dy = p1.x - p0.x, p1.y - p0.y
    ox, oy = p0.x - site.x, p0.y - site.y
    a = dx * dx + dy * dy
    c = ox * ox + oy * oy - site.coverage_radius_m**2
    if a == 0:
        return (0.0, 1.0) if c <= 1e-12 else None
    b = 2 * (ox * dx + oy * dy)
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        return None
    root = math.sqrt(discriminant)
    start = max(0.0, (-b - root) / (2 * a))
    end = min(1.0, (-b + root) / (2 * a))
    return (start, end) if end > start + 1e-12 else None


def highd_contact_windows(
    points: Iterable[HighDPoint], sites: Sequence[RsuSite], *, vehicle_id: str,
    frame_rate: float = 25.0, rate_bps: float = 8_000_000.0,
) -> list[ContactWindow]:
    """Convert adjacent recorded frames to windows relative to first observation.

    Missing frames split a trajectory: interpolation is allowed only when frame
    numbers differ by exactly one. Non-overlapping RSU disks prevent simultaneous
    associations. A stationary sample pair inside a disk remains a valid contact.
    """
    if not vehicle_id:
        raise ValueError("vehicle_id must not be empty")
    if not math.isfinite(frame_rate) or frame_rate <= 0 or not math.isfinite(rate_bps) or rate_bps <= 0:
        raise ValueError("frame rate and transmission rate must be positive and finite")
    if not sites or len({site.rsu_id for site in sites}) != len(sites):
        raise ValueError("sites must be nonempty and have unique IDs")
    for index, left in enumerate(sites):
        for right in sites[index + 1:]:
            if math.hypot(left.x - right.x, left.y - right.y) < left.coverage_radius_m + right.coverage_radius_m - 1e-9:
                raise ValueError("highD RSU coverage disks must not overlap")
    ordered = list(points)
    if not ordered:
        return []
    if any(right.frame <= left.frame for left, right in zip(ordered, ordered[1:])):
        raise ValueError("track frames must be strictly increasing")
    first_frame = ordered[0].frame
    fragments: list[ContactWindow] = []
    for left, right in zip(ordered, ordered[1:]):
        if right.frame != left.frame + 1:
            continue
        for site in sites:
            interval = segment_circle_interval(left, right, site)
            if interval is None:
                continue
            start, end = interval
            fragments.append(ContactWindow(vehicle_id, site.rsu_id,
                (left.frame - first_frame + start) / frame_rate,
                (left.frame - first_frame + end) / frame_rate, rate_bps))
    merged: list[ContactWindow] = []
    for window in sorted(fragments, key=lambda item: (item.start, item.end, item.rsu_id)):
        if merged and window.rsu_id == merged[-1].rsu_id and abs(window.start - merged[-1].end) <= 1e-10:
            previous = merged[-1]
            merged[-1] = ContactWindow(vehicle_id, window.rsu_id, previous.start, window.end, rate_bps)
        else:
            merged.append(window)
    return merged


CONTACT_FIELDS = ("vehicle_id", "rsu_id", "start", "end", "rate_bps")


def write_contact_windows_csv(path: str | Path, windows: Iterable[ContactWindow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for window in windows:
            writer.writerow({field: getattr(window, field) for field in CONTACT_FIELDS})


def load_contact_windows_csv(path: str | Path) -> list[ContactWindow]:
    """Load exact interval CSVs, without the sample-bin extension of csv_trace."""
    windows: list[ContactWindow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(CONTACT_FIELDS):
            raise ValueError("highD contact CSV has unexpected columns")
        for row in reader:
            if not row["vehicle_id"] or not row["rsu_id"]:
                raise ValueError("highD contact IDs must not be empty")
            window = ContactWindow(row["vehicle_id"], row["rsu_id"], float(row["start"]), float(row["end"]), float(row["rate_bps"]))
            if window.duration <= 0:
                raise ValueError("highD contact interval must have positive duration")
            windows.append(window)
    return sorted(windows, key=lambda item: (item.vehicle_id, item.start, item.rsu_id))
