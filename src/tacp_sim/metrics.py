from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import EpisodeResult


def flatten_episode(result: EpisodeResult) -> dict:
    row = result.as_dict()
    row["path"] = json.dumps(row["path"], sort_keys=True)
    row["contact_windows"] = json.dumps(row["contact_windows"], sort_keys=True)
    row["transmissions"] = json.dumps(row["transmissions"], sort_keys=True)
    row["events"] = json.dumps(row["events"], sort_keys=True)
    return row


def append_raw(results: Iterable[EpisodeResult], path: str | Path) -> None:
    rows = [flatten_episode(r) for r in results]
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    exists = target.exists()
    with target.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(results: Iterable[EpisodeResult]) -> dict:
    rows = list(results)
    successes = sum(bool(r.fresh_success) for r in rows)
    return {"episodes": len(rows), "fresh_successes": successes, "P_fresh": successes / len(rows) if rows else 0.0}
