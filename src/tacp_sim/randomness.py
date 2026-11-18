"""Stateless common random numbers for independently simulated trips.

``keyed_v3`` gives each trip/episode its own channel and RSU-failure scenario.
An RSU failure is constant within that trip, not a network-wide outage shared
by vehicles at the same wall-clock time: this simulator runs isolated trips.
Packet outcomes are paired across methods by contact and actual transmission
attempt ordinal. Different packet sizes/policies can use different airtimes,
so ordinal pairing is a variance-reduction device, not a physical channel trace.

The key deliberately excludes method, coding family, shard and parameter-cell
identifiers. Adding an unrelated RSU or skipping a request cannot consume or
shift a later contact's random sequence. The legacy stateful generator remains
available in ``Simulation`` for reproducing already frozen experiments.
"""
from __future__ import annotations

import hashlib
import json


KEY_NAMESPACE = "tacp_sim.keyed_v3"


def keyed_uniform(
    seed: int,
    trip_id: str,
    episode_index: int,
    stream: str,
    **event_fields: object,
) -> float:
    """Return a reproducible uniform variate in [0, 1) from a named event.

    Canonical UTF-8 JSON fixes field ordering and rejects non-finite numbers.
    The first 53 digest bits map exactly onto binary64's uniform grid; dividing
    a full 64-bit integer could instead round a value just below one to one.
    Callers normalize physical times to float before using them in event keys.
    """
    if not str(trip_id) or not str(stream):
        raise ValueError("trip_id and stream must be non-empty")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if (not isinstance(episode_index, int) or isinstance(episode_index, bool)
            or episode_index < 0):
        raise ValueError("episode_index must be a non-negative integer")
    key = {
        "namespace": KEY_NAMESPACE,
        "seed": seed,
        "trip_id": str(trip_id),
        "episode_index": episode_index,
        "stream": str(stream),
        "event": event_fields,
    }
    encoded = json.dumps(key, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / float(1 << 53)
