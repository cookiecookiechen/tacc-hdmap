from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContactWindow:
    vehicle_id: str
    rsu_id: str
    start: float
    end: float
    rate_bps: float

    def __post_init__(self) -> None:
        values = (self.start, self.end, self.rate_bps)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("contact times and rate must be finite")
        if self.start < 0:
            raise ValueError("contact start must be non-negative")
        if self.end < self.start:
            raise ValueError("contact end must be >= start")
        if self.rate_bps <= 0:
            raise ValueError("contact rate must be positive")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def capacity_bytes(self) -> int:
        """Maximum whole bytes transferable during this window.

        Rates in the simulator are explicitly named ``rate_bps`` and are
        interpreted as bits per second.  Keeping this conversion on the
        window avoids repeating (and potentially disagreeing about) the
        units in placement and event code.
        """
        return max(0, math.floor(self.duration * self.rate_bps / 8.0))

    def bytes_for(self, duration_s: float | None = None) -> int:
        """Return whole bytes transferable in at most ``duration_s`` seconds."""
        duration = self.duration if duration_s is None else max(0.0, min(duration_s, self.duration))
        return max(0, math.floor(duration * self.rate_bps / 8.0))


@dataclass(frozen=True)
class ObjectVersion:
    object_id: str
    version: int
    size_bytes: int
    n: int
    k: int
    created_at: float = 0.0
    ttl_s: float = 10.0

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or self.n < 1 or self.k < 1 or self.k > self.n:
            raise ValueError("require size_bytes > 0 and 1 <= k <= n")
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if not math.isfinite(float(self.created_at)):
            raise ValueError("created_at must be finite")
        if not math.isfinite(float(self.ttl_s)) or self.ttl_s <= 0:
            raise ValueError("ttl_s must be positive and finite")

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_s

    @property
    def shard_bytes(self) -> int:
        return (self.size_bytes + self.k - 1) // self.k

    @property
    def padded_size_bytes(self) -> int:
        """Logical encoded payload size after zero padding to ``k`` blocks."""
        return self.shard_bytes * self.k

    @property
    def coded_size_bytes(self) -> int:
        """Storage occupied by all ``n`` coded shards."""
        return self.shard_bytes * self.n

    @property
    def shard_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n))

    def is_expired(self, now: float) -> bool:
        return float(now) >= self.expires_at


@dataclass(frozen=True)
class Shard:
    shard_id: int
    object_id: str
    version: int
    bytes: int

    def __post_init__(self) -> None:
        if self.shard_id < 0:
            raise ValueError("shard_id must be non-negative")
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if self.bytes <= 0:
            raise ValueError("shard bytes must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "object_id": self.object_id,
            "version": self.version,
            "bytes": self.bytes,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow metadata shards to be consumed by old dict-style callers."""
        return self.as_dict()[key]


@dataclass
class TransmissionRecord:
    source_rsu: str
    target_vehicle: str
    object_version: int
    shard_id: int
    start_time: float
    end_time: float
    bytes: int
    innovative: bool
    loss: bool
    failure_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_rsu": self.source_rsu,
            "target_vehicle": self.target_vehicle,
            "object_version": self.object_version,
            "shard_id": self.shard_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "bytes": self.bytes,
            "innovative": self.innovative,
            "loss": self.loss,
            "failure_reason": self.failure_reason,
        }


@dataclass
class EpisodeResult:
    config_hash: str
    seed: int
    vehicle_id: str
    object_id: str
    version: int
    path: list[str]
    contact_windows: list[dict[str, Any]]
    rank: int
    t_decode: float | None
    t_expire: float
    fresh_success: bool
    stale: bool
    storage: int
    backhaul: int
    migration: int
    repair: int
    verification: bool
    transmissions: list[TransmissionRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "vehicle_id": self.vehicle_id,
            "object_id": self.object_id,
            "version": self.version,
            "path": self.path,
            "contact_windows": self.contact_windows,
            "rank": self.rank,
            "T_decode": self.t_decode,
            "T_expire": self.t_expire,
            "fresh_success": self.fresh_success,
            "stale": self.stale,
            "storage": self.storage,
            "backhaul": self.backhaul,
            "migration": self.migration,
            "repair": self.repair,
            "verification": self.verification,
            "transmissions": [t.as_dict() for t in self.transmissions],
            "events": self.events,
        }
