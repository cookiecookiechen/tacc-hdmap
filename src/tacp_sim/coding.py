from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import ObjectVersion, Shard


# GF(256), polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D).
# The octet arithmetic matches RFC 6330 Section 5.7; the B8 code is plain
# random linear coding, not an implementation of the RaptorQ construction.
_GF_EXP = [0] * 510
_GF_LOG = [0] * 256
_field_value = 1
for _exponent in range(255):
    _GF_EXP[_exponent] = _field_value
    _GF_LOG[_field_value] = _exponent
    _field_value <<= 1
    if _field_value & 256:
        _field_value ^= 0x11D
for _exponent in range(255, 510):
    _GF_EXP[_exponent] = _GF_EXP[_exponent - 255]


def gf256_multiply(left: int, right: int) -> int:
    """Multiply two octets using the RFC 6330 finite field."""
    if not 0 <= left <= 255 or not 0 <= right <= 255:
        raise ValueError("GF(256) operands must be octets")
    return _GF_EXP[_GF_LOG[left] + _GF_LOG[right]] if left and right else 0


def _shard_id(value: int | Shard | Mapping[str, Any]) -> int:
    if isinstance(value, Shard):
        return int(value.shard_id)
    if isinstance(value, Mapping):
        if "shard_id" not in value:
            raise ValueError("shard metadata must contain shard_id")
        return int(value["shard_id"])
    return int(value)


@dataclass
class RankTracker:
    """Rank tracker for ideal MDS shards or GF(256) RLNC symbols.

    The original simulator uses shard identities as an ideal MDS rank
    surrogate.  ``vector_mode`` adds a deterministic GF(256) basis so an RLNC
    baseline can carry genuinely dependent symbols without materialising
    payload bytes.
    """

    k: int
    _seen: set[int]
    _basis: dict[int, list[int]]
    vector_mode: bool

    def __init__(self, k: int, *, vector_mode: bool = False) -> None:
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = int(k)
        self._seen = set()
        self._basis = {}
        self.vector_mode = bool(vector_mode)

    @property
    def rank(self) -> int:
        if self.vector_mode:
            return min(self.k, len(self._basis))
        return min(self.k, len(self._seen))

    @property
    def deficit(self) -> int:
        return max(0, self.k - self.rank)

    @property
    def seen(self) -> frozenset[int]:
        return frozenset(self._seen)

    @property
    def decodable(self) -> bool:
        return self.rank >= self.k

    def _reduce_vector(self, vector: Iterable[int] | None) -> list[int]:
        if vector is None:
            raise ValueError("vector is required in vector_mode")
        values = list(vector)
        if len(values) != self.k or any(
            isinstance(value, bool) or int(value) != value or not 0 <= value <= 255
            for value in values
        ):
            raise ValueError("vector must contain exactly k GF(256) octets")
        reduced = [int(value) for value in values]
        for pivot, basis in sorted(self._basis.items()):
            factor = reduced[pivot]
            if factor:
                reduced = [value ^ gf256_multiply(factor, component)
                           for value, component in zip(reduced, basis)]
        return reduced

    def would_increase_rank(
        self, shard_id: int | Shard | Mapping[str, Any], *,
        vector: Iterable[int] | None = None,
    ) -> bool:
        """Inspect a known symbol header without changing receiver state."""
        sid = _shard_id(shard_id)
        if sid < 0:
            raise ValueError("shard_id must be non-negative")
        if self.decodable or sid in self._seen:
            return False
        if not self.vector_mode:
            return True
        return any(self._reduce_vector(vector))

    def observe(
        self,
        shard_id: int | Shard | Mapping[str, Any],
        *,
        linearly_independent: bool = True,
        vector: Iterable[int] | None = None,
    ) -> bool:
        shard_id = _shard_id(shard_id)
        if shard_id < 0:
            raise ValueError("shard_id must be non-negative")
        if not linearly_independent or self.decodable:
            return False
        if not self.vector_mode:
            if shard_id in self._seen:
                return False
            self._seen.add(shard_id)
            return True
        if shard_id in self._seen:
            return False
        reduced = self._reduce_vector(vector)
        self._seen.add(shard_id)
        if not any(reduced):
            return False
        pivot = next(index for index, value in enumerate(reduced) if value)
        inverse = _GF_EXP[255 - _GF_LOG[reduced[pivot]]]
        self._basis[pivot] = [gf256_multiply(value, inverse) for value in reduced]
        return True

    def observe_many(self, shard_ids: Iterable[int | Shard | Mapping[str, Any]]) -> int:
        return sum(self.observe(sid) for sid in shard_ids)

    def reset(self) -> None:
        self._seen.clear()
        self._basis.clear()


@dataclass(frozen=True)
class IdealMDSCode:
    """Metadata-only fixed ``(n, k)`` MDS code description.

    The encoder never materialises an object payload.  Any ``k`` distinct
    shard identifiers are considered linearly independent under this ideal
    model.
    """

    n: int
    k: int

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or isinstance(self.k, bool):
            raise ValueError("n and k must be positive integers")
        if int(self.n) != self.n or int(self.k) != self.k:
            raise ValueError("n and k must be positive integers")
        if self.n < 1 or self.k < 1 or self.k > self.n:
            raise ValueError("require 1 <= k <= n")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "k", int(self.k))

    @staticmethod
    def shard_size(object_bytes: int, k: int) -> int:
        if isinstance(object_bytes, bool) or int(object_bytes) != object_bytes or object_bytes <= 0:
            raise ValueError("object_bytes must be a positive integer")
        return (int(object_bytes) + int(k) - 1) // int(k)

    def encode_metadata(self, object_id: str, version: int, object_bytes: int) -> tuple[Shard, ...]:
        size = self.shard_size(object_bytes, self.k)
        return tuple(
            Shard(shard_id=sid, object_id=str(object_id), version=int(version), bytes=size)
            for sid in range(self.n)
        )

    def encode(
        self,
        obj: ObjectVersion | int | bytes | bytearray | memoryview,
        *,
        object_id: str = "object",
        version: int = 0,
    ) -> tuple[Shard, ...]:
        """Return metadata for an object description or payload length."""
        if isinstance(obj, ObjectVersion):
            if (obj.n, obj.k) != (self.n, self.k):
                raise ValueError("object coding parameters do not match this code")
            return self.encode_metadata(obj.object_id, obj.version, obj.size_bytes)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            object_bytes = len(obj)
        else:
            object_bytes = int(obj)
        return self.encode_metadata(object_id, version, object_bytes)

    def rank(self, shards: Iterable[int | Shard | Mapping[str, Any]]) -> int:
        tracker = RankTracker(self.k)
        tracker.observe_many(shards)
        return tracker.rank

    def can_decode(self, shards: Iterable[int | Shard | Mapping[str, Any]]) -> bool:
        return self.rank(shards) >= self.k


IdealMDSEncoder = IdealMDSCode


def ideal_mds_shards(n: int, object_id: str, version: int, shard_bytes: int):
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(shard_bytes, bool) or int(shard_bytes) != shard_bytes or shard_bytes <= 0:
        raise ValueError("shard_bytes must be a positive integer")
    return [
        {"shard_id": sid, "object_id": str(object_id), "version": int(version), "bytes": int(shard_bytes)}
        for sid in range(int(n))
    ]


def rank_of(shards: Iterable[int | Shard | Mapping[str, Any]], k: int) -> int:
    tracker = RankTracker(k)
    tracker.observe_many(shards)
    return tracker.rank


def reedsolo_roundtrip(payload: bytes, nsym: int = 4) -> bytes:
    """Small consistency helper; never used for event timing or cost."""
    try:
        from reedsolo import RSCodec
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError("reedsolo is required for the consistency helper") from exc
    codec = RSCodec(nsym)
    encoded = codec.encode(payload)
    return bytes(codec.decode(encoded)[0])
