from __future__ import annotations

import hashlib


def metadata_leaf(*, object_id: str, version: int, shard_id: int, rsu_id: str, expires_at: float, shard_hash: str) -> str:
    material = f"{object_id}|{version}|{shard_id}|{rsu_id}|{expires_at:.9f}|{shard_hash}".encode()
    return hashlib.sha256(material).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def verify_metadata(*, expected_object_id: str, expected_version: int, object_id: str, version: int, shard_id: int, rsu_id: str, expires_at: float, shard_hash: str, leaf_digest: str, merkle_root_digest: str | None = None, current_time: float = 0.0) -> bool:
    # TTL is half-open in the event model: the object is expired at the
    # deadline itself, so a proof arriving at ``current_time == expires_at``
    # must not be accepted.
    if object_id != expected_object_id or version != expected_version or shard_id < 0 or not rsu_id or current_time >= expires_at:
        return False
    try:
        leaf = metadata_leaf(object_id=object_id, version=version, shard_id=shard_id, rsu_id=rsu_id, expires_at=expires_at, shard_hash=shard_hash)
        if leaf != leaf_digest:
            return False
        return merkle_root([leaf]) == merkle_root_digest if merkle_root_digest is not None else True
    except (TypeError, ValueError):
        return False
