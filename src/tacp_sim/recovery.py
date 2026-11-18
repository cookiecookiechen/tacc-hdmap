from __future__ import annotations

"""Rank-aware earliest-deadline recovery (manuscript Algorithm 2).

The real-time simulator can consume the legacy boolean deadline gate below.
The planning API in this module ranks candidate RSU/repair actions using the
probability of closing the current rank deficit before expiry and explicitly
filters shards that are already innovative in the decoder state.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_EPS = 1e-12


def _finite(value: Any, name: str, *, nonnegative: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


def algorithm2_should_request(*, rank: int, k: int, now: float, expire_at: float, expected_completion: float) -> bool:
    """Return whether a request can plausibly close the rank deficit before TTL."""
    rank = _int(rank, "rank")
    k = _int(k, "k")
    now = _finite(now, "now")
    expire_at = _finite(expire_at, "expire_at")
    expected_completion = _finite(expected_completion, "expected_completion")
    if rank < 0 or k < 1 or expire_at < now:
        return False
    return rank < k and now + expected_completion <= expire_at + _EPS


def rank_deficit_transmission_expectation(deficit: int, packet_loss: float) -> float:
    """Expected transmissions for ``deficit`` successes under iid erasures."""
    deficit = _int(deficit, "deficit")
    packet_loss = _finite(packet_loss, "packet_loss")
    if deficit < 0:
        raise ValueError("deficit must be non-negative")
    if not 0 <= packet_loss < 1:
        raise ValueError("packet_loss must be in [0,1)")
    return deficit / (1.0 - packet_loss)


@dataclass(frozen=True)
class RecoveryAction:
    """A reachable RSU or repair source candidate.

    An action may expose several coded shard ids; the planner chooses only the
    first id not already present in the decoder rank. ``bytes`` is the payload
    size for one requested coded packet. ``start_time`` is the earliest time a
    request can begin.
    """

    source_rsu: str
    shard_ids: tuple[int, ...] = ()
    bytes: int = 1
    rate_bps: float = 1.0
    start_time: float = 0.0
    success_probability: float = 1.0
    repair_cost_bytes: int = 0
    is_repair: bool = False
    version: int | None = None

    def __post_init__(self) -> None:
        if not str(self.source_rsu):
            raise ValueError("source_rsu must not be empty")
        object.__setattr__(self, "source_rsu", str(self.source_rsu))
        ids = tuple(_int(sid, "shard_id") for sid in self.shard_ids)
        if any(sid < 0 for sid in ids):
            raise ValueError("shard ids must be non-negative")
        object.__setattr__(self, "shard_ids", tuple(dict.fromkeys(ids)))
        object.__setattr__(self, "bytes", _int(self.bytes, "bytes"))
        if self.bytes <= 0:
            raise ValueError("bytes must be positive")
        object.__setattr__(self, "rate_bps", _finite(self.rate_bps, "rate_bps"))
        if self.rate_bps <= 0:
            raise ValueError("rate_bps must be positive")
        object.__setattr__(self, "start_time", _finite(self.start_time, "start_time"))
        probability = _finite(self.success_probability, "success_probability")
        if probability > 1:
            raise ValueError("success_probability must be in [0,1]")
        object.__setattr__(self, "success_probability", probability)
        object.__setattr__(self, "repair_cost_bytes", _int(self.repair_cost_bytes, "repair_cost_bytes"))
        if self.repair_cost_bytes < 0:
            raise ValueError("repair_cost_bytes must be non-negative")
        if self.version is not None:
            object.__setattr__(self, "version", _int(self.version, "version"))
            if self.version < 0:
                raise ValueError("version must be non-negative")

    @property
    def shard_id(self) -> int | None:
        return self.shard_ids[0] if self.shard_ids else None

    @property
    def completion_time_s(self) -> float:
        return self.bytes * 8.0 / self.rate_bps

    @property
    def cost_bytes(self) -> int:
        return self.bytes + self.repair_cost_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_rsu": self.source_rsu,
            "shard_ids": list(self.shard_ids),
            "shard_id": self.shard_id,
            "bytes": self.bytes,
            "rate_bps": self.rate_bps,
            "start_time": self.start_time,
            "success_probability": self.success_probability,
            "repair_cost_bytes": self.repair_cost_bytes,
            "is_repair": self.is_repair,
            "version": self.version,
        }


def _coerce_action(action: RecoveryAction | Mapping[str, Any]) -> RecoveryAction:
    if isinstance(action, RecoveryAction):
        return action
    if not isinstance(action, Mapping):
        raise TypeError("recovery candidates must be RecoveryAction or mapping")
    raw_ids = action.get("shard_ids", action.get("shards"))
    if raw_ids is None:
        sid = action.get("shard_id", action.get("shard"))
        raw_ids = () if sid is None else (sid,)
    if isinstance(raw_ids, (int, str)):
        raw_ids = (raw_ids,)
    return RecoveryAction(
        source_rsu=str(action.get("source_rsu", action.get("source", action.get("rsu_id", "")))),
        shard_ids=tuple(raw_ids),
        bytes=action.get("bytes", action.get("bytes_per_shard", 1)),
        rate_bps=action.get("rate_bps", action.get("rate", 1.0)),
        start_time=action.get("start_time", action.get("available_at", 0.0)),
        success_probability=action.get(
            "success_probability", action.get("packet_success_probability", action.get("q", 1.0))
        ),
        repair_cost_bytes=action.get("repair_cost_bytes", action.get("repair_cost", 0)),
        is_repair=action.get("is_repair", action.get("repair", False)),
        version=action.get("version"),
    )


def _seen_ids(current: Any) -> set[int]:
    if current is None:
        return set()
    if hasattr(current, "seen"):
        return {int(sid) for sid in current.seen}
    if isinstance(current, int):
        # A bare rank conveys no shard identity; use an empty set and let the
        # caller supply ``seen_shards`` when duplicate filtering matters.
        return set()
    return {int(sid.shard_id if hasattr(sid, "shard_id") else sid) for sid in current}


@dataclass(frozen=True)
class RecoveryActionScore:
    action: RecoveryAction
    shard_id: int | None
    start_time: float
    completion_time: float
    innovative: bool
    feasible_before_expiry: bool
    close_probability: float
    utility: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = self.action.as_dict()
        result.update({
            "selected_shard_id": self.shard_id,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "innovative": self.innovative,
            "feasible_before_expiry": self.feasible_before_expiry,
            "close_probability": self.close_probability,
            "utility": self.utility,
            "reason": self.reason,
        })
        return result


def _candidate_score(
    action: RecoveryAction,
    *,
    seen: set[int],
    rank: int,
    k: int,
    now: float,
    expire_at: float,
    lambda_repair: float,
) -> RecoveryActionScore:
    shard = next((sid for sid in action.shard_ids if sid not in seen), None)
    start = max(now, action.start_time)
    completion = start + action.completion_time_s
    innovative = shard is not None and rank < k
    feasible = innovative and completion <= expire_at + _EPS
    if not innovative:
        return RecoveryActionScore(action, shard, start, completion, False, False, 0.0, 0.0, "duplicate_or_dependent")
    if not feasible:
        return RecoveryActionScore(action, shard, start, completion, True, False, 0.0, 0.0, "deadline")
    # One request carries one innovative shard.  This is the exact numerator
    # for a single-packet action in the manuscript's U(a|t) expression.
    probability = min(1.0, max(0.0, action.success_probability))
    denominator = action.bytes + lambda_repair * action.repair_cost_bytes
    utility = probability / max(_EPS, float(denominator))
    if probability <= _EPS:
        return RecoveryActionScore(action, shard, start, completion, True, False, 0.0, 0.0, "zero_success_probability")
    return RecoveryActionScore(action, shard, start, completion, True, True, probability, utility, "")


def rank_recovery_actions(
    candidates: Iterable[RecoveryAction | Mapping[str, Any]],
    *,
    rank: int | None = None,
    k: int,
    now: float,
    expire_at: float,
    seen_shards: Iterable[int] | None = None,
    lambda_repair: float = 1.0,
) -> list[RecoveryActionScore]:
    """Score candidates by deadline-aware innovative-shard utility.

    Results are sorted in descending utility, then earliest completion time and
    stable source/shard ids.  Infeasible/duplicate actions are retained at the
    end for diagnostics and event logging.
    """
    k = _int(k, "k")
    if k < 1:
        raise ValueError("k must be positive")
    now = _finite(now, "now")
    expire_at = _finite(expire_at, "expire_at")
    lambda_repair = _finite(lambda_repair, "lambda_repair")
    if rank is None:
        rank = len(set(int(sid) for sid in (seen_shards or ())))
    rank = _int(rank, "rank")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    seen = {int(sid) for sid in (seen_shards or ())}
    scores = [_candidate_score(_coerce_action(action), seen=seen, rank=rank, k=k,
                               now=now, expire_at=expire_at, lambda_repair=lambda_repair)
              for action in candidates]
    return sorted(scores, key=lambda item: (
        -item.utility if item.feasible_before_expiry else 0.0,
        item.completion_time,
        item.action.source_rsu,
        item.shard_id if item.shard_id is not None else math.inf,
    ))


def select_recovery_action(
    candidates: Iterable[RecoveryAction | Mapping[str, Any]],
    *,
    rank: int,
    k: int,
    now: float,
    expire_at: float,
    seen_shards: Iterable[int] = (),
    lambda_repair: float = 1.0,
) -> RecoveryAction | None:
    """Return the highest-utility feasible innovative action, if one exists."""
    scores = rank_recovery_actions(candidates, rank=rank, k=k, now=now,
                                   expire_at=expire_at, seen_shards=seen_shards,
                                   lambda_repair=lambda_repair)
    return next((score.action for score in scores
                 if score.feasible_before_expiry and score.close_probability > _EPS), None)


@dataclass
class RecoveryResult:
    """Deterministic successful-path plan and probability diagnostics."""

    selected: list[RecoveryActionScore] = field(default_factory=list)
    rank: int = 0
    k: int = 0
    start_time: float = 0.0
    completion_time: float | None = None
    expire_at: float = 0.0
    fresh_probability: float = 0.0
    fresh_success: bool = False
    deadline_miss: bool = False
    bytes: int = 0
    repair_bytes: int = 0
    duplicate_actions: int = 0

    @property
    def actions(self) -> list[RecoveryAction]:
        return [score.action for score in self.selected]

    @property
    def requested_shards(self) -> list[int]:
        return [int(score.shard_id) for score in self.selected if score.shard_id is not None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "k": self.k,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "expire_at": self.expire_at,
            "fresh_probability": self.fresh_probability,
            "fresh_success": self.fresh_success,
            "deadline_miss": self.deadline_miss,
            "bytes": self.bytes,
            "repair_bytes": self.repair_bytes,
            "duplicate_actions": self.duplicate_actions,
            "selected": [score.as_dict() for score in self.selected],
        }


def algorithm2_rank_aware_recovery(
    candidates: Iterable[RecoveryAction | Mapping[str, Any]],
    *,
    k: int,
    rank: int | None = None,
    now: float = 0.0,
    expire_at: float,
    seen_shards: Iterable[int] = (),
    lambda_repair: float = 1.0,
    target_probability: float | None = None,
    current_version: int | None = None,
) -> RecoveryResult:
    """Execute Algorithm 2's earliest-deadline, rank-aware action loop.

    The plan follows the successful innovative path while carrying the product
    of action success probabilities as ``fresh_probability``.  This keeps the
    planner deterministic and lets the event simulator sample packet losses at
    execution time.  Version filtering is applied before ranking when a
    ``current_version`` is supplied.
    """
    k = _int(k, "k")
    if k < 1:
        raise ValueError("k must be positive")
    now = _finite(now, "now")
    expire_at = _finite(expire_at, "expire_at")
    if expire_at < now:
        return RecoveryResult(rank=0, k=k, start_time=now, expire_at=expire_at, deadline_miss=True)
    seen = {int(sid) for sid in seen_shards}
    initial_seen = set(seen)
    planned_rank = max(len(seen), int(rank) if rank is not None else len(seen))
    if planned_rank > k:
        planned_rank = k
    remaining = []
    all_actions: list[RecoveryAction] = []
    for raw in candidates:
        action = _coerce_action(raw)
        all_actions.append(action)
        if current_version is not None and action.version is not None and action.version != current_version:
            continue
        remaining.append(action)
    result = RecoveryResult(rank=planned_rank, k=k, start_time=now, expire_at=expire_at)
    probability = 1.0
    current_time = now
    while planned_rank < k:
        scores = rank_recovery_actions(remaining, rank=planned_rank, k=k, now=current_time,
                                       expire_at=expire_at, seen_shards=seen,
                                       lambda_repair=lambda_repair)
        chosen = next((score for score in scores
                       if score.feasible_before_expiry and score.close_probability > _EPS), None)
        if chosen is None:
            result.deadline_miss = planned_rank < k
            break
        result.selected.append(chosen)
        remaining.remove(chosen.action)
        seen.add(int(chosen.shard_id))
        planned_rank += 1
        probability *= chosen.close_probability
        current_time = chosen.completion_time
        result.bytes += chosen.action.bytes
        if chosen.action.is_repair:
            result.repair_bytes += chosen.action.repair_cost_bytes or chosen.action.bytes
        # A candidate carrying several ids is still requested one packet at a
        # time, so only the selected innovative id enters rank.
    result.rank = planned_rank
    result.completion_time = current_time if result.selected else None
    result.fresh_probability = probability if planned_rank >= k else 0.0
    result.fresh_success = bool(planned_rank >= k and result.completion_time is not None
                                and result.completion_time <= expire_at + _EPS)
    if target_probability is not None:
        target_probability = _finite(target_probability, "target_probability")
        if target_probability > 1:
            raise ValueError("target_probability must be in [0,1]")
        if result.fresh_probability < target_probability:
            result.deadline_miss = True
    result.duplicate_actions = sum(
        1 for action in all_actions
        if not any(shard_id not in initial_seen for shard_id in action.shard_ids)
    )
    return result


# Aliases make the policy easy to discover from notebooks and preserve possible
# naming used by early experiment scripts.
rank_aware_recovery = algorithm2_rank_aware_recovery
algorithm2_recovery = algorithm2_rank_aware_recovery
recover_rank_aware = algorithm2_rank_aware_recovery
