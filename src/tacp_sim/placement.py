from __future__ import annotations

"""Chance-constrained coded placement (Algorithm 1).

The production simulator only needs a deterministic mapping from RSU id to
shard ids.  This module also exposes the planning objects used by small
experiments: each candidate action has an estimated success probability and
explicit storage/backhaul/migration costs.  Probabilities are evaluated with
an exact finite-state rank DP, so no Monte Carlo noise is introduced into the
placement heuristic or the small oracle.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .models import ContactWindow, ObjectVersion


_EPS = 1e-12


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class PlacementAction:
    """One candidate ``(rsu, shard, version)`` placement action.

    ``visit_probability`` and ``packet_success_probability`` are probabilities
    conditional on the candidate being used.  ``success_probability`` may be
    supplied directly when a caller already has an aggregate route estimate.
    Costs are measured in bytes (or byte-normalized units) and are intentionally
    kept separate so each budget can be enforced independently.
    """

    rsu_id: str
    shard_id: int
    version: int = 0
    visit_probability: float = 1.0
    packet_success_probability: float = 1.0
    success_probability: float | None = None
    storage_bytes: int = 0
    backhaul_bytes: int = 0
    migration_bytes: int = 0
    completion_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.rsu_id):
            raise ValueError("rsu_id must not be empty")
        object.__setattr__(self, "rsu_id", str(self.rsu_id))
        object.__setattr__(self, "shard_id", _integer(self.shard_id, "shard_id"))
        object.__setattr__(self, "version", _integer(self.version, "version"))
        visit = _number(self.visit_probability, "visit_probability")
        packet = _number(self.packet_success_probability, "packet_success_probability")
        if visit > 1 or packet > 1:
            raise ValueError("probabilities must be in [0,1]")
        object.__setattr__(self, "visit_probability", visit)
        object.__setattr__(self, "packet_success_probability", packet)
        if self.success_probability is not None:
            probability = _number(self.success_probability, "success_probability")
            if probability > 1:
                raise ValueError("success_probability must be in [0,1]")
            object.__setattr__(self, "success_probability", probability)
        for name in ("storage_bytes", "backhaul_bytes", "migration_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "completion_time_s", _number(self.completion_time_s, "completion_time_s"))

    @property
    def probability(self) -> float:
        if self.success_probability is not None:
            return float(self.success_probability)
        return float(self.visit_probability * self.packet_success_probability)

    @property
    def cost(self) -> int:
        return self.storage_bytes + self.backhaul_bytes + self.migration_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "rsu_id": self.rsu_id,
            "shard_id": self.shard_id,
            "version": self.version,
            "visit_probability": self.visit_probability,
            "packet_success_probability": self.packet_success_probability,
            "success_probability": self.probability,
            "storage_bytes": self.storage_bytes,
            "backhaul_bytes": self.backhaul_bytes,
            "migration_bytes": self.migration_bytes,
            "completion_time_s": self.completion_time_s,
        }


@dataclass
class PlacementResult:
    """Detailed result of a greedy or exhaustive placement plan."""

    selected: list[PlacementAction]
    expected_rank: float
    fresh_probability: float
    storage_bytes: int
    backhaul_bytes: int
    migration_bytes: int
    method: str = "greedy"
    target_probability: float | None = None
    capacity_bytes: int | dict[str, int] | None = None
    backhaul_budget_bytes: int | None = None
    migration_budget_bytes: int | None = None

    @property
    def placement(self) -> dict[str, list[int]]:
        return placement_mapping(self.selected)

    @property
    def mapping(self) -> dict[str, list[int]]:
        return self.placement

    @property
    def objective(self) -> float:
        return self.fresh_probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "placement": self.placement,
            "selected": [a.as_dict() for a in self.selected],
            "expected_rank": self.expected_rank,
            "fresh_probability": self.fresh_probability,
            "storage_bytes": self.storage_bytes,
            "backhaul_bytes": self.backhaul_bytes,
            "migration_bytes": self.migration_bytes,
            "target_probability": self.target_probability,
            "capacity_bytes": self.capacity_bytes,
            "backhaul_budget_bytes": self.backhaul_budget_bytes,
            "migration_budget_bytes": self.migration_budget_bytes,
        }


def _coerce_action(action: PlacementAction | Mapping[str, Any]) -> PlacementAction:
    if isinstance(action, PlacementAction):
        return action
    if not isinstance(action, Mapping):
        raise TypeError("placement actions must be PlacementAction or mapping")
    return PlacementAction(
        rsu_id=str(action.get("rsu_id", action.get("rsu", ""))),
        shard_id=action.get("shard_id", action.get("shard", 0)),
        version=action.get("version", 0),
        visit_probability=action.get("visit_probability", action.get("p_visit", 1.0)),
        packet_success_probability=action.get(
            "packet_success_probability", action.get("q", action.get("success_probability", 1.0))
        ),
        success_probability=action.get("aggregate_success_probability"),
        storage_bytes=action.get("storage_bytes", action.get("storage", 0)),
        backhaul_bytes=action.get("backhaul_bytes", action.get("backhaul", 0)),
        migration_bytes=action.get("migration_bytes", action.get("migration", 0)),
        completion_time_s=action.get("completion_time_s", action.get("completion_time", 0.0)),
    )


def placement_mapping(actions: Iterable[PlacementAction | Mapping[str, Any]]) -> dict[str, list[int]]:
    """Convert selected actions into the simulator's legacy placement mapping."""
    mapping: dict[str, list[int]] = {}
    for raw in actions:
        action = _coerce_action(raw)
        mapping.setdefault(action.rsu_id, []).append(action.shard_id)
    for rsu, shard_ids in mapping.items():
        mapping[rsu] = sorted(dict.fromkeys(shard_ids))
    return dict(sorted(mapping.items()))


def expected_capped_rank(
    actions: Iterable[PlacementAction | Mapping[str, Any]], k: int
) -> float:
    """Return ``E[min(k, rank)]`` under independent action successes."""
    k = _integer(k, "k", nonnegative=False)
    if k < 1:
        raise ValueError("k must be positive")
    # A frozenset state works for n larger than machine-word size and naturally
    # handles duplicate actions for the same coded shard.
    states: dict[frozenset[int], float] = {frozenset(): 1.0}
    for raw in actions:
        action = _coerce_action(raw)
        p = min(1.0, max(0.0, action.probability))
        updated: dict[frozenset[int], float] = {}
        for seen, probability in states.items():
            updated[seen] = updated.get(seen, 0.0) + probability * (1.0 - p)
            if len(seen) < k:
                next_seen = seen | {action.shard_id}
                updated[next_seen] = updated.get(next_seen, 0.0) + probability * p
            else:
                updated[seen] = updated.get(seen, 0.0) + probability * p
        states = updated
    return sum(min(k, len(seen)) * probability for seen, probability in states.items())


def fresh_probability(actions: Iterable[PlacementAction | Mapping[str, Any]], k: int) -> float:
    """Return ``P(rank >= k)`` under independent action successes."""
    k = _integer(k, "k", nonnegative=False)
    if k < 1:
        raise ValueError("k must be positive")
    states: dict[frozenset[int], float] = {frozenset(): 1.0}
    for raw in actions:
        action = _coerce_action(raw)
        p = min(1.0, max(0.0, action.probability))
        updated: dict[frozenset[int], float] = {}
        for seen, probability in states.items():
            updated[seen] = updated.get(seen, 0.0) + probability * (1.0 - p)
            if len(seen) < k:
                next_seen = seen | {action.shard_id}
                updated[next_seen] = updated.get(next_seen, 0.0) + probability * p
            else:
                updated[seen] = updated.get(seen, 0.0) + probability * p
        states = updated
    return min(1.0, max(0.0, sum(probability for seen, probability in states.items() if len(seen) >= k)))


def estimate_expected_rank(actions: Iterable[PlacementAction | Mapping[str, Any]], k: int) -> float:
    return expected_capped_rank(actions, k)


def estimate_fresh_probability(actions: Iterable[PlacementAction | Mapping[str, Any]], k: int) -> float:
    return fresh_probability(actions, k)


def _deadline_window_slots(
    window: ContactWindow | Mapping[str, Any], *, shard_bytes: int,
    created_at: float, expires_at: float,
) -> tuple[str, int, float]:
    """Whole-shard slots inside one observed window and the half-open TTL.

    ContactWindow.capacity_bytes describes the *whole* window and must not
    bypass deadline clipping. Explicit mapping capacities remain an additional
    upper bound. A rate is required because bytes alone cannot locate transfers
    in a partially observed window. Fractions are never pooled across visits.
    """
    rsu = str(_field(window, "rsu_id", _field(window, "rsu", "")))
    start = _number(_field(window, "start", _field(window, "start_time", 0.0)), "window.start")
    end = _number(_field(window, "end", _field(window, "end_time", start)), "window.end")
    available_start = max(start, created_at)
    available_end = min(end, expires_at)
    if available_end <= available_start:
        return rsu, 0, 0.0
    rate = _field(window, "rate_bps", _field(window, "rate", None))
    if rate is None:
        raise ValueError("deadline capacity requires an explicit window rate_bps")
    rate = _number(rate, "window.rate_bps")
    if rate <= 0:
        raise ValueError("deadline capacity requires positive window rate_bps")
    transfer_time = shard_bytes * 8.0 / rate
    slots = max(0, math.floor((available_end - available_start) / transfer_time + _EPS))
    # One time tolerance matches the existing contact-end gate. Expiry itself
    # is excluded, consistently with ObjectVersion.is_expired/verify_metadata.
    while slots > 0 and (
        available_start + slots * transfer_time > end + _EPS
        or available_start + slots * transfer_time >= expires_at
    ):
        slots -= 1
    if isinstance(window, Mapping) and window.get("capacity_bytes") is not None:
        capacity = _number(window["capacity_bytes"], "window.capacity_bytes")
        slots = min(slots, math.floor(capacity / shard_bytes))
    return rsu, slots, transfer_time


def _deadline_slot_limits(
    obj: ObjectVersion, windows: Sequence[ContactWindow | Mapping[str, Any]],
) -> dict[str, int]:
    """Cumulative whole-shard opportunity budget per RSU, without double count.

    The single-radio simulator consumes sequential contacts. Overlapping
    effective windows are rejected in the opt-in model instead of inventing a
    simultaneous-radio model or merging rates. Different windows can be added
    only after their individual whole-packet budgets have been rounded down.
    """
    intervals: dict[str, list[tuple[float, float]]] = {}
    limits: dict[str, int] = {}
    for window in windows:
        start = max(float(_field(window, "start", _field(window, "start_time", 0.0))), obj.created_at)
        end = min(float(_field(window, "end", _field(window, "end_time", start))), obj.expires_at)
        rsu, slots, _ = _deadline_window_slots(window, shard_bytes=obj.shard_bytes,
            created_at=obj.created_at, expires_at=obj.expires_at)
        limits[rsu] = limits.get(rsu, 0) + slots
        if end > start:
            vehicle_id = str(_field(window, "vehicle_id", ""))
            intervals.setdefault(vehicle_id, []).append((start, end))
    if len(intervals) > 1:
        raise ValueError("deadline capacity placement requires one target vehicle")
    for vehicle_intervals in intervals.values():
        ordered = sorted(vehicle_intervals)
        if any(right[0] < left[1] - _EPS for left, right in zip(ordered, ordered[1:])):
            raise ValueError("deadline capacity does not support overlapping contact windows")
    return limits


def _window_success_probability(
    window: ContactWindow | Mapping[str, Any],
    *,
    shard_bytes: int,
    expires_at: float,
    packet_loss: float,
    created_at: float = 0.0,
    enforce_deadline_capacity: bool = False,
) -> tuple[str, float, float] | None:
    rsu = str(_field(window, "rsu_id", _field(window, "rsu", "")))
    start = _number(_field(window, "start", _field(window, "start_time", 0.0)), "window.start")
    end = _number(_field(window, "end", _field(window, "end_time", start)), "window.end")
    if end <= start or start >= expires_at:
        return None
    rate = _field(window, "rate_bps", _field(window, "rate", None))
    capacity = _field(window, "capacity_bytes", None)
    transfer_time = 0.0
    if enforce_deadline_capacity:
        _, slots, transfer_time = _deadline_window_slots(window, shard_bytes=shard_bytes,
            created_at=created_at, expires_at=expires_at)
        if slots < 1:
            return None
    elif capacity is None:
        if rate is None or float(rate) <= 0:
            return None
        transfer_time = shard_bytes * 8.0 / float(rate)
        if start + transfer_time > min(end, expires_at) + _EPS:
            return None
        capacity = (min(end, expires_at) - start) * float(rate) / 8.0
    elif float(capacity) + _EPS < shard_bytes:
        return None
    visit = _field(window, "visit_probability", _field(window, "p_visit", 1.0))
    q = _field(window, "packet_success_probability", _field(window, "q", None))
    direct = _field(window, "success_probability", None)
    if direct is not None:
        probability = _number(direct, "success_probability")
    else:
        if q is None:
            q = 1.0 - packet_loss
        probability = _number(visit, "visit_probability") * _number(q, "packet_success_probability")
    if probability <= 0:
        return None
    if transfer_time <= 0 and rate is not None and float(rate) > 0:
        transfer_time = shard_bytes * 8.0 / float(rate)
    return rsu, min(1.0, probability), transfer_time


def candidate_actions(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow | Mapping[str, Any]],
    *,
    packet_loss: float = 0.0,
    storage_cost_per_byte: float = 1.0,
    backhaul_cost_per_byte: float = 1.0,
    migration_cost_per_byte: float = 1.0,
    action_costs: Mapping[Any, Mapping[str, Any]] | None = None,
    enforce_deadline_capacity: bool = False,
) -> list[PlacementAction]:
    """Generate feasible shard-RSU actions from an ordered contact route.

    Multiple visits to the same RSU are combined as independent opportunities:
    ``p(RSU delivers) = 1 - product(1-p_visit*q)``.  A candidate is emitted
    only when one whole shard fits within at least one contact before expiry.
    """
    if not isinstance(enforce_deadline_capacity, bool):
        raise TypeError("enforce_deadline_capacity must be bool")
    packet_loss = _number(packet_loss, "packet_loss")
    if packet_loss > 1:
        raise ValueError("packet_loss must be in [0,1]")
    storage_cost_per_byte = _number(storage_cost_per_byte, "storage_cost_per_byte")
    backhaul_cost_per_byte = _number(backhaul_cost_per_byte, "backhaul_cost_per_byte")
    migration_cost_per_byte = _number(migration_cost_per_byte, "migration_cost_per_byte")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for window in windows:
        item = _window_success_probability(
            window, shard_bytes=obj.shard_bytes, expires_at=obj.expires_at, packet_loss=packet_loss,
            created_at=obj.created_at, enforce_deadline_capacity=enforce_deadline_capacity,
        )
        if item is None:
            continue
        rsu, probability, _duration = item
        grouped.setdefault(rsu, []).append((probability, _duration))
    actions: list[PlacementAction] = []
    for rsu in sorted(grouped):
        p_rsu = 1.0
        # Complementary success over repeated route opportunities.
        for p, _duration in grouped[rsu]:
            p_rsu *= 1.0 - p
        p_rsu = 1.0 - p_rsu
        min_duration = min(duration for _probability, duration in grouped[rsu])
        for shard_id in obj.shard_ids:
            extras: Mapping[str, Any] = {}
            if action_costs:
                extras = action_costs.get((rsu, shard_id), action_costs.get(rsu, {})) or {}
            storage = _integer(extras.get("storage_bytes", obj.shard_bytes), "storage_bytes")
            backhaul = _integer(extras.get("backhaul_bytes", 0), "backhaul_bytes")
            migration = _integer(extras.get("migration_bytes", 0), "migration_bytes")
            actions.append(
                PlacementAction(
                    rsu_id=rsu,
                    shard_id=shard_id,
                    version=obj.version,
                    success_probability=p_rsu,
                    storage_bytes=storage,
                    backhaul_bytes=backhaul,
                    migration_bytes=migration,
                    completion_time_s=min_duration,
                )
            )
    return actions


def _budget_value(value: int | Mapping[str, int] | None, name: str) -> int | dict[str, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _integer(item, f"{name}[{key}]") for key, item in value.items()}
    return _integer(value, name)


def _fits(
    action: PlacementAction,
    selected: Sequence[PlacementAction],
    *,
    capacity_bytes: int | Mapping[str, int] | None,
    storage_budget_bytes: int | None,
    backhaul_budget_bytes: int | None,
    migration_budget_bytes: int | None,
) -> bool:
    storage = sum(a.storage_bytes for a in selected)
    backhaul = sum(a.backhaul_bytes for a in selected)
    migration = sum(a.migration_bytes for a in selected)
    if capacity_bytes is not None:
        if isinstance(capacity_bytes, Mapping):
            used = sum(a.storage_bytes for a in selected if a.rsu_id == action.rsu_id)
            if used + action.storage_bytes > int(capacity_bytes.get(action.rsu_id, 0)):
                return False
        elif storage + action.storage_bytes > int(capacity_bytes):
            return False
    if storage_budget_bytes is not None and storage + action.storage_bytes > storage_budget_bytes:
        return False
    if backhaul_budget_bytes is not None and backhaul + action.backhaul_bytes > backhaul_budget_bytes:
        return False
    if migration_budget_bytes is not None and migration + action.migration_bytes > migration_budget_bytes:
        return False
    return True


def _result(
    selected: Sequence[PlacementAction],
    obj: ObjectVersion,
    *,
    method: str,
    target_probability: float | None,
    capacity_bytes: int | Mapping[str, int] | None,
    backhaul_budget_bytes: int | None,
    migration_budget_bytes: int | None,
) -> PlacementResult:
    chosen = list(selected)
    return PlacementResult(
        selected=chosen,
        expected_rank=expected_capped_rank(chosen, obj.k),
        fresh_probability=fresh_probability(chosen, obj.k),
        storage_bytes=sum(a.storage_bytes for a in chosen),
        backhaul_bytes=sum(a.backhaul_bytes for a in chosen),
        migration_bytes=sum(a.migration_bytes for a in chosen),
        method=method,
        target_probability=target_probability,
        capacity_bytes=dict(capacity_bytes) if isinstance(capacity_bytes, Mapping) else capacity_bytes,
        backhaul_budget_bytes=backhaul_budget_bytes,
        migration_budget_bytes=migration_budget_bytes,
    )


def algorithm1_greedy_plan(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow | Mapping[str, Any]],
    capacity_bytes: int | Mapping[str, int] | None = None,
    *,
    rsu_capacities: Mapping[str, int] | None = None,
    eta: float | None = None,
    target_probability: float | None = None,
    storage_budget_bytes: int | None = None,
    backhaul_budget_bytes: int | None = None,
    migration_budget_bytes: int | None = None,
    packet_loss: float = 0.0,
    lambda_backhaul: float = 1.0,
    lambda_migration: float = 1.0,
    storage_cost_per_byte: float = 1.0,
    backhaul_cost_per_byte: float = 1.0,
    migration_cost_per_byte: float = 1.0,
    action_costs: Mapping[Any, Mapping[str, Any]] | None = None,
    enforce_deadline_capacity: bool = False,
) -> PlacementResult:
    """Run manuscript Algorithm 1 and return detailed diagnostics."""
    if not isinstance(enforce_deadline_capacity, bool):
        raise TypeError("enforce_deadline_capacity must be bool")
    if rsu_capacities is not None:
        if capacity_bytes is not None:
            raise ValueError("pass either capacity_bytes or rsu_capacities, not both")
        capacity_bytes = rsu_capacities
    if eta is None:
        eta = target_probability
    if eta is not None:
        eta = _number(eta, "eta")
        if eta > 1:
            raise ValueError("eta must be in [0,1]")
    capacity = _budget_value(capacity_bytes, "capacity_bytes")
    storage_budget = None if storage_budget_bytes is None else _integer(storage_budget_bytes, "storage_budget_bytes")
    backhaul_budget = None if backhaul_budget_bytes is None else _integer(backhaul_budget_bytes, "backhaul_budget_bytes")
    migration_budget = None if migration_budget_bytes is None else _integer(migration_budget_bytes, "migration_budget_bytes")
    slot_limits = None
    if enforce_deadline_capacity:
        windows = list(windows)
        slot_limits = _deadline_slot_limits(obj, windows)
    actions = candidate_actions(
        obj,
        windows,
        packet_loss=packet_loss,
        storage_cost_per_byte=storage_cost_per_byte,
        backhaul_cost_per_byte=backhaul_cost_per_byte,
        migration_cost_per_byte=migration_cost_per_byte,
        action_costs=action_costs,
        enforce_deadline_capacity=enforce_deadline_capacity,
    )
    selected: list[PlacementAction] = []
    if eta is not None and eta <= 0:
        return _result(selected, obj, method="greedy", target_probability=eta, capacity_bytes=capacity,
                       backhaul_budget_bytes=backhaul_budget, migration_budget_bytes=migration_budget)
    while actions:
        if eta is not None and fresh_probability(selected, obj.k) + _EPS >= eta:
            break
        feasible: list[tuple[float, PlacementAction]] = []
        for action in actions:
            if slot_limits is not None and sum(a.rsu_id == action.rsu_id for a in selected) >= slot_limits.get(action.rsu_id, 0):
                continue
            if not _fits(action, selected, capacity_bytes=capacity, storage_budget_bytes=storage_budget,
                         backhaul_budget_bytes=backhaul_budget, migration_budget_bytes=migration_budget):
                continue
            before = expected_capped_rank(selected, obj.k)
            gain = expected_capped_rank([*selected, action], obj.k) - before
            if gain <= _EPS:
                continue
            denominator = (
                storage_cost_per_byte * action.storage_bytes
                + lambda_backhaul * backhaul_cost_per_byte * action.backhaul_bytes
                + lambda_migration * migration_cost_per_byte * action.migration_bytes
            )
            denominator = max(_EPS, denominator)
            score = gain / denominator
            feasible.append((score, action))
        if not feasible:
            break
        # Stable route-friendly tie break: higher score/probability first,
        # then prefer an RSU that has not already received a selected action.
        # This preserves the greedy objective while avoiding deterministic
        # pile-up at one RSU under otherwise identical contacts.
        _, chosen = min(
            feasible,
            key=lambda pair: (
                -pair[0],
                -pair[1].probability,
                sum(item.rsu_id == pair[1].rsu_id for item in selected),
                pair[1].shard_id,
                pair[1].rsu_id,
            ),
        )
        selected.append(chosen)
        actions.remove(chosen)
    return _result(selected, obj, method="greedy", target_probability=eta, capacity_bytes=capacity,
                   backhaul_budget_bytes=backhaul_budget, migration_budget_bytes=migration_budget)


def algorithm1_greedy_placement(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow | Mapping[str, Any]],
    capacity_bytes: int | Mapping[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, list[int]] | PlacementResult:
    """Backward-compatible Algorithm 1 entry point returning a mapping."""
    return_result = bool(kwargs.pop("return_result", kwargs.pop("return_details", False)))
    result = algorithm1_greedy_plan(obj, windows, capacity_bytes, **kwargs)
    return result if return_result else result.placement


def algorithm1_chance_constrained(*args: Any, **kwargs: Any) -> PlacementResult:
    """Named manuscript entry point returning the detailed greedy plan."""
    return algorithm1_greedy_plan(*args, **kwargs)


def _better(a: PlacementResult, b: PlacementResult | None) -> bool:
    if b is None:
        return True
    # Fresh probability is the primary objective; expected rank and lower cost
    # break ties so the result remains deterministic.
    lhs = (a.fresh_probability, a.expected_rank, -a.storage_bytes, -a.backhaul_bytes, -a.migration_bytes)
    rhs = (b.fresh_probability, b.expected_rank, -b.storage_bytes, -b.backhaul_bytes, -b.migration_bytes)
    return lhs > rhs


def algorithm1_oracle_placement(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow | Mapping[str, Any]],
    capacity_bytes: int | Mapping[str, int] | None = None,
    *,
    rsu_capacities: Mapping[str, int] | None = None,
    max_actions: int = 20,
    enforce_deadline_capacity: bool = False,
    **kwargs: Any,
) -> PlacementResult:
    """Exhaustive small-instance upper bound under the same action/budgets.

    This is deliberately bounded and intended for hand checks/oracle plots.  A
    caller asking for more than ``max_actions`` gets a clear error instead of
    silently starting an expensive full sweep.
    """
    if not isinstance(enforce_deadline_capacity, bool):
        raise TypeError("enforce_deadline_capacity must be bool")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if rsu_capacities is not None:
        if capacity_bytes is not None:
            raise ValueError("pass either capacity_bytes or rsu_capacities, not both")
        capacity_bytes = rsu_capacities
    eta = kwargs.pop("eta", kwargs.pop("target_probability", None))
    storage_budget = kwargs.pop("storage_budget_bytes", None)
    backhaul_budget = kwargs.pop("backhaul_budget_bytes", None)
    migration_budget = kwargs.pop("migration_budget_bytes", None)
    packet_loss = kwargs.pop("packet_loss", 0.0)
    slot_limits = None
    if enforce_deadline_capacity:
        windows = list(windows)
        slot_limits = _deadline_slot_limits(obj, windows)
    actions = candidate_actions(obj, windows, packet_loss=packet_loss,
        enforce_deadline_capacity=enforce_deadline_capacity, **{
        key: kwargs[key] for key in (
            "storage_cost_per_byte", "backhaul_cost_per_byte", "migration_cost_per_byte", "action_costs"
        ) if key in kwargs
    })
    if len(actions) > max_actions:
        raise ValueError(f"oracle limited to {max_actions} candidate actions; got {len(actions)}")
    capacity = _budget_value(capacity_bytes, "capacity_bytes")
    storage_budget = None if storage_budget is None else _integer(storage_budget, "storage_budget_bytes")
    backhaul_budget = None if backhaul_budget is None else _integer(backhaul_budget, "backhaul_budget_bytes")
    migration_budget = None if migration_budget is None else _integer(migration_budget, "migration_budget_bytes")
    best: PlacementResult | None = None
    for size in range(len(actions) + 1):
        for chosen in itertools.combinations(actions, size):
            if slot_limits is not None and any(
                sum(a.rsu_id == rsu for a in chosen) > limit for rsu, limit in slot_limits.items()
            ):
                continue
            if any(not _fits(action, chosen[:index], capacity_bytes=capacity,
                              storage_budget_bytes=storage_budget,
                              backhaul_budget_bytes=backhaul_budget,
                              migration_budget_bytes=migration_budget)
                   for index, action in enumerate(chosen)):
                continue
            result = _result(chosen, obj, method="oracle", target_probability=eta,
                             capacity_bytes=capacity, backhaul_budget_bytes=backhaul_budget,
                             migration_budget_bytes=migration_budget)
            if _better(result, best):
                best = result
    if best is None:
        best = _result([], obj, method="oracle", target_probability=eta, capacity_bytes=capacity,
                       backhaul_budget_bytes=backhaul_budget, migration_budget_bytes=migration_budget)
    return best


# Short aliases used by experiment scripts and notebooks.
algorithm1_oracle = algorithm1_oracle_placement
oracle_placement = algorithm1_oracle_placement
algorithm1_greedy = algorithm1_greedy_plan
