"""Small, deterministic placement strategies for manuscript baselines.

All strategies share ``{rsu_id: [shard_id, ...]}``, the representation used by
``Simulation``. Whole-file baselines use the first ``k`` systematic ids and
mark the plan ``coded=False`` for storage accounting.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .models import ContactWindow, ObjectVersion
from .placement import algorithm1_greedy_placement, algorithm1_oracle_placement
from .recovery import algorithm2_should_request

BASELINES = (
    "B1_full_replication", "B2_static_ec", "B3_mobility_aware_replication",
    "B4_trajectory_ec_without_freshness", "B5_freshness_aware_uncoded_caching",
    "B6_oracle", "B7_proposed_algorithm1_algorithm2",
)

RELATED_BASELINES = (
    "B8_static_rlnc_fountain", "B9_contact_frequency_coded",
    "A1_no_trajectory", "A2_no_rank_aware",
)


@dataclass(frozen=True)
class PlacementPlan:
    """Placement and policy metadata needed by small comparisons."""
    name: str
    placement: dict[str, list[int]]
    coded: bool
    freshness_aware: bool
    oracle: bool = False
    recovery_policy: str = "deadline_aware"
    storage_bytes: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "placement": {k: list(v) for k, v in self.placement.items()},
            "coded": self.coded,
            "freshness_aware": self.freshness_aware,
            "oracle": self.oracle,
            "recovery_policy": self.recovery_policy,
            "storage_bytes": self.storage_bytes,
            "metadata": dict(self.metadata),
        }


def _windows(windows: Iterable[ContactWindow]) -> list[ContactWindow]:
    result = list(windows)
    if any(not isinstance(w, ContactWindow) for w in result):
        raise TypeError("windows must contain ContactWindow values")
    return sorted(result, key=lambda w: (w.start, w.end, w.rsu_id))


def _rsu_scores(windows: Sequence[ContactWindow], *, until: float | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for w in windows:
        duration = w.duration if until is None else max(0.0, min(w.end, until) - w.start)
        scores[w.rsu_id] = scores.get(w.rsu_id, 0.0) + duration * w.rate_bps
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _capacity_slots(obj: ObjectVersion, capacity_bytes: int | None, *, coded: bool) -> int:
    if capacity_bytes is None:
        return obj.n if coded else obj.k
    if capacity_bytes < 0:
        raise ValueError("capacity_bytes must be non-negative")
    unit = obj.shard_bytes if coded else obj.size_bytes
    return max(0, int(capacity_bytes) // max(1, unit))


def placement_storage_bytes(obj: ObjectVersion, placement: Mapping[str, Sequence[int]], *, coded: bool) -> int:
    """Count duplicate replicas in a placement."""
    if coded:
        return sum(len(shards) for shards in placement.values()) * obj.shard_bytes
    return sum(1 for shards in placement.values() if shards) * obj.size_bytes


def _replication(obj: ObjectVersion, windows: Sequence[ContactWindow], capacity_bytes: int | None,
                 *, ranked: bool, freshness_aware: bool, name: str) -> PlacementPlan:
    order = _rsu_scores(windows, until=obj.expires_at if freshness_aware else None) if ranked else [
        (r, 0.0) for r in sorted({w.rsu_id for w in windows})
    ]
    slots = _capacity_slots(obj, capacity_bytes, coded=False)
    copies = len(order) if capacity_bytes is None else min(len(order), slots)
    placement = {rsu: list(range(obj.k)) for rsu, _ in order[:copies]}
    return PlacementPlan(name=name, placement=placement, coded=False,
                         freshness_aware=freshness_aware,
                         storage_bytes=placement_storage_bytes(obj, placement, coded=False),
                         metadata={"selected_rsus": [r for r, _ in order[:copies]]})


def full_replication_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                               capacity_bytes: int | None = None) -> PlacementPlan:
    """B1: whole-file copies without mobility scoring."""
    return _replication(obj, _windows(windows), capacity_bytes, ranked=False,
                        freshness_aware=False, name="B1_full_replication")


def mobility_aware_replication(obj: ObjectVersion, windows: Iterable[ContactWindow],
                               capacity_bytes: int | None = None) -> PlacementPlan:
    """B3: whole-file copies at RSUs with greatest contact capacity."""
    return _replication(obj, _windows(windows), capacity_bytes, ranked=True,
                        freshness_aware=False, name="B3_mobility_aware_replication")


def _coded_round_robin(obj: ObjectVersion, windows: Sequence[ContactWindow], capacity_bytes: int | None,
                       *, ranked: bool, freshness_aware: bool, oracle: bool = False,
                       name: str = "B2_static_ec") -> PlacementPlan:
    scored = _rsu_scores(windows, until=obj.expires_at if freshness_aware else None)
    order = [r for r, _ in scored] if ranked else sorted({w.rsu_id for w in windows})
    return _coded_from_order(obj, order, capacity_bytes, freshness_aware=freshness_aware,
                             oracle=oracle, name=name,
                             metadata={"ranked_by_contact": ranked})


def _coded_from_order(obj: ObjectVersion, order: Sequence[str], capacity_bytes: int | None,
                      *, freshness_aware: bool, oracle: bool = False,
                      name: str, metadata: Mapping[str, object] | None = None) -> PlacementPlan:
    order = list(dict.fromkeys(str(rsu) for rsu in order))
    if not order:
        return PlacementPlan(name=name, placement={}, coded=True, freshness_aware=freshness_aware,
                             oracle=oracle)
    slots = obj.n if capacity_bytes is None else _capacity_slots(obj, capacity_bytes, coded=True)
    placement: dict[str, list[int]] = {r: [] for r in order}
    for shard_id in range(min(obj.n, slots)):
        placement[order[shard_id % len(order)]].append(shard_id)
    placement = {r: ids for r, ids in placement.items() if ids}
    plan_metadata = dict(metadata or {})
    plan_metadata.setdefault("selected_rsus", order)
    return PlacementPlan(name=name, placement=placement, coded=True,
                         freshness_aware=freshness_aware, oracle=oracle,
                         storage_bytes=placement_storage_bytes(obj, placement, coded=True),
                         metadata=plan_metadata)


def static_ec_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                        capacity_bytes: int | None = None) -> PlacementPlan:
    """B2: fixed (n,k) EC assigned lexicographically."""
    return _coded_round_robin(obj, _windows(windows), capacity_bytes, ranked=False,
                              freshness_aware=False)


def trajectory_ec_without_freshness(obj: ObjectVersion, windows: Iterable[ContactWindow],
                                    capacity_bytes: int | None = None) -> PlacementPlan:
    """B4: trajectory-aware coded placement that ignores object expiry."""
    return _coded_round_robin(obj, _windows(windows), capacity_bytes, ranked=True,
                              freshness_aware=False, name="B4_trajectory_ec_without_freshness")


def freshness_aware_uncoded_caching(obj: ObjectVersion, windows: Iterable[ContactWindow],
                                    capacity_bytes: int | None = None) -> PlacementPlan:
    """B5: whole-file caching ranked by contact capacity before expiry."""
    return _replication(obj, _windows(windows), capacity_bytes, ranked=True,
                        freshness_aware=True, name="B5_freshness_aware_uncoded_caching")


def oracle_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                     capacity_bytes: int | None = None, *, max_actions: int = 20) -> PlacementPlan:
    """B6: use true future contact windows as an upper-bound policy.

    Small instances use the bounded exhaustive oracle from ``placement.py``.
    Larger instances fall back to the deterministic ranked map rather than
    silently launching an expensive search; this guard keeps hand checks cheap.
    """
    ordered = _windows(windows)
    try:
        result = algorithm1_oracle_placement(obj, ordered, capacity_bytes, max_actions=max_actions)
        return PlacementPlan(
            name="B6_oracle", placement=result.placement, coded=True,
            freshness_aware=True, oracle=True, storage_bytes=placement_storage_bytes(
                obj, result.placement, coded=True
            ), metadata={"expected_rank": result.expected_rank,
                        "fresh_probability": result.fresh_probability, "bounded": True},
        )
    except (ValueError, RuntimeError):
        return _coded_round_robin(obj, ordered, capacity_bytes, ranked=True,
                                  freshness_aware=True, oracle=True, name="B6_oracle")


def proposed_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                       capacity_bytes: int | None = None, *,
                       enforce_deadline_capacity: bool = False) -> PlacementPlan:
    """B7: Algorithm 1 placement with Algorithm 2 recovery metadata."""
    ordered = _windows(windows)
    capacity = obj.coded_size_bytes if capacity_bytes is None else int(capacity_bytes)
    placement = algorithm1_greedy_placement(obj, ordered, capacity,
        enforce_deadline_capacity=enforce_deadline_capacity)
    metadata = {"algorithm1": True, "algorithm2": True}
    if enforce_deadline_capacity:
        metadata.update(placement_model="deadline_capacity_v3_1", enforce_deadline_capacity=True)
    return PlacementPlan(name="B7_proposed_algorithm1_algorithm2", placement=placement,
                         coded=True, freshness_aware=True, recovery_policy="algorithm2",
                         storage_bytes=placement_storage_bytes(obj, placement, coded=True),
                         metadata=metadata)


def proposed_runtime_rank_aware_placement(
    obj: ObjectVersion, windows: Iterable[ContactWindow],
    capacity_bytes: int | None = None, *, enforce_deadline_capacity: bool = False,
) -> PlacementPlan:
    """Supplementary B7 reference with request-time innovation filtering.

    Frozen B7 did not filter already received symbols before requesting them.
    Keep that historical baseline unchanged and label this opt-in reference
    separately, so A2 can isolate this one runtime change honestly.
    """
    plan = proposed_placement(obj, windows, capacity_bytes,
        enforce_deadline_capacity=enforce_deadline_capacity)
    return PlacementPlan(
        name="B7_runtime_rank_aware", placement=plan.placement, coded=True,
        freshness_aware=True, recovery_policy="algorithm2_rank_filter",
        storage_bytes=plan.storage_bytes,
        metadata={**plan.metadata, "rank_aware_requests": True,
                  "reference_version": "supplementary_runtime_rank_filter",
                  "placement_matches_frozen_b7": not enforce_deadline_capacity},
    )


def _marginal_expected_rank(counts: Mapping[str, int], probabilities: Mapping[str, float], k: int) -> float:
    """E[min(k, sum_i count_i V_i)] for independent RSU visit indicators.

    All symbols at one RSU share its visit indicator. Treating them as
    independent packets would unfairly devalue placing multiple symbols at a
    commonly visited RSU. This model does not use order, contact capacity or
    TTL and deliberately ignores correlations between different RSU visits.
    """
    state = [1.0] + [0.0] * k
    for rsu, count in sorted(counts.items()):
        probability = probabilities.get(rsu, 0.0)
        updated = [0.0] * (k + 1)
        for rank, mass in enumerate(state):
            updated[rank] += mass * (1.0 - probability)
            updated[min(k, rank + count)] += mass * probability
        state = updated
    return sum(rank * mass for rank, mass in enumerate(state))


def _rlnc_seed_hash(obj: ObjectVersion, experiment_seed: int) -> str:
    return hashlib.sha256(
        f"8309|{experiment_seed}|{obj.object_id}|{obj.version}|{obj.n}|{obj.k}".encode()
    ).hexdigest()


def _rlnc_vectors(obj: ObjectVersion, *, experiment_seed: int = 0) -> dict[str, list[int]]:
    """Generate nonzero uniform GF(256) vectors, without selecting for rank."""
    digest = _rlnc_seed_hash(obj, experiment_seed)
    rng = random.Random(int(digest, 16))
    vectors: dict[str, list[int]] = {}
    for shard_id in range(obj.n):
        vector = [rng.randrange(256) for _ in range(obj.k)]
        while not any(vector):
            vector = [rng.randrange(256) for _ in range(obj.k)]
        vectors[str(shard_id)] = vector
    return vectors


def _fixed_rsu_universe(rsu_universe: Iterable[str] | None) -> list[str]:
    if rsu_universe is None:
        raise ValueError("an exogenous fixed rsu_universe is required")
    result = sorted({str(rsu) for rsu in rsu_universe})
    if not result:
        raise ValueError("rsu_universe must be nonempty")
    return result


def _static_rsu_order(rsu_universe: Iterable[str] | None, experiment_seed: int) -> tuple[list[str], str]:
    """A shared random RSU order, independent of targets and sweep factors."""
    order = _fixed_rsu_universe(rsu_universe)
    digest = hashlib.sha256(
        (f"static-rsu-order|6311|{experiment_seed}|" + "|".join(order)).encode()
    ).hexdigest()
    random.Random(int(digest, 16)).shuffle(order)
    return order, digest


def static_rlnc_fountain_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                                   capacity_bytes: int | None = None, *,
                                   rsu_universe: Iterable[str] | None = None,
                                   experiment_seed: int = 0) -> PlacementPlan:
    """B8: static RSU-only finite-n GF(256) RLNC, not Fountain/RaptorQ.

    The fixed RSU map comes from scenario infrastructure, never from the
    target's visited RSUs. Random coefficients vary with experiment seed and
    are fixed across all vehicles, TTLs, losses and capacities for that code.
    """
    ordered, order_hash = _static_rsu_order(rsu_universe, experiment_seed)
    return _coded_from_order(
        obj, ordered, capacity_bytes, freshness_aware=False,
        name="B8_static_rlnc_fountain",
        metadata={"code_family": "gf256_rlnc", "static": True,
                  "trajectory_aware": False, "candidate_rsu_scope": "fixed_infrastructure",
                  "rsu_universe": sorted(ordered), "static_rsu_order": ordered,
                  "static_order_seed_salt": 6311, "static_order_seed_hash": order_hash,
                  "coefficient_seed_salt": 8309,
                  "experiment_seed": int(experiment_seed),
                  "code_seed_hash": _rlnc_seed_hash(obj, experiment_seed),
                  "symbol_vectors": _rlnc_vectors(obj, experiment_seed=experiment_seed)},
    )


def contact_frequency_coded_placement(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow],
    capacity_bytes: int | None = None,
    *,
    calibration_windows: Iterable[ContactWindow] | None = None,
    rsu_universe: Iterable[str] | None = None,
) -> PlacementPlan:
    """B9: greedy MDS placement using only marginal RSU visit probabilities."""
    target = _windows(windows)
    universe = _fixed_rsu_universe(rsu_universe)
    if calibration_windows is None:
        raise ValueError("B9 requires target-excluding calibration_windows")
    calibration = _windows(calibration_windows)
    target_ids = {str(window.vehicle_id) for window in target}
    if any(str(window.vehicle_id) in target_ids for window in calibration):
        raise ValueError("B9 calibration contains a target vehicle")
    calibration_vehicle_ids = {str(window.vehicle_id) for window in calibration}
    if not calibration_vehicle_ids:
        raise ValueError("B9 requires at least one calibration vehicle")
    visitors: dict[str, set[str]] = {rsu: set() for rsu in universe}
    unknown = {str(window.rsu_id) for window in calibration} - set(universe)
    if unknown:
        raise ValueError(f"calibration RSUs absent from fixed universe: {sorted(unknown)}")
    for window in calibration:
        visitors[str(window.rsu_id)].add(str(window.vehicle_id))
    probabilities = {rsu: len(ids) / len(calibration_vehicle_ids) for rsu, ids in visitors.items()}
    counts: dict[str, int] = {}
    placement: dict[str, list[int]] = {}
    value = 0.0
    slots = obj.n if capacity_bytes is None else _capacity_slots(obj, capacity_bytes, coded=True)
    for symbol in range(min(obj.n, slots)):
        candidates = []
        for rsu in universe:
            trial = dict(counts)
            trial[rsu] = trial.get(rsu, 0) + 1
            candidates.append((_marginal_expected_rank(trial, probabilities, obj.k), rsu))
        # Deterministic ties; the target is never used to break them.
        best_value, best_rsu = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[0]
        if best_value - value <= 1e-12:
            break
        counts[best_rsu] = counts.get(best_rsu, 0) + 1
        placement.setdefault(best_rsu, []).append(symbol)
        value = best_value
    return PlacementPlan(
        name="B9_contact_frequency_coded", placement=placement, coded=True,
        freshness_aware=False, storage_bytes=placement_storage_bytes(obj, placement, coded=True),
        metadata={"code_family": "ideal_mds", "frequency_only": True,
                  "calibration_excludes_target": True,
                  "placement_heuristic": "marginal_visit_probability_expected_rank_greedy",
                  "visit_model": "independent_rsu_bernoulli",
                  "candidate_rsu_scope": "fixed_infrastructure", "rsu_universe": universe,
                  "visit_probabilities": probabilities,
                  "predicted_capped_rank": value,
                  "calibration_vehicle_count": len(calibration_vehicle_ids),
                  "calibration_window_count": len(calibration)},
    )


def no_trajectory_placement(obj: ObjectVersion, windows: Iterable[ContactWindow],
                            capacity_bytes: int | None = None, *,
                            rsu_universe: Iterable[str] | None = None,
                            experiment_seed: int = 0) -> PlacementPlan:
    """A1: B7 coding with a fixed static RSU order."""
    order, order_hash = _static_rsu_order(rsu_universe, experiment_seed)
    # A fixed k-symbol stopping rule avoids confounding no-trajectory with
    # filling all n slots, without consulting B7's target-specific placement.
    static_capacity = obj.k * obj.shard_bytes
    if capacity_bytes is not None:
        static_capacity = min(static_capacity, capacity_bytes)
    plan = _coded_from_order(obj, order, static_capacity, freshness_aware=False,
                             name="A1_no_trajectory")
    return PlacementPlan(name="A1_no_trajectory", placement=plan.placement, coded=True,
                         freshness_aware=False, recovery_policy="algorithm2_rank_filter",
                         storage_bytes=plan.storage_bytes,
                         metadata={"trajectory_aware": False, "algorithm2": True,
                                   "rank_aware_requests": True,
                                   "candidate_rsu_scope": "fixed_infrastructure",
                                   "rsu_universe": sorted(order), "static_rsu_order": order,
                                   "static_order_seed_salt": 6311,
                                   "static_order_seed_hash": order_hash,
                                   "experiment_seed": int(experiment_seed),
                                   "symbol_stop_policy": "min_k_capacity_slots",
                                   "storage_is_budget_upper_bound": True})


def no_rank_aware_recovery(obj: ObjectVersion, windows: Iterable[ContactWindow],
                           capacity_bytes: int | None = None, *,
                           enforce_deadline_capacity: bool = False) -> PlacementPlan:
    """A2: remove only request-time innovation filtering from corrected B7.

    This is intentionally equivalent to frozen B7: that implementation did
    not filter duplicates before requests. There are no added requests, cache
    copies, or post-decode transmissions to manufacture an ablation effect.
    """
    plan = proposed_placement(obj, windows, capacity_bytes,
        enforce_deadline_capacity=enforce_deadline_capacity)
    return PlacementPlan(name="A2_no_rank_aware", placement=plan.placement, coded=True,
                         freshness_aware=True, recovery_policy="algorithm2",
                         storage_bytes=plan.storage_bytes,
                         metadata={**plan.metadata,
                                   "rank_aware_requests": False,
                                   "matches_frozen_b7": not enforce_deadline_capacity,
                                   "ablation_reference": "B7_runtime_rank_aware"})


def should_request(baseline: str, *, rank: int, k: int, now: float,
                   expire_at: float, expected_completion: float = 0.0) -> bool:
    """Common recovery gate for small baseline comparisons."""
    if rank >= k:
        return False
    key = str(baseline).lower()
    if "without_freshness" in key or key in {"b4", "trajectory_ec"}:
        return True
    if key.startswith(("b1", "b2", "b3")):
        return True
    if key.startswith(("b5", "b6", "b7", "b8", "b9", "a1", "a2")) or "proposed" in key:
        return algorithm2_should_request(rank=rank, k=k, now=now, expire_at=expire_at,
                                         expected_completion=expected_completion)
    raise ValueError(f"unknown baseline: {baseline}")


def build_baseline(name: str, obj: ObjectVersion, windows: Iterable[ContactWindow],
                   capacity_bytes: int | None = None, *,
                   rsu_universe: Iterable[str] | None = None,
                   calibration_windows: Iterable[ContactWindow] | None = None,
                   experiment_seed: int = 0,
                   enforce_deadline_capacity: bool = False) -> PlacementPlan:
    """Build one named baseline using a uniform call signature."""
    if not isinstance(enforce_deadline_capacity, bool):
        raise TypeError("enforce_deadline_capacity must be bool")
    aliases = {
        "b1": full_replication_placement, "b1_full_replication": full_replication_placement,
        "b2": static_ec_placement, "b2_static_ec": static_ec_placement,
        "b3": mobility_aware_replication, "b3_mobility_aware_replication": mobility_aware_replication,
        "b4": trajectory_ec_without_freshness,
        "b4_trajectory_ec_without_freshness": trajectory_ec_without_freshness,
        "b5": freshness_aware_uncoded_caching,
        "b5_freshness_aware_uncoded_caching": freshness_aware_uncoded_caching,
        "b6": oracle_placement, "b6_oracle": oracle_placement,
        "b7": proposed_placement, "b7_proposed_algorithm1_algorithm2": proposed_placement,
        "b7_runtime_rank_aware": proposed_runtime_rank_aware_placement,
        "b8": static_rlnc_fountain_placement,
        "b8_static_rlnc_fountain": static_rlnc_fountain_placement,
        "b9": contact_frequency_coded_placement,
        "b9_contact_frequency_coded": contact_frequency_coded_placement,
        "a1": no_trajectory_placement, "a1_no_trajectory": no_trajectory_placement,
        "a2": no_rank_aware_recovery, "a2_no_rank_aware": no_rank_aware_recovery,
    }
    try:
        method = aliases[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown baseline: {name}") from exc
    kwargs: dict[str, object] = {}
    if method in (static_rlnc_fountain_placement, contact_frequency_coded_placement,
                  no_trajectory_placement):
        kwargs["rsu_universe"] = rsu_universe
    if method in (static_rlnc_fountain_placement, no_trajectory_placement):
        kwargs["experiment_seed"] = experiment_seed
    if method is contact_frequency_coded_placement:
        kwargs["calibration_windows"] = calibration_windows
    if method in (proposed_placement, proposed_runtime_rank_aware_placement, no_rank_aware_recovery):
        kwargs["enforce_deadline_capacity"] = enforce_deadline_capacity
    return method(obj, windows, capacity_bytes, **kwargs)


make_baseline = build_baseline


def all_baselines(obj: ObjectVersion, windows: Iterable[ContactWindow],
                  capacity_bytes: int | None = None, *,
                  enforce_deadline_capacity: bool = False) -> dict[str, PlacementPlan]:
    """Return all seven methods for one shared small instance."""
    saved = list(windows)
    return {name: build_baseline(name, obj, saved, capacity_bytes,
        enforce_deadline_capacity=enforce_deadline_capacity) for name in BASELINES}


def all_related_baselines(
    obj: ObjectVersion,
    windows: Iterable[ContactWindow],
    capacity_bytes: int | None = None,
    *,
    calibration_windows: Iterable[ContactWindow] | None = None,
    rsu_universe: Iterable[str] | None = None,
    experiment_seed: int = 0,
    enforce_deadline_capacity: bool = False,
) -> dict[str, PlacementPlan]:
    """Build the four approved related-work methods for one target vehicle."""
    saved = list(windows)
    universe = _fixed_rsu_universe(rsu_universe)
    return {
        "B8_static_rlnc_fountain": static_rlnc_fountain_placement(
            obj, saved, capacity_bytes, rsu_universe=universe, experiment_seed=experiment_seed),
        "B9_contact_frequency_coded": contact_frequency_coded_placement(
            obj, saved, capacity_bytes, calibration_windows=calibration_windows,
            rsu_universe=universe,
        ),
        "A1_no_trajectory": no_trajectory_placement(
            obj, saved, capacity_bytes, rsu_universe=universe, experiment_seed=experiment_seed),
        "A2_no_rank_aware": no_rank_aware_recovery(obj, saved, capacity_bytes,
            enforce_deadline_capacity=enforce_deadline_capacity),
    }

# Explicit aliases make the mapping easy to discover from notebooks and keep
# baseline labels close to the manuscript notation.
b1_full_replication = full_replication_placement
b2_static_ec = static_ec_placement
b3_mobility_aware_replication = mobility_aware_replication
b4_trajectory_ec_without_freshness = trajectory_ec_without_freshness
b5_freshness_aware_uncoded_caching = freshness_aware_uncoded_caching
b6_oracle = oracle_placement
b7_proposed_algorithm1_algorithm2 = proposed_placement
