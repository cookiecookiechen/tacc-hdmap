from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import simpy

from .coding import RankTracker
from .io import config_hash
from .models import ContactWindow, EpisodeResult, ObjectVersion, TransmissionRecord
from .mobility import (
    contact_windows_for_vehicle,
    csv_trace,
    sumo_fcd_trace,
    synthetic_line_trace,
    synthetic_markov_trace,
)
from .placement import algorithm1_greedy_placement
from .randomness import KEY_NAMESPACE, keyed_uniform
from .recovery import algorithm2_should_request
from .verification import merkle_root, metadata_leaf, verify_metadata


def _window_dict(window: ContactWindow) -> dict[str, object]:
    return {
        "vehicle_id": window.vehicle_id,
        "rsu_id": window.rsu_id,
        "start": window.start,
        "end": window.end,
        "rate_bps": window.rate_bps,
    }


def _trace_from_config(config: Mapping[str, object], seed: int) -> list[ContactWindow]:
    """Build a contact trace without allocating payloads."""
    kind = str(config.get("trace_kind", "synthetic_line")).lower()
    rate = float(config.get("link_rate_bps", config.get("rate_bps", 1_000_000)))
    if kind in {"synthetic_line", "line", "synthetic"}:
        return list(synthetic_line_trace(
            int(config.get("vehicles", 1)), int(config.get("rsus", 1)),
            float(config.get("contact_duration_s", 1.0)),
            float(config.get("inter_contact_s", 0.0)), rate))
    if kind in {"synthetic_markov", "markov"}:
        return list(synthetic_markov_trace(
            int(config.get("vehicles", 1)), int(config.get("rsus", 1)),
            float(config.get("contact_duration_s", 1.0)), rate, seed=seed))
    if kind == "csv":
        trace_path = config.get("trace_path")
        if not trace_path:
            raise ValueError("trace_path is required when trace_kind=csv")
        return csv_trace(
            Path(str(trace_path)), default_rate_bps=rate,
            sample_interval_s=float(config.get("sample_interval_s", config.get("trace_sample_interval_s", 1.0))),
            merge_consecutive=bool(config.get("merge_consecutive", config.get("trace_merge_consecutive", True))))
    if kind in {"sumo_fcd", "sumo", "fcd"}:
        trace_path = config.get("trace_path")
        rsu_sites = config.get("rsus")
        if not trace_path or not isinstance(rsu_sites, Mapping):
            raise ValueError("trace_path and mapping rsus are required when trace_kind=sumo_fcd")
        return sumo_fcd_trace(
            Path(str(trace_path)), rsu_sites,
            coverage_radius_m=(float(config["coverage_radius_m"])
                               if config.get("coverage_radius_m") is not None else None),
            default_rate_bps=rate,
            sample_interval_s=float(config.get("sample_interval_s", config.get("trace_sample_interval_s", 1.0))),
            merge_consecutive=bool(config.get("merge_consecutive", config.get("trace_merge_consecutive", True))),
            vehicle_ids=config.get("vehicle_ids"),
        )
    if kind in {"traci", "live_traci"}:
        raise ValueError("TraCI requires an active connection; use sumo_fcd for XML")
    if kind == "hand":
        return list(synthetic_line_trace(
            int(config.get("vehicles", 1)), int(config.get("rsus", 1)),
            float(config.get("contact_duration_s", 1.0)),
            float(config.get("inter_contact_s", 0.0)), rate))
    raise ValueError(f"unknown trace_kind: {kind}")


@dataclass
class Simulation:
    config: dict
    seed: int

    def run_episode(
        self,
        episode_index: int = 0,
        *,
        windows: Iterable[ContactWindow] | None = None,
        placement_override: Mapping[str, Sequence[int]] | None = None,
        object_override: ObjectVersion | None = None,
        symbol_vectors: Mapping[object, Sequence[int]] | None = None,
        allow_dependent_requests: bool | None = None,
        rank_aware_requests: bool | None = None,
    ) -> EpisodeResult:
        """Run one event-driven episode.

        ``windows`` and ``placement_override`` are public hooks for the small
        deterministic hand checks.  Ordinary runs use the configured synthetic
        or CSV trace and the Phase 1 placement hook.

        ``random_model='keyed_v3'`` opts into independent trip scenarios with
        stateless draws paired across methods by RSU/contact/attempt. Failures
        are per trip rather than a common global-time infrastructure outage.
        Omission (or ``'legacy'``) preserves the frozen historical draw order.
        """
        random_model = str(self.config.get("random_model", "legacy"))
        if random_model not in {"legacy", "keyed_v3"}:
            raise ValueError(f"unknown random_model: {random_model}")
        placement_model = str(self.config.get("placement_model", "legacy"))
        if placement_model not in {"legacy", "deadline_capacity_v3_1"}:
            raise ValueError(f"unknown placement_model: {placement_model}")
        enforce_deadline_capacity = placement_model == "deadline_capacity_v3_1"
        rng = random.Random(self.seed + episode_index)
        vehicle_id = str(self.config.get("vehicle_id", f"v{episode_index % int(self.config.get('vehicles', 1))}"))
        all_windows = list(windows) if windows is not None else _trace_from_config(self.config, self.seed + episode_index)
        own_windows = contact_windows_for_vehicle(all_windows, vehicle_id)
        if windows is not None and not own_windows:
            vehicle_id = all_windows[0].vehicle_id if all_windows else vehicle_id
            own_windows = contact_windows_for_vehicle(all_windows, vehicle_id)

        base_obj = object_override or ObjectVersion(
            str(self.config.get("object_id", "map-0")), int(self.config.get("version", 1)),
            int(self.config["object_bytes"]), int(self.config["n"]), int(self.config["k"]),
            created_at=float(self.config.get("created_at", 0.0)), ttl_s=float(self.config["ttl_s"]))
        if placement_override is None:
            placement = algorithm1_greedy_placement(
                base_obj, own_windows,
                int(self.config.get("placement_capacity_bytes", self.config.get("object_bytes", base_obj.size_bytes))),
                enforce_deadline_capacity=enforce_deadline_capacity)
        else:
            placement = {str(rsu): [int(sid) for sid in shard_ids] for rsu, shard_ids in placement_override.items()}

        env = simpy.Environment()
        configured_vectors = symbol_vectors if symbol_vectors is not None else self.config.get("coded_symbol_vectors")
        if configured_vectors is not None and not isinstance(configured_vectors, Mapping):
            raise ValueError("coded_symbol_vectors must be a mapping")
        vector_map = configured_vectors or {}
        # False preserves the frozen B1--B7 request sequence exactly. The
        # corrected B7 reference opts into pre-request innovation filtering.
        rank_filter = (bool(self.config.get("rank_aware_requests", False))
                       if rank_aware_requests is None else bool(rank_aware_requests))
        # Kept for caller compatibility; this switch may disable filtering,
        # but it must never disable the common stop-on-decode condition.
        if (bool(self.config.get("allow_dependent_requests", False))
                if allow_dependent_requests is None else bool(allow_dependent_requests)):
            rank_filter = False
        tracker = RankTracker(base_obj.k, vector_mode=bool(vector_map))
        records: list[TransmissionRecord] = []
        events: list[dict[str, object]] = []
        state: dict[str, object] = {"version": base_obj.version, "decode_time": None, "stale_decode": False, "expired": False}
        counters = {"backhaul": 0, "migration": 0, "repair": 0}

        def event(kind: str, time: float, **fields: object) -> None:
            events.append({"event": kind, "time": float(time), **fields})

        if random_model == "keyed_v3":
            event("stochastic_model", 0, model=random_model, key_namespace=KEY_NAMESPACE,
                  seed=self.seed, trip_id=vehicle_id, episode_index=episode_index,
                  failure_scope="isolated_trip", packet_pairing="contact_attempt_ordinal")
        failed_rsus = {str(x) for x in self.config.get("failed_rsus", [])}
        fail_probability = float(self.config.get("rsu_failure_probability", 0.0))
        for rsu_id in sorted({w.rsu_id for w in own_windows}):
            failure_uniform = (keyed_uniform(self.seed, vehicle_id, episode_index,
                                            "rsu_failure", rsu_id=rsu_id)
                               if random_model == "keyed_v3" else rng.random())
            if random_model == "keyed_v3":
                event("rsu_failure_draw", 0, rsu_id=rsu_id, uniform=failure_uniform,
                      probability=fail_probability, sampled_failed=failure_uniform < fail_probability)
            if failure_uniform < fail_probability:
                failed_rsus.add(rsu_id)

        def version_update_process():
            update_at = self.config.get("version_update_time_s")
            if update_at is None:
                return
            update_at = float(update_at)
            if update_at < 0 or update_at > base_obj.expires_at:
                return
            if update_at > env.now:
                yield env.timeout(update_at - env.now)
            if bool(state["expired"]):
                return
            old_version = int(state["version"])
            new_version = int(self.config.get("updated_version", old_version + 1))
            if new_version == old_version:
                return
            if state["decode_time"] is not None:
                state["stale_decode"] = True
            state["version"] = new_version
            tracker.reset()
            state["decode_time"] = None
            event("version_update", env.now, old_version=old_version, new_version=new_version)

        def expiry_process():
            if base_obj.expires_at > env.now:
                yield env.timeout(base_obj.expires_at - env.now)
            state["expired"] = True
            event("expiry", env.now, version=int(state["version"]))

        def contact_event_process(window: ContactWindow):
            if window.start > env.now:
                yield env.timeout(window.start - env.now)
            event("contact_start", env.now, vehicle_id=vehicle_id, rsu_id=window.rsu_id, end=window.end)
            if window.end > env.now:
                yield env.timeout(window.end - env.now)
            event("contact_end", env.now, vehicle_id=vehicle_id, rsu_id=window.rsu_id)

        def valid_metadata(shard_id: int, rsu_id: str, version: int, now: float) -> bool:
            if not bool(self.config.get("verification", False)):
                return True
            shard_hash = hashlib.sha256(f"{base_obj.object_id}|{version}|{shard_id}".encode()).hexdigest()
            if shard_id in {int(x) for x in self.config.get("invalid_metadata_shards", [])}:
                shard_hash = "invalid-" + shard_hash
            leaf = metadata_leaf(object_id=base_obj.object_id, version=version, shard_id=shard_id,
                                 rsu_id=rsu_id, expires_at=base_obj.expires_at, shard_hash=shard_hash)
            accepted = verify_metadata(
                expected_object_id=base_obj.object_id, expected_version=int(state["version"]),
                object_id=base_obj.object_id, version=version, shard_id=shard_id, rsu_id=rsu_id,
                expires_at=base_obj.expires_at, shard_hash=shard_hash, leaf_digest=leaf,
                merkle_root_digest=merkle_root([leaf]), current_time=now)
            event("metadata_verification", now, rsu_id=rsu_id, shard_id=shard_id,
                  version=version, accepted=accepted)
            return accepted

        def symbol_vector(shard_id: int) -> Sequence[int] | None:
            if not vector_map:
                return None
            value = vector_map.get(shard_id, vector_map.get(str(shard_id)))
            if value is None:
                raise ValueError(f"missing coded symbol vector for shard {shard_id}")
            return value

        def transmission_process():
            if enforce_deadline_capacity and env.now < base_obj.created_at:
                yield env.timeout(base_obj.created_at - env.now)
            for window in own_windows:
                if bool(state["expired"]) or env.now >= base_obj.expires_at:
                    break
                if env.now < window.start:
                    yield env.timeout(window.start - env.now)
                if window.rsu_id in failed_rsus:
                    event("rsu_failure", env.now, rsu_id=window.rsu_id, reason="configured_or_sampled_failure")
                    continue
                contact_attempt_index = 0
                for shard_id in placement.get(window.rsu_id, []):
                    if tracker.decodable or bool(state["expired"]) or env.now >= base_obj.expires_at:
                        break
                    if rank_filter and not tracker.would_increase_rank(
                        shard_id, vector=symbol_vector(shard_id)
                    ):
                        event("request_skipped", env.now, rsu_id=window.rsu_id,
                              shard_id=shard_id, reason="known_duplicate_or_dependent")
                        continue
                    tx_bytes = base_obj.shard_bytes
                    duration = tx_bytes * 8.0 / window.rate_bps
                    request_allowed = algorithm2_should_request(
                        rank=tracker.rank, k=base_obj.k, now=env.now,
                        expire_at=base_obj.expires_at, expected_completion=duration,
                    )
                    if enforce_deadline_capacity and env.now + duration >= base_obj.expires_at:
                        request_allowed = False
                    if not request_allowed:
                        event("shard_transmission", env.now, rsu_id=window.rsu_id, shard_id=shard_id,
                              version=int(state["version"]), bytes=0, innovative=False, loss=False,
                              failure_reason="deadline_gate")
                        break
                    if env.now + duration > window.end + 1e-12:
                        event("shard_transmission", env.now, rsu_id=window.rsu_id, shard_id=shard_id,
                              version=int(state["version"]), bytes=0, innovative=False, loss=False,
                              failure_reason="contact_capacity")
                        break
                    start = float(env.now)
                    tx_version = int(state["version"])
                    yield env.timeout(duration)
                    packet_uniform = (keyed_uniform(
                        self.seed, vehicle_id, episode_index, "packet_loss",
                        rsu_id=window.rsu_id, contact_start=float(window.start),
                        contact_end=float(window.end), attempt_index=contact_attempt_index,
                    ) if random_model == "keyed_v3" else rng.random())
                    stochastic_fields = ({"packet_uniform": packet_uniform,
                                          "contact_attempt_index": contact_attempt_index}
                                         if random_model == "keyed_v3" else {})
                    contact_attempt_index += 1  # Only an actual transfer advances this contact.
                    lost = packet_uniform < float(self.config.get("packet_loss", 0.0))
                    reason = "packet_loss" if lost else ""
                    innovative = False
                    if lost:
                        event("packet_loss", env.now, rsu_id=window.rsu_id, shard_id=shard_id, version=tx_version)
                    elif tx_version != int(state["version"]):
                        reason = "stale_version"
                        event("stale_version", env.now, rsu_id=window.rsu_id, shard_id=shard_id,
                              version=tx_version, expected_version=int(state["version"]))
                    elif not valid_metadata(shard_id, window.rsu_id, tx_version, env.now):
                        reason = "metadata_rejected"
                    else:
                        innovative = tracker.observe(shard_id, vector=symbol_vector(shard_id))
                        if not innovative:
                            reason = "duplicate_or_dependent"
                    records.append(TransmissionRecord(window.rsu_id, vehicle_id, tx_version, shard_id,
                                                      start, env.now, tx_bytes, innovative, lost, reason))
                    event("shard_transmission", env.now, rsu_id=window.rsu_id, shard_id=shard_id,
                          version=tx_version, start=start, end=float(env.now), bytes=tx_bytes,
                          innovative=innovative, loss=lost, failure_reason=reason,
                          **stochastic_fields)
                    if tracker.decodable and state["decode_time"] is None:
                        state["decode_time"] = float(env.now)
                        event("decode", env.now, version=int(state["version"]), rank=tracker.rank)

        def recovery_process():
            enabled = bool(self.config.get("repair_enabled", False) or self.config.get("migration_enabled", False))
            if not enabled:
                return
            start = float(self.config.get("repair_start_s", self.config.get("migration_start_s", 0.0)))
            if enforce_deadline_capacity:
                start = max(start, base_obj.created_at)
            if start > env.now:
                yield env.timeout(start - env.now)
            source = "migration" if bool(self.config.get("migration_enabled", False)) else "repair"
            rate = float(self.config.get("backhaul_rate_bps", self.config.get("link_rate_bps", 1_000_000)))
            while tracker.deficit > 0 and not bool(state["expired"]):
                missing = next((sid for sid in base_obj.shard_ids if sid not in tracker.seen), None)
                if missing is None:
                    break
                duration = base_obj.shard_bytes * 8.0 / rate
                if (env.now + duration > base_obj.expires_at
                    or (enforce_deadline_capacity and env.now + duration >= base_obj.expires_at)):
                    event(source, env.now, shard_id=missing, bytes=0, accepted=False, reason="deadline")
                    break
                start_time = float(env.now)
                yield env.timeout(duration)
                innovative = tracker.observe(missing, vector=symbol_vector(missing))
                counters["backhaul"] += base_obj.shard_bytes
                counters[source] += base_obj.shard_bytes
                records.append(TransmissionRecord(source, vehicle_id, int(state["version"]), missing,
                                                  start_time, env.now, base_obj.shard_bytes, innovative, False,
                                                  "" if innovative else "duplicate_or_dependent"))
                event(source, env.now, shard_id=missing, bytes=base_obj.shard_bytes, accepted=innovative,
                      start=start_time, end=float(env.now))
                if tracker.decodable and state["decode_time"] is None:
                    state["decode_time"] = float(env.now)
                    event("decode", env.now, version=int(state["version"]), rank=tracker.rank)
                break

        for window in own_windows:
            env.process(contact_event_process(window))
        env.process(version_update_process())
        env.process(expiry_process())
        env.process(transmission_process())
        env.process(recovery_process())
        horizon = max(base_obj.expires_at, max((w.end for w in own_windows), default=0.1)) + 1e-6
        env.run(until=horizon)

        decode_time = state["decode_time"]
        fresh = tracker.decodable and decode_time is not None and float(decode_time) <= base_obj.expires_at and not bool(state["stale_decode"])
        events.sort(key=lambda item: (float(item.get("time", 0.0)), str(item.get("event", ""))))
        return EpisodeResult(
            config_hash=config_hash(dict(self.config)), seed=self.seed + episode_index,
            vehicle_id=vehicle_id, object_id=base_obj.object_id, version=int(state["version"]),
            path=[w.rsu_id for w in own_windows], contact_windows=[_window_dict(w) for w in own_windows],
            rank=tracker.rank, t_decode=float(decode_time) if decode_time is not None else None,
            t_expire=base_obj.expires_at, fresh_success=bool(fresh),
            stale=bool(state["stale_decode"] or (tracker.decodable and not fresh)),
            storage=sum(len(v) for v in placement.values()) * base_obj.shard_bytes,
            backhaul=counters["backhaul"], migration=counters["migration"], repair=counters["repair"],
            verification=bool(self.config.get("verification", False)), transmissions=records, events=events)


def run_episodes(config: dict, seed: int, episodes: int) -> list[EpisodeResult]:
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    sim = Simulation(config, seed)
    return [sim.run_episode(i) for i in range(episodes)]
