"""Replicate shards: one calibration job split over disjoint replicate ranges.

A calibration test draws its replicates through `map_replicates`, and replicate `k`'s
outcome is a function of `k` alone (the test derives its seeds from the index), so the
replicates `a..b` of a job can be computed apart from the rest. The harness
(crates/antecedent/tests/common/calibration.rs) already extends a finished run: a run
at `ANTECEDENT_CALIBRATION_REPLICATE_START=k` seeds its tallies from the first run's
tally file and folds replicates `k..n` on top, in replicate order, printing exactly the
lines a fresh run at `n` prints. A shard is the same run bounded above:
`ANTECEDENT_CALIBRATION_REPLICATE_END=e` computes `a..e` on empty tallies, asserts and
prints nothing, and writes each tally's raw state, with every interval length in
replicate order, to its tally file.

`merge` folds the states of consecutive shards onto the prior state of the range's
start (or onto nothing at replicate 0) and produces the prior of an ordinary extension
over the last, short, range, the tail. That run does all the asserting and printing on
the merged counts, so the log, the tally sidecar and the decision are those of one run
over the whole range. The merge is exact because every field is either an integer count,
a min/max, a concatenation in replicate order (the replicate fingerprints), or the
interval-length sum, which is re-folded in replicate order from the shard's lengths
rather than adding partial sums (floating-point addition is not associative).

A shard set is refused, and the caller falls back to the plain run, when a state is
missing its replicate range, ranges do not join, two states disagree on a construction,
level or decision context, or a tally exists in a shard but not in the prior.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MAX_DIGEST_PREFIX = 1000  # replicate fingerprints a tally retains (calibration.rs)
HEX_PER_FINGERPRINT = 32  # 16 bytes, hex

START_ENV = "ANTECEDENT_CALIBRATION_REPLICATE_START"
END_ENV = "ANTECEDENT_CALIBRATION_REPLICATE_END"
PRIOR_ENV = "ANTECEDENT_CALIBRATION_PRIOR_TALLIES"
TALLY_ENV = "ANTECEDENT_CALIBRATION_TALLY_OUT"

# Fewest replicates a range needs before it is worth splitting.
MIN_SHARD_REPLICATES = 4


class ShardError(ValueError):
    """The shard states cannot be merged into the state of one run."""


def plan_ranges(start: int, end: int, shards: int, tail: int | None = None
                ) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Split `start..end` into up to `shards` parallel ranges and a short tail.

    The tail is the extension that does the asserting once the others are merged: it is
    about 5% of the range (at least one replicate per shard) so it adds little wall time,
    and it runs on every core the shards used. `None` when the range is too small to split."""
    length = end - start
    if shards < 2 or length < MIN_SHARD_REPLICATES * shards:
        return None
    tail = tail if tail is not None else max(shards, math.ceil(length / 20))
    body = length - tail
    if body < shards or tail < 1:
        return None
    ranges, begin = [], start
    for i in range(shards):
        take = math.ceil((start + body - begin) / (shards - i))
        ranges.append((begin, begin + take))
        begin += take
    return ranges, (begin, end)


def load_states(path: Path) -> dict[str, dict]:
    """The tally states in a tally file, by key; a key written twice is refused."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        if state["key"] in out:
            raise ShardError(f"two tallies wrote the key {state['key']}")
        out[state["key"]] = state
    return out


def _min_opt(a, b):
    return b if a is None else a if b is None else min(a, b)


def _merge_record(key: str, acc: dict | None, new: dict | None) -> dict | None:
    if acc is None and new is None:
        return None
    if acc is None or new is None:
        raise ShardError(f"{key}: a record backs the tally in one run and not in another")
    for field in ("construction", "validation"):
        a, b = acc.get(field), new.get(field)
        if a is not None and b is not None and a != b:
            raise ShardError(f"{key}: replicates were keyed under different {field}s")
    return {
        "construction": acc.get("construction") if acc.get("construction") is not None else new.get("construction"),
        "validation": acc.get("validation") if acc.get("validation") is not None else new.get("validation"),
        "n_min": min(acc["n_min"], new["n_min"]),
        "n_max": max(acc["n_max"], new["n_max"]),
        "replicates_min": _min_opt(acc.get("replicates_min"), new.get("replicates_min")),
        "posterior_draws_min": _min_opt(acc.get("posterior_draws_min"), new.get("posterior_draws_min")),
        "unidentified_mass_max": max(acc["unidentified_mass_max"], new["unidentified_mass_max"]),
        "bound": acc["bound"] + new["bound"],
    }


def merge(prior: dict[str, dict] | None, shards: list[tuple[int, int, dict[str, dict]]]) -> dict[str, dict]:
    """The tally states after the shards `(start, end, states)`, in range order, folded onto `prior`.

    `prior` is the state of the run that computed replicates `0..first start` (`None` when the
    first shard starts at 0). The result is what a single run over `0..last end` writes, except
    for the prefix digests, which the next run recomputes from the fingerprints."""
    if not shards:
        raise ShardError("no shards to merge")
    first = shards[0][0]
    if prior is None and first != 0:
        raise ShardError(f"the first shard starts at {first}, not 0, and there is no prior state")
    acc: dict[str, dict] = {}
    for key, state in (prior or {}).items():
        if state.get("map_end") != first:
            raise ShardError(f"{key}: the prior computed {state.get('map_end')} replicates, not {first}")
        acc[key] = {**state, "length_sum": state["length_sum"], "event_digests": state.get("event_digests", ""),
                    "prefix_digests": {}}
    cursor = first
    for start, end, states in shards:
        if start != cursor or end <= start:
            raise ShardError(f"shard {start}..{end} does not join the previous range ending at {cursor}")
        for key, state in states.items():
            if state.get("map_end") != end:
                raise ShardError(f"{key}: the shard {start}..{end} reports map_end {state.get('map_end')}")
            if key not in acc:
                if prior is not None:
                    raise ShardError(f"{key}: written by a shard but not by the run it extends")
                acc[key] = {
                    "key": key, "level": state["level"], "covered": 0, "scored": 0, "skipped": 0,
                    "with_interval": 0, "length_sum": 0.0, "map_end": start, "event_digests": "",
                    "prefix_digests": {}, "decision_context": state.get("decision_context"),
                    "record": None if state["record"] is None else {
                        "construction": None, "validation": None, "n_min": 2**64 - 1, "n_max": 0,
                        "replicates_min": None, "posterior_draws_min": None,
                        "unidentified_mass_max": 0.0, "bound": 0},
                }
            mine = acc[key]
            if mine["level"] != state["level"]:
                raise ShardError(f"{key}: levels differ between runs")
            if mine.get("decision_context") != state.get("decision_context"):
                raise ShardError(f"{key}: decision contexts differ between runs")
            for field in ("covered", "scored", "skipped", "with_interval"):
                mine[field] += state[field]
            length_sum = mine["length_sum"]
            for length in state.get("lengths", ()):
                length_sum += length  # replicate order, as a single run adds them
            mine["length_sum"] = length_sum
            if len(state.get("lengths", ())) != state["with_interval"]:
                raise ShardError(f"{key}: the shard did not keep every interval length")
            events = mine["event_digests"] + state.get("event_digests", "")
            mine["event_digests"] = events[: MAX_DIGEST_PREFIX * HEX_PER_FINGERPRINT]
            mine["record"] = _merge_record(key, mine["record"], state["record"])
        for mine in acc.values():
            mine["map_end"] = end
        cursor = end
    return acc


def write_states(path: Path, states: dict[str, dict]) -> None:
    """A tally file of merged states, in the form the harness reads as a prior."""
    lines = []
    for key in sorted(states):
        state = {k: v for k, v in states[key].items() if k not in ("lengths",)}
        if state.get("decision_context") is None:
            state.pop("decision_context", None)
        lines.append(json.dumps(state, sort_keys=True))
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
