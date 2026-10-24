"""Decide whether a lane goal was met from retained repeats. Pure functions, no I/O.

A cell is one control run by one arm at one row tier and thread count:

    {"outcome": "completed" | "unsuccessful" | "oom",
     "repeats": [{"section_ns", "peak_rss_bytes", "data_digest", "accounting", ...}, ...]}

Every repeat counts. None is dropped for being slow, fast or failed. A goal is met only when
both arms completed, ran the same data, and agree exactly on the integer and label fields of
their results (replicate counts, support statuses, test counts): a speedup that comes with
different accounting did different work and is rejected.
"""

from __future__ import annotations

import random

MIN_REPEATS_FOR_INTERVAL = 5
GOALS = {"first_answer": 2.0, "reuse_throughput": 10.0}
CAPACITY_TIER_STEPS = {"row_capacity": 1, "exact_capacity": 2}  # tiers are decades


def median(values: list[int]) -> int:
    """Median; an even count takes the floor of the midpoint of the two central values."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of no values")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def spread(values: list[int]) -> dict[str, int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def column(cell: dict, key: str) -> list[int]:
    return [rep[key] for rep in cell["repeats"] if rep.get(key) is not None]


def completed(cell: dict | None) -> bool:
    return bool(cell) and cell["outcome"] == "completed" and bool(cell["repeats"])


def relative_change(
    frozen_ns: list[int],
    candidate_ns: list[int],
    *,
    seed: int = 0,
    resamples: int = 2000,
) -> dict[str, float] | None:
    """Candidate time relative to frozen, as the Criterion comparison reports it: the
    change of the median (negative is faster) and the lower end of its 95 percent
    percentile-bootstrap interval. Absent when either side has fewer than five repeats,
    because an interval from fewer is not an interval."""
    if min(len(frozen_ns), len(candidate_ns)) < MIN_REPEATS_FOR_INTERVAL:
        return None
    rng = random.Random(seed)
    point = median(candidate_ns) / median(frozen_ns) - 1.0
    draws = sorted(
        median(rng.choices(candidate_ns, k=len(candidate_ns)))
        / median(rng.choices(frozen_ns, k=len(frozen_ns)))
        - 1.0
        for _ in range(resamples)
    )
    return {"point_estimate": point, "lower_bound": draws[int(0.025 * resamples)]}


def same_work(first: dict, second: dict) -> tuple[bool, str]:
    """Whether two completed cells ran the same data and agree on exact result fields."""
    if not (completed(first) and completed(second)):
        return False, "an arm did not complete"
    digests = {
        rep["data_digest"] for cell in (first, second) for rep in cell["repeats"]
    }
    if len(digests) != 1:
        return False, "the arms ran different data"
    exact = {
        repr(sorted(rep["accounting"].items()))
        for cell in (first, second)
        for rep in cell["repeats"]
    }
    if len(exact) != 1:
        return False, "the arms disagree on counts or statuses in the result"
    return True, "same data and same result accounting"


def speedup(slow: dict, fast: dict) -> float:
    return median(column(slow, "section_ns")) / median(column(fast, "section_ns"))


def first_answer_verdict(frozen: dict, candidate: dict, goal: float) -> dict:
    ok, why = same_work(frozen, candidate)
    if not ok:
        return {"target_met": False, "reason": why}
    gained = speedup(frozen, candidate)
    return {
        "target_met": gained >= goal,
        "speedup": gained,
        "goal": goal,
        "reason": f"median speedup {gained:.3f} against goal {goal:g}",
    }


def reuse_verdict(
    frozen_solo: dict | None, frozen_prepared: dict | None, candidate: dict, goal: float
) -> dict:
    """The candidate must beat both the independent solo calls and the prepared API."""
    parts = {}
    for name, baseline in (("solo", frozen_solo), ("prepared", frozen_prepared)):
        if baseline is None:
            return {"target_met": False, "reason": f"no {name} baseline was run"}
        parts[name] = first_answer_verdict(baseline, candidate, goal)
    return {
        "target_met": all(part["target_met"] for part in parts.values()),
        "goal": goal,
        "against": parts,
        "reason": "; ".join(
            f"{name}: {part['reason']}" for name, part in parts.items()
        ),
    }


def largest_completed(cells: dict[str, dict], tiers: tuple[str, ...]) -> int | None:
    done = [index for index, tier in enumerate(tiers) if completed(cells.get(tier))]
    return max(done) if done else None


def capacity_verdict(
    frozen: dict[str, dict],
    candidate: dict[str, dict],
    tiers: tuple[str, ...],
    steps: int,
    time_factor: float = 1.0,
) -> dict:
    """`steps` decades more rows than the frozen build reaches, in the frozen build's peak
    resident set and within `time_factor` times its time at its own largest tier."""
    top_frozen = largest_completed(frozen, tiers)
    top_candidate = largest_completed(candidate, tiers)
    if top_frozen is None:
        return {"target_met": False, "reason": "the frozen build completed no tier"}
    if top_candidate is None:
        return {"target_met": False, "reason": "the candidate completed no tier"}
    wanted = top_frozen + steps
    if wanted >= len(tiers):
        return {
            "target_met": False,
            "reason": (
                f"the frozen build reached {tiers[top_frozen]}; the tier {steps} decade(s) "
                "above it is not registered, so the goal cannot be tested"
            ),
        }
    if top_candidate < wanted or not completed(candidate.get(tiers[wanted])):
        return {
            "target_met": False,
            "reason": f"the candidate reached {tiers[top_candidate]}, goal {tiers[wanted]}",
        }
    base, reach = frozen[tiers[top_frozen]], candidate[tiers[wanted]]
    ok, why = (
        same_work(base, candidate[tiers[top_frozen]])
        if completed(candidate.get(tiers[top_frozen]))
        else (False, "the candidate has no run at the frozen build's largest tier")
    )
    ram_ok = median(column(reach, "peak_rss_bytes")) <= median(
        column(base, "peak_rss_bytes")
    )
    time_ok = median(column(reach, "section_ns")) <= time_factor * median(
        column(base, "section_ns")
    )
    return {
        "target_met": ok and ram_ok and time_ok,
        "frozen_tier": tiers[top_frozen],
        "candidate_tier": tiers[wanted],
        "same_work": ok,
        "within_peak_rss": ram_ok,
        "within_time": time_ok,
        "reason": (
            f"candidate {tiers[wanted]} vs frozen {tiers[top_frozen]}: {why}; "
            f"peak RSS {'within' if ram_ok else 'over'} the frozen peak; "
            f"time {'within' if time_ok else 'over'} {time_factor:g}x the frozen time"
        ),
    }
