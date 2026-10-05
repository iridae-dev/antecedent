"""Conditional odds ratio from matched case-control sets (point only).

The data are matched sets sampled on the outcome: each stratum holds one or more
cases and their matched controls, with a binary exposure. The one estimand is the
**conditional odds ratio** of exposure, estimated by the exact conditional
likelihood (conditional logistic regression), which removes the set-specific
intercepts that outcome-dependent sampling distorts. The declaration is explicit:
``sampling="matched_case_control"`` is required.

Sets that cannot inform the odds ratio are counted, never silently dropped: sets
with no case or no control (singletons included) and sets whose members all share
one exposure value contribute nothing to the likelihood.

``result.export()`` writes a JSON artifact carrying the set-shape sufficient statistic
and a digest; :func:`replay_matched_case_control` rebuilds the sets from it and
recomputes the estimate and counts bit for bit.

Refused, with a registered reason code:

* any risk-scale estimand (``population_risk``, ``absolute_risk``, ``risk_difference``,
  ``risk_ratio``): ``effect_not_identified``, because the case fraction is fixed by the
  design and no prevalence or selection information is taken;
* an interval (``level=...``): ``cell_not_licensed``, because the asymptotic
  conditions of a Wald interval are not measured by any coverage record;
* a set collection with no informative set (``effect_not_identified``) or whose
  conditional likelihood has no finite maximizer (``route_not_supported``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._native import CancellationToken
from ._native import matched_case_control_odds_ratio as _matched_case_control_odds_ratio
from .errors import CausalResourceError, CausalSerializationError, CausalTypeError

__all__ = [
    "MatchedCaseControlResult",
    "MatchedSetCounts",
    "matched_case_control_odds_ratio",
    "replay_matched_case_control",
]

_FORMAT = "matched_case_control_v1"
_MAX_REPLAY_ROWS = 10_000_000


@dataclass(frozen=True, slots=True)
class MatchedSetCounts:
    """What every matched set contributed; the last three classes carry no information.

    ``total == informative + exposure_concordant + outcome_degenerate``.
    ``singleton`` counts sets of one member and is a subset of ``outcome_degenerate``.
    """

    total: int
    informative: int
    exposure_concordant: int
    outcome_degenerate: int
    singleton: int


@dataclass(frozen=True, slots=True)
class MatchedCaseControlResult:
    """The conditional odds ratio; ``claim`` is ``point_only`` and no interval exists.

    ``set_types`` is the sufficient statistic of the declared sets: one
    ``(members, cases, exposed, exposed_cases, sets)`` row per distinct set shape,
    sorted. The conditional likelihood depends on the data only through it.
    """

    odds_ratio: float
    log_odds_ratio: float
    counts: MatchedSetCounts
    set_types: tuple[tuple[int, int, int, int, int], ...]
    estimand: str = "conditional_odds_ratio"
    sampling: str = "matched_case_control"
    claim: str = "point_only"

    def export(self) -> str:
        """A self-describing JSON artifact that :func:`replay_matched_case_control` re-derives."""
        body: dict[str, Any] = {
            "format": _FORMAT,
            "sampling": self.sampling,
            "estimand": self.estimand,
            "claim": self.claim,
            "set_types": [list(row) for row in self.set_types],
            "counts": _counts_dict(self.counts),
            "log_odds_ratio": self.log_odds_ratio,
            "odds_ratio": self.odds_ratio,
        }
        return json.dumps({**body, "digest": _digest(body)}, sort_keys=True)


def _counts_dict(counts: MatchedSetCounts) -> dict[str, int]:
    return {
        "total": counts.total,
        "informative": counts.informative,
        "exposure_concordant": counts.exposure_concordant,
        "outcome_degenerate": counts.outcome_degenerate,
        "singleton": counts.singleton,
    }


def _digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _set_types(
    labels: list[str], case: list[float], exposed: list[float]
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Distinct set shapes ``(members, cases, exposed, exposed_cases)`` with their multiplicity."""
    tallies: dict[str, list[int]] = {}
    for label, y, x in zip(labels, case, exposed, strict=True):
        tally = tallies.setdefault(label, [0, 0, 0, 0])
        tally[0] += 1
        tally[1] += int(y)
        tally[2] += int(x)
        tally[3] += int(y and x)
    shapes: dict[tuple[int, int, int, int], int] = {}
    for n, k, t, a in tallies.values():
        shapes[(n, k, t, a)] = shapes.get((n, k, t, a), 0) + 1
    return tuple((n, k, t, a, sets) for (n, k, t, a), sets in sorted(shapes.items()))


def _floats(name: str, values: Iterable[Any]) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{name} must be a sequence of 0/1 numbers") from error


def matched_case_control_odds_ratio(
    *,
    stratum: Iterable[Any],
    case: Iterable[Any],
    exposed: Iterable[Any],
    sampling: str,
    estimand: str = "conditional_odds_ratio",
    level: float | None = None,
    cancel: CancellationToken | None = None,
) -> MatchedCaseControlResult:
    """Conditional odds ratio of matched sets by exact conditional likelihood.

    ``stratum``, ``case`` (1 = case, 0 = control) and ``exposed`` (1 = exposed) are
    parallel row sequences; stratum labels are compared as their ``str``. ``sampling``
    must be ``"matched_case_control"`` (sets sampled on the outcome). ``level`` exists
    only so that asking for an interval is a typed refusal.
    """
    labels = [str(s) for s in stratum]
    case_values = _floats("case", case)
    exposed_values = _floats("exposed", exposed)
    estimate = _matched_case_control_odds_ratio(
        labels,
        case_values,
        exposed_values,
        sampling=sampling,
        estimand=estimand,
        interval=level is not None,
        cancel=cancel,
    )
    log_or, odds_ratio, total, informative, concordant, degenerate, singleton = estimate
    return MatchedCaseControlResult(
        set_types=_set_types(labels, case_values, exposed_values),
        odds_ratio=odds_ratio,
        log_odds_ratio=log_or,
        counts=MatchedSetCounts(
            total=total,
            informative=informative,
            exposure_concordant=concordant,
            outcome_degenerate=degenerate,
            singleton=singleton,
        ),
    )


def _bad(why: str) -> CausalSerializationError:
    return CausalSerializationError(f"matched case-control artifact: {why}")


def _expanded_rows(set_types: object, max_rows: int) -> tuple[list[str], list[float], list[float]]:
    """Rebuild one representative row list from the stored sufficient statistic."""
    if not isinstance(set_types, list):
        raise _bad("set_types must be a list")
    stratum: list[str] = []
    case: list[float] = []
    exposed: list[float] = []
    for row in set_types:
        if not (
            isinstance(row, list)
            and len(row) == 5
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in row)
        ):
            raise _bad("a set type must be five non-negative integers")
        n, k, t, a, sets = row
        if not (0 <= a <= min(k, t) and t - a <= n - k and k <= n and n >= 1 and sets >= 1):
            raise _bad("a set type is not a possible matched set")
        if len(stratum) + n * sets > max_rows:
            raise CausalResourceError(
                f"matched case-control artifact expands past max_rows={max_rows}"
            )
        members = [(1.0, 1.0)] * a + [(1.0, 0.0)] * (k - a)
        members += [(0.0, 1.0)] * (t - a) + [(0.0, 0.0)] * (n - k - (t - a))
        for _ in range(sets):
            label = str(len(stratum))
            for y, x in members:
                stratum.append(label)
                case.append(y)
                exposed.append(x)
    return stratum, case, exposed


def replay_matched_case_control(
    artifact: str,
    *,
    max_rows: int = _MAX_REPLAY_ROWS,
    cancel: CancellationToken | None = None,
) -> MatchedCaseControlResult:
    """Re-derive an exported result from its stored sufficient statistic.

    The digest is checked, the sets are rebuilt from ``set_types`` and the conditional
    odds ratio and every count are recomputed and compared bit for bit with the stored
    ones. A tampered, unknown-format or non-reproducing artifact is refused with
    :class:`~antecedent.errors.CausalSerializationError`; one that would expand past
    ``max_rows`` rows is refused with ``CausalResourceError`` before any work.
    """
    try:
        body = json.loads(artifact)
    except (TypeError, ValueError) as error:
        raise _bad("not valid JSON") from error
    if not isinstance(body, dict) or body.get("format") != _FORMAT:
        raise _bad(f"format must be {_FORMAT!r}")
    stored = body.pop("digest", None)
    if stored != _digest(body):
        raise _bad("digest does not match the content")
    expected_keys = {
        "format",
        "sampling",
        "estimand",
        "claim",
        "set_types",
        "counts",
        "log_odds_ratio",
        "odds_ratio",
    }
    if set(body) != expected_keys or body["claim"] != "point_only":
        raise _bad("unexpected fields or claim")
    if not (isinstance(body["sampling"], str) and isinstance(body["estimand"], str)):
        raise _bad("sampling and estimand must be strings")
    stratum, case, exposed = _expanded_rows(body["set_types"], max_rows)
    again = matched_case_control_odds_ratio(
        stratum=stratum,
        case=case,
        exposed=exposed,
        sampling=body["sampling"],
        estimand=body["estimand"],
        cancel=cancel,
    )
    if (
        again.log_odds_ratio != body["log_odds_ratio"]
        or again.odds_ratio != body["odds_ratio"]
        or _counts_dict(again.counts) != body["counts"]
        or json.loads(json.dumps(again.set_types)) != body["set_types"]
    ):
        raise _bad("the stored result does not reproduce from its sufficient statistic")
    return again
