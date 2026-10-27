"""Tier-aware overlap and E-value diagnostics of an executed tiered result.

A tiered study (``analyze(data, query, graph=TieredBackground(...))``) runs one of two licensed
average-effect cells. :func:`tier_diagnostics` reads only what the executed result carries and
states which diagnostic is available for it; nothing is refit and no within-tier graph is built::

    from antecedent import tier_diagnostics

    result = antecedent.analyze(data, query, graph=background)
    report = tier_diagnostics.tier_diagnostics(result)
    report.design.kind            # "co_determined_closure" or "unknown_scenarios"
    report.overlap.available      # False with a typed reason when none is carried
    report.evalue.point.value     # the point E-value, when the tier cell attached it
    report.evalue.interval        # always typed unavailable: no coverage record measures it

* **Overlap.** Available for a ``CoDetermined`` result whose estimator fitted a propensity score.
  Typed unavailable for an ``Unknown`` result and for an estimator that fits none.
* **E-value, point.** The ``CoDetermined`` cell's own point value with its method. Typed
  unavailable for ``Unknown``, and when the value on the result came from the E-value refuter
  (it embeds a converted interval endpoint).
* **E-value, interval.** Never reported.

The E-value is a sensitivity summary of the stated no-latent-path premise, never a test of it.
Every number, availability decision and refusal is computed in Rust.

A result that is not a licensed single-treatment tiered average-effect result is refused with
``cell_not_licensed`` (``tier_diagnostics.result_not_licensed``) or ``invalid_argument``
(``tier_diagnostics.not_a_tiered_average_result``); a cancelled ``cancel`` token stops the read with
``cancelled_no_claim`` (``tier_diagnostics.cancelled``) and no diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._scientific import ScientificResult
from .errors import CausalTypeError

__all__ = [
    "Availability",
    "TierDesign",
    "TierDiagnostics",
    "TierEvalue",
    "TierScenario",
    "tier_diagnostics",
]


@dataclass(frozen=True, slots=True)
class Availability(ScientificResult):
    """A diagnostic that is either carried (``value``) or typed unavailable.

    ``value`` is the diagnostic's own field mapping when ``available``; otherwise ``code``,
    ``detail`` and ``reason`` say what the executed result does not carry.
    """

    available: bool
    value: Mapping[str, Any] | None = None
    code: str | None = None
    detail: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TierScenario(ScientificResult):
    """One declared orientation scenario of an ``Unknown`` tiered result."""

    method: str
    adjustment: tuple[str, ...]
    effect: float | None


@dataclass(frozen=True, slots=True)
class TierDesign(ScientificResult):
    """The design the tiered result executed.

    ``kind`` is ``"co_determined_closure"`` (``adjustment``, ``rows``, ``propensity_scored``) or
    ``"unknown_scenarios"`` (``scenarios``; never one adjustment set and never one graph).
    """

    kind: str
    adjustment: tuple[str, ...] = ()
    rows: int | None = None
    propensity_scored: bool = False
    scenarios: tuple[TierScenario, ...] = ()


@dataclass(frozen=True, slots=True)
class TierEvalue(ScientificResult):
    """The point E-value where licensed; the interval value is always unavailable."""

    point: Availability
    interval: Availability


@dataclass(frozen=True, slots=True)
class TierDiagnostics(ScientificResult):
    """The diagnostics a tiered result carries: design, overlap and E-value."""

    design: TierDesign
    overlap: Availability
    evalue: TierEvalue


def _availability(wire: Mapping[str, Any]) -> Availability:
    if wire.get("available"):
        value = {k: v for k, v in wire.items() if k != "available"}
        return Availability(True, value)
    return Availability(
        False,
        None,
        str(wire["code"]),
        str(wire["detail"]),
        str(wire["reason"]),
    )


def _design(wire: Mapping[str, Any]) -> TierDesign:
    if wire["kind"] == "co_determined_closure":
        return TierDesign(
            "co_determined_closure",
            adjustment=tuple(wire["adjustment"]),
            rows=wire["rows"],
            propensity_scored=bool(wire["propensity_scored"]),
        )
    return TierDesign(
        "unknown_scenarios",
        scenarios=tuple(
            TierScenario(str(s["method"]), tuple(s["adjustment"]), s["effect"])
            for s in wire["scenarios"]
        ),
    )


def tier_diagnostics(result: Any, *, cancel: Any = None) -> TierDiagnostics:
    """Read the tier diagnostics of the most recent estimate of a tiered study.

    Args:
        result: The :class:`~antecedent.results.AnalysisResult` of a tiered study (its retained
            study is read) or the :class:`~antecedent.prepared.PreparedAnalysis` that ran it.
            The diagnostics are those of the study's most recent estimate.
        cancel: An optional cancellation token; observed on entry and before each read.

    Raises:
        CausalUnsupportedError: the result is not a licensed single-treatment tiered
            average-effect result, or the read was cancelled (see the module notes).
        CausalTypeError: ``result`` carries no prepared study.
    """
    study = getattr(result, "study", result)
    native = getattr(getattr(study, "_engine", study), "_native", None)
    reader = getattr(native, "tier_diagnostics_json", None)
    if reader is None:
        raise CausalTypeError(
            "tier_diagnostics needs the AnalysisResult or PreparedAnalysis of an estimated study"
        )
    wire = json.loads(reader(cancel=cancel))
    return TierDiagnostics(
        design=_design(wire["design"]),
        overlap=_availability(wire["overlap"]),
        evalue=TierEvalue(
            point=_availability(wire["evalue"]["point"]),
            interval=_availability(wire["evalue"]["interval"]),
        ),
    )
