"""Canonical portable projection for structured result families.

The live result carries a rich Python dataclass per family (``result.panel_did``,
``result.survival``, ...); an exported artifact carries a leaner Rust wire body
whose field names differ (``effect`` for ``estimate``, ``graphless_support_status``
for ``support_status``, ...). :attr:`Answer.structured` must be identical whether it
came from the live result or from ``load(result.export())`` -- the golden-path
invariant ``load(result.export()).answer == result.answer``.

This module is the single projection both sides use. It maps each side onto one
canonical portable claim dict (wire names, JSON-native types), so the two agree by
construction. It is deliberately a projection of the *portable claim numbers*: the
descriptive metadata a family also carries (assumptions, diagnostics, design labels,
cohort/period identity, full bands) stays on ``result.<family>`` and is not part of
the portable answer. Nothing is removed from the result object; this only governs
``Answer.structured``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


def _normalize(value: Any) -> Any:
    """JSON-native, order-stable form so live (tuples) and wire (lists) compare equal."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            item = _normalize(item)
            if item is None or item == [] or item == ():
                continue  # matches the wire's omit-when-empty for conditional fields
            out[str(key)] = item
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


class _FamilySpec:
    """How one structured family projects onto its canonical portable claim dict.

    Most families reconcile with a rename table plus an exclusion set. A family whose
    live dataclass and wire body decompose differently (e.g. a list of typed effects
    vs. parallel arrays) supplies ``live_fn`` / ``wire_fn`` callables instead.
    """

    __slots__ = ("live_renames", "exclude", "live_fn", "wire_fn")

    def __init__(
        self,
        *,
        live_renames: Mapping[str, str] | None = None,
        exclude: frozenset[str] = frozenset(),
        live_fn: Any = None,
        wire_fn: Any = None,
    ) -> None:
        # live dataclass field name -> canonical (wire) name, for the twins whose
        # spellings differ across the FFI boundary.
        self.live_renames = dict(live_renames or {})
        # canonical names that are one-sided descriptive metadata (present on only one
        # side, always), excluded from the portable claim on both sides.
        self.exclude = exclude
        self.live_fn = live_fn
        self.wire_fn = wire_fn

    def from_live(self, section: Any) -> dict[str, Any]:
        if self.live_fn is not None:
            return _normalize(self.live_fn(section))
        raw = (
            asdict(section)
            if is_dataclass(section) and not isinstance(section, type)
            else dict(section)
        )
        renamed = {self.live_renames.get(key, key): item for key, item in raw.items()}
        return _normalize({k: v for k, v in renamed.items() if k not in self.exclude})

    def from_wire(self, wire: Mapping[str, Any]) -> dict[str, Any]:
        if self.wire_fn is not None:
            return _normalize(self.wire_fn(wire))
        return _normalize({k: v for k, v in wire.items() if k not in self.exclude})


# --- panel_did: one slot, three decompositions (plain DiD, augmented, event study) ---

_PANEL_PLAIN = _FamilySpec(
    live_renames={
        "estimate": "effect",
        "control_subjects": "comparison_subjects",
        "support_status": "graphless_support_status",
    },
    exclude=frozenset({"assumptions", "cohort", "design", "period"}),
)


def _event_study_uncertainty(any_interval: bool) -> str:
    return (
        "event_time_pointwise_normal_intervals_independent_clusters"
        if any_interval
        else "cluster_robust_se_only_pointwise_cr1_unlicensed"
    )


def _panel_from_live(section: Any) -> dict[str, Any]:
    if getattr(section, "effective_control_sample_size", None) is not None:  # augmented DiD
        return {
            "effect": section.estimate,
            "treated_subjects": section.treated_subjects,
            "comparison_subjects": section.control_subjects,
            "clusters": section.clusters,
            "uncertainty": section.uncertainty,
            "propensity_min": section.propensity_min,
            "propensity_max": section.propensity_max,
            "effective_control_sample_size": section.effective_control_sample_size,
            "nuisance_predictions_cross_fitted": section.nuisance_predictions_cross_fitted,
        }
    effects = getattr(section, "effects", None)
    if effects is None:
        return _PANEL_PLAIN.from_live(section)
    rows = [
        {
            "cohort": e.cohort,
            "period": e.period,
            "event_time": e.event_time,
            "estimate": e.estimate,
            "standard_error": e.standard_error,
            "treated_subjects": e.treated_subjects,
            "control_subjects": e.control_subjects,
            "clusters": e.clusters,
            "interval_95": e.interval_95,
        }
        for e in effects
    ]
    any_interval = any(e.interval_95 is not None for e in effects)
    return {"event_time_effects": rows, "uncertainty": _event_study_uncertainty(any_interval)}


def _panel_from_wire(wire: Mapping[str, Any]) -> dict[str, Any]:
    augmented = wire.get("augmented")
    if augmented is not None:  # augmented DiD: propensity_min, max, ESS, cross_fitted
        p_min, p_max, ess, declared = augmented
        return {
            "effect": wire.get("effect"),
            "treated_subjects": wire.get("treated_subjects"),
            "comparison_subjects": wire.get("comparison_subjects"),
            "clusters": wire.get("clusters"),
            "uncertainty": wire.get("uncertainty"),
            "propensity_min": p_min,
            "propensity_max": p_max,
            "effective_control_sample_size": ess,
            "nuisance_predictions_cross_fitted": declared,
        }
    tuples = wire.get("event_time_effects")
    if not tuples:
        return _PANEL_PLAIN.from_wire(wire)
    intervals = wire.get("event_time_intervals_95") or [None] * len(tuples)
    rows = [
        {
            "cohort": t[0],
            "period": t[1],
            "event_time": t[2],
            "estimate": t[3],
            "standard_error": t[6],
            "treated_subjects": t[4],
            "control_subjects": t[5],
            "clusters": t[7],
            "interval_95": interval,
        }
        for t, interval in zip(tuples, intervals, strict=True)
    ]
    any_interval = any(interval is not None for interval in intervals)
    return {"event_time_effects": rows, "uncertainty": _event_study_uncertainty(any_interval)}


#: slot on :class:`AnalysisResult` -> its portable-claim projection.
STRUCTURED_FAMILIES: dict[str, _FamilySpec] = {
    "policy_value": _FamilySpec(
        live_renames={
            "policy_value_interval_95": "policy_interval_95",
            "incremental_value_interval_95": "incremental_interval_95",
            "policy_value_standard_error": "policy_standard_error",
            "incremental_value_standard_error": "incremental_standard_error",
            "reference_value_standard_error": "reference_standard_error",
            "total_treatment_cost": "total_cost",
            "support_status": "graphless_support_status",
        },
        exclude=frozenset({"assumptions", "diagnostics", "evaluation_method"}),
    ),
    "panel_did": _FamilySpec(live_fn=_panel_from_live, wire_fn=_panel_from_wire),
    "survival": _FamilySpec(
        live_renames={
            "control_survival": "control",
            "treated_survival": "treated",
            "survival_at_tau_difference_interval": "difference_at_tau_interval",
        },
        exclude=frozenset(
            {
                # live-only descriptive / derived
                "assumptions",
                "censoring_survival_provenance",
                "difference_band",
                "rmst_difference",
                "support_status",
                # wire-only diagnostics
                "assignment_counts",
                "minimum_event_risk_set",
                "target_cause",
            }
        ),
    ),
    "longitudinal_regime": _FamilySpec(
        exclude=frozenset(
            {
                "observed_subjects",
                "period_effects",
                "period_intervals_95",
                "rule_id",
                "rule_provenance",
                "rule_version",
                "stabilizing_numerator_probabilities",
                "standard_errors",
                "support_status",
            }
        ),
    ),
    "continuous_dose_response": _FamilySpec(),
}


def canonical_from_live(slot: str, section: Any) -> dict[str, Any]:
    """Portable claim dict from a live result section dataclass."""
    return STRUCTURED_FAMILIES[slot].from_live(section)


def canonical_from_wire(slot: str, wire: Mapping[str, Any]) -> dict[str, Any]:
    """Portable claim dict from an exported artifact's wire body."""
    return STRUCTURED_FAMILIES[slot].from_wire(wire)
