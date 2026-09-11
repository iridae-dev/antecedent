"""Handoffs to external estimators that consume an adjustment set.

Antecedent identifies. The adapter emits the set and status for estimands that
actually are adjustment estimands. It does not wrap EconML learners or absorb
ML CATE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .errors import CausalUnsupportedError, CausalValueError
from .estimation import IdentifyResult
from .identify import Identification
from .ids import Identifier
from .results import AnalysisResult, CausalResponseView, IdentificationView

_ADJUSTMENT_IDENTIFIERS = frozenset(
    {
        Identifier.BACKDOOR_ADJUSTMENT,
        Identifier.BACKDOOR_EFFICIENT,
        Identifier.GENERALIZED_ADJUSTMENT,
        Identifier.RESPONSE_BACKDOOR,
        Identifier.TEMPORAL_BACKDOOR_UNFOLDED,
    }
)

_GRAPH_POSTERIOR_MARKERS = ("posterior", "mcmc")


@dataclass(frozen=True, slots=True)
class EconMLSpec:
    """Adjustment-set handoff for an EconML-style CATE estimator.

    ``confounders`` is ``W`` in EconML's ``fit(Y, T, X=X, W=W)``. Heterogeneity
    features ``X`` are the caller's: identification does not name them.
    """

    treatment: str | tuple[str, ...]
    outcome: str
    confounders: tuple[str, ...]
    identifier: str
    status: str
    treatment_offsets: tuple[int, ...] = ()
    outcome_offset: int = 0
    confounder_offsets: tuple[int, ...] = ()
    modifiers: tuple[str, ...] = ()
    temporal: bool = False
    target_weights: Any | None = None
    outcome_functional: Any | None = None

    @property
    def treatments(self) -> tuple[str, ...]:
        """All jointly intervened variables, in certificate order."""
        return (self.treatment,) if isinstance(self.treatment, str) else self.treatment

    def columns(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Return aligned Y, T, W and certified conditioning features X.

        For temporal specs the input is one regularly sampled series in row order.
        Offset k selects row origin+k. Boundary rows are trimmed, never wrapped.
        ``origins`` identifies retained rows relative to the input series; separate
        subjects must be aligned separately to avoid crossing panel boundaries.
        """
        names = (self.outcome, *self.treatments, *self.confounders, *self.modifiers)
        missing = [name for name in names if name not in data]
        if missing:
            raise CausalValueError(f"EconML handoff missing columns: {missing}")
        arrays = {name: np.asarray(data[name]) for name in names}
        if any(col.ndim != 1 for col in arrays.values()):
            raise CausalValueError("EconML handoff requires one-dimensional columns")
        if len({len(col) for col in arrays.values()}) != 1:
            raise CausalValueError("EconML handoff columns must have the same number of rows")
        t_offsets = self.treatment_offsets or (0,) * len(self.treatments)
        w_offsets = self.confounder_offsets or (0,) * len(self.confounders)
        offsets = (0, self.outcome_offset, *t_offsets, *w_offsets)
        n = len(arrays[self.outcome])
        origins = np.arange(max(0, -min(offsets)), min(n, n - max(offsets)))
        if self.temporal and not len(origins):
            raise CausalValueError("EconML handoff has no rows after temporal alignment")
        targets = [
            arrays[name][origins + offset]
            for name, offset in zip(self.treatments, t_offsets, strict=True)
        ]
        cols = {
            "Y": arrays[self.outcome][origins + self.outcome_offset],
            "T": targets[0] if len(targets) == 1 else np.column_stack(targets),
            "W": np.column_stack(
                [
                    arrays[name][origins + offset]
                    for name, offset in zip(self.confounders, w_offsets, strict=True)
                ]
            )
            if self.confounders
            else None,
        }
        if self.modifiers:
            cols["X"] = np.column_stack([arrays[name][origins] for name in self.modifiers])
        if self.outcome_functional is not None:
            from .query import coerce_outcome_functional

            functional = coerce_outcome_functional(self.outcome_functional)
            if functional is not None:
                y = np.asarray(cols["Y"])
                if functional["kind"] == "exceedance":
                    cols["Y"] = (y > float(cast(float, functional["threshold"]))).astype(float)
                elif functional["kind"] == "exceedance_grid":
                    cols["Y"] = np.column_stack(
                        [(y > c).astype(float) for c in cast(list[float], functional["thresholds"])]
                    )
        if self.target_weights is not None:
            weights = np.asarray(self.target_weights, dtype=float)
            if (
                weights.ndim != 1
                or len(weights) != n
                or not np.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise CausalValueError(
                    "target weights must be finite non-negative weights aligned to input rows"
                )
            aligned = weights[origins]
            if aligned.sum() <= 0:
                raise CausalValueError("target weights have no mass on aligned rows")
            cols["sample_weight"] = aligned
        if self.temporal:
            cols["origins"] = origins
        return cols


def econml(
    result: AnalysisResult | CausalResponseView | Identification | IdentifyResult,
    *,
    treatment: str | tuple[str, ...] | None = None,
    outcome: str | None = None,
    structure_source: str | None = None,
    modifiers: str | tuple[str, ...] | Sequence[str] | None = None,
    target_weights: Any | None = None,
    target_depends_on: Sequence[str] = (),
    outcome_functional: Any | None = None,
) -> EconMLSpec:
    """Emit an EconML adjustment-set spec, or refuse.

    Licensed only for point-identified backdoor / generalized-adjustment
    estimands on an explicit or accepted graph. Front-door, IV, general ID,
    general temporal ID functionals,
    partial identification, and graph-posterior mixtures refuse rather than
    pretending they are a single adjustment set.
    """
    if structure_source == "graph_posterior" or _graph_posterior_result(result):
        raise CausalUnsupportedError(
            "EconML handoff refuses graph-posterior mixtures; there is no single "
            "adjustment set to pass to another estimator"
        )
    status, method, adjustment, identifier, treatment, outcome = _unpack(
        result, treatment=treatment, outcome=outcome
    )
    resolved = identifier or method
    if resolved not in _ADJUSTMENT_IDENTIFIERS:
        raise CausalUnsupportedError(
            f"EconML handoff requires a backdoor or generalized-adjustment "
            f"identifier; got {resolved!r}"
        )
    view = IdentificationView(
        status=status,
        method=method,
        adjustment_set=list(adjustment),
        assumption_count=0,
        derivation_step_count=0,
    )
    if (
        "partial" in status.lower()
        or "graphdependent" in status.lower()
        or "graph-dependent" in status.lower()
        or not view
    ):
        raise CausalUnsupportedError(
            f"EconML handoff requires point identification; got status {status!r}"
        )
    query = getattr(result, "query", getattr(result, "estimand", None))
    certificate = getattr(result, "certificate", None)
    temporal = resolved == Identifier.TEMPORAL_BACKDOOR_UNFOLDED or bool(
        certificate and str(certificate["graph_class"]).lower().startswith("temporal")
    )
    kwargs: dict[str, Any] = {}
    if certificate:
        sets = [
            case["adjustment_coordinates"][0]
            for case in certificate["cases"]
            if case["adjustment_coordinates"]
        ]
        if not sets or not all(item == sets[0] for item in sets):
            raise CausalUnsupportedError(
                "EconML handoff requires a common certified adjustment set"
            )
        adjustment = [item["name"] for item in sets[0]]
        kwargs["confounder_offsets"] = tuple(item["offset"] for item in sets[0])
        kwargs["treatment_offsets"] = tuple(item["offset"] for item in certificate["treatments"])
        kwargs["outcome_offset"] = certificate["outcome"]["offset"]
    elif temporal:
        raise CausalUnsupportedError(
            "EconML handoff needs an identification certificate preserving temporal offsets; "
            "use identify(graph=..., query=...)"
        )
    declared = _modifier_set(query, modifiers)
    forbidden = {treatment} if isinstance(treatment, str) else set(treatment)
    forbidden.add(outcome)
    if any(name in forbidden for name in declared):
        raise CausalValueError("EconML modifiers must be pre-treatment and not the outcome")
    certified_modifiers = set(adjustment) | set(_modifier_set(query, None))
    if any(name not in certified_modifiers for name in declared):
        raise CausalUnsupportedError(
            "EconML modifiers need a pre-treatment certificate: identify the modifier context first or use certified adjustment variables"
        )
    if any(name not in set(adjustment) or name in forbidden for name in target_depends_on):
        raise CausalValueError("target weights must depend only on certified adjustment variables")
    if target_weights is not None:
        weights = np.asarray(target_weights, dtype=float)
        if (
            weights.ndim != 1
            or not weights.size
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise CausalValueError(
                "target weights must be a finite non-negative vector with positive mass"
            )
        if not target_depends_on and not np.all(weights == weights[0]):
            raise CausalValueError("nonconstant target weights require declared target_depends_on")
    if outcome_functional is None:
        outcome_functional = getattr(query, "outcome_functional", None)
    return EconMLSpec(
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(adjustment),
        identifier=resolved,
        status=status,
        modifiers=declared,
        temporal=temporal,
        target_weights=None if target_weights is None else np.asarray(target_weights),
        outcome_functional=outcome_functional,
        **kwargs,
    )


def _modifier_set(
    query: Any, modifiers: str | tuple[str, ...] | Sequence[str] | None
) -> tuple[str, ...]:
    if modifiers is None:
        single = getattr(query, "modifier", None)
        many = getattr(query, "effect_modifiers", None)
        if isinstance(single, str):
            return (single,)
        if many:
            return tuple(str(name) for name in many)
        return ()
    if isinstance(modifiers, str):
        return (modifiers,)
    return tuple(str(name) for name in modifiers)


def _unpack(
    result: AnalysisResult | CausalResponseView | Identification | IdentifyResult,
    *,
    treatment: str | tuple[str, ...] | None,
    outcome: str | None,
) -> tuple[str, str, list[str], str | None, str | tuple[str, ...], str]:
    if not isinstance(result, (AnalysisResult, CausalResponseView, Identification, IdentifyResult)):
        raise CausalValueError(
            f"EconML handoff expects an identification or analysis result; got {type(result)!r}"
        )
    query = getattr(result, "query", getattr(result, "estimand", None))
    certificate = getattr(result, "certificate", None)
    certified_t = getattr(query, "treatment", None)
    certified_y = getattr(query, "outcome", None)
    if certificate:
        names = tuple(item["name"] for item in certificate["treatments"])
        certified_t = names[0] if len(names) == 1 else names
        certified_y = certificate["outcome"]["name"]
    elif query is not None and hasattr(query, "interventions"):
        names = tuple(item.variable for item in query.interventions)
        certified_t = names[0] if len(names) == 1 else names
    if not certified_t or not certified_y:
        raise CausalValueError(
            "EconML handoff needs a result carrying its certified treatment and outcome; "
            "caller-supplied names cannot establish query identity"
        )
    requested = (treatment,) if isinstance(treatment, str) else treatment
    expected = (certified_t,) if isinstance(certified_t, str) else certified_t
    if (requested is not None and requested != expected) or (
        outcome is not None and outcome != certified_y
    ):
        raise CausalValueError(
            "EconML handoff treatment/outcome must match the certified query, including every joint treatment"
        )
    if isinstance(result, (AnalysisResult, CausalResponseView)):
        ident = result.identification
        plan = getattr(result, "plan", None)
        plan_id = plan.identifier if plan is not None else None
        if plan_id is None and certificate:
            plan_id = certificate["method"]
        return (
            ident.status,
            ident.method,
            list(ident.adjustment_set),
            plan_id,
            certified_t,
            certified_y,
        )
    return (
        result.status,
        result.method,
        list(result.adjustment_set),
        getattr(result, "identifier", None),
        certified_t,
        certified_y,
    )


def _graph_posterior_result(result: object) -> bool:
    plan = getattr(result, "plan", None)
    source = getattr(plan, "structure_source", None) if plan is not None else None
    if source == "graph_posterior":
        return True
    algo = getattr(plan, "discovery_algorithm", None) if plan is not None else None
    return isinstance(algo, str) and any(
        marker in algo.lower() for marker in _GRAPH_POSTERIOR_MARKERS
    )


__all__ = ["EconMLSpec", "econml"]
