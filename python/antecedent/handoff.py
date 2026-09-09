"""Handoffs to external estimators that consume an adjustment set.

Antecedent identifies. The adapter emits the set and status for estimands that
actually are adjustment estimands. It does not wrap EconML learners or absorb
ML CATE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import CausalUnsupportedError, CausalValueError
from .estimation import IdentifyResult
from .identify import Identification
from .ids import Identifier
from .results import AnalysisResult, IdentificationView

_ADJUSTMENT_IDENTIFIERS = frozenset(
    {
        Identifier.BACKDOOR_ADJUSTMENT,
        Identifier.BACKDOOR_EFFICIENT,
        Identifier.GENERALIZED_ADJUSTMENT,
        Identifier.RESPONSE_BACKDOOR,
    }
)

_GRAPH_POSTERIOR_MARKERS = ("posterior", "mcmc")


@dataclass(frozen=True, slots=True)
class EconMLSpec:
    """Adjustment-set handoff for an EconML-style CATE estimator.

    ``confounders`` is ``W`` in EconML's ``fit(Y, T, X=X, W=W)``. Heterogeneity
    features ``X`` are the caller's: identification does not name them.
    """

    treatment: str
    outcome: str
    confounders: tuple[str, ...]
    identifier: str
    status: str

    def columns(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Pull ``Y``, ``T``, and ``W`` columns from a name→array mapping."""
        missing = [
            name for name in (self.outcome, self.treatment, *self.confounders) if name not in data
        ]
        if missing:
            raise CausalValueError(f"EconML handoff missing columns: {missing}")
        arrays = {
            name: np.asarray(data[name])
            for name in (self.outcome, self.treatment, *self.confounders)
        }
        if any(col.ndim != 1 for col in arrays.values()):
            raise CausalValueError("EconML handoff requires one-dimensional columns")
        if len({len(col) for col in arrays.values()}) != 1:
            raise CausalValueError("EconML handoff columns must have the same number of rows")
        return {
            "Y": arrays[self.outcome],
            "T": arrays[self.treatment],
            "W": None
            if not self.confounders
            else np.column_stack([arrays[name] for name in self.confounders]),
        }


def econml(
    result: AnalysisResult | Identification | IdentifyResult,
    *,
    treatment: str | None = None,
    outcome: str | None = None,
    structure_source: str | None = None,
) -> EconMLSpec:
    """Emit an EconML adjustment-set spec, or refuse.

    Licensed only for point-identified backdoor / generalized-adjustment
    estimands on an explicit or accepted graph. Front-door, IV, general ID,
    temporal estimands (whose offsets this spec cannot represent),
    partial identification, and graph-posterior mixtures refuse rather than
    pretending they are a single adjustment set.
    """
    status, method, adjustment, identifier, treatment, outcome = _unpack(
        result, treatment=treatment, outcome=outcome
    )
    resolved = identifier or method
    if structure_source == "graph_posterior" or _graph_posterior_result(result):
        raise CausalUnsupportedError(
            "EconML handoff refuses graph-posterior mixtures; there is no single "
            "adjustment set to pass to another estimator"
        )
    if resolved == Identifier.TEMPORAL_BACKDOOR_UNFOLDED:
        raise CausalUnsupportedError(
            "EconML handoff cannot preserve temporal offsets; construct explicitly aligned "
            "treatment, outcome, and adjustment columns before fitting an external estimator"
        )
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
    return EconMLSpec(
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(adjustment),
        identifier=resolved,
        status=status,
    )


def _unpack(
    result: AnalysisResult | Identification | IdentifyResult,
    *,
    treatment: str | None,
    outcome: str | None,
) -> tuple[str, str, list[str], str | None, str, str]:
    if isinstance(result, Identification):
        treatment, outcome = _require_pair(
            treatment or getattr(result.query, "treatment", None),
            outcome or getattr(result.query, "outcome", None),
        )
        return (
            result.status,
            result.method,
            list(result.adjustment_set),
            result.identifier,
            treatment,
            outcome,
        )
    if isinstance(result, IdentifyResult):
        treatment, outcome = _require_pair(treatment, outcome)
        return result.status, result.method, list(result.adjustment_set), None, treatment, outcome
    if isinstance(result, AnalysisResult):
        ident = result.identification
        plan_id = result.plan.identifier if result.plan is not None else None
        treatment, outcome = _require_pair(treatment, outcome)
        return ident.status, ident.method, list(ident.adjustment_set), plan_id, treatment, outcome
    raise CausalValueError(
        f"EconML handoff expects AnalysisResult, Identification, or IdentifyResult; "
        f"got {type(result)!r}"
    )


def _require_pair(treatment: str | None, outcome: str | None) -> tuple[str, str]:
    if not treatment or not outcome:
        raise CausalValueError(
            "EconML handoff needs treatment= and outcome= (the result does not carry them)"
        )
    return str(treatment), str(outcome)


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
