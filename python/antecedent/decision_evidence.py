"""Experimental evidence for decision-relevant outcomes.

An experiment often estimates several effects at once (order frequency, basket value, unit
cost) while the decision turns on a quantity built from all of them. This module keeps what each
step from estimate to decision may say:

* :class:`JointEvidence` holds several effects of *one* experiment with the covariance of the
  arm-mean estimators, or states that only marginals exist; marginals are never promoted to a
  joint law.
* :class:`DerivedQuantity` is a function of those outcomes (a nonlinear margin, say). Its
  inputs are measured components, values bounded by a declared assumption, or declared missing.
  :func:`evaluate` reports a delta-method sampling interval and a separate assumption range;
  neither inherits a support license or calibration from its components.
* A :class:`DecisionFrame` classifies the decision as ``robust``, ``assumption_dependent`` or
  ``unresolved`` and names what is unresolved.
* ``candidates`` rank next evidence under a declared planning model.
* :func:`reuse_receipt` states which stages a change invalidates and which it reuses.

Every number, identity and refusal is computed by the Rust core. A refusal raises
:class:`DecisionEvidenceRefusal`, a typed refusal that names the next attempt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalValueError

__all__ = [
    "Assumed",
    "DecisionEvidenceRefusal",
    "DecisionFrame",
    "DerivedQuantity",
    "EvidenceLedger",
    "Evaluation",
    "Expr",
    "Fixed",
    "JointEvidence",
    "Measured",
    "Missing",
    "ReuseReceipt",
    "compile_experiment",
    "evaluate",
    "reuse_receipt",
]

STAGES: tuple[str, ...] = ("premise", "adjustment", "estimate", "derived_quantity", "decision")


class DecisionEvidenceRefusal(StructuredRefusal):
    """A typed refusal of composed experimental evidence.

    ``detail`` is a ``decision_evidence.*`` reason and ``remedy`` the next attempt.
    """


def _call(result: tuple[str | None, str | None]) -> dict[str, Any]:
    value, refusal = result
    raise_refusal(refusal, DecisionEvidenceRefusal)
    if value is None:  # pragma: no cover - the native layer returns one or the other
        raise CausalValueError("the native layer returned neither a result nor a refusal")
    parsed: dict[str, Any] = json.loads(value)
    return parsed


@dataclass(frozen=True)
class Expr:
    """An expression over the inputs of a derived quantity (the decision contract's language)."""

    wire: Mapping[str, Any]

    @staticmethod
    def const(value: float) -> Expr:
        """A constant."""
        return Expr({"const": float(value)})

    @staticmethod
    def input(index: int) -> Expr:
        """The ``index``th input of the quantity."""
        return Expr({"input": int(index)})

    def _binary(self, op: str, other: Expr | float) -> Expr:
        return Expr({"op": op, "args": [dict(self.wire), dict(_lift(other).wire)]})

    def __add__(self, other: Expr | float) -> Expr:
        return self._binary("add", other)

    def __sub__(self, other: Expr | float) -> Expr:
        return self._binary("sub", other)

    def __mul__(self, other: Expr | float) -> Expr:
        return self._binary("mul", other)

    def __neg__(self) -> Expr:
        return Expr({"op": "neg", "args": [dict(self.wire)]})

    def minimum(self, other: Expr | float) -> Expr:
        """Pointwise minimum; the delta method is then unavailable."""
        return self._binary("min", other)

    def maximum(self, other: Expr | float) -> Expr:
        """Pointwise maximum; the delta method is then unavailable."""
        return self._binary("max", other)


def _lift(value: Expr | float) -> Expr:
    return value if isinstance(value, Expr) else Expr.const(float(value))


@dataclass(frozen=True)
class Measured:
    """An input read from the arm mean of a measured component."""

    component: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "measured", "component": self.component}


@dataclass(frozen=True)
class Fixed:
    """An input known without uncertainty."""

    value: float

    def _wire(self) -> dict[str, Any]:
        return {"kind": "fixed", "value": float(self.value)}


@dataclass(frozen=True)
class Assumed:
    """An input not measured, bounded by a declared assumption (never turned into a probability)."""

    lower: float
    upper: float
    assumption_id: str
    shared_across_arms: bool = True

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "assumed",
            "lower": float(self.lower),
            "upper": float(self.upper),
            "assumption_id": self.assumption_id,
            "shared_across_arms": self.shared_across_arms,
        }


@dataclass(frozen=True)
class Missing:
    """An input that is needed and unavailable; the answer is unresolved, never defaulted."""

    what: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "missing", "what": self.what}


@dataclass(frozen=True)
class DerivedQuantity:
    """A function of experimental outcomes, evaluated in each arm at that arm's means.

    The contrast is the treated value minus the control value: a function of means, not the mean
    of a function.
    """

    name: str
    units: str
    expr: Expr
    inputs: Sequence[Measured | Fixed | Assumed | Missing]

    def _wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "expr": dict(self.expr.wire),
            "inputs": [i._wire() for i in self.inputs],
        }


@dataclass(frozen=True)
class DecisionFrame:
    """Two actions and the threshold on the contrast that separates them."""

    treated_action: str
    control_action: str
    threshold: float = 0.0

    def _wire(self) -> dict[str, Any]:
        return {
            "treated_action": self.treated_action,
            "control_action": self.control_action,
            "threshold": float(self.threshold),
        }


def _exposure(kind: str, assumption_id: str | None, bias_bounds: Sequence[float] | None) -> dict[str, Any]:
    if kind == "unspecified":
        return {"kind": "unspecified"}
    body: dict[str, Any] = {"kind": kind, "assumption_id": assumption_id}
    if kind == "bounded_spillover":
        body["bias_bounds"] = [float(b) for b in bias_bounds or ()]
    return body


@dataclass(frozen=True)
class JointEvidence:
    """Several effects of one experiment, kept together.

    Build one from cluster totals with :meth:`from_cluster_totals`, which estimates the arm
    means and the covariance of their estimators in the Rust core.
    """

    wire: Mapping[str, Any]

    @property
    def evidence_id(self) -> str:
        """Identity of the observations and assignment the estimates came from."""
        return str(self.wire["evidence_id"])

    @staticmethod
    def from_cluster_totals(
        *,
        evidence_id: str,
        population_id: str,
        treated_regime_id: str,
        control_regime_id: str,
        horizon: int,
        components: Mapping[str, tuple[str, str]],
        totals: Sequence[Sequence[float]],
        assigned: Sequence[bool],
        outcome_rows: int,
        exposure: str = "unspecified",
        exposure_assumption_id: str | None = None,
        spillover_bias_bounds: Sequence[float] | None = None,
        adjustments: Sequence[Mapping[str, Any]] = (),
    ) -> JointEvidence:
        """Estimate cluster-randomized outcomes jointly.

        ``components`` maps each outcome name to ``(variable_id, units)``; ``totals[k][c]`` is
        outcome ``k`` summed over the rows of cluster ``c`` (adjusted already by any fixed
        pre-assignment coefficient declared in ``adjustments``); ``assigned[c]`` is the realized
        arm. ``exposure`` is ``"no_between_unit_interference"``, ``"bounded_spillover"`` or
        ``"unspecified"``; the first two are assumptions you declare, not findings.
        """
        if len(components) != len(totals):
            raise CausalValueError("components must align with the outcome totals")
        raw = _native.decision_evidence_cluster_contrast(
            [[float(v) for v in row] for row in totals], [bool(a) for a in assigned], int(outcome_rows)
        )
        if raw is None:
            raise CausalValueError(
                "the totals cannot support a joint estimate: need aligned finite totals, at least "
                "two clusters in each arm and a positive row count"
            )
        c = json.loads(raw)
        return JointEvidence(
            {
                "evidence_id": evidence_id,
                "population_id": population_id,
                "treated_regime_id": treated_regime_id,
                "control_regime_id": control_regime_id,
                "horizon": int(horizon),
                "components": [
                    {"name": name, "variable_id": vid, "units": units}
                    for name, (vid, units) in components.items()
                ],
                "treated_means": c["treated_means"],
                "control_means": c["control_means"],
                "dependence": {
                    "kind": "joint",
                    "treated": c["treated_covariance"],
                    "control": c["control_covariance"],
                },
                "design": {
                    "assignment_design": "cluster",
                    "treated_units": c["treated_clusters"],
                    "control_units": c["control_clusters"],
                    "outcome_rows": int(outcome_rows),
                },
                "adjustments": [dict(a) for a in adjustments],
                "exposure": _exposure(exposure, exposure_assumption_id, spillover_bias_bounds),
            }
        )

    def marginals_only(self) -> JointEvidence:
        """The same evidence with the cross-component covariance withheld.

        A quantity that combines several components is then unresolved
        (``joint_law_missing``): dependence is never assumed away.
        """
        wire = dict(self.wire)
        k = len(wire["components"])
        dep = wire["dependence"]
        wire["dependence"] = {
            "kind": "marginals_only",
            "treated": [dep["treated"][i * k + i] for i in range(k)],
            "control": [dep["control"][i * k + i] for i in range(k)],
        }
        return JointEvidence(wire)


def compile_experiment(
    *,
    outcomes: Mapping[str, tuple[str, str]],
    assignment_design: str,
    population_id: str,
    treated_regime_id: str,
    control_regime_id: str,
    horizon: int,
    exposure: str,
    exposure_assumption_id: str | None = None,
    adjustments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """State the design, admissible adjustments and obligations before anything is fitted."""
    spec = {
        "outcomes": [
            {"name": n, "variable_id": vid, "units": units} for n, (vid, units) in outcomes.items()
        ],
        "assignment_design": assignment_design,
        "population_id": population_id,
        "treated_regime_id": treated_regime_id,
        "control_regime_id": control_regime_id,
        "horizon": int(horizon),
        "adjustments": [dict(a) for a in adjustments],
        "exposure": _exposure(exposure, exposure_assumption_id, None),
    }
    return _call(_native.decision_evidence_compile(json.dumps(spec)))


@dataclass(frozen=True)
class Evaluation:
    """The answer for a derived quantity, and for a decision when a frame was given."""

    raw: Mapping[str, Any]

    @property
    def derived(self) -> Mapping[str, Any]:
        """Contrast, sampling interval, assumption range, unresolved reasons and standing."""
        return self.raw["decision"]["derived"] if "decision" in self.raw else self.raw["derived"]

    @property
    def verdict(self) -> str | None:
        """``robust``, ``assumption_dependent`` or ``unresolved``; ``None`` without a frame."""
        return self.raw["decision"]["verdict"]["class"] if "decision" in self.raw else None

    @property
    def verdict_detail(self) -> Mapping[str, Any] | None:
        """The classification with its action, assumptions or reasons."""
        return self.raw["decision"]["verdict"] if "decision" in self.raw else None

    @property
    def plan(self) -> Mapping[str, Any] | None:
        """Ranked next evidence when candidates were given."""
        return self.raw.get("plan")

    @property
    def stage_digests(self) -> Mapping[str, str] | None:
        """Digest of each stage of the composed procedure, for :func:`reuse_receipt`."""
        return self.raw.get("stage_digests")


def evaluate(
    evidence: JointEvidence,
    quantity: DerivedQuantity,
    *,
    population_id: str | None = None,
    treated_regime_id: str | None = None,
    control_regime_id: str | None = None,
    horizon: int | None = None,
    deployment: bool = False,
    bridge_assumption_id: str | None = None,
    frame: DecisionFrame | None = None,
    candidates: Sequence[Mapping[str, Any]] = (),
) -> Evaluation:
    """Derive ``quantity`` over ``evidence`` and, with ``frame``, classify the decision.

    The request defaults to the evidence's own population, regimes and horizon; naming anything
    else is refused. ``deployment=True`` asks about a broader policy and is refused unless a
    structural ``bridge_assumption_id`` is declared. ``candidates`` are dictionaries with
    ``kind`` of ``"more_units"`` (``additional_units_per_arm``) or ``"resolve_assumption"``
    (``assumption_id``, ``narrowed_lower``, ``narrowed_upper``), each with an ``id`` and an
    optional ``cost``.
    """
    w = evidence.wire
    request: dict[str, Any] = {
        "population_id": population_id or w["population_id"],
        "treated_regime_id": treated_regime_id or w["treated_regime_id"],
        "control_regime_id": control_regime_id or w["control_regime_id"],
        "horizon": int(horizon if horizon is not None else w["horizon"]),
        "target": {"kind": "deployment", "bridge_assumption_id": bridge_assumption_id}
        if deployment
        else {"kind": "experimental_contrast"},
    }
    payload: dict[str, Any] = {
        "evidence": dict(w),
        "request": request,
        "quantity": quantity._wire(),
    }
    if frame is not None:
        payload["frame"] = frame._wire()
    if candidates:
        payload["candidates"] = [dict(c) for c in candidates]
    return Evaluation(_call(_native.decision_evidence_evaluate(json.dumps(payload))))


@dataclass(frozen=True)
class ReuseReceipt:
    """Which stages of a composed procedure a change reuses and which it invalidates."""

    status: Mapping[str, str]

    def reused(self, stage: str) -> bool:
        """Whether ``stage`` may be reused."""
        return self.status[stage] == "reused"


def reuse_receipt(previous: Evaluation, current: Evaluation) -> ReuseReceipt:
    """Compare two evaluations stage by stage (each digest covers the stages before it)."""
    a, b = previous.stage_digests, current.stage_digests
    if a is None or b is None:
        raise CausalValueError("reuse needs evaluations made with a decision frame")
    return ReuseReceipt({s: "reused" if a[s] == b[s] else "invalidated" for s in STAGES})


@dataclass
class EvidenceLedger:
    """Which evidence is in force; withdrawing evidence withdraws what was composed from it."""

    _status: dict[str, str | None] = field(default_factory=dict)

    def record(self, evidence_id: str) -> None:
        """Record evidence as active."""
        self._status.setdefault(evidence_id, None)

    def withdraw(self, evidence_id: str, reason: str) -> None:
        """Withdraw evidence with a reason."""
        if evidence_id not in self._status:
            raise CausalValueError(f"evidence `{evidence_id}` was never recorded")
        self._status[evidence_id] = reason

    def unsupported_parts(self, composed_evidence_id: str) -> list[tuple[str, str]]:
        """Parts of a composed id (joined by ``+``) that are withdrawn or unrecorded."""
        out: list[tuple[str, str]] = []
        for part in composed_evidence_id.split("+"):
            if part not in self._status:
                out.append((part, "not recorded in the ledger"))
            elif self._status[part] is not None:
                out.append((part, str(self._status[part])))
        return out
