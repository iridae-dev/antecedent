"""Consume an assumption-sensitivity result in a durable decision (F17).

A 2.2 joint mechanism sensitivity answers "what could the effect be if the
assumption holds anywhere inside this declared range?". This module carries that
answer into a decision: the surface becomes a :class:`SensitivityArtifact`
(assumption coordinate, effect surface, tipping coordinates, coordinate support,
uncertainty relationship, provenance), and :func:`decide` evaluates a decision
contract over it::

    artifact = sensitivity_decision.SensitivityArtifact.from_joint_sensitivity(
        stage, JointDeviation({"outcome_kernel": 0.2}),
        effect=effect, actions=[act, wait], causal_contract_id="contract-1",
    )
    result = sensitivity_decision.decide(artifact.contract(), artifact)
    result.kind       # "invariant_action" | "assumption_dependent" | "no_robust_action"
    result.switch     # the tipping coordinate of an assumption-dependent switch

Three things stay distinct and are never merged:

* the **assumption range** (``lower``, ``upper`` at every grid point) is what a
  quantity may be if the assumption holds anywhere in the declared range; it is not
  a probability and not a confidence interval, so ``weights`` on a ranged surface
  refuse;
* an **identified bound** is a separate bound from partial identification;
* a **sampling interval** is reported next to the decision and never inside the
  range. Composing the two refuses with ``reason_code == "cell_not_licensed"``
  (``sensitivity_decision_composition.composition_not_licensed``) because no
  composition method is licensed.

Decisions are evaluated only at the declared grid points; nothing between them is
evaluated, so the interpolated crossing of a switch is labelled an interpolation.
Rust owns the artifact, its identity digests, the evaluation, the classification
and every refusal; this module builds declarations and raises each refusal as
:class:`SensitivityRefusal`, a :class:`~antecedent.errors.CausalUnsupportedError`
with its registered ``reason_code``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._native import SensitivityArtifact as _NativeSensitivityArtifact
from ._native import sensitivity_artifact_from_surface as _from_surface
from ._native import sensitivity_artifact_from_z_joint as _from_z_joint
from ._native import sensitivity_contract as _contract
from ._native import sensitivity_decide as _decide
from .decision import Contract, StructuralPolicy
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .joint_distribution import ScientificQuantity

ASSUMPTION_RANGE_STATEMENT = (
    "assumption range over the declared coordinate set; not a probability, "
    "not a confidence interval and not a sampling interval"
)
PointSupport = Literal["supported", "unsupported", "unevaluated"]
DecisionKind = Literal["invariant_action", "assumption_dependent", "no_robust_action", "unresolved"]


class SensitivityRefusal(CausalUnsupportedError):
    """A sensitivity-decision refusal carrying the structured Rust fields.

    ``reason_code`` is registered. ``detail`` is the namespaced
    ``sensitivity_decision_composition.<slot>`` (or the engine's own detail);
    ``message`` is the human-readable context. Typical details:
    ``composition_not_licensed`` (a sampling interval composed with the range),
    ``wrong_contract`` (mixed estimands, unlike units, weights on a ranged surface,
    an input the surface does not carry), ``unsupported_coordinate`` (an unsupported
    grid point inside the evaluated range), ``invalid_surface`` and
    ``bounds_exceeded``.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        message = refusal.get("message") or refusal.get("offending") or ""
        text = str(refusal["detail"]) + (f": {message}" if message else "")
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = refusal.get("stage", "")
        self.detail: str = refusal["detail"]
        self.message: str = str(message)


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise SensitivityRefusal(json.loads(refusal))


# --------------------------------------------------------------------------- terms


@dataclass(frozen=True, slots=True)
class Utility:
    """A closed utility expression over named surface quantities; build with operators."""

    _wire_value: Any

    def __add__(self, other: Utility | float) -> Utility:
        return Utility({"add": [self._wire_value, _coerce(other)._wire_value]})

    def __radd__(self, other: float) -> Utility:
        return _coerce(other) + self

    def __sub__(self, other: Utility | float) -> Utility:
        return Utility({"sub": [self._wire_value, _coerce(other)._wire_value]})

    def __rsub__(self, other: float) -> Utility:
        return _coerce(other) - self

    def __mul__(self, other: Utility | float) -> Utility:
        return Utility({"mul": [self._wire_value, _coerce(other)._wire_value]})

    def __rmul__(self, other: float) -> Utility:
        return _coerce(other) * self

    def __neg__(self) -> Utility:
        return Utility({"neg": self._wire_value})

    def __repr__(self) -> str:
        return f"Utility({json.dumps(self._wire_value, sort_keys=True)})"


def _coerce(value: Utility | float) -> Utility:
    if isinstance(value, Utility):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError("a utility combines Utility values and numbers")
    return const(float(value))


def quantity(variable_id: str) -> Utility:
    """The surface quantity with this ``variable_id``."""
    return Utility({"quantity": variable_id})


def const(value: float) -> Utility:
    """A constant."""
    return Utility({"const": float(value)})


def maximum(left: Utility | float, right: Utility | float) -> Utility:
    """Pointwise maximum (not multilinear: vertices then certify only the vertices)."""
    return Utility({"max": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


def minimum(left: Utility | float, right: Utility | float) -> Utility:
    """Pointwise minimum (not multilinear: vertices then certify only the vertices)."""
    return Utility({"min": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


@dataclass(frozen=True, slots=True)
class Action:
    """One action and the utility it reads from the surface quantities."""

    id: str
    utility: Utility

    def _wire(self) -> dict[str, Any]:
        return {"id": self.id, "utility": self.utility._wire_value}


# ------------------------------------------------------------------ declarations


@dataclass(frozen=True, slots=True)
class AssumptionCoordinate:
    """The declared assumption coordinate (for example ``gamma``) and its range."""

    id: str
    scale: str
    units: str
    minimum: float
    maximum: float

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scale": self.scale,
            "units": self.units,
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class SurfaceQuantity:
    """A surface quantity and its assumption range at every grid point.

    ``upper`` defaults to ``lower`` (a point surface).
    """

    quantity: ScientificQuantity
    lower: tuple[float, ...]
    upper: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        lower = tuple(float(v) for v in self.lower)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(
            self,
            "upper",
            lower if self.upper is None else tuple(float(v) for v in self.upper),
        )

    @property
    def ranged(self) -> bool:
        """Whether the assumption range is not a single point somewhere on the grid."""
        return self.lower != self.upper

    def _wire(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity._wire(),
            "lower": list(self.lower),
            "upper": list(self.upper or self.lower),
        }


@dataclass(frozen=True, slots=True)
class SamplingInterval:
    """A sampling interval of one surface quantity, kept apart from the assumption range.

    ``composed`` names a composition with the range; no method is licensed, so any
    value refuses with ``cell_not_licensed``.
    """

    quantity: str
    level: float
    method: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    composed: str | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "reported": {
                "quantity": self.quantity,
                "level": float(self.level),
                "method": self.method,
                "composed": self.composed,
                "lower": [float(v) for v in self.lower],
                "upper": [float(v) for v in self.upper],
            }
        }


@dataclass(frozen=True, slots=True)
class SamplingWithheld:
    """No sampling interval exists; the registered reason says why."""

    reason_code: str
    detail: str

    def _wire(self) -> dict[str, Any]:
        return {"withheld": {"reason_code": self.reason_code, "detail": self.detail}}


@dataclass(frozen=True, slots=True)
class IdentifiedBound:
    """A separate identified (partial-identification) bound of one surface quantity."""

    quantity: str
    lower: float
    upper: float
    source: str


@dataclass(frozen=True, slots=True)
class SourceTipping:
    """A tipping result of the producing analysis, retained as provenance (not recomputed)."""

    factor: str
    status: str
    lower: float | None
    upper: float | None
    analytic: float | None


@dataclass(frozen=True, slots=True)
class SurfaceProvenance:
    """Provenance of the numbers of a declared surface."""

    source_kind: str
    query_binding: str
    provider_snapshot: str
    source_regime: str
    method: str
    causal_contract_id: str
    decision_threshold: float | None = None
    source_tipping: tuple[SourceTipping, ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "query_binding": self.query_binding,
            "provider_snapshot": self.provider_snapshot,
            "source_regime": self.source_regime,
            "method": self.method,
            "causal_contract_id": self.causal_contract_id,
            "decision_threshold": self.decision_threshold,
            "source_tipping": [
                {
                    "factor": t.factor,
                    "status": t.status,
                    "lower": t.lower,
                    "upper": t.upper,
                    "analytic": t.analytic,
                }
                for t in self.source_tipping
            ],
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> SurfaceProvenance:
        return cls(
            source_kind=wire["source_kind"],
            query_binding=wire["query_binding"],
            provider_snapshot=wire["provider_snapshot"],
            source_regime=wire["source_regime"],
            method=wire["method"],
            causal_contract_id=wire["causal_contract_id"],
            decision_threshold=wire["decision_threshold"],
            source_tipping=tuple(SourceTipping(**item) for item in wire["source_tipping"]),
        )


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """The three uncertainty kinds, kept distinct."""

    assumption_range: str
    identified_bound: IdentifiedBound | None
    sampling: SamplingInterval | SamplingWithheld

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> Uncertainty:
        sampling = wire["sampling"]
        parsed: SamplingInterval | SamplingWithheld
        if "withheld" in sampling:
            parsed = SamplingWithheld(**sampling["withheld"])
        else:
            body = sampling["reported"]
            parsed = SamplingInterval(
                quantity=body["quantity"],
                level=body["level"],
                method=body["method"],
                lower=tuple(body["lower"]),
                upper=tuple(body["upper"]),
                composed=body["composed"],
            )
        bound = wire["identified_bound"]
        return cls(
            assumption_range=wire["assumption_range"]["interpretation"],
            identified_bound=None if bound is None else IdentifiedBound(**bound),
            sampling=parsed,
        )


# --------------------------------------------------------------------- outcomes


@dataclass(frozen=True, slots=True)
class DecisionSwitch:
    """A change of leader between grid points.

    ``lower``/``upper`` bracket the tipping coordinate; ``exact`` means an exact tie
    at one grid coordinate (``lower == upper``). ``interpolated`` is the linear
    crossing between the two bracketing grid points: it is exact only if the utility
    difference is linear between them, and is present only for a point surface.
    """

    from_actions: tuple[str, ...]
    to_actions: tuple[str, ...]
    lower: float
    upper: float
    exact: bool
    interpolated: float | None

    @property
    def bracket(self) -> tuple[float, float]:
        """The coordinate interval that contains the switch."""
        return (self.lower, self.upper)

    @property
    def tipping_coordinate(self) -> float | None:
        """The exact tie coordinate, else the labelled interpolation, else ``None``."""
        return self.lower if self.exact else self.interpolated

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> DecisionSwitch:
        return cls(
            from_actions=tuple(wire["from"]),
            to_actions=tuple(wire["to"]),
            lower=float(wire["lower"]),
            upper=float(wire["upper"]),
            exact=bool(wire["exact"]),
            interpolated=wire["interpolated"],
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    """The classification of the decision along the assumption coordinate.

    ``kind`` is ``invariant_action`` (one action is uniquely best at every grid point
    under every range vertex), ``assumption_dependent`` (the unique leader changes
    exactly once: ``switch``), ``no_robust_action`` (``reason`` says why) or
    ``unresolved`` (a grid point was not evaluated, so invariance is not checked).
    """

    kind: DecisionKind
    action: str | None = None
    switch: DecisionSwitch | None = None
    switches: tuple[DecisionSwitch, ...] = ()
    reason: str | None = None
    coordinates: tuple[float, ...] = ()
    unresolved_coordinate: float | None = None

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> Outcome:
        ((kind, body),) = wire.items()
        if kind == "invariant_action":
            return cls("invariant_action", action=body["action"])
        if kind == "assumption_dependent":
            return cls("assumption_dependent", switch=DecisionSwitch._from_wire(body["switch"]))
        if kind == "unresolved":
            return cls("unresolved", unresolved_coordinate=float(body["coordinate"]))
        reason = body["reason"]
        name: str
        coordinates: tuple[float, ...] = ()
        if isinstance(reason, str):
            name = reason
        else:
            ((name, detail),) = reason.items()
            coordinates = tuple(float(c) for c in detail["coordinates"])
        return cls(
            "no_robust_action",
            switches=tuple(DecisionSwitch._from_wire(s) for s in body["switches"]),
            reason=name,
            coordinates=coordinates,
        )

    def __str__(self) -> str:
        if self.kind == "invariant_action":
            return f"{self.action!r} is the invariant best action"
        if self.kind == "assumption_dependent":
            assert self.switch is not None
            return _switch_text(self.switch)
        if self.kind == "no_robust_action":
            return f"no action is robust over the assumption range ({self.reason})"
        return f"unresolved at coordinate {self.unresolved_coordinate}"


def _switch_text(switch: DecisionSwitch) -> str:
    before = " / ".join(repr(a) for a in switch.from_actions)
    after = " / ".join(repr(a) for a in switch.to_actions)
    if switch.exact:
        where = f"an exact tie at {switch.lower:g}"
    else:
        where = f"between {switch.lower:g} and {switch.upper:g}"
        if switch.interpolated is not None:
            where += f" (interpolated crossing {switch.interpolated:g})"
    return f"the best action changes from {before} to {after} at {where}"


@dataclass(frozen=True, slots=True)
class SamplingReport:
    """Sampling uncertainty reported next to the decision; ``composed`` is always ``False``."""

    status: Literal["withheld", "separate_not_composed"]
    reason_code: str | None = None
    detail: str | None = None
    quantity: str | None = None
    level: float | None = None
    method: str | None = None
    coordinates: tuple[float, ...] = ()
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    composed: bool = False


# ---------------------------------------------------------------------- artifact


class SensitivityArtifact:
    """A composition-ready assumption-sensitivity surface with its identity.

    The identity keeps the premises (coordinate, support, quantities, actions,
    uncertainty kinds, provenance) apart from the data (the surface numbers):
    :attr:`identity` holds ``premises_digest``, ``data_digest`` and ``digest``.
    """

    def __init__(self, native: _NativeSensitivityArtifact) -> None:
        self._native = native
        self._summary: dict[str, Any] = json.loads(native.summary_json)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_surface(
        cls,
        *,
        coordinate: AssumptionCoordinate,
        grid: Sequence[float],
        quantities: Sequence[SurfaceQuantity],
        actions: Sequence[Action],
        provenance: SurfaceProvenance,
        support: Sequence[PointSupport] | None = None,
        sampling: SamplingInterval | SamplingWithheld | None = None,
        identified_bound: IdentifiedBound | None = None,
        interpretation: str = ASSUMPTION_RANGE_STATEMENT,
    ) -> SensitivityArtifact:
        """Declare a surface directly (any order of grid points; they are sorted).

        Refuses with :class:`SensitivityRefusal` on a malformed surface
        (``invalid_surface``), mixed estimands or unlike units (``wrong_contract``), a
        grid point outside the declared range (``unsupported_coordinate``) and a
        sampling interval composed with the range (``composition_not_licensed``).
        """
        grid_values = [float(g) for g in grid]
        sampling_wire = (
            sampling
            if sampling is not None
            else SamplingWithheld(
                "cell_not_licensed", "sensitivity_decision_composition.sampling_not_reported"
            )
        )
        surface = {
            "coordinate": coordinate._wire(),
            "grid": grid_values,
            "support": list(support) if support is not None else ["supported"] * len(grid_values),
            "quantities": [q._wire() for q in quantities],
            "actions": [a._wire() for a in actions],
            "uncertainty": {
                "assumption_range": {"kind": "assumption_range", "interpretation": interpretation},
                "identified_bound": (
                    None
                    if identified_bound is None
                    else {
                        "quantity": identified_bound.quantity,
                        "lower": float(identified_bound.lower),
                        "upper": float(identified_bound.upper),
                        "source": identified_bound.source,
                    }
                ),
                "sampling": sampling_wire._wire(),
            },
            "provenance": provenance._wire(),
        }
        native, refusal = _from_surface(json.dumps(surface))
        _raise(refusal)
        assert native is not None
        return cls(native)

    @classmethod
    def from_joint_sensitivity(
        cls,
        stage: Any,
        deviation: Any,
        *,
        effect: ScientificQuantity,
        actions: Sequence[Action],
        causal_contract_id: str,
        grid_points: int | None = None,
        memory_bytes: int | None = None,
        cancel: Any = None,
    ) -> SensitivityArtifact:
        """Adapt the 2.2 joint mechanism sensitivity of a prepared z stage.

        ``stage`` and ``deviation`` are exactly what
        :func:`antecedent.transport.joint_mechanism_sensitivity` takes (a prepared
        z-transport stage and a :class:`~antecedent.transport.JointDeviation`); the
        exact range is evaluated on the stage's retained laws and filled on
        ``grid_points`` equally spaced contamination fractions (one declared factor,
        default 9) or at the two exact points of the declared box (two factors,
        which must use 2). The 2.2 withheld sampling status is carried unchanged: the
        range stays an assumption range, never a confidence interval.
        """
        from .transport._impl import _optional_non_negative
        from .transport._joint_sensitivity import _deviation, _stage

        deviation = _deviation(deviation)
        stage = _stage(stage)
        points = (
            grid_points if grid_points is not None else (9 if len(deviation._pairs) == 1 else 2)
        )
        native, refusal = _from_z_joint(
            stage,
            list(deviation._pairs),
            json.dumps(effect._wire()),
            int(points),
            json.dumps([a._wire() for a in actions]),
            causal_contract_id,
            **deviation._kwargs(),
            memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
            cancel=cancel,
        )
        _raise(refusal)
        assert native is not None
        return cls(native)

    @classmethod
    def consume(
        cls, data: bytes, *, expected_identity: Mapping[str, str] | None = None
    ) -> SensitivityArtifact:
        """Consume by recomputation; refuses a resealed mutation.

        The declarations and numbers are rebuilt, the identity digests and the decision
        outcome are recomputed and must equal the stored ones, and, when the consumer
        retained ``expected_identity`` (the dict of :attr:`identity`) independently of
        the bytes, that identity must match too.
        """
        if not isinstance(data, bytes):
            raise CausalTypeError("artifact must be bytes")
        native, refusal = _NativeSensitivityArtifact.consume(
            data, None if expected_identity is None else json.dumps(dict(expected_identity))
        )
        _raise(refusal)
        assert native is not None
        return cls(native)

    def export(self, *, artifact_id: str = "sensitivity-decision") -> bytes:
        """Serialize through the bounded, checksummed container."""
        return bytes(self._native.export(artifact_id))

    # -- views -----------------------------------------------------------------

    @property
    def identity(self) -> dict[str, str]:
        """``premises_digest``, ``data_digest`` and ``digest``: retain it to consume."""
        return dict(self._summary["identity"])

    @property
    def coordinate(self) -> AssumptionCoordinate:
        return AssumptionCoordinate(**self._summary["coordinate"])

    @property
    def grid(self) -> tuple[float, ...]:
        return tuple(self._summary["grid"])

    @property
    def support(self) -> tuple[PointSupport, ...]:
        return tuple(self._summary["support"])

    @property
    def quantities(self) -> tuple[SurfaceQuantity, ...]:
        return tuple(
            SurfaceQuantity(
                ScientificQuantity._from_wire(q["quantity"]), tuple(q["lower"]), tuple(q["upper"])
            )
            for q in self._summary["quantities"]
        )

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(Action(a["id"], Utility(a["utility"])) for a in self._summary["actions"])

    @property
    def utility_units(self) -> str | None:
        return self._summary["utility_units"]

    @property
    def uncertainty(self) -> Uncertainty:
        return Uncertainty._from_wire(self._summary["uncertainty"])

    @property
    def provenance(self) -> SurfaceProvenance:
        return SurfaceProvenance._from_wire(self._summary["provenance"])

    @property
    def outcome(self) -> Outcome | None:
        """The stored decision over the whole grid, recomputed on consumption.

        ``None`` when a grid point is unsupported.
        """
        wire = self._summary["outcome"]
        return None if wire is None else Outcome._from_wire(wire)

    def outcome_over(self, range: tuple[float, float] | None = None) -> Outcome:  # noqa: A002
        """Recompute the classification over an inclusive coordinate sub-range from the
        surface and the declared action utilities alone (no contract needed)."""
        text, refusal = self._native.outcome_json(None if range is None else _range(range))
        _raise(refusal)
        assert text is not None
        return Outcome._from_wire(json.loads(text))

    def contract(self, policy: StructuralPolicy = "require_invariant_best_action") -> Contract:
        """The :class:`~antecedent.decision.Contract` the artifact's own declared actions define."""
        text, refusal = _contract(self._native, policy)
        _raise(refusal)
        assert text is not None
        return Contract._from_wire(json.loads(text))

    def __repr__(self) -> str:
        return (
            f"<SensitivityArtifact {self.coordinate.id} x{len(self.grid)} "
            f"{self.identity['digest'][:12]}>"
        )


def _range(value: tuple[float, float]) -> tuple[float, float]:
    low, high = value
    if not (math.isfinite(low) and math.isfinite(high)):
        raise CausalValueError("a coordinate range is two finite numbers")
    return (float(low), float(high))


# ---------------------------------------------------------------------- decision


@dataclass(frozen=True, slots=True)
class AtomView:
    """One grid-point by range-vertex scenario of the structural evaluation."""

    id: str
    probability: float | None
    status: Literal["evaluated", "unidentified", "unevaluated"]
    leaders: tuple[str, ...] = ()
    values: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None


class SensitivityDecision:
    """A decision over the assumption range of a sensitivity surface."""

    def __init__(self, contract: Contract, artifact: SensitivityArtifact, body: Mapping[str, Any]):
        self._contract = contract
        self._artifact = artifact
        self._body = body
        self._outcome = Outcome._from_wire(body["outcome"])

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def kind(self) -> DecisionKind:
        return self._outcome.kind

    @property
    def invariant_action(self) -> str | None:
        """The action that is uniquely best everywhere, when there is one."""
        return self._outcome.action

    @property
    def switch(self) -> DecisionSwitch | None:
        """The single assumption-dependent switch with its tipping coordinate, when there is one."""
        return self._outcome.switch

    @property
    def robust(self) -> bool:
        """``True`` only for an invariant action; an assumption-dependent switch is not robust."""
        return self._outcome.kind == "invariant_action"

    @property
    def coordinates(self) -> tuple[float, ...]:
        """The evaluated grid coordinates, ascending."""
        return tuple(self._body["coordinates"])

    @property
    def coverage(self) -> str:
        """``point_surface``, ``vertex_certified`` (multilinear utilities: the vertices
        certify the whole range) or ``vertices_only`` (a minimum or maximum: interior
        range values are not certified)."""
        return str(self._body["coverage"])

    @property
    def artifact_identity(self) -> str:
        return str(self._body["artifact_identity"])

    @property
    def sampling(self) -> SamplingReport:
        wire = self._body["sampling"]
        if "withheld" in wire:
            return SamplingReport("withheld", **wire["withheld"])
        body = wire["separate_not_composed"]
        return SamplingReport(
            "separate_not_composed",
            quantity=body["quantity"],
            level=body["level"],
            method=body["method"],
            coordinates=tuple(body["coordinates"]),
            lower=tuple(body["lower"]),
            upper=tuple(body["upper"]),
        )

    @property
    def structural_verdict(self) -> Mapping[str, Any]:
        """The contract's structural policy applied across the scenarios (``kind`` and fields)."""
        return dict(self._body["structural"]["verdict"])

    @property
    def atoms(self) -> tuple[AtomView, ...]:
        return tuple(
            AtomView(
                id=a["id"],
                probability=a["probability"],
                status=a["status"],
                leaders=tuple(a.get("leaders", ())),
                values={v["id"]: float(v["expected_utility"]) for v in a.get("values", ())},
                reason=a.get("reason"),
            )
            for a in self._body["structural"]["atoms"]
        )

    @property
    def interpretation(self) -> str:
        return str(self._body["interpretation"])

    def explain(self) -> str:
        """The decision, what it ranges over and what it is not."""
        coordinate = self._artifact.coordinate
        points = self.coordinates
        text = (
            f"Over {len(points)} grid point(s) of {coordinate.id} in "
            f"[{points[0]:g}, {points[-1]:g}] {coordinate.units}: {self._outcome}"
        )
        if self._outcome.kind == "no_robust_action" and self._outcome.coordinates:
            where = ", ".join(f"{c:g}" for c in self._outcome.coordinates)
            text += f" at {where}"
        sampling = self.sampling
        if sampling.status == "separate_not_composed":
            text += (
                f"; a {sampling.level:g} sampling interval ({sampling.method}) is reported "
                "next to the decision and is not composed with the assumption range"
            )
        else:
            text += "; no sampling interval exists (" + str(sampling.detail) + ")"
        return text + "; " + self.interpretation + "."

    def __repr__(self) -> str:
        return f"<SensitivityDecision {self._outcome}>"


def decide(
    contract: Contract,
    artifact: SensitivityArtifact,
    *,
    range: tuple[float, float] | None = None,  # noqa: A002
    weights: Sequence[float] | None = None,
    sampling_composition: str | None = None,
) -> SensitivityDecision:
    """Evaluate ``contract`` over the surface of ``artifact``.

    Every grid point by range-vertex scenario is one structural atom, so the contract's
    ``structural_policy`` decides how the assumption scenarios combine
    (``require_invariant_best_action``, ``maximin``, ``bayes_over_structures`` over
    *declared* ``weights``, ``report_only``). ``range`` is an inclusive coordinate
    sub-range. ``weights`` are genuine probabilities, one per evaluated grid point in
    ascending order, and only for a point surface: an assumption range is never a
    probability, so weights on a ranged surface refuse. ``sampling_composition``
    requests composing the sampling interval with the range and always refuses with
    ``cell_not_licensed`` today.
    """
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be a decision.Contract")
    if not isinstance(artifact, SensitivityArtifact):
        raise CausalTypeError("artifact must be a SensitivityArtifact")
    spec = {
        "range": None if range is None else list(_range(range)),
        "weights": None if weights is None else [float(w) for w in weights],
        "sampling_composition": sampling_composition,
    }
    text, refusal = _decide(json.dumps(contract._wire()), artifact._native, json.dumps(spec))
    _raise(refusal)
    assert text is not None
    return SensitivityDecision(contract, artifact, json.loads(text))


__all__ = [
    "ASSUMPTION_RANGE_STATEMENT",
    "Action",
    "AssumptionCoordinate",
    "AtomView",
    "DecisionSwitch",
    "IdentifiedBound",
    "Outcome",
    "SamplingInterval",
    "SamplingReport",
    "SamplingWithheld",
    "SensitivityArtifact",
    "SensitivityDecision",
    "SensitivityRefusal",
    "SourceTipping",
    "SurfaceProvenance",
    "SurfaceQuantity",
    "Uncertainty",
    "Utility",
    "const",
    "decide",
    "maximum",
    "minimum",
    "quantity",
]
