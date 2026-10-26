"""Transported responses: a value carried to a target population, and the decision made on one.

A transported response is a value (one mean per coordinate, or a law) about a *target*
population that was derived from evidence of other populations under a declared transport. It
carries the receipt that says how: the route, the populations it was learned in and carried to,
the selection-diagram identities, the assumptions, an extrapolative support label and the
source model's own standard error, which is a diagnostic and never an interval::

    from antecedent import transported_response as tr

    declaration = tr.TransportDeclaration(
        route="declared",
        source_identity="transport-1",
        source_populations=("trial",),
        target_population="clinic_b",
        transport_identities=("diagram-1",),
        assumptions=("selection acts on age only",),
        provider_id="transport-provider",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    response = tr.TransportedResponse.declared_point([effect], [2.0], declaration)
    decided = tr.evaluate_transported(
        contract, response, analysis=prepared,
        support_policy=composition.SupportPolicy.require_all("extrapolative"),
    )
    decided.decision.verdict.selected, decided.receipt.target_population

What this is, and is not:

* **Never native target data.** The receipt travels into every result; ``native_target_data`` is
  always ``False``. A consumer that accepts only native target data refuses the response.
* **A point answers an affine expected utility and nothing else.** A law, a probability, a
  quantile, a hard constraint or an interval of a transported point is refused with its own
  reason; an interval is never licensed for a transported value.
* **The support label is at most ``extrapolative``.** The default support policy (``supported``)
  refuses it; a policy that accepts ``extrapolative`` states that the decision rests on a
  declared transport.
* **Refresh.** New outcome-only data of the source population is a new response: decide again
  from it. The contract, the rules and the policy are kept; every value and the identity are
  recomputed; a response about another population, quantity or arm is refused by the coordinate
  check. A decision is never refreshed in place.

Rust owns the receipt checks, the admissibility rules, the decision, the identities and every
refusal; this module declares the value and presents the result. Refusals raise
:class:`TransportedResponseRefusal` (response construction) or
:class:`~antecedent.decision.DecisionRefusal` (a decision), each a
:class:`~antecedent.errors.StructuredRefusal` with a registered ``reason_code``, a namespaced
``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._columns import coerce_fields
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import choice, real, text
from .errors import CausalTypeError, StructuredRefusal
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import outcome_operation, require_success

__all__ = [
    "TransportDeclaration",
    "TransportReceipt",
    "TransportedDecision",
    "TransportedResponse",
    "TransportedResponseRefusal",
    "evaluate_transported",
]

TransportedRoute = Literal[
    "cohort_effect",
    "shifted_cohort",
    "policy",
    "synthesis",
    "mechanism",
    "temporal_policy",
    "adaptive_trial",
    "surrogate_bridge",
    "declared",
]
Demand = Literal["expected_utility", "law", "interval"]

_ROUTES = (
    "cohort_effect",
    "shifted_cohort",
    "policy",
    "synthesis",
    "mechanism",
    "temporal_policy",
    "adaptive_trial",
    "surrogate_bridge",
    "declared",
)
_DEMANDS = ("expected_utility", "law", "interval")
_SUPPORT = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)


class TransportedResponseRefusal(StructuredRefusal):
    """A refusal of a transported response, its sensitivity range or its revalidation.

    ``reason_code`` is registered. ``detail`` is the namespaced ``family.slot``
    (``transported_decision.*``, ``transport_sensitivity.*``, ``transport_revalidation.*``).
    ``expected`` and ``supplied`` carry the compared semantics where the refusal names them.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        message = str(refusal["detail"]) + (f": {offending}" if offending else "")
        super().__init__(refusal, text=message)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


def _strings(values: Sequence[str], what: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of strings")
    return tuple(text(value, what, non_empty=True) for value in values)


@dataclass(frozen=True, slots=True)
class TransportDeclaration:
    """What a route declares when it hands a value to the decision boundary.

    Every field is a declaration or an identity of the producing route; none is evidence that
    the premises hold. ``support`` defaults to ``extrapolative``, the strongest label a
    transported value may carry. ``uncalibrated_standard_error`` is the source model's own
    standard error, a diagnostic that no decision inherits.

    A route with a typed producer (``shifted_cohort``, ``policy``, ``synthesis``) cannot be
    declared by hand: its populations and identities come from the producing result.
    """

    route: TransportedRoute
    source_identity: str
    source_populations: tuple[str, ...]
    target_population: str
    transport_identities: tuple[str, ...]
    assumptions: tuple[str, ...]
    provider_id: str
    snapshot_id: str
    causal_contract_id: str
    support: str = "extrapolative"
    uncalibrated_standard_error: float | None = None

    def __post_init__(self) -> None:
        choice(self.route, "route", _ROUTES)
        choice(self.support, "support", _SUPPORT)
        coerce_fields(
            self,
            source_populations=_strings(self.source_populations, "source_populations"),
            transport_identities=_strings(self.transport_identities, "transport_identities"),
            assumptions=_strings(self.assumptions, "assumptions"),
        )
        if self.uncalibrated_standard_error is not None:
            coerce_fields(
                self,
                uncalibrated_standard_error=real(
                    self.uncalibrated_standard_error, "uncalibrated_standard_error", finite=True
                ),
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "source_identity": self.source_identity,
            "source_populations": list(self.source_populations),
            "target_population": self.target_population,
            "transport_identities": list(self.transport_identities),
            "assumptions": list(self.assumptions),
            "support": self.support,
            "uncalibrated_standard_error": self.uncalibrated_standard_error,
            "provider_id": self.provider_id,
            "snapshot_id": self.snapshot_id,
            "causal_contract_id": self.causal_contract_id,
        }


@dataclass(frozen=True, slots=True)
class TransportReceipt:
    """How a value was carried to its target population.

    ``support`` is never ``supported``. ``representation`` is ``point_only`` (one mean per
    coordinate) or ``law`` (aligned draws under a declared transport model).
    ``uncalibrated_standard_error`` is a diagnostic, never an interval.
    """

    route: TransportedRoute
    source_identity: str
    source_populations: tuple[str, ...]
    target_population: str
    transport_identities: tuple[str, ...]
    assumptions: tuple[str, ...]
    support: str = "extrapolative"
    representation: Literal["point_only", "law"] = "point_only"
    uncalibrated_standard_error: float | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            source_populations=tuple(self.source_populations),
            transport_identities=tuple(self.transport_identities),
            assumptions=tuple(self.assumptions),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "source_identity": self.source_identity,
            "source_populations": list(self.source_populations),
            "target_population": self.target_population,
            "transport_identities": list(self.transport_identities),
            "assumptions": list(self.assumptions),
            "support": self.support,
            "representation": self.representation,
            "uncalibrated_standard_error": self.uncalibrated_standard_error,
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> TransportReceipt:
        return cls(
            route=wire["route"],
            source_identity=wire["source_identity"],
            source_populations=tuple(wire["source_populations"]),
            target_population=wire["target_population"],
            transport_identities=tuple(wire["transport_identities"]),
            assumptions=tuple(wire["assumptions"]),
            support=wire["support"],
            representation=wire["representation"],
            uncalibrated_standard_error=wire.get("uncalibrated_standard_error"),
        )


def _unwrap(pair: tuple[Any, str | None], cls: type[StructuredRefusal]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, cls)
    return value


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportedResponse(ScientificResult):
    """A value carried to a target population, with the receipt that says how.

    ``means`` is the transported point per coordinate (``None`` for a law). ``label`` is
    ``point_only`` or ``law``; ``native_target_data`` is always ``False``. Build with
    :meth:`declared_point`, :meth:`declared_law` or :meth:`from_shifted_cohort`.
    """

    receipt: TransportReceipt
    coordinates: tuple[ScientificQuantity, ...]
    means: tuple[float, ...] | None
    label: str
    native_target_data: bool
    identity: str
    _handle: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def _of(cls, handle: Any) -> TransportedResponse:
        receipt = TransportReceipt._from_wire(json.loads(handle.receipt_json()))
        value = json.loads(handle.value_json())
        return cls(
            receipt=receipt,
            coordinates=tuple(ScientificQuantity._from_wire(q) for q in value["coordinates"]),
            means=tuple(float(m) for m in value["means"]) if value["kind"] == "point" else None,
            label=receipt.representation,
            native_target_data=False,
            identity=handle.identity(),
            _handle=handle,
        )

    @classmethod
    def declared_point(
        cls,
        coordinates: Sequence[ScientificQuantity],
        means: Sequence[float],
        declaration: TransportDeclaration,
    ) -> TransportedResponse:
        """A transported point: one finite mean per coordinate, all in the target population.

        Raises:
            TransportedResponseRefusal: an incomplete transport identity (no source
                population, a source equal to the target, no assumption or identity), a
                support label stronger than ``extrapolative``, a route that needs a typed
                producer, no coordinate, a repeated coordinate, a count mismatch, a non-finite
                mean, or a coordinate outside the target population.
        """
        quantities = _quantities(coordinates)
        values = [real(m, "mean", finite=False) for m in means]
        _declaration(declaration)
        handle = _unwrap(
            _native.transported_response_declared_point(
                json.dumps([q._wire() for q in quantities]),
                values,
                json.dumps(declaration._wire()),
            ),
            TransportedResponseRefusal,
        )
        return cls._of(handle)

    @classmethod
    def declared_law(
        cls, law: JointDistributionArtifact, declaration: TransportDeclaration
    ) -> TransportedResponse:
        """A transported law: aligned draws under a declared transport model.

        A law labelled native, exact or measured is refused: a transport never raises a
        calibration or trust label.
        """
        law = require_success(law)
        if not isinstance(law, JointDistributionArtifact):
            raise CausalTypeError("law must be a JointDistributionArtifact")
        _declaration(declaration)
        handle = _unwrap(
            _native.transported_response_declared_law(law._native, json.dumps(declaration._wire())),
            TransportedResponseRefusal,
        )
        return cls._of(handle)

    @classmethod
    def from_shifted_cohort(cls, source: Any) -> TransportedResponse:
        """The cohort-mean contrast of a cohort source built under a declared covariate shift.

        The source is re-derived from its artifact under its identity before the response is
        formed. A source in the model's own population is refused (``not_transported``).
        """
        from .cohort_decision import CohortDecisionSource

        source = require_success(source)
        if not isinstance(source, CohortDecisionSource):
            raise CausalTypeError("source must be a cohort_decision.CohortDecisionSource")
        handle = _unwrap(
            _native.transported_response_from_shifted_cohort(
                bytes(source.export()), source.identity
            ),
            TransportedResponseRefusal,
        )
        return cls._of(handle)

    def interval(self) -> None:
        """Always refuses: no interval is licensed for a transported value."""
        raise TransportedResponseRefusal(
            {
                "code": "cell_not_licensed",
                "stage": "transported_decision",
                "detail": "transported_decision.interval_not_licensed",
                "message": "a transported value is a point under a declared transport",
                "offending": None,
                "remedy": "decide on native target evidence of the required representation",
            }
        )


def _quantities(coordinates: Sequence[ScientificQuantity]) -> tuple[ScientificQuantity, ...]:
    if isinstance(coordinates, (str, ScientificQuantity)) or not isinstance(coordinates, Sequence):
        raise CausalTypeError("coordinates must be a sequence of ScientificQuantity")
    if not all(isinstance(q, ScientificQuantity) for q in coordinates):
        raise CausalTypeError("coordinates must be a sequence of ScientificQuantity")
    return tuple(coordinates)


def _declaration(declaration: object) -> None:
    if not isinstance(declaration, TransportDeclaration):
        raise CausalTypeError("declaration must be a TransportDeclaration")


@dataclass(repr=False, frozen=True, slots=True)
class TransportedDecision(ScientificResult):
    """A decision made on a transported response, bound to the causal contract.

    ``decision`` is the support-aware decision (a :class:`~antecedent.composition.
    SupportedDecision`: dispositions, outcomes, verdict). ``receipt`` is the transport the
    value travelled with; ``assumptions`` list the decision's own followed by the transport's.
    ``identity`` binds the bound contract, the receipt and the lineage of the value: another
    transport at the same values is another decision. ``inference`` is always ``point_only``
    and ``native_target_data`` always ``False``.
    """

    decision: Any
    receipt: TransportReceipt
    assumptions: tuple[str, ...]
    inference: str
    native_target_data: bool
    identity: str
    decision_identity: str
    bound_identity: str
    causal_target: str
    causal_identification: str

    @property
    def verdict(self) -> Any:
        """The verdict over the evaluated actions."""
        return self.decision.verdict

    @property
    def outcomes(self) -> Any:
        """The outcomes of the evaluated actions, in declaration order."""
        return self.decision.outcomes

    def explain(self) -> str:
        """The verdict and the transport it rests on."""
        r = self.receipt
        return (
            f"{self.decision.explain()} The value was carried from "
            f"{', '.join(r.source_populations)} to {r.target_population} by route {r.route} "
            f"under declared assumptions; it is not native target data and the decision is "
            f"point-only."
        )


def _prepared_native(analysis: Any) -> Any:
    study = analysis if hasattr(analysis, "_native") else getattr(analysis, "study", None)
    native = getattr(study, "_native", None)
    if native is None:
        raise CausalTypeError(
            "analysis must be a prepared analysis or an analysis result that retains one"
        )
    return native


@outcome_operation
def evaluate_transported(
    contract: Any,
    response: TransportedResponse,
    *,
    analysis: Any,
    demand: Demand = "expected_utility",
    support_policy: Any = None,
) -> TransportedDecision:
    """Decide ``contract`` on a transported response, bound to the identities of ``analysis``.

    ``contract`` is a :class:`~antecedent.decision.Contract` or a
    :class:`~antecedent.decision_robust.AdmissibleContract`; the admissibility rules apply
    first (a declared exclusion removes its action, a support rule the transported label does
    not meet refuses, an uncertainty requirement a point cannot meet refuses). ``analysis`` is
    the prepared analysis (or an analysis result retaining it) whose target and identification
    the decision is bound to. ``support_policy`` defaults to the policy that accepts only
    ``supported`` values, which refuses an ``extrapolative`` transported label; pass
    ``SupportPolicy.require_all("extrapolative")`` to decide on it explicitly.

    Raises:
        ~antecedent.decision.DecisionRefusal: an interval or a law the value cannot supply
            (``transported_decision.interval_not_licensed``, ``.law_not_supplied``), a support
            label below the policy (``.transported_not_accepted``) or below a rule
            (``.support_rule_unmet``), an uncertainty requirement a point cannot meet
            (``.uncertainty_requirement``), exclusions that leave no comparison, and any
            refusal of the decision boundary.
    """
    from . import decision_robust
    from .composition import SupportPolicy
    from .decision import Contract, DecisionRefusal

    contract = require_success(contract)
    response = require_success(response)
    if isinstance(contract, Contract):
        admissible = decision_robust.admissible_contract(contract)
    elif isinstance(contract, decision_robust.AdmissibleContract):
        admissible = contract
    else:
        raise CausalTypeError("contract must be a decision.Contract or an AdmissibleContract")
    if not isinstance(response, TransportedResponse):
        raise CausalTypeError("response must be a TransportedResponse")
    choice(demand, "demand", _DEMANDS)
    policy = SupportPolicy.require_all("supported") if support_policy is None else support_policy
    if not isinstance(policy, SupportPolicy):
        raise CausalTypeError("support_policy must be a composition.SupportPolicy")
    payload = _unwrap(
        _native.transported_decision_evaluate(
            _prepared_native(analysis),
            admissible._declaration(),
            response._handle,
            demand,
            policy.unsupported,
            policy.weakest_support,
        ),
        DecisionRefusal,
    )
    wire = json.loads(payload)
    from .composition import _supported_from_wire

    return TransportedDecision(
        decision=_supported_from_wire(wire["decision"]),
        receipt=TransportReceipt._from_wire(wire["receipt"]),
        assumptions=tuple(wire["assumptions"]),
        inference=str(wire["inference"]),
        native_target_data=bool(wire["native_target_data"]),
        identity=str(wire["identity"]),
        decision_identity=str(wire["decision_identity"]),
        bound_identity=str(wire["bound_identity"]),
        causal_target=str(wire["causal_target"]),
        causal_identification=str(wire["causal_identification"]),
    )
