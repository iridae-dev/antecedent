"""Declare a decision problem and evaluate it on aligned joint draws.

A decision names its actions by stable ID, the scientific quantities each
action's utility reads, and a closed utility expression you build with ordinary
operators. Hard constraints exclude actions; they never become penalties::

    p, q = decision.x(0), decision.x(1)
    contract = decision.Contract(
        actions=[
            decision.Action("treat", inputs=(benefit, cost), utility=p * q),
            decision.Action("wait", inputs=(baseline,), utility=decision.x(0)),
        ],
        utility_units="qaly",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    result = contract.evaluate(joint_distribution)   # aligned draws, not marginals
    print(result.explain())

The draws are a :class:`~antecedent.joint_distribution.JointDistributionArtifact`:
a nonlinear utility over several quantities needs genuine joint rows, and
independent marginals refuse with ``joint_law_required``. Rust owns the
identity, validation, evaluation, artifacts and refusal rules; this module
builds declarations and raises each refusal as
:class:`DecisionRefusal`, a :class:`~antecedent.errors.CausalUnsupportedError`
with its registered ``reason_code``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from ._native import composition_lineage as _composition_lineage
from ._native import decision_contract_normalize as _normalize
from ._native import decision_source_digest as _source_digest
from ._native import evaluate_decision as _evaluate
from ._native import evaluate_decision_means as _evaluate_means
from ._native import export_decision_contract as _export_contract
from ._native import export_decision_result as _export_result
from ._native import load_decision_contract as _load_contract
from ._native import replay_decision_result as _replay
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError, StructuredRefusal
from .external import BoundExternalClaim, LineageLink
from .joint_distribution import JointDistributionArtifact, ScientificQuantity

if TYPE_CHECKING:
    from ._native import AteAnalysisResult
    from .program_claims import ProgramBinding
    from .source_evidence import SourceEvidence

RESULT_LINK_ID = "decision_result"
ActionKind = Literal["intervention", "policy", "regime", "study", "external"]
StructuralPolicy = Literal[
    "require_invariant_best_action", "maximin", "bayes_over_structures", "report_only"
]


class DecisionRefusal(StructuredRefusal):
    """A decision refusal carrying the structured Rust fields.

    A :class:`~antecedent.errors.StructuredRefusal`: ``code`` (alias ``reason_code``) and
    ``remedy`` are the registered fields. ``stage`` is the stage that refused. ``detail`` is
    the namespaced ``family.slot``; ``offending`` names the action input (``action[input]``) or
    action when there is one; ``expected`` and ``supplied`` carry the compared
    semantics (for example ``joint`` against ``independent_marginals``).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f" at {offending}" if offending else "")
        super().__init__(refusal, text=text)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise DecisionRefusal(json.loads(refusal))


@dataclass(frozen=True, slots=True)
class Expr:
    """A closed utility expression over an action's inputs; build it with operators."""

    _wire_value: Any

    def __add__(self, other: Expr | float) -> Expr:
        return Expr({"add": [self._wire_value, _coerce(other)._wire_value]})

    def __radd__(self, other: float) -> Expr:
        return _coerce(other) + self

    def __sub__(self, other: Expr | float) -> Expr:
        return Expr({"sub": [self._wire_value, _coerce(other)._wire_value]})

    def __rsub__(self, other: float) -> Expr:
        return _coerce(other) - self

    def __mul__(self, other: Expr | float) -> Expr:
        return Expr({"mul": [self._wire_value, _coerce(other)._wire_value]})

    def __rmul__(self, other: float) -> Expr:
        return _coerce(other) * self

    def __neg__(self) -> Expr:
        return Expr({"neg": self._wire_value})

    def __repr__(self) -> str:
        return f"Expr({json.dumps(self._wire_value, sort_keys=True)})"


def _leaf_kinds(expr: Expr) -> frozenset[str]:
    """The leaf kinds an expression reads: ``input``, ``quantity`` and/or ``const``.

    A decision utility reads ``input`` leaves (:func:`x`); a sensitivity utility reads
    ``quantity`` leaves (:func:`antecedent.sensitivity_decision.quantity`). The same
    :class:`Expr` type carries both, and each declaration refuses the other's leaves.
    """
    found: set[str] = set()
    stack: list[Any] = [expr._wire_value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("input", "quantity", "const"):
                    found.add(key)
                else:
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(found)


def _coerce(value: Expr | float) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalValueError("a utility expression combines Expr values and numbers")
    return const(float(value))


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise CausalTypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def x(index: int) -> Expr:
    """The ``index``th input quantity of the action."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CausalValueError("an input index is a non-negative integer")
    return Expr({"input": index})


def const(value: float) -> Expr:
    """A constant."""
    return Expr({"const": float(value)})


def maximum(left: Expr | float, right: Expr | float) -> Expr:
    """Pointwise maximum."""
    return Expr({"max": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


def minimum(left: Expr | float, right: Expr | float) -> Expr:
    """Pointwise minimum."""
    return Expr({"min": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


@dataclass(frozen=True, slots=True)
class Criterion:
    """How actions are ranked. Build with the classmethods; CVaR is not offered."""

    kind: str
    threshold: float | None = None
    p: float | None = None

    @classmethod
    def expected_utility(cls) -> Criterion:
        return cls("posterior_expected_utility")

    @classmethod
    def expected_loss(cls) -> Criterion:
        """Minimize posterior expected loss; the contract's utilities are losses."""
        return cls("posterior_expected_loss")

    @classmethod
    def threshold_probability(cls, threshold: float) -> Criterion:
        """Maximize ``P(utility >= threshold)``; ``threshold`` is a number in utility units."""
        return cls("threshold_probability", threshold=_number("threshold", threshold))

    @classmethod
    def quantile(cls, p: float) -> Criterion:
        """Maximize the ``p`` quantile of the utility; ``p`` is a number."""
        return cls("quantile", p=_number("p", p))

    @classmethod
    def minimax_over_identified_set(cls) -> Criterion:
        return cls("minimax_over_identified_set")

    @classmethod
    def maximin_over_structures(cls) -> Criterion:
        return cls("maximin_over_structures")

    @classmethod
    def regret(cls) -> Criterion:
        return cls("regret")

    @classmethod
    def expected_regret(cls) -> Criterion:
        return cls("expected_regret")

    def _wire(self) -> Any:
        if self.kind == "threshold_probability":
            return {"threshold_probability": {"threshold": self.threshold}}
        if self.kind == "quantile":
            return {"quantile": {"p": self.p}}
        return self.kind

    @classmethod
    def _from_wire(cls, wire: Any) -> Criterion:
        if isinstance(wire, str):
            return cls(wire)
        ((kind, args),) = wire.items()
        return cls(kind, threshold=args.get("threshold"), p=args.get("p"))


@dataclass(frozen=True, slots=True)
class Action:
    """One action: a stable ID, the quantities its utility reads, and the utility."""

    id: str
    inputs: tuple[ScientificQuantity, ...]
    utility: Expr
    kind: ActionKind = "intervention"

    def __post_init__(self) -> None:
        _text("Action.id", self.id)
        if isinstance(self.inputs, (str, bytes)) or not isinstance(self.inputs, Iterable):
            raise CausalTypeError("Action.inputs must be a sequence of ScientificQuantity")
        inputs = tuple(self.inputs)
        if any(not isinstance(item, ScientificQuantity) for item in inputs):
            raise CausalTypeError("Action.inputs must hold only ScientificQuantity values")
        if not isinstance(self.utility, Expr):
            raise CausalTypeError(
                "Action.utility must be an Expr built from decision.x(i), decision.const(v) "
                f"and operators, not {type(self.utility).__name__}; wrap a number with "
                "decision.const(value)"
            )
        if "quantity" in _leaf_kinds(self.utility):
            raise CausalTypeError(
                "Action.utility reads inputs by position (decision.x(i)); a "
                "sensitivity_decision.quantity(...) leaf belongs to "
                "sensitivity_decision.SensitivityAction"
            )
        if self.kind not in get_args(ActionKind):
            raise CausalValueError(f"Action.kind must be one of {get_args(ActionKind)}")
        object.__setattr__(self, "inputs", inputs)

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "inputs": [q._wire() for q in self.inputs],
            "utility": self.utility._wire_value,
        }


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard constraint ``P(expr <= bound) >= min_probability``.

    A violated constraint excludes the action; it never enters the utility.
    ``applies_to`` empty applies to every action.
    """

    id: str
    expr: Expr
    bound: float
    units: str
    min_probability: float = 1.0
    applies_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text("Constraint.id", self.id)
        if not isinstance(self.expr, Expr):
            raise CausalTypeError("Constraint.expr must be an Expr built from decision.x(i)")
        _number("Constraint.bound", self.bound)
        _number("Constraint.min_probability", self.min_probability)
        _text("Constraint.units", self.units)
        if isinstance(self.applies_to, (str, bytes)):
            raise CausalTypeError("Constraint.applies_to must be a sequence of action ids")
        object.__setattr__(self, "applies_to", tuple(self.applies_to))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expr": self.expr._wire_value,
            "bound": float(self.bound),
            "min_probability": float(self.min_probability),
            "units": self.units,
            "applies_to": list(self.applies_to),
        }


@dataclass(frozen=True, slots=True)
class ConstraintExclusion:
    """A hard constraint an action failed, with the probability it reached."""

    constraint_id: str
    probability: float
    required: float


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """One action's outcome under the contract."""

    id: str
    admissible: bool
    exclusions: tuple[ConstraintExclusion, ...]
    expected_utility: float
    value: float
    standard_error: float | None
    expected_regret: float | None
    max_regret: float | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the evaluation can claim about the choice.

    ``kind`` is ``uniquely_optimal`` under the supplied finite law,
    ``indistinguishable`` for equal criterion values, or ``no_admissible_action``.
    Sampled-law rankings carry no statistical ranking guarantee.
    ``actions`` are the leader, then those it ties.
    """

    kind: Literal["uniquely_optimal", "indistinguishable", "no_admissible_action"]
    actions: tuple[str, ...]


def _expr_from_wire(wire: Any) -> Expr:
    return Expr(wire)


@dataclass(frozen=True, slots=True)
class MeanSource:
    """One mean per coordinate, with the provenance the decision records.

    It answers only the expectation of an affine utility: any other criterion refuses
    with ``decision_contract_unsatisfied``, because a mean is never an outcome law.
    Build it from a native response with
    :meth:`antecedent.program_claims.NativeClaim.as_decision_source`.
    """

    coordinates: tuple[ScientificQuantity, ...]
    means: tuple[float, ...]
    provider_id: str
    snapshot_id: str
    causal_contract_id: str
    rng_id: str = "none:mean_grid"
    _original_native: Any = field(default=None, repr=False, compare=False)
    _original_effect: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "means", tuple(float(m) for m in self.means))

    def _validated_original(self):
        if self._original_effect is not None:
            original, value, units, population = self._original_effect
            raw = original._raw
            wire, refusal = raw.effect_source_json(units, population, value)
            if refusal is not None:
                from .external import ExternalRefusal

                raise ExternalRefusal(json.loads(refusal))
            actual = json.loads(wire)
            actual.pop("query", None)
            supplied = {
                "coordinates": [q._wire() for q in self.coordinates],
                "means": list(self.means),
                "provider_id": self.provider_id,
                "snapshot_id": self.snapshot_id,
                "causal_contract_id": self.causal_contract_id,
                "rng_id": self.rng_id,
            }
            if json.loads(json.dumps(supplied)) != actual:
                raise CausalValueError(
                    "a native effect mean source must retain its original fields"
                )
            return raw
        if self._original_native is None:
            return None
        from . import _native

        return _native.source_backed_native_mean(
            self._original_native._native,
            json.dumps(
                {
                    "coordinates": [q._wire() for q in self.coordinates],
                    "means": list(self.means),
                    "provider_id": self.provider_id,
                    "snapshot_id": self.snapshot_id,
                    "causal_contract_id": self.causal_contract_id,
                    "rng_id": self.rng_id,
                }
            ),
        )


def _result_mean_source(
    contract: Contract,
    source: object,
    *,
    program: ProgramBinding | None,
    outcome_units: str | None,
    dose_units: str | None,
    population: str,
) -> MeanSource | JointDistributionArtifact | None:
    """The mean source an analysis result makes for ``contract``, or ``None`` for other sources.

    Reuses the native-claim path (``program_claims.native_claim`` then
    ``NativeClaim.as_decision_source``) so support labels, trust, snapshot identity and every
    refusal stay Rust-owned.
    """
    from .results._execution import ResultAPI

    if not isinstance(source, ResultAPI):
        if program is not None or outcome_units is not None or dose_units is not None:
            raise CausalTypeError(
                "program=, outcome_units= and dose_units= apply only when evaluating an "
                f"analysis result, not {type(source).__name__}"
            )
        return None
    from .results import AnalysisResult

    if isinstance(source, AnalysisResult):
        if program is not None or dose_units is not None:
            raise CausalValueError(
                "a scalar effect uses a contrast coordinate, not a response-grid program"
            )
        quantities = tuple(q for action in contract.actions for q in action.inputs)
        declarations = {(q.units, q.population_id) for q in quantities}
        if outcome_units is None:
            if len(declarations) != 1:
                raise CausalValueError("effect evaluation needs one declared unit/population")
            outcome_units, population = next(iter(declarations))
        native = getattr(source._raw, "effect_source_json", None)
        if native is None:
            raise CausalUnsupportedError(
                "this result retains no original native static effect source",
                reason_code="route_not_supported",
            )
        value = source.as_point()
        wire, refusal = native(outcome_units, population, value)
        if refusal is not None:
            from .external import ExternalRefusal

            raise ExternalRefusal(json.loads(refusal))
        fields = json.loads(wire)
        return MeanSource(
            coordinates=tuple(ScientificQuantity._from_wire(q) for q in fields["coordinates"]),
            means=tuple(fields["means"]),
            provider_id=fields["provider_id"],
            snapshot_id=fields["snapshot_id"],
            causal_contract_id=fields["causal_contract_id"],
            rng_id=fields["rng_id"],
            _original_effect=(source, value, outcome_units, population),
        )
    from .results.response import CausalResponseView

    if not isinstance(source, CausalResponseView):
        raise CausalUnsupportedError(
            f"a {type(source).__name__} carries no per-dose mean response coordinates, so a "
            "decision contract cannot read it; analyze a ResponseCurve query for a mean "
            "source, or supply a JointDistributionArtifact of aligned draws",
            reason_code="route_not_supported",
            remedy="run antecedent.analyze with a ResponseCurve query, or pass joint draws",
        )
    from . import program_claims

    retained = source.program_binding
    if program is None and outcome_units is None and dose_units is None and retained is not None:
        program = retained
    if program is None:
        # A contract declares units; it does not discover them from numbers. Require
        # one outcome/population across its inputs, then let the original native
        # claim compare the full descriptors, regimes and source projection.
        quantities = tuple(q for action in contract.actions for q in action.inputs)
        response_declarations = {(q.variable_id, q.units, q.population_id) for q in quantities}
        if outcome_units is None:
            if len(response_declarations) != 1:
                raise CausalValueError(
                    "analysis-result evaluation needs one declared outcome/unit/population "
                    "in the contract, or an explicit program= binding"
                )
            _, outcome_units, population = next(iter(response_declarations))
        if dose_units is None:
            dose_units = "native_numeric_scale"
        assert outcome_units is not None
        program = program_claims.ProgramBinding.from_response(
            source, outcome_units=outcome_units, dose_units=dose_units, population=population
        )
    elif outcome_units is not None or dose_units is not None:
        raise CausalValueError("pass program= or the units, not both")
    claim = program_claims.native_claim(source, program)
    return claim.as_decision_source(contract).source


@dataclass(frozen=True, slots=True)
class Contract:
    """A durable decision problem."""

    actions: tuple[Action, ...]
    utility_units: str
    criterion: Criterion
    target_population: str
    horizon: int = 0
    constraints: tuple[Constraint, ...] = ()
    structural_policy: StructuralPolicy = "report_only"

    def __post_init__(self) -> None:
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Iterable):
            raise CausalTypeError("Contract.actions must be a sequence of Action")
        actions = tuple(self.actions)
        if any(not isinstance(action, Action) for action in actions):
            raise CausalTypeError("Contract.actions must hold only Action values")
        if isinstance(self.constraints, (str, bytes)) or not isinstance(self.constraints, Iterable):
            raise CausalTypeError("Contract.constraints must be a sequence of Constraint")
        constraints = tuple(self.constraints)
        if any(not isinstance(constraint, Constraint) for constraint in constraints):
            raise CausalTypeError("Contract.constraints must hold only Constraint values")
        _text("Contract.utility_units", self.utility_units)
        _text("Contract.target_population", self.target_population)
        if not isinstance(self.criterion, Criterion):
            raise CausalTypeError(
                "Contract.criterion must be a Criterion such as Criterion.expected_utility(), "
                f"not {type(self.criterion).__name__}"
            )
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise CausalTypeError("Contract.horizon must be an integer")
        if self.structural_policy not in get_args(StructuralPolicy):
            raise CausalValueError(
                f"Contract.structural_policy must be one of {get_args(StructuralPolicy)}"
            )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "constraints", constraints)

    def _wire(self) -> dict[str, Any]:
        return {
            "version": 1,
            "actions": [a._wire() for a in self.actions],
            "utility_units": self.utility_units,
            "criterion": self.criterion._wire(),
            "constraints": [c._wire() for c in self.constraints],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self.structural_policy,
        }

    def _normalized(self) -> dict[str, Any]:
        normalized, refusal = _normalize(json.dumps(self._wire()))
        _raise(refusal)
        assert normalized is not None
        return dict(json.loads(normalized))

    @property
    def identity(self) -> str:
        """Canonical digest: unchanged by reordering actions, changed by any semantic edit."""
        return str(self._normalized()["identity"])

    def evaluate(
        self,
        source: JointDistributionArtifact | BoundExternalClaim | MeanSource | Any,
        *,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
    ) -> Decision:
        """Evaluate on aligned joint draws, a bound external response grid, a mean source
        or an analysis result.

        ``source`` is one of:

        * a :class:`~antecedent.joint_distribution.JointDistributionArtifact` (aligned draws:
          every criterion, replayable);
        * a :class:`~antecedent.external.BoundExternalClaim` or a :class:`MeanSource` (one
          mean per coordinate: the expectation of an affine utility only);
        * a checked static AverageEffect result: its single ``mean_difference`` contrast
          coordinate is an affine mean input, never two reconstructed outcome means;
        * a response-curve analysis result (the :class:`~antecedent.results.CausalResponseView`
          that ``antecedent.analyze(..., query=ResponseCurve(...))`` returns). Its response
          values are turned into a native mean claim through
          :func:`antecedent.program_claims.native_claim` and evaluated as a mean source, so
          the same restrictions hold and the result is not replayable. Units are declared once: ``analyze`` can retain a program binding, or the
          contract supplies an unambiguous outcome/unit/population declaration. Undeclared
          dose units mean ``native_numeric_scale``: the original numeric intervention
          scale, with no physical unit claim or conversion. Override with ``program=`` (a
          :class:`~antecedent.program_claims.ProgramBinding`) or ``outcome_units=`` and
          ``dose_units=`` (``population=`` names the target population). A result that is not
          a point-identified response curve refuses with a typed
          :class:`~antecedent.external.ExternalRefusal` or
          :class:`~antecedent.errors.CausalUnsupportedError` naming what is missing, and a
          contract that needs a joint law (a probability, a quantile, a nonlinear utility, a
          hard constraint) refuses ``decision_contract_unsatisfied`` because a mean
          response retains no outcome draws.

        A :class:`MeanSource` (for example from a native response claim) answers the
        same affine expectation and refuses everything else, exactly as below.

        Raises:
            CausalTypeError: ``source`` is none of the accepted types.
            CausalValueError: scientific declarations are ambiguous or conflict with a binding.

        A :class:`~antecedent.external.BoundExternalClaim` supplies one mean per
        coordinate, so it answers only the expectation of an affine utility: a
        nonlinear utility, a hard constraint or any other criterion refuses with
        ``decision_contract_unsatisfied`` (``decision_evaluation.mean_source_insufficient``).
        Regret and EVPI are unavailable for such a decision.
        """
        result_source = _result_mean_source(
            self,
            source,
            program=program,
            outcome_units=outcome_units,
            dose_units=dose_units,
            population=population,
        )
        if result_source is not None:
            source = result_source
        if isinstance(source, BoundExternalClaim):
            identity = source.identity_fields
            result, refusal = _evaluate_means(
                json.dumps(self._wire()),
                json.dumps([q._wire() for q in source.quantities]),
                [float(v) for v in source.values],
                str(identity["provider_id"]),
                str(identity["snapshot_id"]),
                str(identity["causal_contract_id"]),
            )
            _raise(refusal)
            assert result is not None
            return Decision(self, source, json.loads(result))
        if isinstance(source, MeanSource):
            source._validated_original()
            result, refusal = _evaluate_means(
                json.dumps(self._wire()),
                json.dumps([q._wire() for q in source.coordinates]),
                list(source.means),
                source.provider_id,
                source.snapshot_id,
                source.causal_contract_id,
            )
            _raise(refusal)
            assert result is not None
            return Decision(self, source, json.loads(result))
        if not isinstance(source, JointDistributionArtifact):
            hint = (
                "; a NativeClaim is not a source itself: pass "
                "claim.as_decision_source(contract).source"
                if type(source).__name__ == "NativeClaim"
                else ""
            )
            raise CausalTypeError(
                "Contract.evaluate needs a JointDistributionArtifact (aligned draws), a "
                "BoundExternalClaim or MeanSource (means), or a supported response/effect analysis "
                f"result (antecedent.analyze(...)), not {type(source).__name__}{hint}"
            )
        result, refusal = _evaluate(json.dumps(self._wire()), source._native)
        _raise(refusal)
        assert result is not None
        return Decision(self, source, json.loads(result))

    def export(self, *, artifact_id: str = "decision-contract") -> bytes:
        return bytes(_export_contract(json.dumps(self._wire()), artifact_id))

    @classmethod
    def load(cls, data: bytes, *, expected_identity: str) -> Contract:
        """Load only under the identity the consumer retained independently."""
        return cls._from_wire(json.loads(_load_contract(data, expected_identity)))

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> Contract:
        return cls(
            actions=tuple(
                Action(
                    id=a["id"],
                    kind=a["kind"],
                    inputs=tuple(ScientificQuantity._from_wire(q) for q in a["inputs"]),
                    utility=_expr_from_wire(a["utility"]),
                )
                for a in wire["actions"]
            ),
            utility_units=wire["utility_units"],
            criterion=Criterion._from_wire(wire["criterion"]),
            target_population=wire["target_population"],
            horizon=wire["horizon"],
            constraints=tuple(
                Constraint(
                    id=c["id"],
                    expr=_expr_from_wire(c["expr"]),
                    bound=c["bound"],
                    units=c["units"],
                    min_probability=c["min_probability"],
                    applies_to=tuple(c["applies_to"]),
                )
                for c in wire["constraints"]
            ),
            structural_policy=wire["structural_policy"],
        )


class Decision:
    """A decision result with the evidence behind it; exportable and replayable."""

    def __init__(
        self,
        contract: Contract,
        source: JointDistributionArtifact | BoundExternalClaim | MeanSource,
        body: Mapping[str, Any],
    ) -> None:
        self._contract = contract
        self._source = source
        self._body = body

    @property
    def contract_identity(self) -> str:
        return str(self._body["contract_identity"])

    @property
    def source_digest(self) -> str:
        return str(self._body["source_digest"])

    @property
    def actions(self) -> tuple[ActionOutcome, ...]:
        def number(value: float | None) -> float:
            return float("nan") if value is None else float(value)

        return tuple(
            ActionOutcome(
                id=a["id"],
                admissible=a["admissible"],
                exclusions=tuple(
                    ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
                    for e in a["exclusions"]
                ),
                expected_utility=float(a["expected_utility"]),
                value=number(a["value"]),
                standard_error=a["standard_error"],
                expected_regret=a["expected_regret"],
                max_regret=a["max_regret"],
            )
            for a in self._body["actions"]
        )

    @property
    def verdict(self) -> Verdict:
        wire = self._body["verdict"]
        if wire == "no_admissible_action":
            return Verdict("no_admissible_action", ())
        ((kind, value),) = wire.items()
        if kind == "uniquely_optimal":
            return Verdict("uniquely_optimal", (value,))
        return Verdict("indistinguishable", tuple(value))

    @property
    def selected(self) -> tuple[str, ...]:
        """The leader, and any actions it cannot be told apart from; empty if none admissible."""
        return self.verdict.actions

    @property
    def evpi(self) -> float | None:
        """Expected value of perfect information over the admissible actions."""
        value = self._body["evpi"]
        return None if value is None else float(value)

    @property
    def n_draws(self) -> int:
        return int(self._body["n_draws"])

    @property
    def effective_draws(self) -> float:
        return float(self._body["effective_draws"])

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(self._body["assumptions"])

    @property
    def original_execution(self) -> AteAnalysisResult | None:
        """Original native scalar execution behind a contrast mean, or ``None``.

        Keeps its checked identification, diagnostics, support and uncertainty
        available for inspection. The decision is an affine point ranking, not an
        interval for utility; the original effect interval is never transferred.
        """
        if isinstance(self._source, MeanSource) and self._source._original_effect is not None:
            self._source._validated_original()
            return self._source._original_effect[0]._raw
        return None

    @property
    def source_evidence(self) -> tuple[SourceEvidence, ...]:
        """Original external support/request ancestry; no inferred numeric diagnostic or native authority."""
        sources: list[SourceEvidence] = []
        if isinstance(self._source, BoundExternalClaim):
            sources.append(self._source.source_evidence)
        elif isinstance(self._source, MeanSource) and self._source._original_native is not None:
            self._source._validated_original()
            sources.append(self._source._original_native.source_evidence)
        elif (
            isinstance(self._source, JointDistributionArtifact)
            and self._source.source_evidence is not None
        ):
            sources.append(self._source.source_evidence)
        return tuple(
            source.project(self._contract, [action.id for action in self.actions])
            for source in sources
        )

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        """Derivation chain behind the decision, parents before children.

        For a joint-draw source Rust derives it (contract, provider, distribution,
        decision contract, ``decision_result``). For an external claim it is the
        claim's own lineage plus the decision contract and ``decision_result``,
        re-chained natively so every link, including the two appended ones, has
        its Merkle digest (Python cannot hash).
        """
        if isinstance(self._source, BoundExternalClaim):
            decision_id = f"decision:{self.contract_identity}"
            rows = [[link.id, link.stage, list(link.parents)] for link in self._source.lineage]
            rows.append([decision_id, "decision_contract", []])
            rows.append([RESULT_LINK_ID, "claim", ["claim", decision_id]])
            return tuple(
                LineageLink(
                    item["id"],
                    item["stage"],
                    tuple(item["parents"]),
                    item["digest"],
                    tuple(item["parent_digests"]),
                )
                for item in json.loads(_composition_lineage(json.dumps(rows)))
            )
        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in self._body["lineage"]
        )

    def stages_behind(self, link: str = RESULT_LINK_ID) -> frozenset[str]:
        """Stages standing behind ``link`` (default: the reported decision)."""
        by_id = {item.id: item for item in self.lineage}
        if link not in by_id:
            raise CausalValueError(f"unknown lineage link {link!r}")
        seen: set[str] = set()
        stack = [link]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(by_id[current].parents)
        return frozenset(by_id[item].stage for item in seen)

    def explain(self) -> str:
        """Why this action, or why none: the choice, its evidence and its limits."""
        verdict = self.verdict
        by_id = {a.id: a for a in self.actions}
        criterion = self._contract.criterion.kind.replace("_", " ")
        if verdict.kind == "no_admissible_action":
            reasons = "; ".join(
                f"{a.id} failed {e.constraint_id} (held {e.probability:.3g}, required {e.required:.3g})"
                for a in self.actions
                for e in a.exclusions
            )
            return f"No action satisfies every hard constraint: {reasons}."
        leader = by_id[verdict.actions[0]]
        text = f"{leader.id!r} ranks first by {criterion} (value {leader.value:.4g}"
        if leader.standard_error is not None:
            text += f", standard error {leader.standard_error:.3g}"
        text += ")"
        if verdict.kind == "indistinguishable":
            others = ", ".join(repr(a) for a in verdict.actions[1:])
            text += f"; it ties {others} at the same criterion value"
        excluded = [a for a in self.actions if not a.admissible]
        if excluded:
            text += "; excluded by a hard constraint: " + ", ".join(repr(a.id) for a in excluded)
        if self.evpi is not None:
            text += f"; perfect information would be worth {self.evpi:.4g}"
        if isinstance(self._source, BoundExternalClaim):
            text += (
                "; values came from an external mean grid supplied by "
                f"{self._source.provenance_label}, not estimated natively, and carry no "
                "sampling error"
            )
        if any("point ranking" in assumption for assumption in self.assumptions):
            text += "; sampled-law point ranking only; Monte Carlo error is not certified"
        return text + "."

    def export(self, *, artifact_id: str = "decision-result") -> bytes:
        """The result bound to its contract identity and source digest, replayable.

        Only a joint-draw source can be exported and replayed; a decision computed
        from an external or native mean grid refuses with ``route_not_supported``.
        """
        if isinstance(self._source, (BoundExternalClaim, MeanSource)):
            raise DecisionRefusal(
                {
                    "code": "route_not_supported",
                    "stage": "export",
                    "detail": "decision_evaluation.mean_source_not_replayable",
                    "offending": None,
                    "expected": "joint_draws",
                    "supplied": "mean",
                    "remedy": "export a decision evaluated on aligned joint draws",
                }
            )
        return bytes(
            _export_result(json.dumps(self._contract._wire()), self._source._native, artifact_id)
        )

    def __repr__(self) -> str:
        return f"<Decision {self.verdict.kind} {self.selected}>"


def replay(data: bytes, *, contract: Contract, source: JointDistributionArtifact) -> Decision:
    """Recompute a stored decision from its inputs and require an exact match.

    The contract and source are the consumer's own: a result computed under a
    different contract or from different draws, or one whose numbers were edited,
    refuses.
    """
    if not isinstance(contract, Contract):
        raise CausalTypeError(f"replay needs a decision.Contract, not {type(contract).__name__}")
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError(
            "replay needs the JointDistributionArtifact of aligned draws the result was computed "
            f"from, not {type(source).__name__}; only a joint-draw decision replays"
        )
    result, refusal = _replay(data, json.dumps(contract._wire()), source._native)
    _raise(refusal)
    assert result is not None
    return Decision(contract, source, json.loads(result))


def source_digest(source: JointDistributionArtifact) -> str:
    """Digest of a source's aligned draws, to retain alongside a result."""
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError(
            f"source_digest needs a JointDistributionArtifact, not {type(source).__name__}"
        )
    return str(_source_digest(source._native))


__all__ = [
    "Action",
    "ActionOutcome",
    "Constraint",
    "ConstraintExclusion",
    "Contract",
    "Criterion",
    "Decision",
    "DecisionRefusal",
    "Expr",
    "MeanSource",
    "Verdict",
    "const",
    "maximum",
    "minimum",
    "replay",
    "source_digest",
    "x",
]
