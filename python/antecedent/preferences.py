"""Decide when the trade-offs between outcomes are stated as ranges rather than single numbers.

A decision that weighs several outcomes (benefit, harm, cost) needs to say how much of one is
worth how much of another. Here you state those trade-offs explicitly: each one is a named
parameter with a declared range (:func:`strength`) or a fixed value (:func:`fixed`), and every
range names the judgment it stands on. The decision is then answered across every admissible
value of the trade-offs at once, and, when the analysis supplies several causal structures,
across those structures too::

    from antecedent import preferences as pf

    benefit = pf.attribute("benefit", units="qaly")
    harm = pf.attribute("harm", units="events")
    w = pf.strength("harm_weight", units="qaly/event", lower=0.5, upper=4.0,
                    assumption="clinical_panel_2026")
    model = pf.ValueModel(benefit - w * harm, utility_units="qaly")
    contract = pf.Contract(
        model=model,
        actions=[
            pf.Action("treat", inputs={"benefit": treat_benefit, "harm": treat_harm}),
            pf.Action("wait", inputs={"benefit": wait_benefit, "harm": 0.0}),
        ],
        target_population="target",
    )
    decision = antecedent.require_success(contract.evaluate(joint_draws))
    print(decision.statement)      # e.g. "Action treat is preferred for every admissible value ..."
    for switch in decision.switches:
        print(switch.first, switch.second, switch.threshold)

What a result is and is not:

* A range of trade-off values is a declared assumption, not a probability. A verdict over the
  range says what holds for every admissible value, or where the choice changes; it never says
  how likely a value is.
* A share is reported only for a supplied :class:`Measure` and is that measure's weight, not a
  probability of anyone's preferences unless the measure is one.
* When the value is affine in the trade-offs and the criterion is an expectation, the verdict is
  exact (linear programs). Otherwise the set is subdivided into boxes and each box is certified
  with interval arithmetic; part of the set may be left undecided, and the result says how much.

Stating the trade-offs is the whole capability. Learning them from recorded choices is optional
and lives in :mod:`antecedent.preference_learning`; anything it produces plugs in here as
``Contract(preferences=...)``. This module does not depend on it.

Rust owns validation, identity, the route, every verdict, artifacts and refusals; a refusal is a
:class:`PreferenceRefusal` carrying the registered ``code`` and ``detail``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

from antecedent import _reason_details

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._native import decide_preferences_joint as _decide_joint
from ._native import decide_preferences_means as _decide_means
from ._native import decide_preferences_structures as _decide_structures
from ._native import export_preference_decision as _export_decision
from ._native import preference_chain_joint_law as _chain_joint_law
from ._native import preference_compile_across_studies as _compile_across
from ._native import preference_compile_for_study as _compile_for_study
from ._native import preference_contract_describe as _describe_contract
from ._native import preference_experiment_quantity as _experiment_quantity
from ._native import preference_experiment_source as _experiment_source
from ._native import preference_joint_law_of as _joint_law_of
from ._native import preference_model_describe as _describe_model
from ._native import replay_preference_decision as _replay_decision
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import real, text
from .decision import Constraint, ConstraintExclusion, Criterion
from .decision import Expr as _DecisionExpr
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, resolve_execution
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import Answer, outcome_operation, require_success
from .results.sources import MeanSource

if TYPE_CHECKING:
    from .decision_robust import Claim
    from .program_claims import ProgramBinding

ActionKind = Literal["intervention", "policy", "regime", "study", "external"]
StructuralPolicy = Literal[
    "require_invariant_best_action", "maximin", "bayes_over_structures", "report_only"
]
StructureKind = Literal["point", "graph_dependent", "weighted_graph_posterior", "finite_scenarios"]
Sense = Literal["at_least", "at_most", "equal"]

_SENSES: dict[str, Sense] = {">=": "at_least", "<=": "at_most", "==": "equal"}
_SYMBOLS = {"at_least": ">=", "at_most": "<=", "equal": "=="}


class PreferenceRefusal(StructuredRefusal):
    """A refusal of a value model, a preference contract or a decision across preferences.

    A :class:`~antecedent.errors.StructuredRefusal`: ``code`` (alias ``reason_code``) is the
    registered refusal code and ``detail`` the namespaced ``family.slot`` (for example
    ``preference_decision.measure_required``); ``offending`` names the parameter, action or
    record when there is one, and ``expected`` / ``supplied`` carry the compared values.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        message = refusal.get("message")
        detail = str(refusal["detail"])
        text_ = (
            detail + (f": {message}" if message else "") + (f" at {offending}" if offending else "")
        )
        super().__init__(refusal, text=text_)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


def _raise(payload: str | None) -> None:
    raise_refusal(payload, PreferenceRefusal)


def _number(value: object, what: str) -> float:
    return real(value, what, finite=True)


# --------------------------------------------------------------------------- expressions


def _merge(*groups: tuple[Any, ...]) -> tuple[Any, ...]:
    seen: list[Any] = []
    for group in groups:
        for item in group:
            if not any(item is known or item == known for known in seen):
                seen.append(item)
    return tuple(seen)


class _Algebra:
    """Arithmetic shared by attributes, trade-off parameters, curves and expressions."""

    __slots__ = ()

    def _expr(self) -> Expr:
        raise NotImplementedError

    def __add__(self, other: Operand) -> Expr:
        return _binary("add", self, other)

    def __radd__(self, other: Operand) -> Expr:
        return _binary("add", other, self)

    def __sub__(self, other: Operand) -> Expr:
        return _binary("sub", self, other)

    def __rsub__(self, other: Operand) -> Expr:
        return _binary("sub", other, self)

    def __mul__(self, other: Operand) -> Expr:
        return _binary("mul", self, other)

    def __rmul__(self, other: Operand) -> Expr:
        return _binary("mul", other, self)

    def __truediv__(self, other: float) -> Expr:
        divisor = _number(other, "a divisor")
        if divisor == 0.0:
            raise CausalValueError("an expression is divided only by a nonzero number")
        return _binary("mul", self, 1.0 / divisor)

    def __neg__(self) -> Expr:
        inner = self._expr()
        return Expr({"neg": inner._node}, inner._leaves)

    def __pos__(self) -> Expr:
        return self._expr()

    def __ge__(self, other: Operand) -> Relation:
        return _relation_of(self, other, "at_least")

    def __le__(self, other: Operand) -> Relation:
        return _relation_of(self, other, "at_most")


Operand: TypeAlias = "_Algebra | float"


def _coerce(value: Any) -> Expr:
    if isinstance(value, _Algebra):
        return value._expr()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalValueError(
            "a value expression combines attributes, trade-off parameters, curves, expressions "
            f"and numbers, not {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError("a constant in a value expression is finite")
    return Expr({"const": number}, ())


def _binary(op: str, left: Any, right: Any) -> Expr:
    a, b = _coerce(left), _coerce(right)
    return Expr({op: [a._node, b._node]}, _merge(a._leaves, b._leaves))


@dataclass(frozen=True, slots=True, eq=False)
class Expr(_Algebra):
    """A closed expression over attributes, trade-off parameters and numbers.

    Build it with ``+``, ``-``, ``*``, ``/`` by a number, unary ``-``, :func:`exp`, :func:`ln`,
    :func:`maximum`, :func:`minimum` and the shapes. Comparing two parameter expressions with
    ``>=`` or ``<=`` gives a :class:`Relation`, not a boolean.
    """

    _node: Any
    _leaves: tuple[Any, ...] = ()

    def _expr(self) -> Expr:
        return self

    def __bool__(self) -> bool:
        raise CausalTypeError("an expression has no truth value")

    def __repr__(self) -> str:
        return f"Expr({json.dumps(self._node, sort_keys=True)})"


@dataclass(frozen=True, slots=True)
class Attribute(_Algebra):
    """One outcome attribute the value reads (for example ``harm`` in ``events``)."""

    id: str
    units: str

    def __post_init__(self) -> None:
        text(self.id, "Attribute.id", non_empty=True)
        text(self.units, "Attribute.units", non_empty=True)

    def _expr(self) -> Expr:
        return Expr({"attribute": self.id}, (self,))


@dataclass(frozen=True, slots=True)
class Parameter(_Algebra):
    """A trade-off parameter with its admissible range and the judgment the range stands on.

    The range is a declared assumption, not a probability distribution. ``lower == upper`` fixes
    the parameter.
    """

    id: str
    units: str
    lower: float
    upper: float
    assumption: str

    def __post_init__(self) -> None:
        text(self.id, "Parameter.id", non_empty=True)
        text(self.units, "Parameter.units", non_empty=True)
        text(self.assumption, "Parameter.assumption", non_empty=True)
        object.__setattr__(self, "lower", _number(self.lower, "Parameter.lower"))
        object.__setattr__(self, "upper", _number(self.upper, "Parameter.upper"))

    @property
    def is_fixed(self) -> bool:
        """Whether the range is a single value."""
        return self.lower == self.upper

    def _expr(self) -> Expr:
        return Expr({"parameter": self.id}, (self,))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "units": self.units,
            "lower": self.lower,
            "upper": self.upper,
            "assumption": self.assumption,
        }


def attribute(id: str, *, units: str) -> Attribute:
    """An outcome attribute: what an action's input supplies for the value to read."""
    return Attribute(id, units)


def strength(id: str, *, units: str, lower: float, upper: float, assumption: str) -> Parameter:
    """A trade-off whose admissible values are ``[lower, upper]``.

    ``units`` is utility per unit of what it multiplies (``qaly/event``); ``assumption`` names
    the judgment the range stands on. Every value in the range is treated as admissible; none
    is more likely than another.
    """
    return Parameter(id, units, lower, upper, assumption)


def fixed(id: str, *, units: str, value: float, assumption: str = "declared_value") -> Parameter:
    """A trade-off known as one value (a range of width zero)."""
    return Parameter(id, units, value, value, assumption)


def exp(value: Operand) -> Expr:
    """Natural exponential."""
    inner = _coerce(value)
    return Expr({"exp": inner._node}, inner._leaves)


def ln(value: Operand) -> Expr:
    """Natural logarithm; the argument must stay positive wherever it is evaluated."""
    inner = _coerce(value)
    return Expr({"ln": inner._node}, inner._leaves)


def maximum(left: Operand, right: Operand) -> Expr:
    """Pointwise maximum."""
    return _binary("max", left, right)


def minimum(left: Operand, right: Operand) -> Expr:
    """Pointwise minimum."""
    return _binary("min", left, right)


def _attribute_arg(value: object, what: str) -> Attribute:
    if not isinstance(value, Attribute):
        raise CausalTypeError(f"{what} is an Attribute made by preferences.attribute(...)")
    return value


def excess(of: Attribute, above: float) -> Expr:
    """``max(x - above, 0)``: how far the attribute exceeds a threshold."""
    x = _attribute_arg(of, "excess(of=...)")
    return maximum(x - _number(above, "above"), 0.0)


def shortfall(of: Attribute, below: float) -> Expr:
    """``max(below - x, 0)``: how far the attribute falls short of a threshold."""
    x = _attribute_arg(of, "shortfall(of=...)")
    return maximum(_number(below, "below") - x, 0.0)


def segment(of: Attribute, start: float, end: float) -> Expr:
    """``clamp(x, start, end) - start``: the part of the attribute inside ``[start, end]``."""
    x = _attribute_arg(of, "segment(of=...)")
    lo, hi = _number(start, "start"), _number(end, "end")
    return minimum(maximum(x, lo), hi) - lo


def interaction(a: Attribute, b: Attribute) -> Expr:
    """``x_a * x_b``."""
    return _attribute_arg(a, "interaction(a=...)") * _attribute_arg(b, "interaction(b=...)")


def both(a: Attribute, b: Attribute) -> Expr:
    """``min(x_a, x_b)``: the attribute that is lower."""
    return minimum(_attribute_arg(a, "both(a=...)"), _attribute_arg(b, "both(b=...)"))


# --------------------------------------------------------------------------- relations


@dataclass(frozen=True, slots=True)
class Relation:
    """A linear restriction between trade-off parameters: ``sum(c * p) (>= | <= | ==) bound``.

    Write it naturally (``harm_weight >= 2 * cost_weight``) or with :func:`relation` to give it a
    stable id. It cuts the admissible set; it is a declared judgment, never evidence.
    """

    id: str
    terms: tuple[tuple[str, float], ...]
    sense: Sense
    bound: float
    origin: Literal["declared", "shape_rule"] = "declared"
    parameters: tuple[Parameter, ...] = field(default=(), compare=False, repr=False)

    def __bool__(self) -> bool:
        raise CausalTypeError(
            "a Relation is a declaration, not a truth value; pass it to ValueModel(relations=...)"
        )

    def _wire(self, index: Mapping[str, int]) -> dict[str, Any]:
        terms = []
        for name, coefficient in self.terms:
            if name not in index:
                raise CausalValueError(
                    f"relation {self.id!r} reads parameter {name!r}, which the value never reads"
                )
            terms.append([index[name], coefficient])
        return {
            "id": self.id,
            "terms": terms,
            "sense": self.sense,
            "bound": self.bound,
            "origin": {"kind": self.origin},
        }


def _linear(node: Any) -> tuple[dict[str, float], float]:
    """``(coefficients, constant)`` of an expression affine in the parameters."""
    ((op, arg),) = node.items()
    if op == "const":
        return {}, float(arg)
    if op == "parameter":
        return {arg: 1.0}, 0.0
    if op == "attribute":
        raise CausalValueError(
            f"a relation reads only trade-off parameters, not the attribute {arg!r}"
        )
    if op == "neg":
        terms, constant = _linear(arg)
        return {k: -v for k, v in terms.items()}, -constant
    if op in ("add", "sub"):
        (lt, lc), (rt, rc) = _linear(arg[0]), _linear(arg[1])
        sign = 1.0 if op == "add" else -1.0
        merged = dict(lt)
        for name, value in rt.items():
            merged[name] = merged.get(name, 0.0) + sign * value
        return merged, lc + sign * rc
    if op == "mul":
        (lt, lc), (rt, rc) = _linear(arg[0]), _linear(arg[1])
        if lt and rt:
            raise CausalValueError("a relation is linear: a parameter never multiplies another")
        if not lt:
            return {k: lc * v for k, v in rt.items()}, lc * rc
        return {k: rc * v for k, v in lt.items()}, lc * rc
    raise CausalValueError(f"a relation is linear in the parameters; {op} is not")


def _relation_of(left: Any, right: Any, sense: Sense, id: str | None = None) -> Relation:
    difference = _binary("sub", left, right)
    terms, constant = _linear(difference._node)
    pairs = tuple((name, value) for name, value in terms.items() if value != 0.0)
    if not pairs:
        raise CausalValueError("a relation reads at least one trade-off parameter")
    bound = 0.0 - constant
    if id is None:
        lhs = "".join(f"{value:+g}*{name}" for name, value in pairs)
        id = f"{lhs}{_SYMBOLS[sense]}{bound:g}"
    parameters = tuple(p for p in difference._leaves if isinstance(p, Parameter))
    return Relation(id, pairs, sense, bound, "declared", parameters)


def relation(
    id: str,
    left: Relation | Operand,
    sense: Literal[">=", "<=", "=="] | None = None,
    right: Operand | None = None,
) -> Relation:
    """A relation with a stable id.

    ``relation("harm_over_cost", harm_weight >= 2 * cost_weight)`` names a relation written with
    an operator; ``relation("equal_weights", a, "==", b)`` states one with an explicit sense
    (equality has no operator form).
    """
    text(id, "relation id", non_empty=True)
    if sense is None:
        if not isinstance(left, Relation) or right is not None:
            raise CausalTypeError("relation(id, rel) takes a Relation, or (id, left, sense, right)")
        return replace(left, id=id)
    if sense not in _SENSES:
        raise CausalValueError("a relation sense is '>=', '<=' or '=='")
    if right is None:
        raise CausalTypeError("relation(id, left, sense, right) needs the right side")
    return _relation_of(left, right, _SENSES[sense], id)


@dataclass(frozen=True, slots=True, eq=False)
class Curve(_Algebra):
    """A piecewise-linear value of one attribute: one slope parameter per segment.

    The curve is zero at the first breakpoint and flat outside the first and last ones. Its
    shape rules (:meth:`increasing`, :meth:`decreasing`, :meth:`diminishing`,
    :meth:`accelerating`) return relations to pass to :class:`ValueModel`; they restrict the
    slopes, so they are judgments like the ranges.
    """

    of: Attribute
    breakpoints: tuple[float, ...]
    slopes: tuple[Parameter, ...]
    id: str | None = None

    def __post_init__(self) -> None:
        _attribute_arg(self.of, "Curve.of")
        points = tuple(_number(b, "a breakpoint") for b in self.breakpoints)
        slopes = tuple(self.slopes)
        if any(not isinstance(s, Parameter) for s in slopes):
            raise CausalTypeError("Curve.slopes are Parameters made by strength(...) or fixed(...)")
        if len(points) < 2 or len(slopes) + 1 != len(points):
            raise CausalValueError("a curve has at least two breakpoints and one slope per segment")
        if any(a >= b for a, b in zip(points, points[1:], strict=False)):
            raise CausalValueError("curve breakpoints strictly increase")
        object.__setattr__(self, "breakpoints", points)
        object.__setattr__(self, "slopes", slopes)

    @property
    def _prefix(self) -> str:
        return self.id or f"{self.of.id}.curve"

    def _expr(self) -> Expr:
        parts = [
            slope * segment(self.of, a, b)
            for slope, a, b in zip(
                self.slopes, self.breakpoints, self.breakpoints[1:], strict=False
            )
        ]
        return reduce(lambda x, y: x + y, parts)

    def _single(self, rule: str, sense: Sense) -> tuple[Relation, ...]:
        return tuple(
            Relation(f"{self._prefix}.{rule}.{i}", ((s.id, 1.0),), sense, 0.0, "shape_rule", (s,))
            for i, s in enumerate(self.slopes)
        )

    def _adjacent(self, rule: str, sign: float) -> tuple[Relation, ...]:
        return tuple(
            Relation(
                f"{self._prefix}.{rule}.{i}",
                ((a.id, sign), (b.id, -sign)),
                "at_least",
                0.0,
                "shape_rule",
                (a, b),
            )
            for i, (a, b) in enumerate(zip(self.slopes, self.slopes[1:], strict=False))
            if a.id != b.id
        )

    def increasing(self) -> tuple[Relation, ...]:
        """Every slope is at least zero."""
        return self._single("increasing", "at_least")

    def decreasing(self) -> tuple[Relation, ...]:
        """Every slope is at most zero."""
        return self._single("decreasing", "at_most")

    def diminishing(self) -> tuple[Relation, ...]:
        """Each slope is at least the next one (each further unit is worth no more)."""
        return self._adjacent("diminishing", 1.0)

    def accelerating(self) -> tuple[Relation, ...]:
        """Each slope is at most the next one."""
        return self._adjacent("accelerating", -1.0)


# --------------------------------------------------------------------------- value model


def _resolve(node: Any, attributes: Mapping[str, int], parameters: Mapping[str, int]) -> Any:
    ((op, arg),) = node.items()
    if op == "attribute":
        return {"attribute": attributes[arg]}
    if op == "parameter":
        return {"parameter": parameters[arg]}
    if op == "const":
        return node
    if isinstance(arg, list):
        return {op: [_resolve(item, attributes, parameters) for item in arg]}
    return {op: _resolve(arg, attributes, parameters)}


def _declarations(kind: type, found: Iterable[Any], given: Sequence[Any] | None) -> tuple[Any, ...]:
    by_id: dict[str, Any] = {}
    for candidate in (*(given or ()), *found):
        if not isinstance(candidate, kind):
            continue
        item: Any = candidate
        known = by_id.get(item.id)
        if known is not None and known != item:
            raise CausalValueError(
                f"{kind.__name__.lower()} {item.id!r} is declared twice with different values"
            )
        by_id.setdefault(item.id, item)
    return tuple(by_id.values())


def _flatten(relations: Iterable[Any]) -> tuple[Relation, ...]:
    out: list[Relation] = []
    for item in relations:
        if isinstance(item, Relation):
            out.append(item)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            out.extend(_flatten(item))
        else:
            raise CausalTypeError("ValueModel.relations holds Relations (or groups of them)")
    return tuple(out)


@dataclass(frozen=True, slots=True, eq=False)
class ValueModel:
    """How the outcome attributes combine into one value, with explicit trade-offs.

    ``value`` is an expression over attributes and trade-off parameters. The attributes and
    parameters are read from it in order of first appearance unless ``attributes=`` or
    ``parameters=`` fixes the order. ``relations`` restrict the parameters jointly. The model is
    complete on its own: it is a declaration of judgments, not an estimate.
    """

    value: Any
    utility_units: str
    relations: tuple[Relation, ...] = ()
    attributes: tuple[Attribute, ...] | None = None
    parameters: tuple[Parameter, ...] | None = None

    def __post_init__(self) -> None:
        expr = _coerce(self.value)
        text(self.utility_units, "ValueModel.utility_units", non_empty=True)
        relations = _flatten(self.relations)
        attributes = _declarations(Attribute, expr._leaves, self.attributes)
        parameters = _declarations(
            Parameter,
            (*expr._leaves, *(p for r in relations for p in r.parameters)),
            self.parameters,
        )
        object.__setattr__(self, "value", expr)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "parameters", parameters)

    def _wire(self) -> dict[str, Any]:
        assert self.attributes is not None and self.parameters is not None
        attributes = {a.id: k for k, a in enumerate(self.attributes)}
        parameters = {p.id: j for j, p in enumerate(self.parameters)}
        return {
            "attributes": [{"id": a.id, "units": a.units} for a in self.attributes],
            "parameters": [p._wire() for p in self.parameters],
            "relations": [r._wire(parameters) for r in self.relations],
            "value": _resolve(self.value._node, attributes, parameters),
            "utility_units": self.utility_units,
        }

    def _preference_wire(self) -> dict[str, Any]:
        """The protocol a :class:`Contract` reads from ``preferences=``: the model and no measure."""
        return {"model": self._wire(), "measure": None}

    def _described(self) -> dict[str, Any]:
        body, refusal = _describe_model(json.dumps(self._wire()))
        _raise(refusal)
        assert body is not None
        return dict(json.loads(body))

    @property
    def identity(self) -> str:
        """Canonical digest; any semantic edit (a range, a relation, the expression) changes it."""
        return str(self._described()["identity"])

    @property
    def shape_parameters(self) -> tuple[str, ...]:
        """Parameters the value is not affine in. Non-empty means the certified route answers."""
        return tuple(self._described()["shape_parameters"])

    def validate(self) -> ValueModel:
        """Check the model natively; raise :class:`PreferenceRefusal` when it is invalid."""
        self._described()
        return self


# --------------------------------------------------------------------------- contract


@dataclass(frozen=True, slots=True)
class Action:
    """One candidate action: what it supplies for each attribute, by attribute id.

    A value is a :class:`~antecedent.joint_distribution.ScientificQuantity` whose law the source
    supplies, or a plain number for an input known without uncertainty (in the attribute's
    units). At least one input is a quantity.
    """

    id: str
    inputs: Mapping[str, ScientificQuantity | float]
    kind: ActionKind = "intervention"

    def __post_init__(self) -> None:
        text(self.id, "Action.id", non_empty=True)
        if not isinstance(self.inputs, Mapping):
            raise CausalTypeError("Action.inputs maps attribute ids to quantities or numbers")
        inputs: dict[str, ScientificQuantity | float] = {}
        for key, value in self.inputs.items():
            name = key.id if isinstance(key, Attribute) else key
            if not isinstance(name, str):
                raise CausalTypeError("Action.inputs keys are attribute ids or Attributes")
            if not isinstance(value, ScientificQuantity):
                value = _number(value, f"Action {self.id!r} input {name!r}")
            inputs[name] = value
        if self.kind not in get_args(ActionKind):
            raise CausalValueError(f"Action.kind is one of {get_args(ActionKind)}")
        object.__setattr__(self, "inputs", MappingProxyType(inputs))

    @property
    def quantities(self) -> tuple[ScientificQuantity, ...]:
        """The scientific quantities the action reads, in input order."""
        return tuple(v for v in self.inputs.values() if isinstance(v, ScientificQuantity))

    def _wire(self, attributes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        names = [a["id"] for a in attributes]
        missing = [n for n in names if n not in self.inputs]
        extra = [n for n in self.inputs if n not in names]
        if missing or extra:
            raise CausalValueError(
                f"Action {self.id!r} supplies every attribute of the model exactly once"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {extra}" if extra else "")
            )
        wire = []
        for attr in attributes:
            value = self.inputs[attr["id"]]
            if isinstance(value, ScientificQuantity):
                wire.append({"quantity": value._wire()})
            else:
                wire.append({"fixed": {"value": float(value), "units": attr["units"]}})
        return {"id": self.id, "kind": self.kind, "inputs": wire}


@dataclass(frozen=True, slots=True)
class Policy:
    """How a result across the admissible trade-offs may be consumed.

    * :meth:`invariant_best` chooses only an action best at every admissible value;
    * :meth:`maximin` the action with the best worst-case value over the set;
    * :meth:`minimax_regret` the action with the smallest worst-case regret;
    * :meth:`weighted` weighs values by a supplied :class:`Measure` (refused without one);
    * :meth:`report_only` reports the answer across the set and chooses nothing.
    """

    kind: Literal[
        "require_invariant_best_action", "maximin", "minimax_regret", "weighted", "report_only"
    ]

    @classmethod
    def invariant_best(cls) -> Policy:
        return cls("require_invariant_best_action")

    @classmethod
    def maximin(cls) -> Policy:
        return cls("maximin")

    @classmethod
    def minimax_regret(cls) -> Policy:
        return cls("minimax_regret")

    @classmethod
    def weighted(cls) -> Policy:
        return cls("weighted")

    @classmethod
    def report_only(cls) -> Policy:
        return cls("report_only")


@dataclass(frozen=True, slots=True, eq=False)
class Measure:
    """A weighted set of trade-off values over the admissible set.

    Each point gives a value per parameter id (a mapping) or one value per parameter in model
    order (a sequence). ``labels`` names points (named viewpoints, see :meth:`viewpoints`).
    ``assumption`` names the judgment behind the weights; a measure estimated from recorded
    choices carries ``estimate`` instead. ``unresolved`` names parameters the measure says
    nothing about; their declared range still applies. A share computed from a measure is that
    measure's weight, not a probability of preferences unless the measure is one.
    """

    points: tuple[Mapping[str, float] | tuple[float, ...], ...]
    weights: tuple[float, ...] | None = None
    labels: tuple[str, ...] | None = None
    assumption: str | None = None
    estimate: str | None = None
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.assumption is None) == (self.estimate is None):
            raise CausalValueError("a Measure names its assumption= (or an estimate=), exactly one")
        object.__setattr__(self, "points", tuple(self.points))
        if self.weights is not None:
            object.__setattr__(
                self, "weights", tuple(_number(w, "a measure weight") for w in self.weights)
            )
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))

    @classmethod
    def viewpoints(
        cls,
        views: Mapping[str, Mapping[str, float]],
        *,
        assumption: str,
        weights: Mapping[str, float] | None = None,
    ) -> Measure:
        """Named viewpoints, each a trade-off value per parameter, equally weighted by default."""
        labels = tuple(views)
        return cls(
            points=tuple(views[label] for label in labels),
            weights=None if weights is None else tuple(weights[label] for label in labels),
            labels=labels,
            assumption=assumption,
        )

    def _wire(self, parameters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ids = [p["id"] for p in parameters]
        unknown = [name for name in self.unresolved if name not in ids]
        if unknown:
            raise CausalValueError(f"Measure.unresolved names unknown parameters {unknown}")
        points = []
        for index, point in enumerate(self.points):
            if isinstance(point, Mapping):
                extra = [name for name in point if name not in ids]
                if extra:
                    raise CausalValueError(
                        f"measure point {index} names unknown parameters {extra}"
                    )
                row = []
                for p in parameters:
                    if p["id"] in point:
                        row.append(_number(point[p["id"]], f"measure point {index}"))
                    elif p["lower"] == p["upper"] or p["id"] in self.unresolved:
                        row.append((p["lower"] + p["upper"]) / 2.0)
                    else:
                        raise CausalValueError(
                            f"measure point {index} gives no value for parameter {p['id']!r}"
                        )
                points.append(row)
            else:
                points.append([_number(v, f"measure point {index}") for v in point])
        origin = (
            {"kind": "declared", "assumption": self.assumption}
            if self.assumption is not None
            else {"kind": "choice_estimate", "estimate": self.estimate}
        )
        return {
            "points": points,
            "weights": None if self.weights is None else list(self.weights),
            "labels": None if self.labels is None else list(self.labels),
            "origin": origin,
            "unresolved": [ids.index(name) for name in self.unresolved],
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any], parameter_ids: Sequence[str]) -> Measure:
        origin = wire["origin"]
        return cls(
            points=tuple(tuple(point) for point in wire["points"]),
            weights=None if wire.get("weights") is None else tuple(wire["weights"]),
            labels=None if wire.get("labels") is None else tuple(wire["labels"]),
            assumption=origin.get("assumption"),
            estimate=origin.get("estimate"),
            unresolved=tuple(parameter_ids[j] for j in wire.get("unresolved", ())),
        )


@dataclass(frozen=True, slots=True)
class Structures:
    """Several causal structures, each with its own evidence, decided together with the trade-offs.

    ``claims`` are :class:`antecedent.decision_robust.Claim` values (evaluated with draws,
    unidentified, or unevaluated with a reason), read as ``kind`` reads them in
    :mod:`antecedent.decision_robust`. Unidentified and unevaluated mass is reported, never
    renormalized away.
    """

    kind: StructureKind
    claims: tuple[Claim, ...]

    def __post_init__(self) -> None:
        if self.kind not in get_args(StructureKind):
            raise CausalValueError(f"Structures.kind is one of {get_args(StructureKind)}")
        object.__setattr__(self, "claims", tuple(self.claims))


def constraint(
    id: str,
    expr: Operand,
    *,
    at_most: float,
    units: str,
    min_probability: float = 1.0,
    applies_to: Sequence[str] = (),
) -> Constraint:
    """A hard constraint ``P(expr <= at_most) >= min_probability`` over attributes.

    It never reads a trade-off parameter, so whether it excludes an action does not depend on
    preferences. Returns a :class:`antecedent.decision.Constraint` to pass to ``constraints=``.
    """
    inner = _coerce(expr)
    if any(isinstance(leaf, Parameter) for leaf in inner._leaves):
        raise CausalValueError("a hard constraint reads attributes only, never a trade-off")
    return Constraint(
        id=id,
        expr=_DecisionExpr(inner._node),
        bound=at_most,
        units=units,
        min_probability=min_probability,
        applies_to=tuple(applies_to),
    )


def _constraint_node(node: Any, attributes: Mapping[str, int]) -> Any:
    ((op, arg),) = node.items()
    if op == "attribute":
        if arg not in attributes:
            raise CausalValueError(f"a constraint reads attribute {arg!r}, which the model lacks")
        return {"input": attributes[arg]}
    if op in ("const", "input"):
        return node
    if op in ("parameter", "exp", "ln"):
        raise CausalValueError(f"a hard constraint cannot read {op}")
    if isinstance(arg, list):
        return {op: [_constraint_node(item, attributes) for item in arg]}
    return {op: _constraint_node(arg, attributes)}


def _options(
    tie_tolerance: float | None,
    vertex_cap: int | None,
    max_boxes: int | None,
    min_width_fraction: float | None,
) -> dict[str, Any]:
    values = {
        "tie_tolerance": tie_tolerance,
        "vertex_cap": vertex_cap,
        "max_boxes": max_boxes,
        "min_width_fraction": min_width_fraction,
    }
    return {key: value for key, value in values.items() if value is not None}


def _mean_wire(source: MeanSource | BoundExternalClaim) -> dict[str, Any]:
    if isinstance(source, BoundExternalClaim):
        identity = source.identity_fields
        return {
            "coordinates": [q._wire() for q in source.quantities],
            "means": [float(v) for v in source.values],
            "provider_id": str(identity["provider_id"]),
            "snapshot_id": str(identity["snapshot_id"]),
            "causal_contract_id": str(identity["causal_contract_id"]),
            "rng_id": "none:mean_grid",
        }
    source._validated_original()
    return {
        "coordinates": [q._wire() for q in source.coordinates],
        "means": list(source.means),
        "provider_id": source.provider_id,
        "snapshot_id": source.snapshot_id,
        "causal_contract_id": source.causal_contract_id,
        "rng_id": source.rng_id,
    }


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Contract:
    """A decision whose value comes from a model with explicit trade-offs.

    Give the model as ``model=`` (a :class:`ValueModel`), or as ``preferences=``: any object
    exposing ``_preference_wire()``, such as a :class:`ValueModel` or the accepted preferences
    of :mod:`antecedent.preference_learning` (which then also supply their measure, if any).
    ``criterion`` and ``constraints`` are those of :mod:`antecedent.decision`; a constraint reads
    attributes (build one with :func:`constraint`, or with ``decision.x(k)`` for attribute
    ``k``). ``policy`` says how the answer across the trade-offs is consumed.
    """

    actions: tuple[Action, ...]
    target_population: str
    model: ValueModel | None = None
    preferences: Any = None
    criterion: Criterion = field(default_factory=Criterion.expected_utility)
    policy: Policy = field(default_factory=Policy.invariant_best)
    constraints: tuple[Constraint, ...] = ()
    horizon: int = 0
    structural_policy: StructuralPolicy = "report_only"

    def __post_init__(self) -> None:
        if (self.model is None) == (self.preferences is None):
            raise CausalValueError("a Contract takes model= or preferences=, exactly one")
        if self.model is not None and not isinstance(self.model, ValueModel):
            raise CausalTypeError("Contract.model is a ValueModel")
        if self.preferences is not None and not callable(
            getattr(self.preferences, "_preference_wire", None)
        ):
            raise CausalTypeError(
                "Contract.preferences is a ValueModel or accepted preferences "
                "(an object with _preference_wire())"
            )
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Iterable):
            raise CausalTypeError("Contract.actions is a sequence of Action")
        actions = tuple(self.actions)
        if any(not isinstance(a, Action) for a in actions):
            raise CausalTypeError("Contract.actions holds only Action values")
        constraints = tuple(self.constraints)
        if any(not isinstance(c, Constraint) for c in constraints):
            raise CausalTypeError("Contract.constraints holds decision.Constraint values")
        if not isinstance(self.criterion, Criterion):
            raise CausalTypeError("Contract.criterion is a decision.Criterion")
        if not isinstance(self.policy, Policy):
            raise CausalTypeError("Contract.policy is a Policy such as Policy.maximin()")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise CausalTypeError("Contract.horizon is an integer")
        if self.structural_policy not in get_args(StructuralPolicy):
            raise CausalValueError(
                f"Contract.structural_policy is one of {get_args(StructuralPolicy)}"
            )
        text(self.target_population, "Contract.target_population", non_empty=True)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "constraints", constraints)

    def _preferences(self) -> dict[str, Any]:
        source = self.model if self.model is not None else self.preferences
        return dict(source._preference_wire())

    def _wire(self) -> dict[str, Any]:
        model = self._preferences()["model"]
        attributes = {a["id"]: k for k, a in enumerate(model["attributes"])}
        return {
            "model": model,
            "actions": [a._wire(model["attributes"]) for a in self.actions],
            "criterion": self.criterion._wire(),
            "constraints": [
                {
                    **c._wire(),
                    "expr": _constraint_node(c.expr._wire_value, attributes),
                }
                for c in self.constraints
            ],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self.structural_policy,
            "preference_policy": self.policy.kind,
        }

    def _described(self) -> dict[str, Any]:
        body, refusal = _describe_contract(json.dumps(self._wire()))
        _raise(refusal)
        assert body is not None
        return dict(json.loads(body))

    @property
    def identity(self) -> str:
        """Canonical digest of the contract, to retain for replay."""
        return str(self._described()["identity"])

    @property
    def route(self) -> Literal["exact", "certified"]:
        """``exact`` (linear programs) or ``certified`` (interval subdivision)."""
        return self._described()["route"]

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(p["id"] for p in self._preferences()["model"]["parameters"])

    def _measure(self, measure: Measure | None) -> dict[str, Any] | None:
        carried = self._preferences().get("measure")
        if measure is None:
            return carried
        if carried is not None:
            raise CausalValueError(
                "these accepted preferences carry their own measure; do not pass measure= too"
            )
        if not isinstance(measure, Measure):
            raise CausalTypeError("measure= is a preferences.Measure")
        return measure._wire(self._preferences()["model"]["parameters"])

    def _proxy(self) -> Any:
        """An ordinary decision contract over the same quantities, so an analysis result is read
        exactly as :meth:`antecedent.decision.Contract.evaluate` reads it."""
        from . import decision

        needs = self._described()["needs"]
        actions = []
        for action in self.actions:
            quantities = action.quantities
            utility = reduce(lambda a, b: a + b, (decision.x(i) for i in range(len(quantities))))
            actions.append(decision.Action(action.id, inputs=quantities, utility=utility))
        return decision.Contract(
            actions=tuple(actions),
            utility_units=self._preferences()["model"]["utility_units"],
            criterion=(
                Criterion.expected_utility() if needs == "means" else Criterion.quantile(0.5)
            ),
            target_population=self.target_population,
            horizon=self.horizon,
        )

    def _source(
        self,
        source: Any,
        *,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
    ) -> Any:
        """The source to decide on: joint draws, a mean source or structures."""
        from .decision import _result_mean_source
        from .results.sources import adapt_mean_source, is_mean_producer

        source = require_success(source)
        if isinstance(source, Structures):
            return source
        if isinstance(source, (list, tuple)):
            return joint_law(*source)
        from .decision_evidence import JointEvidence

        if isinstance(source, JointEvidence):
            return experiment_means(source)
        from .propagation_chain import PropagationChain

        if isinstance(source, PropagationChain):
            return chain_law(source)
        if is_mean_producer(source):
            source = adapt_mean_source(source)
        cohort_source = getattr(source, "_cohort_mean_source", None)
        if cohort_source is not None:
            source = cohort_source()
        from .results._execution import ResultAPI

        if isinstance(source, ResultAPI):
            source = _result_mean_source(
                self._proxy(),
                source,
                program=program,
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=population,
            )
        if isinstance(source, (JointDistributionArtifact, MeanSource, BoundExternalClaim)):
            return source
        raise CausalTypeError(
            "preferences.Contract.evaluate reads a JointDistributionArtifact (aligned draws), a "
            "sequence of them whose rows pair, a PropagationChain with joint outputs, the "
            "JointEvidence of one experiment (its arm means), a "
            "MeanSource or BoundExternalClaim (means), Structures, or a supported analysis "
            f"result, not {type(source).__name__}"
        )

    @outcome_operation
    def compile(
        self,
        source: Any,
        *,
        draws: int | None = None,
        seed: int = 0,
        shared: Mapping[tuple[int, int], Sequence[float]] | None = None,
        refits: int | None = None,
        execution: Execution | None = None,
    ) -> CompiledDecision:
        """Bind this contract to the analysis that estimates its quantities and compile it.

        This is the way in. ``source`` is the analysis, prepared with
        :func:`antecedent.prepare`: one prepared study (its mean response curve, under its
        structure or each structure of its graph posterior), a sequence of prepared studies
        (one outcome each), or a compiled design of several outcomes
        (``joint_panel.prepare_native_joint_design``). Every quantity an action reads must be
        a coordinate the analysis targets. The decision is lowered into the analysis's own
        plan, so :meth:`CompiledDecision.inspect` shows the quantities, the source the
        contract needs, the trade-off ranges and the identification before anything is
        fitted; what cannot be answered (a quantity not identified, a source the analysis
        cannot supply, studies that share observations without a declared covariance) is
        refused here. :meth:`CompiledDecision.execute` then runs each plan once.

        Several studies meet by their means when means suffice. A value that needs one law
        of their outcomes needs ``draws=`` (the aligned rows the propagation chain draws,
        under ``seed=``) and, for studies that share observations, ``shared=``: the declared
        covariance between the estimates of each such pair ``(i, j)``, rows of study ``i``'s
        coordinates by columns of study ``j``'s. A static-regime design reads its arm means,
        or with ``refits=`` (at least 99, under ``seed=``) its aligned whole-subject refits.
        """
        from ._joint_panel_designs import PreparedNativeJointDesign
        from ._panel_execution import run_panel
        from .design.ranking import _bound_native
        from .transport._controls import call_native

        source = require_success(source)
        contract_json = json.dumps(self._wire())
        if isinstance(source, PreparedNativeJointDesign):
            bootstrap = None if refits is None else json.dumps({"replicates": refits, "seed": seed})
            compiled, refusal = run_panel(
                execution,
                source._handle.compile_for_preference_decision,
                contract_json,
                bootstrap,
            )
        elif isinstance(source, (list, tuple)):
            natives = [_bound_native(require_success(study)) for study in source]
            if shared is not None:
                kind = "shared_evidence"
            elif draws is not None:
                kind = "distinct_evidence"
            else:
                kind = "undeclared"
            dependence = {
                "kind": kind,
                "draws": 2000 if draws is None else int(draws),
                "seed": int(seed),
                "cross": [
                    {"first": int(i), "second": int(j), "covariance": [float(v) for v in block]}
                    for (i, j), block in (shared or {}).items()
                ],
            }
            compiled, refusal = call_native(
                execution,
                _compile_across,
                natives,
                contract_json,
                json.dumps(dependence),
                seed="seed",
            )
        else:
            compiled, refusal = call_native(
                execution, _compile_for_study, _bound_native(source), contract_json, seed="seed"
            )
        _raise(refusal)
        assert compiled is not None
        return CompiledDecision(self, compiled)

    @outcome_operation
    def evaluate(
        self,
        source: Any,
        *,
        measure: Measure | None = None,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
        execution: Execution | None = None,
    ) -> PreferenceDecision:
        """Decide across every admissible value of the trade-offs, on a source already in hand.

        The way in is :meth:`compile`, which binds the contract to the analysis, shows the plan
        and refuses what cannot be answered before anything is fitted. This is the lower layer
        that route calls: it reads a source the caller already holds.

        ``source`` is what :meth:`antecedent.decision.Contract.evaluate` reads (aligned joint
        draws; a mean source or bound external claim; a supported analysis result, read as its
        mean source with ``program=``, ``outcome_units=``, ``dose_units=`` and ``population=``
        as there), or :class:`Structures` to decide across causal structures as well. Several
        separately estimated outcomes are read as one law: a sequence of
        :class:`~antecedent.joint_distribution.JointDistributionArtifact` whose rows already pair
        (see :func:`joint_law`), or a :class:`~antecedent.propagation_chain.PropagationChain`
        whose declared joint outputs join them under the chain's declared dependence (see
        :func:`chain_law`). The :class:`~antecedent.decision_evidence.JointEvidence` of one
        experiment is read as its arm means (:func:`experiment_means`); name each action's
        inputs with :func:`experiment_quantity`. Means answer only a value affine in the
        attributes; a kinked or otherwise nonlinear term refuses.

        A mean source answers only the exact route (an affine value and an expectation
        criterion); anything else refuses with ``preference_evaluation.joint_draws_required``.
        ``measure`` weighs trade-off values for :meth:`Policy.weighted` and reports shares; it
        is refused when the accepted preferences already carry one. ``tie_tolerance``,
        ``vertex_cap``, ``max_boxes`` and ``min_width_fraction`` tune the routes. The decision
        reads the execution's cancellation token at its work boundaries; a cancelled run is
        refused with ``cancelled_no_claim``.
        """
        execution = resolve_execution(execution)
        resolved = self._source(
            source,
            program=program,
            outcome_units=outcome_units,
            dose_units=dose_units,
            population=population,
        )
        measure_wire = self._measure(measure)
        options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
        contract_json = json.dumps(self._wire())
        measure_json = None if measure_wire is None else json.dumps(measure_wire)
        options_json = json.dumps(options)
        if isinstance(resolved, Structures):
            claims = [c._wire() for c in resolved.claims]
            sources = [None if c.source is None else c.source._native for c in resolved.claims]
            body, refusal = _decide_structures(
                contract_json,
                measure_json,
                options_json,
                resolved.kind,
                json.dumps(claims),
                sources,
                execution.cancel,
            )
        elif isinstance(resolved, JointDistributionArtifact):
            body, refusal = _decide_joint(
                contract_json, measure_json, options_json, resolved._native, execution.cancel
            )
        else:
            body, refusal = _decide_means(
                contract_json,
                measure_json,
                options_json,
                json.dumps(_mean_wire(resolved)),
                execution.cancel,
            )
        _raise(refusal)
        assert body is not None
        return PreferenceDecision(self, resolved, json.loads(body), measure_wire, options)


class CompiledDecision(ScientificResult):
    """A decision across preferences compiled with the analysis that estimates it.

    :meth:`inspect` reads what the compile states before anything is fitted: the bound
    contract identity, each quantity the actions read, the source representation the contract
    needs, the carrier the plan demands, every structure with its identification and weight,
    the unidentified mass, the declared joint law of separately estimated outcomes, and every
    compiled plan (whose ``evidence.decision.preference`` lists the quantities, the trade-off
    parameters with their ranges and the bound contract). :meth:`execute` runs each plan once
    and returns the :class:`PreferenceDecision`.
    """

    def __init__(self, contract: Contract, native: Any) -> None:
        self._contract = contract
        self._native = native

    @property
    def contract(self) -> Contract:
        return self._contract

    @property
    def bound_identity(self) -> str:
        """Identity of the contract bound to the analysis's causal contracts."""
        return str(self._native.bound_identity())

    def inspect(self) -> Mapping[str, Any]:
        """What the compile states, read without fitting anything."""
        return MappingProxyType(json.loads(self._native.describe()))

    @property
    def plans(self) -> tuple[Mapping[str, Any], ...]:
        """Every compiled plan, each with the decision in its evidence layer."""
        return tuple(
            MappingProxyType(plan) for plan in json.loads(self._native.describe())["plans"]
        )

    @property
    def unidentified_mass(self) -> float:
        """Share of the structural mass under which the quantities are not identified."""
        return float(self.inspect()["unidentified_mass"])

    def _summary_fields(self) -> dict[str, Any]:
        described = self.inspect()
        return {
            "bound_identity": described["bound_identity"],
            "representation": described["demand"]["representation"],
            "structures": len(described["structures"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self._native.describe()))

    @outcome_operation
    def execute(
        self,
        measure: Measure | None = None,
        execution: Execution | None = None,
        *,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
    ) -> PreferenceDecision:
        """Run every compiled plan once and decide across the admissible trade-offs.

        ``measure`` and the tuning options are those of :meth:`Contract.evaluate`. The decision
        reads the execution's cancellation token at its work boundaries.
        """
        from .transport._controls import call_native

        measure_wire = self._contract._measure(measure)
        options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
        body, refusal, law = call_native(
            execution,
            self._native.execute,
            None if measure_wire is None else json.dumps(measure_wire),
            json.dumps(options),
            seed="seed",
        )
        _raise(refusal)
        assert body is not None
        source = self if law is None else JointDistributionArtifact._from_native(law)
        return PreferenceDecision(self._contract, source, json.loads(body), measure_wire, options)

    def __repr__(self) -> str:
        return f"<CompiledDecision {self.bound_identity[:12]}>"


def joint_law(*laws: JointDistributionArtifact) -> JointDistributionArtifact:
    """One joint law from laws whose rows already pair.

    Rows pair when every law has joint alignment and the same source, provider, generator,
    snapshot, causal contract, number of draws, meaning, weights, calibration and trust: row
    ``i`` of each is the same draw. Laws that were sampled separately do not pair and refuse
    (``joint_law_required``); join those under a declared dependence with a propagation chain
    and :func:`chain_law`. Nothing is resampled or re-estimated.
    """
    laws = tuple(require_success(law) for law in laws)
    if not laws or any(not isinstance(law, JointDistributionArtifact) for law in laws):
        raise CausalTypeError("joint_law takes one or more JointDistributionArtifact values")
    native, refusal = _joint_law_of([law._native for law in laws])
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def chain_law(chain: Any) -> JointDistributionArtifact:
    """The joint output law of a sealed propagation chain, as aligned joint draws.

    The chain's record is resealed (every stage recomputed under its declared dependence) and
    its declared ``joint_outputs`` become the columns, so separately fitted results enter one
    decision with the dependence the chain states, not an assumed independence. A chain read
    neither as a posterior nor as a predictive law, or without joint outputs, refuses.
    """
    from .propagation_chain import PropagationChain

    chain = require_success(chain)
    if not isinstance(chain, PropagationChain):
        raise CausalTypeError(f"chain_law takes a PropagationChain, not {type(chain).__name__}")
    native, refusal = _chain_joint_law(chain._record, None)
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def _evidence_wire(evidence: Any) -> str:
    from .decision_evidence import JointEvidence

    evidence = require_success(evidence)
    if not isinstance(evidence, JointEvidence):
        raise CausalTypeError(
            f"an experiment is a decision_evidence.JointEvidence, not {type(evidence).__name__}"
        )
    return json.dumps(dict(evidence.wire))


def experiment_quantity(
    evidence: Any,
    outcome: str,
    arm: Literal["treated", "control"],
    *,
    execution: Any = None,
) -> ScientificQuantity:
    """The coordinate of one outcome of an experiment in one arm, for an :class:`Action` input.

    ``evidence`` is the :class:`~antecedent.decision_evidence.JointEvidence` of one experiment;
    ``outcome`` names one of its outcome components and ``arm`` is ``treated`` or ``control``.
    The coordinate is the arm mean (functional ``mean``) in the experiment's population,
    regime and horizon, as the experiment's mean source (:func:`experiment_means`) carries it.
    """
    from ._panel_execution import run_panel

    if arm not in ("treated", "control"):
        raise CausalValueError("arm is 'treated' or 'control'")
    body, refusal = run_panel(
        execution, _experiment_quantity, _evidence_wire(evidence), outcome, arm == "treated"
    )
    _raise(refusal)
    assert body is not None
    return ScientificQuantity._from_wire(json.loads(body))


def experiment_means(evidence: Any, *, execution: Any = None) -> MeanSource:
    """The arm means of every outcome of one experiment, as one mean source.

    The experiment is rebuilt and checked against its compiled source first. Arm means are
    point estimates read as means only under exposure declared free of between-unit
    interference; any other exposure refuses. A mean source carries no joint law, so it answers
    the exact route (a value affine in the attributes, an expectation criterion) only.
    """
    from ._panel_execution import run_panel

    wire = _evidence_wire(evidence)
    body, refusal = run_panel(execution, _experiment_source, wire)
    _raise(refusal)
    assert body is not None
    fields_ = json.loads(body)
    return MeanSource(
        coordinates=tuple(ScientificQuantity._from_wire(q) for q in fields_["coordinates"]),
        means=tuple(fields_["means"]),
        provider_id=fields_["provider_id"],
        snapshot_id=fields_["snapshot_id"],
        causal_contract_id=fields_["causal_contract_id"],
        rng_id=fields_["rng_id"],
        evidence=(json.loads(wire)["evidence_id"],),
    )


# --------------------------------------------------------------------------- result


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One action across the admissible trade-offs.

    Ranges are over declared assumptions, never confidence intervals. On the exact route
    ``value_range`` is exact; on the certified route ``value_range``, ``worst_case_range`` and
    ``worst_case_regret_range`` are enclosures and ``best_fraction`` is the volume fraction of
    the set on which the action is certified best. ``measure_share`` is a measure's weight, not
    a probability unless the measure is one.
    """

    id: str
    admissible: bool
    exclusions: tuple[ConstraintExclusion, ...]
    value_range: tuple[float, float] | None
    necessarily_best: bool | None
    uniquely_best: bool | None
    dominated_by: tuple[str, ...]
    possibly_best: bool | None
    worst_case_regret: float | None
    measure_share: float | None
    measure_value: float | None
    best_fraction: float | None
    worst_case_range: tuple[float, float] | None
    worst_case_regret_range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class SwitchThreshold:
    """Where the order of a pair changes, when it depends on a single trade-off."""

    parameter: str
    value: float
    first_leads_above: bool


@dataclass(frozen=True, slots=True)
class Switch:
    """A pair of actions whose order changes inside the admissible set.

    ``favours_first_at`` and ``favours_second_at`` are admissible trade-off values on each side;
    ``threshold`` is the value where the order changes when one parameter decides it.
    """

    first: str
    second: str
    margin_range: tuple[float, float]
    favours_first_at: Mapping[str, float]
    favours_second_at: Mapping[str, float]
    threshold: SwitchThreshold | None


@dataclass(frozen=True, slots=True)
class Region:
    """One box of trade-off values the certified route kept: ``best`` when one action is proven
    best everywhere in it, otherwise the ``candidates`` still possibly best."""

    lower: Mapping[str, float]
    upper: Mapping[str, float]
    best: str | None
    candidates: tuple[str, ...]
    touches_relation_boundary: bool


@dataclass(frozen=True, slots=True)
class StructureOutcome:
    """What became of one causal structure: ``evaluated`` (with its own verdict),
    ``unidentified`` or ``unevaluated`` (with the reason)."""

    id: str
    probability: float | None
    status: Literal["evaluated", "unidentified", "unevaluated"]
    reason: str | None
    kind: str | None
    selected: str | None


def _pair(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


def _verdict_action(verdict: Mapping[str, Any]) -> str | None:
    action = verdict.get("action")
    return None if action is None else str(action)


_STATEMENTS = {
    "necessarily_best": "Action {action} is preferred for every admissible value of the declared trade-offs.",
    "never_beaten": "Action {action} is never beaten over the declared trade-offs; it ties {ties} somewhere.",
    "preference_dependent": "The preferred action depends on the declared trade-offs: {actions} are each preferred somewhere.",
    "worst_case_choice": "Action {action} has the best worst-case value over the declared trade-offs.",
    "minimax_regret_choice": "Action {action} has the smallest worst-case regret over the declared trade-offs.",
    "measure_choice": "Action {action} has the best value weighted by the supplied measure.",
    "tied": "Actions {actions} tie across the declared trade-offs.",
    "undetermined": "The certified subdivision does not settle the choice among {actions}.",
    "report_only": "No action is chosen: the answer across the declared trade-offs is reported only.",
    "no_admissible_action": "Every action is excluded by a hard constraint.",
    "structures_disagree": "The evaluated structures give different verdicts, so no action is preferred across structures.",
    "insufficient_science": "No action is chosen across structures: {reason}.",
}


class PreferenceDecision(ScientificResult):
    """The decision across the admissible trade-offs (and structures, when supplied).

    ``kind`` is the verdict in snake case: ``necessarily_best`` (one action strictly ahead
    everywhere), ``never_beaten`` (never behind, tying some rival somewhere),
    ``preference_dependent`` (the choice changes inside the set; see :attr:`switches`),
    ``worst_case_choice``, ``minimax_regret_choice``, ``measure_choice``, ``tied``,
    ``undetermined`` (certified route only), ``report_only``, ``no_admissible_action``, and
    across structures ``structures_disagree`` or ``insufficient_science``. Every range in it is
    over declared assumptions, not a confidence interval.
    """

    def __init__(
        self,
        contract: Contract,
        source: Any,
        body: Mapping[str, Any],
        measure: Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> None:
        self._contract = contract
        self._source = source
        self._body = body
        self._measure_wire = measure
        self._options = dict(options)
        self._report = {"kind": self.kind, "label": self.route, "status": self.selected}

    # -- verdict -----------------------------------------------------------------

    @property
    def _table(self) -> Mapping[str, Any] | None:
        """The single-source table: the body itself, or the combination across structures."""
        if "structural_verdict" in self._body:
            combined = self._body.get("combined")
            return combined if isinstance(combined, Mapping) else None
        return self._body

    @property
    def _verdict(self) -> Mapping[str, Any]:
        structural = self._body.get("structural_verdict")
        if structural is None:
            return dict(self._body["verdict"])
        if structural["kind"] == "combined":
            return dict(structural["verdict"])
        return dict(structural)

    @property
    def kind(self) -> str:
        return str(self._verdict["kind"])

    @property
    def selected(self) -> str | None:
        """The action the verdict chooses, or ``None`` when it chooses none."""
        return _verdict_action(self._verdict)

    @property
    def answer(self) -> Answer:
        """A structured answer whose ``detail`` is the verdict kind (``"necessarily_best"``, ...)."""
        return Answer(
            "structured",
            detail=self.kind,
            structured={"selected": self.selected, "contract_identity": self.contract_identity},
        )

    @property
    def statement(self) -> str:
        """The verdict in plain words."""
        stated = self._body.get("statement")
        if isinstance(stated, str) and stated:
            return stated
        verdict = self._verdict
        actions = ", ".join(verdict.get("actions") or verdict.get("candidates") or ())
        template = _STATEMENTS.get(self.kind, "Verdict {kind}.")
        return template.format(
            action=verdict.get("action"),
            actions=actions,
            ties=", ".join(verdict.get("ties_with") or ()),
            reason=verdict.get("reason"),
            kind=self.kind,
        )

    def explain(self) -> str:
        """The statement, followed by the route and the assumptions it stands on."""
        text_ = self.statement
        text_ += f" Answered by the {self.route} route"
        if self.undecided_fraction:
            text_ += f", with a volume fraction of {self.undecided_fraction:.3g} left undecided"
        text_ += "."
        if self.assumptions:
            text_ += " It stands on: " + ", ".join(self.assumptions) + "."
        return text_

    @property
    def route(self) -> Literal["exact", "certified"]:
        return self._body["route"]

    @property
    def route_reason(self) -> str | None:
        """The stable label naming why this route answered, when known."""
        reason = self._body.get("route_reason")
        return None if reason is None else str(reason)

    @property
    def contract_identity(self) -> str:
        return str(self._body["contract_identity"])

    @property
    def policy(self) -> str:
        return str(self._body["policy"])

    # -- tables ------------------------------------------------------------------

    @property
    def parameters(self) -> tuple[str, ...]:
        table = self._table
        return () if table is None else tuple(table["parameter_ids"])

    @property
    def parameter_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Each parameter's range over the admissible set (declared ranges cut by relations)."""
        table = self._table
        if table is None or table.get("parameter_ranges") is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                name: (float(lo), float(hi))
                for name, (lo, hi) in zip(self.parameters, table["parameter_ranges"], strict=False)
            }
        )

    def _point(self, values: Sequence[float]) -> Mapping[str, float]:
        return MappingProxyType(
            {name: float(v) for name, v in zip(self.parameters, values, strict=False)}
        )

    @property
    def actions(self) -> tuple[ActionRecord, ...]:
        table = self._table
        if table is None:
            return ()
        return tuple(
            ActionRecord(
                id=a["id"],
                admissible=bool(a["admissible"]),
                exclusions=tuple(
                    ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
                    for e in a["exclusions"]
                ),
                value_range=_pair(a.get("value_range", a.get("value_bounds"))),
                necessarily_best=a.get("necessarily_best"),
                uniquely_best=a.get("uniquely_best"),
                dominated_by=tuple(a.get("dominated_by") or ()),
                possibly_best=a.get("possibly_best"),
                worst_case_regret=a.get("worst_case_regret"),
                measure_share=a.get("measure_share"),
                measure_value=a.get("measure_value"),
                best_fraction=a.get("best_fraction"),
                worst_case_range=_pair(a.get("worst_case_bounds")),
                worst_case_regret_range=_pair(a.get("worst_case_regret_bounds")),
            )
            for a in table["actions"]
        )

    @property
    def switches(self) -> tuple[Switch, ...]:
        """Pairs whose order changes inside the set (exact route), each with a threshold when
        one parameter decides it."""
        table = self._table
        if table is None:
            return ()
        out = []
        for s in table.get("switches") or ():
            threshold = s.get("threshold")
            out.append(
                Switch(
                    first=s["first"],
                    second=s["second"],
                    margin_range=(float(s["margin_range"][0]), float(s["margin_range"][1])),
                    favours_first_at=self._point(s["favours_first_at"]),
                    favours_second_at=self._point(s["favours_second_at"]),
                    threshold=None
                    if threshold is None
                    else SwitchThreshold(
                        threshold["parameter_id"],
                        float(threshold["value"]),
                        bool(threshold["first_leads_above"]),
                    ),
                )
            )
        return tuple(out)

    @property
    def undecided_fraction(self) -> float | None:
        """Volume fraction of the set the certified route left undecided; ``None`` on the exact
        route, which decides every point."""
        table = self._table
        value = None if table is None else table.get("undecided_fraction")
        return None if value is None else float(value)

    @property
    def regions(self) -> tuple[Region, ...]:
        """The boxes the certified route kept (empty on the exact route)."""
        table = self._table
        if table is None:
            return ()
        out = []
        for r in table.get("regions") or ():
            status = r["status"]
            out.append(
                Region(
                    lower=self._point(r["lower"]),
                    upper=self._point(r["upper"]),
                    best=status.get("action") if status["kind"] == "best" else None,
                    candidates=tuple(status.get("candidates") or ()),
                    touches_relation_boundary=bool(r["touches_relation_boundary"]),
                )
            )
        return tuple(out)

    @property
    def structures(self) -> tuple[StructureOutcome, ...]:
        """Per-structure outcomes when the decision was across structures (else empty)."""
        out = []
        for atom in self._body.get("structures") or ():
            result = atom.get("result")
            verdict = None if result is None else result["verdict"]
            out.append(
                StructureOutcome(
                    id=atom["id"],
                    probability=atom.get("probability"),
                    status=atom["status"],
                    reason=atom.get("reason"),
                    kind=None if verdict is None else verdict["kind"],
                    selected=None if verdict is None else _verdict_action(verdict),
                )
            )
        return tuple(out)

    def _mass(self, name: str) -> float | None:
        value = self._body.get(name)
        return None if value is None else float(value)

    @property
    def unidentified_mass(self) -> float | None:
        return self._mass("unidentified_mass")

    @property
    def unevaluated_mass(self) -> float | None:
        return self._mass("unevaluated_mass")

    @property
    def measure(self) -> Measure | None:
        """The measure the decision was read with, if any."""
        wire = self._measure_wire or self._body.get("measure")
        if wire is None:
            return None
        ids = [p["id"] for p in self._contract._preferences()["model"]["parameters"]]
        return Measure._from_wire(wire, ids)

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Parameter assumptions, relation origins and the measure origin the verdict stands on."""
        table = self._table
        found = list((table or {}).get("assumptions") or ())
        for atom in self._body.get("structures") or ():
            result = atom.get("result")
            if result is not None:
                found.extend(result.get("assumptions") or ())
        return tuple(sorted(set(found)))

    @property
    def draw_share_best_everywhere(self) -> float | None:
        """Weight of the draws in which the necessarily-best action also has the highest
        per-draw value at every vertex of the admissible set.

        A point diagnostic of the supplied law, not a calibrated probability that the action
        is best. ``None`` for a mean source, when no action is necessarily best, when vertex
        enumeration exceeds its cap, on the certified route and for structural combinations.
        """
        table = self._table
        value = None if table is None else table.get("draw_share_best_everywhere")
        return None if value is None else float(value)

    @property
    def n_draws(self) -> int:
        table = self._table
        return 0 if table is None else int(table.get("n_draws", 0))

    @property
    def contract(self) -> Contract:
        return self._contract

    @property
    def source(self) -> Any:
        return self._source

    def to_dict(self) -> dict[str, Any]:
        """The native result, with the verdict kind, the selected action and the statement."""
        body = json.loads(json.dumps(self._body))
        body.update({"kind": self.kind, "selected": self.selected, "statement": self.statement})
        return dict(body)

    @artifact_export
    def export(self, *, artifact_id: str = "preference-decision") -> ExportedArtifact:
        """The decision bound to its contract identity and the digest of its draws, replayable.

        Only a decision on aligned joint draws is exported; a mean-source or structural
        decision refuses with ``route_not_supported``.
        """
        if not isinstance(self._source, JointDistributionArtifact):
            raise PreferenceRefusal(
                {
                    "code": "route_not_supported",
                    "stage": "export",
                    "detail": _reason_details.decision_evaluation.MEAN_SOURCE_NOT_REPLAYABLE,
                    "offending": None,
                    "expected": "joint_draws",
                    "supplied": type(self._source).__name__,
                    "remedy": "export a decision across preferences evaluated on aligned joint draws",
                }
            )
        data, refusal = _export_decision(
            json.dumps(self._contract._wire()),
            None if self._measure_wire is None else json.dumps(self._measure_wire),
            json.dumps(self._options),
            self._source._native,
            artifact_id,
        )
        _raise(refusal)
        assert data is not None
        return artifact_from_receipt(self, bytes(data))

    def __repr__(self) -> str:
        return f"<PreferenceDecision {self.kind} {self.selected!r} ({self.route})>"


@outcome_operation
@artifact_consumer(
    proof=(
        "Independent native replay loads the record under the consumer's contract identity and "
        "draw digest, recomputes the decision and compares every stored field."
    ),
    replayable=True,
    identity=("contract", "source"),
    input="data",
)
def replay(
    data: bytes,
    *,
    contract: Contract,
    source: JointDistributionArtifact,
    execution: Execution | None = None,
) -> PreferenceDecision:
    """Recompute a stored decision across preferences from the consumer's own inputs.

    The contract and draws are the consumer's: a record computed under another contract, from
    other draws, or with edited numbers refuses. The stored measure and options are reused. The
    recomputation reads the execution's cancellation token at its work boundaries; a cancelled
    replay is refused with ``cancelled_no_claim``.
    """
    execution = resolve_execution(execution)
    data = container_bytes(data)
    source = require_success(source)
    if not isinstance(contract, Contract):
        raise CausalTypeError(f"replay needs a preferences.Contract, not {type(contract).__name__}")
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError(
            "replay needs the JointDistributionArtifact of aligned draws the decision was "
            f"computed from, not {type(source).__name__}"
        )
    body, refusal = _replay_decision(
        data, json.dumps(contract._wire()), source._native, execution.cancel
    )
    _raise(refusal)
    assert body is not None
    decoded = json.loads(body)
    return PreferenceDecision(contract, source, decoded, decoded.get("measure"), {})


__all__ = [
    "Action",
    "ActionRecord",
    "Attribute",
    "CompiledDecision",
    "Contract",
    "Criterion",
    "Constraint",
    "Curve",
    "Expr",
    "Measure",
    "Parameter",
    "Policy",
    "PreferenceDecision",
    "PreferenceRefusal",
    "Region",
    "Relation",
    "StructureOutcome",
    "Structures",
    "Switch",
    "SwitchThreshold",
    "ValueModel",
    "attribute",
    "both",
    "chain_law",
    "constraint",
    "excess",
    "exp",
    "experiment_means",
    "experiment_quantity",
    "fixed",
    "interaction",
    "joint_law",
    "ln",
    "maximum",
    "minimum",
    "relation",
    "replay",
    "segment",
    "shortfall",
    "strength",
]
