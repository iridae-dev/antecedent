"""Declarations of a decision across trade-off ranges: expressions over attributes and
trade-off parameters, relations, the value model, actions, policies, measures, structures
and hard constraints. Public names are exported from :mod:`antecedent.preferences`."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

from ._native import preference_model_describe as _describe_model
from ._refusal import raise_refusal
from ._validate import real, text
from .decision import ActionKind, Constraint
from .decision import Expr as _DecisionExpr
from .decision import StructuralPolicy as StructuralPolicyKind
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .joint_distribution import ScientificQuantity

if TYPE_CHECKING:
    from .decision_robust import Claim

PolicyKind = Literal[
    "require_invariant_best_action", "maximin", "minimax_regret", "weighted", "report_only"
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

    kind: PolicyKind

    def __post_init__(self) -> None:
        if self.kind not in get_args(PolicyKind):
            raise CausalValueError(
                f"Policy kind {self.kind!r} is not one of {get_args(PolicyKind)}; build one with "
                "Policy.invariant_best(), .maximin(), .minimax_regret(), .weighted() or "
                ".report_only()"
            )

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


@dataclass(frozen=True, slots=True)
class StructuralPolicy:
    """How the results under several causal structures combine (see :class:`Structures`).

    * :meth:`invariant_best` requires every structure to be evaluated and their verdicts to
      agree; otherwise the result is ``structures_disagree``;
    * :meth:`maximin` scores each action by its worst value over structures at each trade-off,
      then applies the preference :class:`Policy`;
    * :meth:`bayes_over_structures` weighs the per-structure values by the supplied structure
      probabilities, never renormalized;
    * :meth:`report_only` reports each structure's result with no combination.

    ``Contract(structural_policy=...)`` also takes the kind as a string.
    """

    kind: StructuralPolicyKind

    def __post_init__(self) -> None:
        if self.kind not in get_args(StructuralPolicyKind):
            raise CausalValueError(
                f"StructuralPolicy kind {self.kind!r} is not one of "
                f"{get_args(StructuralPolicyKind)}"
            )

    @classmethod
    def invariant_best(cls) -> StructuralPolicy:
        return cls("require_invariant_best_action")

    @classmethod
    def maximin(cls) -> StructuralPolicy:
        return cls("maximin")

    @classmethod
    def bayes_over_structures(cls) -> StructuralPolicy:
        return cls("bayes_over_structures")

    @classmethod
    def report_only(cls) -> StructuralPolicy:
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
    unidentified, or unevaluated with a reason). Build the set by how the claims are read, as
    in :mod:`antecedent.decision_robust`: :meth:`point` (one structure),
    :meth:`graph_dependent` (one answer per graph, no probabilities), :meth:`weighted_atoms`
    (graph-posterior atoms whose probabilities weigh them) or :meth:`finite_scenarios`
    (supplied scenarios, always an envelope). ``Structures(kind, claims)`` with the kind as a
    string is the same. Unidentified and unevaluated mass is reported, never renormalized away.
    """

    kind: StructureKind
    claims: tuple[Claim, ...]

    def __post_init__(self) -> None:
        if self.kind not in get_args(StructureKind):
            raise CausalValueError(f"Structures.kind is one of {get_args(StructureKind)}")
        object.__setattr__(self, "claims", tuple(self.claims))

    @classmethod
    def point(cls, claim: Claim) -> Structures:
        """One structure: a single exact law or answer."""
        return cls("point", (claim,))

    @classmethod
    def graph_dependent(cls, claims: Iterable[Claim]) -> Structures:
        """One answer per graph, with no probability over the graphs."""
        return cls("graph_dependent", tuple(claims))

    @classmethod
    def weighted_atoms(cls, claims: Iterable[Claim]) -> Structures:
        """Graph-posterior atoms whose probabilities weigh the structures."""
        return cls("weighted_graph_posterior", tuple(claims))

    @classmethod
    def finite_scenarios(cls, claims: Iterable[Claim]) -> Structures:
        """A finite set of supplied scenarios, read as a structural envelope."""
        return cls("finite_scenarios", tuple(claims))


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
