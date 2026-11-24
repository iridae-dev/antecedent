"""Declarations of a value over several outcome attributes: attributes, trade-offs and the
expressions over them, relations between trade-offs, the value, trade-off policies, measures
over the trade-offs and the causal structures a decision is taken across. Public names are
exported from :mod:`antecedent.decision`."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

from ._native import preference_model_describe as _describe_model
from ._refusal import raise_refusal
from ._validate import real, text
from .errors import CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from .decision_robust import Claim

ActionKind = Literal["intervention", "policy", "regime", "study", "external"]
StructuralPolicyKind = Literal[
    "require_invariant_best_action", "maximin", "bayes_over_structures", "report_only"
]
PolicyKind = Literal[
    "require_invariant_best_action", "maximin", "minimax_regret", "weighted", "report_only"
]
StructureKind = Literal["point", "graph_dependent", "weighted_graph_posterior", "finite_scenarios"]
Sense = Literal["at_least", "at_most", "equal"]

_SENSES: dict[str, Sense] = {">=": "at_least", "<=": "at_most", "==": "equal"}
_SYMBOLS = {"at_least": ">=", "at_most": "<=", "equal": "=="}


def _raise(payload: str | None) -> None:
    """Raise the native refusal in ``payload`` as a :class:`~antecedent.decision.PreferenceRefusal`."""
    if payload is None:
        return
    from .decision import PreferenceRefusal

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

    def _expr(self) -> ValueExpr:
        raise NotImplementedError

    def __add__(self, other: Operand) -> ValueExpr:
        return _binary("add", self, other)

    def __radd__(self, other: Operand) -> ValueExpr:
        return _binary("add", other, self)

    def __sub__(self, other: Operand) -> ValueExpr:
        return _binary("sub", self, other)

    def __rsub__(self, other: Operand) -> ValueExpr:
        return _binary("sub", other, self)

    def __mul__(self, other: Operand) -> ValueExpr:
        return _binary("mul", self, other)

    def __rmul__(self, other: Operand) -> ValueExpr:
        return _binary("mul", other, self)

    def __truediv__(self, other: float) -> ValueExpr:
        divisor = _number(other, "a divisor")
        if divisor == 0.0:
            raise CausalValueError("an expression is divided only by a nonzero number")
        return _binary("mul", self, 1.0 / divisor)

    def __neg__(self) -> ValueExpr:
        inner = self._expr()
        return ValueExpr({"neg": inner._node}, inner._leaves)

    def __pos__(self) -> ValueExpr:
        return self._expr()

    def __ge__(self, other: Operand) -> Relation:
        return _relation_of(self, other, "at_least")

    def __le__(self, other: Operand) -> Relation:
        return _relation_of(self, other, "at_most")


Operand: TypeAlias = "_Algebra | float"


def _coerce(value: Any) -> ValueExpr:
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
    return ValueExpr({"const": number}, ())


def _binary(op: str, left: Any, right: Any) -> ValueExpr:
    a, b = _coerce(left), _coerce(right)
    return ValueExpr({op: [a._node, b._node]}, _merge(a._leaves, b._leaves))


@dataclass(frozen=True, slots=True, eq=False)
class ValueExpr(_Algebra):
    """A closed expression over attributes, trade-off parameters and numbers.

    Build it with ``+``, ``-``, ``*``, ``/`` by a number, unary ``-``, :func:`exp`, :func:`ln`,
    :func:`maximum`, :func:`minimum` and the shapes. Comparing two parameter expressions with
    ``>=`` or ``<=`` gives a :class:`Relation`, not a boolean.
    """

    _node: Any
    _leaves: tuple[Any, ...] = ()

    def _expr(self) -> ValueExpr:
        return self

    def __bool__(self) -> bool:
        raise CausalTypeError("an expression has no truth value")

    def __repr__(self) -> str:
        return f"ValueExpr({json.dumps(self._node, sort_keys=True)})"


@dataclass(frozen=True, slots=True)
class Attribute(_Algebra):
    """One outcome attribute the value reads (for example ``harm`` in ``events``)."""

    id: str
    units: str

    def __post_init__(self) -> None:
        text(self.id, "Attribute.id", non_empty=True)
        text(self.units, "Attribute.units", non_empty=True)

    def _expr(self) -> ValueExpr:
        return ValueExpr({"attribute": self.id}, (self,))


@dataclass(frozen=True, slots=True)
class Tradeoff(_Algebra):
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
        text(self.id, "Tradeoff.id", non_empty=True)
        text(self.units, "Tradeoff.units", non_empty=True)
        text(self.assumption, "Tradeoff.assumption", non_empty=True)
        object.__setattr__(self, "lower", _number(self.lower, "Tradeoff.lower"))
        object.__setattr__(self, "upper", _number(self.upper, "Tradeoff.upper"))

    @property
    def is_fixed(self) -> bool:
        """Whether the range is a single value."""
        return self.lower == self.upper

    def _expr(self) -> ValueExpr:
        return ValueExpr({"parameter": self.id}, (self,))

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


def tradeoff(
    id: str,
    *,
    units: str,
    lower: float | None = None,
    upper: float | None = None,
    value: float | None = None,
    assumption: str | None = None,
) -> Tradeoff:
    """A trade-off whose admissible values are ``[lower, upper]``.

    The range is a declared assumption, not a probability: every value in it is treated as
    admissible and none is more likely than another. ``units`` is utility per unit of what the
    trade-off multiplies (``qaly/event``) and ``assumption`` names the judgment the range stands
    on. ``value=`` fixes the trade-off at one value as a named trade-off (its assumption is
    ``declared_value`` unless named). A plain number in the expression is a fixed trade-off with
    no name: a value with only plain numbers is an ordinary contract.
    """
    if value is not None:
        if lower is not None or upper is not None:
            raise CausalValueError(
                "a trade-off takes lower= and upper= (a range) or value= (one value), not both"
            )
        return Tradeoff(
            id, units, value, value, "declared_value" if assumption is None else assumption
        )
    if lower is None or upper is None:
        raise CausalValueError(
            "a trade-off takes lower= and upper= for a range of admissible values, or value= "
            "for one value"
        )
    if assumption is None:
        raise CausalValueError(
            f"the range of trade-off {id!r} names the judgment it stands on: give assumption="
        )
    return Tradeoff(id, units, lower, upper, assumption)


def exp(value: Operand) -> ValueExpr:
    """Natural exponential."""
    inner = _coerce(value)
    return ValueExpr({"exp": inner._node}, inner._leaves)


def ln(value: Operand) -> ValueExpr:
    """Natural logarithm; the argument must stay positive wherever it is evaluated."""
    inner = _coerce(value)
    return ValueExpr({"ln": inner._node}, inner._leaves)


def maximum(left: Operand, right: Operand) -> ValueExpr:
    """Pointwise maximum."""
    return _binary("max", left, right)


def minimum(left: Operand, right: Operand) -> ValueExpr:
    """Pointwise minimum."""
    return _binary("min", left, right)


def _attribute_arg(value: object, what: str) -> Attribute:
    if not isinstance(value, Attribute):
        raise CausalTypeError(f"{what} is an Attribute made by decision.attribute(...)")
    return value


def excess(of: Attribute, above: float) -> ValueExpr:
    """``max(x - above, 0)``: how far the attribute exceeds a threshold."""
    x = _attribute_arg(of, "excess(of=...)")
    return maximum(x - _number(above, "above"), 0.0)


def shortfall(of: Attribute, below: float) -> ValueExpr:
    """``max(below - x, 0)``: how far the attribute falls short of a threshold."""
    x = _attribute_arg(of, "shortfall(of=...)")
    return maximum(_number(below, "below") - x, 0.0)


def segment(of: Attribute, start: float, end: float) -> ValueExpr:
    """``clamp(x, start, end) - start``: the part of the attribute inside ``[start, end]``."""
    x = _attribute_arg(of, "segment(of=...)")
    lo, hi = _number(start, "start"), _number(end, "end")
    return minimum(maximum(x, lo), hi) - lo


def interaction(a: Attribute, b: Attribute) -> ValueExpr:
    """``x_a * x_b``."""
    return _attribute_arg(a, "interaction(a=...)") * _attribute_arg(b, "interaction(b=...)")


def both(a: Attribute, b: Attribute) -> ValueExpr:
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
    parameters: tuple[Tradeoff, ...] = field(default=(), compare=False, repr=False)

    def __bool__(self) -> bool:
        raise CausalTypeError(
            "a Relation is a declaration, not a truth value; pass it to Value(relations=...)"
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
    parameters = tuple(p for p in difference._leaves if isinstance(p, Tradeoff))
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
    :meth:`accelerating`) return relations to pass to :class:`Value`; they restrict the
    slopes, so they are judgments like the ranges.
    """

    of: Attribute
    breakpoints: tuple[float, ...]
    slopes: tuple[Tradeoff, ...]
    id: str | None = None

    def __post_init__(self) -> None:
        _attribute_arg(self.of, "Curve.of")
        points = tuple(_number(b, "a breakpoint") for b in self.breakpoints)
        slopes = tuple(self.slopes)
        if any(not isinstance(s, Tradeoff) for s in slopes):
            raise CausalTypeError("Curve.slopes are trade-offs made by decision.tradeoff(...)")
        if len(points) < 2 or len(slopes) + 1 != len(points):
            raise CausalValueError("a curve has at least two breakpoints and one slope per segment")
        if any(a >= b for a, b in zip(points, points[1:], strict=False)):
            raise CausalValueError("curve breakpoints strictly increase")
        object.__setattr__(self, "breakpoints", points)
        object.__setattr__(self, "slopes", slopes)

    @property
    def _prefix(self) -> str:
        return self.id or f"{self.of.id}.curve"

    def _expr(self) -> ValueExpr:
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
            raise CausalTypeError("Value.relations holds Relations (or groups of them)")
    return tuple(out)


class Value:
    """How the outcome attributes combine into one value, with its trade-offs.

    ``expr`` is an expression over attributes (:func:`attribute`), trade-offs (:func:`tradeoff`)
    and numbers; ``units`` names the utility it is in. The attributes and trade-offs are read
    from the expression in order of first appearance unless ``attributes=`` or ``tradeoffs=``
    fixes the order. ``relations`` restrict the trade-offs jointly. A value is a declaration of
    judgments, not an estimate.

    A value whose trade-offs are all plain numbers is the utility of an ordinary contract;
    one with a named trade-off is decided across every admissible value of its trade-offs.
    """

    __slots__ = ("_expr_value", "_units", "_relations", "_attributes", "_tradeoffs")

    def __init__(
        self,
        expr: Any,
        *,
        units: str,
        relations: Iterable[Any] = (),
        attributes: Sequence[Attribute] | None = None,
        tradeoffs: Sequence[Tradeoff] | None = None,
    ) -> None:
        value = _coerce(expr)
        text(units, f"{type(self).__name__} units", non_empty=True)
        flat = _flatten(relations)
        found_attributes = _declarations(Attribute, value._leaves, attributes)
        found_tradeoffs = _declarations(
            Tradeoff, (*value._leaves, *(p for r in flat for p in r.parameters)), tradeoffs
        )
        object.__setattr__(self, "_expr_value", value)
        object.__setattr__(self, "_units", units)
        object.__setattr__(self, "_relations", flat)
        object.__setattr__(self, "_attributes", found_attributes)
        object.__setattr__(self, "_tradeoffs", found_tradeoffs)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @property
    def expr(self) -> ValueExpr:
        """The value expression."""
        return self._expr_value

    @property
    def units(self) -> str:
        """The utility units the value is in."""
        return self._units

    @property
    def relations(self) -> tuple[Relation, ...]:
        """Linear relations between the trade-offs."""
        return self._relations

    @property
    def attributes(self) -> tuple[Attribute, ...]:
        """The attributes the value reads, in order."""
        return self._attributes

    @property
    def tradeoffs(self) -> tuple[Tradeoff, ...]:
        """The named trade-offs the value reads, in order."""
        return self._tradeoffs

    def _wire(self) -> dict[str, Any]:
        attributes = {a.id: k for k, a in enumerate(self._attributes)}
        parameters = {p.id: j for j, p in enumerate(self._tradeoffs)}
        return {
            "attributes": [{"id": a.id, "units": a.units} for a in self._attributes],
            "parameters": [p._wire() for p in self._tradeoffs],
            "relations": [r._wire(parameters) for r in self._relations],
            "value": _resolve(self._expr_value._node, attributes, parameters),
            "utility_units": self._units,
        }

    def _preference_wire(self) -> dict[str, Any]:
        """What a contract reads from its value: the model and no measure."""
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
        """Trade-offs the value is not affine in. Non-empty means the certified route answers."""
        return tuple(self._described()["shape_parameters"])

    def validate(self) -> Value:
        """Check the value natively; raise :class:`~antecedent.decision.PreferenceRefusal` when
        it is invalid."""
        self._described()
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._expr_value!r}, units={self._units!r}, "
            f"tradeoffs={tuple(t.id for t in self._tradeoffs)!r})"
        )


# --------------------------------------------------------------------------- policies


@dataclass(frozen=True, slots=True)
class TradeoffPolicy:
    """How a result across the admissible trade-offs may be consumed (``Contract(policy=...)``).

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
                f"TradeoffPolicy kind {self.kind!r} is not one of {get_args(PolicyKind)}; build "
                "one with TradeoffPolicy.invariant_best(), .maximin(), .minimax_regret(), "
                ".weighted() or .report_only()"
            )

    @classmethod
    def invariant_best(cls) -> TradeoffPolicy:
        return cls("require_invariant_best_action")

    @classmethod
    def maximin(cls) -> TradeoffPolicy:
        return cls("maximin")

    @classmethod
    def minimax_regret(cls) -> TradeoffPolicy:
        return cls("minimax_regret")

    @classmethod
    def weighted(cls) -> TradeoffPolicy:
        return cls("weighted")

    @classmethod
    def report_only(cls) -> TradeoffPolicy:
        return cls("report_only")


@dataclass(frozen=True, slots=True)
class StructuralPolicy:
    """How the results under several causal structures combine (see :class:`Structures`).

    * :meth:`invariant_best` requires every structure to be evaluated and their verdicts to
      agree; otherwise the result is ``structures_disagree``;
    * :meth:`maximin` scores each action by its worst value over structures at each trade-off,
      then applies the :class:`TradeoffPolicy`;
    * :meth:`bayes_over_structures` weighs the per-structure values by the supplied structure
      probabilities, never renormalized;
    * :meth:`report_only` reports each structure's result with no combination.

    ``Contract(structural_policy=...)`` takes the kind as a string and also accepts this class.
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


def _constraint_node(node: Any, attributes: Mapping[str, int]) -> Any:
    ((op, arg),) = node.items()
    if op == "attribute":
        if arg not in attributes:
            raise CausalValueError(f"a constraint reads attribute {arg!r}, which the value lacks")
        return {"input": attributes[arg]}
    if op in ("const", "input"):
        return node
    if op in ("parameter", "exp", "ln"):
        raise CausalValueError(f"a hard constraint cannot read {op}")
    if isinstance(arg, list):
        return {op: [_constraint_node(item, attributes) for item in arg]}
    return {op: _constraint_node(arg, attributes)}


def _constraint_node_of(expr: Any) -> Any:
    """The expression node of a hard constraint over attributes; a trade-off is refused, so
    whether a constraint excludes an action never depends on the trade-offs."""
    inner = _coerce(expr)
    if any(isinstance(leaf, Tradeoff) for leaf in inner._leaves):
        raise CausalValueError("a hard constraint reads attributes only, never a trade-off")
    return inner._node


__all__ = [
    "ActionKind",
    "Attribute",
    "Curve",
    "Measure",
    "Relation",
    "StructuralPolicy",
    "StructuralPolicyKind",
    "Structures",
    "Tradeoff",
    "TradeoffPolicy",
    "Value",
    "ValueExpr",
    "attribute",
    "both",
    "excess",
    "exp",
    "interaction",
    "ln",
    "maximum",
    "minimum",
    "relation",
    "segment",
    "shortfall",
    "tradeoff",
]
