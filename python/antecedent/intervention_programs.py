"""Closed-form intervention programs for fitted graphical causal models.

A program is a *transform* or an *assignment* of one variable by an expression:

* :func:`transform` runs after the node's own mechanism and replaces the value it generated, which
  the expression reads as :func:`generated`. The node keeps its incoming edges and its noise:
  ``price := max(10, 0.85 * price)`` is ``transform("price", maximum(10, 0.85 * generated()))``.
* :func:`assign` discards the mechanism: the node's value is an expression of other variables
  (a computed ``do``).

Expressions are built from :func:`var` (another variable's current value), :func:`lag` (its value
``k`` steps back, temporal routes only), :func:`generated`, numbers, arithmetic, comparisons,
``&`` / ``|`` / ``~`` on conditions, :func:`maximum`, :func:`minimum`, :func:`clip`, :func:`where`
and :func:`indicator`. Python never evaluates one: the program text goes to the library, which
validates it, compiles it once and runs the same executor on every route -- forward sampling
(:func:`sample_policy`), counterfactual execution on abduced noise
(:func:`counterfactual_policy`) and a temporal rollout that reads earlier steps
(:func:`sample_temporal_policy`).

A program has a versioned portable encoding (:func:`encode_policy_program`) and an identity
(:func:`policy_program_identity`) that moves with any change to an expression, a lag, a target
variable or the program kind. A decoded program is re-verified: an edited expression is an
identity mismatch, edited declared reads are refused, and a future encoding version is refused
before its body is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from . import _native
from .errors import CausalValueError

__all__ = [
    "Expr",
    "Program",
    "assign",
    "clip",
    "const",
    "counterfactual_policy",
    "decode_policy_program",
    "encode_policy_program",
    "generated",
    "indicator",
    "lag",
    "maximum",
    "minimum",
    "policy_program_identity",
    "sample_policy",
    "sample_temporal_policy",
    "transform",
    "var",
    "where",
]

Mode = Literal["transform", "assign"]


@dataclass(frozen=True, slots=True)
class Expr:
    """A node of an expression tree, in the library's wire form (variables by name)."""

    node: Any

    def _binary(self, op: str, other: Any, *, reflected: bool = False) -> Expr:
        lhs, rhs = (_coerce(other), self) if reflected else (self, _coerce(other))
        return Expr({"arith": {"op": op, "lhs": lhs.node, "rhs": rhs.node}})

    def _compare(self, op: str, other: Any) -> Expr:
        return Expr({"compare": {"op": op, "lhs": self.node, "rhs": _coerce(other).node}})

    def _logic(self, op: str, other: Expr) -> Expr:
        if not isinstance(other, Expr):
            raise CausalValueError("a condition combines only with another condition")
        return Expr({"logic": {"op": op, "lhs": self.node, "rhs": other.node}})

    def __add__(self, other: Any) -> Expr:
        return self._binary("add", other)

    def __radd__(self, other: Any) -> Expr:
        return self._binary("add", other, reflected=True)

    def __sub__(self, other: Any) -> Expr:
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> Expr:
        return self._binary("sub", other, reflected=True)

    def __mul__(self, other: Any) -> Expr:
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> Expr:
        return self._binary("mul", other, reflected=True)

    def __truediv__(self, other: Any) -> Expr:
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> Expr:
        return self._binary("div", other, reflected=True)

    def __neg__(self) -> Expr:
        return Expr({"neg": {"operand": self.node}})

    def __lt__(self, other: Any) -> Expr:
        return self._compare("lt", other)

    def __le__(self, other: Any) -> Expr:
        return self._compare("le", other)

    def __gt__(self, other: Any) -> Expr:
        return self._compare("gt", other)

    def __ge__(self, other: Any) -> Expr:
        return self._compare("ge", other)

    def equals(self, other: Any) -> Expr:
        """Condition ``self == other`` (``==`` itself stays Python equality of trees)."""
        return self._compare("eq", other)

    def not_equals(self, other: Any) -> Expr:
        """Condition ``self != other``."""
        return self._compare("ne", other)

    def __and__(self, other: Expr) -> Expr:
        return self._logic("and", other)

    def __or__(self, other: Expr) -> Expr:
        return self._logic("or", other)

    def __invert__(self) -> Expr:
        return Expr({"not": {"operand": self.node}})


def _coerce(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise CausalValueError(
            f"expected an expression or a number, got {type(value).__name__}; "
            "convert a condition with indicator()"
        )
    return const(float(value))


def const(value: float) -> Expr:
    """A numeric constant (finite, checked by the library)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise CausalValueError(f"a constant must be a number, got {type(value).__name__}")
    return Expr({"const": {"value": float(value)}})


def var(name: str) -> Expr:
    """The current-draw value of variable ``name``, after any intervention on it."""
    return Expr({"var": {"variable": _name(name)}})


def lag(name: str, steps: int) -> Expr:
    """The value of variable ``name`` ``steps`` steps back (``steps >= 1``; temporal routes)."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise CausalValueError("a lag must be an integer of at least one step")
    return Expr({"lag": {"variable": _name(name), "lag": steps}})


def generated() -> Expr:
    """The value the node's own mechanism generated (transforms only)."""
    return Expr("generated")


def maximum(a: Any, b: Any) -> Expr:
    """The larger of two expressions."""
    return _coerce(a)._binary("max", b)


def minimum(a: Any, b: Any) -> Expr:
    """The smaller of two expressions."""
    return _coerce(a)._binary("min", b)


def clip(value: Any, lower: Any, upper: Any) -> Expr:
    """``value`` limited to ``[lower, upper]``."""
    return Expr(
        {
            "clip": {
                "value": _coerce(value).node,
                "lower": _coerce(lower).node,
                "upper": _coerce(upper).node,
            }
        }
    )


def where(condition: Expr, then: Any, otherwise: Any) -> Expr:
    """``then`` where ``condition`` holds, else ``otherwise`` (both evaluated per row)."""
    if not isinstance(condition, Expr):
        raise CausalValueError("where() needs a condition built from comparisons")
    return Expr(
        {
            "if": {
                "condition": condition.node,
                "then_branch": _coerce(then).node,
                "else_branch": _coerce(otherwise).node,
            }
        }
    )


def indicator(condition: Expr) -> Expr:
    """``1`` where ``condition`` holds, else ``0`` (a condition is not a node value)."""
    if not isinstance(condition, Expr):
        raise CausalValueError("indicator() needs a condition built from comparisons")
    return Expr({"indicator": {"operand": condition.node}})


def _name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise CausalValueError("a variable is named by a non-empty string")
    return name


@dataclass(frozen=True, slots=True)
class Program:
    """One transform or assignment of ``variable`` by ``expr``."""

    mode: Mode
    variable: str
    expr: Expr


def transform(variable: str, expr: Expr) -> Program:
    """Replace ``variable``'s generated value by ``expr`` (read it as :func:`generated`)."""
    return Program("transform", _name(variable), _coerce(expr))


def assign(variable: str, expr: Expr) -> Program:
    """Replace ``variable``'s mechanism by ``expr`` of other variables."""
    return Program("assign", _name(variable), _coerce(expr))


def _text(programs: Sequence[Program]) -> str:
    items = list(programs)
    if not items or any(not isinstance(p, Program) for p in items):
        raise CausalValueError("pass one or more programs built with transform() or assign()")
    return json.dumps(
        [{"mode": p.mode, "variable": p.variable, "expr": p.expr.node} for p in items],
        allow_nan=False,
    )


def _programs(text: str) -> tuple[Program, ...]:
    return tuple(
        Program(item["mode"], item["variable"], Expr(item["expr"])) for item in json.loads(text)
    )


def sample_policy(
    fitted: _native.FittedGcm,
    programs: Sequence[Program],
    n: int,
    *,
    seed: int = 0,
    threads: int | None = None,
) -> _native.GcmSampleResult:
    """Draw ``n`` rows from ``fitted`` with ``programs`` applied (forward ancestral sampling)."""
    return _native.sample_policy(fitted, _text(programs), n, seed, threads)


def counterfactual_policy(
    fitted: _native.FittedGcm,
    programs: Sequence[Program],
    outcomes: Sequence[str],
    *,
    seed: int = 0,
    threads: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Each factual row's ``outcomes`` had ``programs`` applied, on its own abduced noise."""
    return _native.counterfactual_policy(fitted, _text(programs), list(outcomes), seed, threads)


def sample_temporal_policy(
    fitted: _native.FittedGcm,
    programs: Sequence[Program],
    *,
    n_rows: int,
    n_steps: int,
    window: int,
    start: int = 0,
    history: Sequence[Mapping[str, Sequence[float]]] | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> list[_native.GcmSampleResult]:
    """Roll ``fitted`` out for ``n_steps`` steps with ``programs`` active from step ``start``.

    A :func:`lag` read sees up to ``window`` earlier steps; ``history`` supplies steps before time
    zero, most recent first, as ``{variable: values}`` over every variable. A look-back that
    reaches before the first available step is refused (``policy_history_unavailable``).
    """
    steps = None
    if history is not None:
        steps = [{name: [float(v) for v in values] for name, values in h.items()} for h in history]
    return _native.sample_temporal_policy(
        fitted, _text(programs), n_rows, n_steps, window, seed, start, steps, threads
    )


def policy_program_identity(names: Sequence[str], programs: Sequence[Program]) -> str:
    """Hex identity of ``programs`` with variables spelled by ``names`` (graph order)."""
    return _native.policy_program_identity(list(names), _text(programs))


def encode_policy_program(names: Sequence[str], programs: Sequence[Program]) -> bytes:
    """Versioned portable encoding of ``programs``; refuses a program that cannot leave the process."""
    return _native.encode_policy_program(list(names), _text(programs))


def decode_policy_program(names: Sequence[str], data: bytes) -> tuple[Program, ...]:
    """Decode and re-verify an encoded program (version, identity and validity)."""
    return _programs(_native.decode_policy_program(list(names), bytes(data)))
