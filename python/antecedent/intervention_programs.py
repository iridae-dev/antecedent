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
(:func:`sample_temporal_policy`), plus its replay over a factual panel
(:func:`counterfactual_temporal_policy`).

The licence of each route for a program is reported by :func:`policy_support` (simulation,
counterfactual execution and observational identification, each with the reason code of a refusal
or the assumption of a conditional licence), :func:`policy_value` runs the program against the
unintervened model on the same noise, and :class:`PreparedPolicy` compiles and licenses a
program once for any number of runs; the practitioner handle (:class:`PolicyProgramValue` through
``antecedent.prepare``) is the same retained program reached from data and a graph. Identification routes cannot take a transform or a computed
assignment (its interventional law is not a point mass), so those stay model-based: the values
are properties of the fitted model, with only their simulation error quantified.

A program has a versioned portable encoding (:func:`encode_policy_program`) and an identity
(:func:`policy_program_identity`) that moves with any change to an expression, a lag, a target
variable or the program kind. A decoded program is re-verified: an edited expression is an
identity mismatch, edited declared reads are refused, and a future encoding version is refused
before its body is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from . import _native
from ._artifact_protocol import ArtifactAcceptance, container_bytes
from ._scientific import ScientificResult
from .errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)

__all__ = [
    "Expr",
    "ConsumedPolicyProgram",
    "consume_policy_program",
    "PolicyContract",
    "PolicyPreparation",
    "PolicyProgramResult",
    "PolicyProgramValue",
    "PolicySupport",
    "PolicyValueResult",
    "PreparedPolicy",
    "Program",
    "RouteDecision",
    "assign",
    "clip",
    "const",
    "counterfactual_policy",
    "counterfactual_temporal_policy",
    "decode_policy_program",
    "encode_policy_program",
    "generated",
    "indicator",
    "lag",
    "maximum",
    "minimum",
    "policy_program_identity",
    "policy_support",
    "policy_value",
    "prepare_policy",
    "sample_policy",
    "sample_temporal_policy",
    "transform",
    "var",
    "where",
]

Mode = Literal["transform", "assign"]

PROVENANCE_ID = "model.intervention_programs"


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


def counterfactual_temporal_policy(
    fitted: _native.FittedGcm,
    programs: Sequence[Program],
    steps: Sequence[Mapping[str, Sequence[float]]],
    *,
    window: int,
    start: int = 0,
    history: Sequence[Mapping[str, Sequence[float]]] | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> list[_native.GcmSampleResult]:
    """Replay a factual panel under ``programs`` active from step ``start``, step by step.

    ``steps[t]`` maps **every** model variable to the factual values of the same units at step
    ``t``. The exogenous noise is abduced from each step's table and replayed through the same
    rollout executor as :func:`sample_temporal_policy`, so a :func:`lag` read sees the
    counterfactual earlier steps (up to ``window`` of them) and a program without a lag read
    gives, at each step, what :func:`counterfactual_policy` gives for that step's table.
    ``history`` supplies steps before the panel starts, most recent first; a look-back that
    reaches before the first available step is refused (``policy_history_unavailable``).
    The abduction's rank-preservation disclosure applies as on the static route.
    """
    if not steps:
        raise CausalValueError("pass at least one panel step")
    tables = [
        {name: np.ascontiguousarray(values, dtype=np.float64) for name, values in step.items()}
        for step in steps
    ]
    previous = None
    if history is not None:
        previous = [
            {name: [float(v) for v in values] for name, values in h.items()} for h in history
        ]
    return _native.counterfactual_temporal_policy(
        fitted, _text(programs), tables, window, seed, start, previous, threads
    )


def policy_program_identity(names: Sequence[str], programs: Sequence[Program]) -> str:
    """Hex identity of ``programs`` with variables spelled by ``names`` (graph order)."""
    return _native.policy_program_identity(list(names), _text(programs))


def encode_policy_program(names: Sequence[str], programs: Sequence[Any]) -> bytes:
    """Versioned portable encoding of ``programs``; refuses a program that cannot leave the process.

    ``programs`` may mix closed-form programs with
    :class:`~antecedent.intervention_callbacks.CallbackProgram` applications of callbacks declared
    from source; those are encoded as their recovery records and never their code (a runtime
    binding is refused with ``policy_callback_not_portable``). A program without callbacks keeps
    the encoding and identity it has always had.
    """
    from .intervention_callbacks import CallbackProgram, encode_programs

    if any(isinstance(p, CallbackProgram) for p in programs):
        return encode_programs(names, programs)
    return _native.encode_policy_program(list(names), _text(programs))


def decode_policy_program(
    names: Sequence[str],
    data: bytes,
    *,
    callbacks: Mapping[str, Any] | Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    """Decode and re-verify an encoded program (version, identity and validity).

    An artifact that carries callback records needs ``callbacks``: the implementations, rebuilt
    with :func:`~antecedent.intervention_callbacks.recover_callback`, keyed by name or in a
    sequence. Each is verified against its record again before the program is returned
    (``policy_callback_unrecoverable`` on any difference, or when one is missing).
    """
    data = container_bytes(data)
    from .intervention_callbacks import decode_programs

    if callbacks is not None:
        return decode_programs(names, data, callbacks)
    return _programs(_native.decode_policy_program(list(names), bytes(data)))


@dataclass(frozen=True, slots=True, repr=False)
class ConsumedPolicyProgram(ScientificResult):
    """An independently verified program declaration, with its original encoding.

    The ordered names are supplied by the consumer: historical program bytes
    identify variables by position and do not attest their external labels.
    This object contains no fitted mechanisms, observations or policy value.
    """

    identity: str
    names: tuple[str, ...]
    programs: tuple[Program, ...]
    acceptance: ArtifactAcceptance
    _artifact_bytes: bytes = field(repr=False, compare=False)

    @property
    def answer(self) -> Any:
        from .outcomes import Answer

        return Answer("unavailable", detail="verified_program_declaration_has_no_policy_value")

    def inspect(self) -> Any:
        from .results._report import InspectionReport, SlotModel

        unavailable = SlotModel(
            available=False, reason="program_declaration_only", summary="no execution retained"
        )
        return InspectionReport(
            identification=unavailable,
            support=SlotModel(
                available=True,
                summary="native decoder verified program version, identity and declared reads",
            ),
            uncertainty=unavailable,
            assumptions=SlotModel(
                available=True,
                summary="variable labels supplied by consumer in native variable order",
                payload={"names": self.names},
            ),
            answer=self.answer,
            program_id=self.identity,
            family="policy_program",
            stage="consumed",
            operations=("inspect", "to_dict", "export"),
            details={"replayable": False, "resumable": False},
        )


def consume_policy_program(
    artifact: bytes, *, expected_identity: str | None = None, supplied: Any = None
) -> ConsumedPolicyProgram:
    """Verify a historical policy encoding under explicitly supplied ordered names.

    ``supplied`` is an ordered sequence of variable names, or ``{"names": ...}``.
    No labels are inferred from a live study. The native decoder checks the
    encoding version, expression identity, variable references and declared reads.
    An optional independently retained identity is then checked against the
    decoded program's native identity. Original bytes are retained unchanged.
    """
    names = supplied.get("names") if isinstance(supplied, Mapping) else supplied
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)):
        raise CausalTypeError("policy program consumption needs supplied ordered variable names")
    names = tuple(names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise CausalValueError("supplied names must be a nonempty ordered sequence of names")
    if len(set(names)) != len(names):
        raise CausalValueError("supplied variable names must be unique")
    if expected_identity is not None and (
        not isinstance(expected_identity, str)
        or len(expected_identity) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_identity)
    ):
        raise CausalValueError(
            "expected_identity must be a 64-character hexadecimal program identity"
        )
    data = container_bytes(artifact)
    # Both operations go through native verification; Python performs no expression evaluation.
    programs = decode_policy_program(names, data)
    identity = policy_program_identity(names, programs)
    if expected_identity is not None and identity != expected_identity.lower():
        raise CausalSerializationError("policy program does not match the expected identity")
    return ConsumedPolicyProgram(
        identity=identity,
        names=names,
        programs=programs,
        acceptance=ArtifactAcceptance(
            "verified",
            expected_identity is not None,
            False,
            False,
            proof="native policy decoder verifies version, expression digest and declared reads; "
            "native identity recomputation binds the optional consumer expectation",
        ),
        _artifact_bytes=data,
    )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One route's licence for a program.

    ``licensed`` with an ``assumption`` is a conditional licence (counterfactual execution of a
    callback assumes it is a function of its declared inputs). A refusal carries the registered
    ``reason_code``, a namespaced ``detail`` and the full ``message``.
    """

    licensed: bool
    assumption: str | None
    reason_code: str | None
    detail: str | None
    message: str | None


@dataclass(frozen=True, slots=True)
class PolicySupport:
    """The licence of each execution route for one program."""

    simulation: RouteDecision
    counterfactual: RouteDecision
    identification: RouteDecision

    @property
    def provenance_id(self) -> str:
        """The provenance record (``provenance/<id>.toml``) the routes' definitions come from."""
        return PROVENANCE_ID


@dataclass(frozen=True, slots=True)
class PolicyValueResult:
    """A policy's model-based value on one outcome, against the same noise unintervened.

    ``policy_simulation_stderr`` and ``difference_simulation_stderr`` are Monte Carlo errors of the
    draws only; the fitted model's own uncertainty is ``"unquantified"`` (not zero) and no interval
    is claimed.
    """

    outcome: str
    n_draws: int
    policy_mean: float
    baseline_mean: float
    mean_difference: float
    policy_simulation_stderr: float
    difference_simulation_stderr: float
    support: PolicySupport
    model_uncertainty: Literal["unquantified"] = "unquantified"

    @property
    def provenance_id(self) -> str:
        """The provenance record (``provenance/<id>.toml``) the executor is defined in."""
        return PROVENANCE_ID


def _support(fields: Mapping[str, Mapping[str, Any]]) -> PolicySupport:
    return PolicySupport(
        **{route: RouteDecision(**fields[route]) for route in PolicySupport.__slots__}
    )


def policy_support(fitted: _native.FittedGcm, programs: Sequence[Program]) -> PolicySupport:
    """The licence of each route for ``programs`` on ``fitted``, before anything runs.

    A refusal on a route names its registered reason code; for example a transform of the
    generated value is licensed for simulation and counterfactual execution and refused for
    observational identification with ``policy_route_not_licensed``.
    """
    return _support(_native.policy_support(fitted, _text(programs)))


def policy_value(
    fitted: _native.FittedGcm,
    programs: Sequence[Program],
    outcome: str,
    *,
    n: int,
    seed: int = 0,
    threads: int | None = None,
) -> PolicyValueResult:
    """Value of ``programs`` on ``outcome`` under the fitted model.

    Draws ``n`` rows under the policy together with their noise, replays the same noise with no
    intervention, and reports both means and their paired difference with the simulation error.
    A policy the simulation route refuses is refused here with the same reason code.
    """
    return _value(_native.policy_value(fitted, _text(programs), outcome, n, seed, threads))


def _value(fields: Mapping[str, Any]) -> PolicyValueResult:
    values = dict(fields)
    return PolicyValueResult(**{**values, "support": _support(values["support"])})


class PreparedPolicy:
    """A program compiled and licensed once against one fitted model.

    Every run reuses the prepared program: :meth:`sample`, :meth:`counterfactual` and
    :meth:`value` are the library's executor on the same program, and equal the one-shot
    :func:`sample_policy`, :func:`counterfactual_policy` and :func:`policy_value`. The licence is
    fixed at preparation (:attr:`support`) and the program's identity is the one
    :func:`policy_program_identity` reports (``None`` for a program that cannot leave the
    process).
    """

    __slots__ = ("_native",)

    def __init__(self, fitted: _native.FittedGcm, programs: Sequence[Program]) -> None:
        self._native = _native.PreparedPolicy(fitted, _text(programs))

    @property
    def identity(self) -> str | None:
        """Hex identity of the program."""
        return self._native.identity

    @property
    def support(self) -> PolicySupport:
        """The licence of each route, fixed at preparation."""
        return _support(self._native.support())

    @property
    def provenance_id(self) -> str:
        """The provenance record the executor is defined in."""
        return PROVENANCE_ID

    @property
    def program_compilations(self) -> int:
        """How many times the program was compiled: one, however often it is rerun or rebound."""
        return int(self._native.program_compilations)

    @property
    def model_bindings(self) -> int:
        """How many fitted models the program's overlay and licence were resolved against."""
        return int(self._native.model_bindings)

    def rebind(self, fitted: _native.FittedGcm) -> PreparedPolicy:
        """The same program bound to another fitted model over the same variables.

        For a model refitted on refreshed data: the overlay and the licence are resolved against
        ``fitted``; the program is not compiled again (:attr:`program_compilations` is
        unchanged, :attr:`model_bindings` grows by one).
        """
        rebound = object.__new__(PreparedPolicy)
        rebound._native = self._native.rebind(fitted)
        return rebound

    def sample(
        self, n: int, *, seed: int = 0, threads: int | None = None
    ) -> _native.GcmSampleResult:
        """Draw ``n`` rows under the program."""
        return self._native.sample(n, seed, threads)

    def counterfactual(
        self, outcomes: Sequence[str], *, seed: int = 0, threads: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Each factual row's ``outcomes`` had the program been applied, on its own noise."""
        return self._native.counterfactual(list(outcomes), seed, threads)

    def value(
        self, outcome: str, *, n: int, seed: int = 0, threads: int | None = None
    ) -> PolicyValueResult:
        """The program's value on ``outcome`` (see :func:`policy_value`)."""
        return _value(self._native.value(outcome, n, seed, threads))


def prepare_policy(fitted: _native.FittedGcm, programs: Sequence[Program]) -> PreparedPolicy:
    """Compile and license ``programs`` once on ``fitted``; see :class:`PreparedPolicy`."""
    return PreparedPolicy(fitted, programs)


@dataclass(frozen=True, slots=True)
class PolicyProgramValue:
    """The value of ``programs`` on ``outcome`` under a structural model fitted to the data.

    The question ``antecedent.prepare(data, query=..., graph=...)`` takes to answer a policy
    question through the retained handle: the model is fitted on ``data`` along ``graph`` once
    (``candidates`` / ``constraints`` / ``splits`` / ``purge`` are :func:`antecedent.model.fit_gcm`
    options), the program is compiled and licensed against it once, and every ``estimate`` /
    ``refresh`` reruns :func:`policy_value` on ``n_draws`` draws without compiling the program
    again. The value is model-based: only its simulation error is quantified.
    """

    programs: tuple[Program, ...]
    outcome: str
    n_draws: int
    candidates: tuple[str, ...] | None = None
    constraints: Mapping[str, Mapping[str, Any]] | None = None
    splits: str = "shuffled"
    purge: int = 0

    def __post_init__(self) -> None:
        _text(self.programs)
        object.__setattr__(self, "programs", tuple(self.programs))
        if not isinstance(self.outcome, str) or not self.outcome:
            raise CausalValueError("outcome must be the name of a variable")
        if isinstance(self.n_draws, bool) or not isinstance(self.n_draws, int) or self.n_draws < 2:
            raise CausalValueError("n_draws must be an integer of at least two")
        if self.candidates is not None:
            object.__setattr__(self, "candidates", tuple(self.candidates))

    def fit_options(self) -> dict[str, Any]:
        """The keyword options of the model fit this question asks for."""
        options: dict[str, Any] = {"splits": self.splits, "purge": self.purge}
        if self.candidates is not None:
            options["candidates"] = list(self.candidates)
        if self.constraints is not None:
            options["constraints"] = {k: dict(v) for k, v in self.constraints.items()}
        return options


@dataclass(frozen=True, slots=True)
class PolicyContract:
    """What a policy value rests on: the route it ran on and every assumption it needs.

    ``route`` is the licence the value was produced under (``"simulation"``); the licence of the
    other routes is in :attr:`PolicyProgramResult.support`. ``assumptions`` are the structural
    assumptions of a model-based value plus the conditional licences of the route (for example a
    callback's), ``mechanisms`` the fitted family of each variable, and ``model_uncertainty`` is
    ``"unquantified"`` (not zero): the fitted parameters are taken as exact.
    """

    route: Literal["simulation"]
    assumptions: tuple[str, ...]
    mechanisms: Mapping[str, str]
    program_identity: str | None
    model_uncertainty: Literal["unquantified"] = "unquantified"


@dataclass(frozen=True, slots=True)
class PolicyPreparation:
    """What the retained handle has done: the receipt that a rerun compiled nothing.

    ``program_compilations`` stays 1 however many runs and refreshes follow; ``model_bindings``
    counts the fitted models the compiled overlay and licence were resolved against (one more per
    refresh); ``runs`` counts the executions of the value.
    """

    program_compilations: int
    model_bindings: int
    runs: int


@dataclass(frozen=True, slots=True, repr=False)
class PolicyProgramResult(ScientificResult):
    """One run of the prepared policy question: the value, its licence, contract and provenance.

    ``value`` equals, bit for bit, :func:`policy_value` on the same fitted model, program and
    seed; ``support`` is the decision of each route (simulation, counterfactual execution,
    observational identification) with the assumption of a conditional licence or the reason code
    of a refusal.
    """

    value: PolicyValueResult
    contract: PolicyContract
    preparation: PolicyPreparation
    _native: Any = field(default=None, repr=False, compare=False)
    _prepared: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ._freeze import freeze

        object.__setattr__(self, "contract", freeze(self.contract))

    @property
    def answer(self) -> Any:
        from .outcomes import Answer

        return Answer(
            kind="policy_value",
            value=self.value.policy_mean,
            detail="model-based policy mean; Monte Carlo error only",
        )

    @property
    def route(self) -> Any:
        return None if self._native is None else self._native.last_route

    def inspect(self) -> Any:
        from .results._report import InspectionReport, SlotModel

        return InspectionReport(
            identification=SlotModel(available=True, summary="model-based policy value"),
            support=SlotModel(
                available=True,
                summary="simulation route licensed",
                payload={"support": self.support},
            ),
            uncertainty=SlotModel(
                available=True,
                summary="Monte Carlo error only; model uncertainty unquantified",
                payload={
                    "policy_simulation_stderr": self.value.policy_simulation_stderr,
                    "difference_simulation_stderr": self.value.difference_simulation_stderr,
                },
            ),
            assumptions=SlotModel(
                available=True,
                summary=f"{len(self.contract.assumptions)} assumptions",
                payload={"assumptions": self.contract.assumptions},
            ),
            answer=self.answer,
            program_id=self.contract.program_identity,
            stage="executed",
            family="policy_program",
            operations=("inspect", "to_dict") + (("export",) if self._prepared is not None else ()),
            details={"export_quantities": ("program",)},
        )

    def export(self, *, quantity: str) -> Any:
        """Export the retained checked program; this does not export a fitted value."""
        if quantity != "program" or self._prepared is None:
            raise CausalUnsupportedError(
                "this policy result exports only its retained program with quantity='program'",
                reason_code="route_not_supported",
            )
        from ._artifact_protocol import artifact_from_receipt

        return artifact_from_receipt(self, self._prepared._native.export_program())

    @property
    def support(self) -> PolicySupport:
        """The licence of each route for the program."""
        return self.value.support

    @property
    def provenance_id(self) -> str:
        """The provenance record (``provenance/<id>.toml``) the executor is defined in."""
        return PROVENANCE_ID
