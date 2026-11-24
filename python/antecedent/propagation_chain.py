"""Uncertainty carried through a chain of transformations.

A chain runs three or more steps in sequence -- a measurement mapping, a transport, a pool, a
derived quantity, a nonlinear decision utility, a preposterior update for a future study -- and
the uncertainty has to survive the sequence:

* every source declares the rows, clusters, models, priors, disturbances and periods it uses
  (``factors``); a stage inherits what its inputs use. An independent pool of overlapping
  evidence is refused with ``evidence_double_counted`` and names the dependent pool; an
  undeclared overlap is ``evidence_overlap_unknown``;
* nonlinear stages read the **joint law** of their inputs from one aligned draw table, never a
  marginal approximation (``g(E[X]) != E[g(X)]``); draws of unrelated tables are never paired by
  index (``joint_law_required``), and :func:`independent_product_table` builds the table for
  independent sources;
* intervals run by interval arithmetic into an **enclosure** that holds under every coupling;
  ``assumption_envelope`` combines an interval assumption with aligned draws and retains
  one lower and upper enclosure per row, with no measured coverage;
* a changed assumption re-weights (:meth:`PropagationChain.reweight`) or invalidates
  (:meth:`PropagationChain.withdraw`) exactly the stages that depend on the source; every other
  stage keeps its stored numbers bit for bit;
* a ``preposterior`` stage simulates the future study per row, updates the posterior draws by its
  likelihood and re-evaluates the decision under them;
* where the representation cannot carry the uncertainty (a point asked for a spread, an interval
  for a posterior, a sampling law for a posterior, an interval for a pool), the refusal is typed
  and ``next_attempt`` names what would carry it.

A chain is compiled before it runs: :func:`plan_view` shows the six-layer lowered program
(route ``PropagationChain``) with nothing computed, :func:`propagate` executes that retained
program and stamps its ``program_identity`` on the result, and every change (:meth:`withdraw`,
:meth:`reweight`) is compiled and executed as a new program.

The record is retained (``propagation_chain_v1`` for point/spread/enclosure readings,
``propagation_chain_law_v2`` for posterior/predictive readings, and
``propagation_chain_assumption_v3`` for draw-indexed assumption enclosures, and
``propagation_chain_joint_law_v4`` for several aligned output quantities) and
:func:`consume_propagation_chain` checks the rule again and replays every stage and output law to
identical bits. Every rule, number, identity and refusal is computed in Rust; this module declares
chains and decodes results.

A decision is compiled together with the chain whose law it reads: :func:`compile_chain_decision`
binds a posterior or predictive chain (or a transported law carried through the chain compiler)
and a :class:`~antecedent.decision.Contract` into one six-layer program, and
:func:`compile_chain_robust_decision` binds an ``assumption_envelope`` chain and a
maximin-over-structures contract into a conservative dominance decision. Each executed decision
exports an artifact that :func:`consume_chain_decision` or :func:`consume_chain_robust_decision`
recompiles and re-executes under the program identity the consumer retained.

A value under a supported chain is labelled ``point_only``, its spread ``uncalibrated_spread``,
posterior draws ``posterior_propagation``, predictive draws ``predictive_propagation``, and an
enclosure ``propagation_enclosure``. No label is a coverage statement:
:meth:`PropagationChain.interval` raises ``uncertainty_representation_insufficient``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native, _reason_details, _retained
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._float_bits import float_bits as _bits
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import finite_float_protocol as _real
from .decision import ActionOutcome, ConstraintExclusion, Contract, Verdict
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, native_controls, resolve_execution
from .joint_distribution import ScientificQuantity
from .outcomes import outcome_operation, require_success

if TYPE_CHECKING:
    from .composition import SupportPolicy
    from .transported_response import TransportedResponse

__all__ = [
    "ChainSource",
    "ChainStage",
    "ChainLaw",
    "ChainJointLaw",
    "ChainAssumptionEnvelope",
    "ChainDecision",
    "ChainDecisionRefusal",
    "ChainRobustDecision",
    "CompiledChainDecision",
    "CompiledChainRobustDecision",
    "ConsumedPropagationChain",
    "FamilyEstimate",
    "FamilyTable",
    "PropagationChain",
    "PropagationChainRefusal",
    "RobustActionBound",
    "RobustConstraintBound",
    "StageResult",
    "affine",
    "binary",
    "compile_chain_decision",
    "compile_chain_robust_decision",
    "consume_chain_decision",
    "consume_chain_robust_decision",
    "consume_propagation_chain",
    "divide",
    "draws",
    "expression",
    "family_table",
    "fixed",
    "independent_product_table",
    "input_coefficient",
    "interval",
    "plan_view",
    "point",
    "pool",
    "preposterior",
    "propagate",
    "reweight_stage",
    "term_add",
    "term_constant",
    "term_divide",
    "term_input",
    "term_multiply",
    "term_require_range",
    "term_subtract",
    "unary",
]

_COUNTS = frozenset({"n_rows", "verdict_code", "robust_verdict_code"})
_VERDICTS = {0: "act", 1: "withhold", 2: "unresolved"}


class PropagationChainRefusal(StructuredRefusal):
    """A typed refusal of a chain.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``evidence_double_counted``, ``evidence_overlap_unknown``, ``joint_law_required``,
    ``uncertainty_representation_insufficient``, ``distribution_meaning_mismatch``,
    ``uncertainty_representation_insufficient``, ``posterior_draws_below_supported``, ``invalid_argument``) and
    a ``propagation_chain.*`` ``detail``. :attr:`next_attempt` is the stable name of the attempt
    that would carry the uncertainty and ``remedy`` is its sentence.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: Stable name of the attempt the refusal names.
        self.next_attempt: str | None = refusal.get("next_attempt")


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, PropagationChainRefusal)


def _float(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", int(bits)))[0])


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty string")
    return value


def _floats(values: Sequence[float] | np.ndarray, what: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise CausalValueError(f"{what} must be finite")
    return tuple(float(v) for v in array)


def _factors(factors: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    out = []
    for item in factors:
        kind, ident = item
        out.append((_text(kind, "factor kind"), _text(ident, "factor id")))
    return tuple(out)


# ----------------------------------------------------------------- declarations


@dataclass(frozen=True)
class ChainSource:
    """One source of the chain: how it carries its uncertainty and the evidence it uses."""

    id: str
    representation: str
    values: tuple[float, ...]
    meaning: str | None = None
    alignment: str | None = None
    complete_disclosure: bool = True
    factors: tuple[tuple[str, str], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "representation": self.representation,
            "meaning": self.meaning,
            "alignment": self.alignment,
            "complete_disclosure": self.complete_disclosure,
            "factors": [{"kind": k, "id": i} for k, i in self.factors],
            "values": [_bits(v) for v in self.values],
        }


def draws(
    id: str,  # noqa: A002 - the declaration's own field name
    values: Sequence[float] | np.ndarray,
    *,
    meaning: str,
    alignment: str,
    factors: Sequence[tuple[str, str]] = (),
    complete_disclosure: bool = True,
) -> ChainSource:
    """One column of an aligned draw table.

    ``meaning`` is ``"posterior"`` (a posterior under a declared prior), ``"predictive"``
    (a future quantity law), ``"innovation"`` (a model's outcome disturbance), or
    ``"sampling"`` (a sampling or bootstrap law). ``alignment`` is the identity of the table:
    columns that share it are rows of one joint law. ``factors`` are ``(kind, id)`` pairs,
    kind one of ``row``,
    ``cluster``, ``model``, ``parameter``, ``prior``, ``disturbance``, ``period``.
    """
    if meaning not in ("posterior", "predictive", "innovation", "sampling"):
        raise CausalValueError("meaning must be posterior, predictive, innovation or sampling")
    return ChainSource(
        id=_text(id, "id"),
        representation="draws",
        values=_floats(values, "values"),
        meaning=meaning,
        alignment=_text(alignment, "alignment"),
        complete_disclosure=bool(complete_disclosure),
        factors=_factors(factors),
    )


def point(
    id: str,  # noqa: A002
    value: float,
    *,
    factors: Sequence[tuple[str, str]] = (),
) -> ChainSource:
    """A value fixed by declaration. It carries no uncertainty."""
    return ChainSource(
        id=_text(id, "id"),
        representation="point",
        values=(_real(value, "value"),),
        factors=_factors(factors),
    )


def interval(
    id: str,  # noqa: A002
    lower: float,
    upper: float,
    *,
    factors: Sequence[tuple[str, str]] = (),
) -> ChainSource:
    """A declared interval. It is not a law: it runs into an enclosure."""
    low, high = _real(lower, "lower"), _real(upper, "upper")
    if low > high:
        raise CausalValueError("lower must not exceed upper")
    return ChainSource(
        id=_text(id, "id"),
        representation="interval",
        values=(low, high),
        factors=_factors(factors),
    )


def fixed(value: float) -> dict[str, Any]:
    """A constant coefficient of an :func:`affine` stage."""
    return {"kind": "fixed", "value": _bits(_real(value, "value"))}


def input_coefficient(position: int) -> dict[str, Any]:
    """The stage's input at ``position`` (after the first) as an uncertain coefficient."""
    if isinstance(position, bool) or not isinstance(position, int) or position < 1:
        raise CausalValueError("position must be an integer of at least one")
    return {"kind": "input", "position": position}


@dataclass(frozen=True)
class ChainStage:
    """One stage of the chain."""

    id: str
    kind: str
    inputs: tuple[str, ...]
    op: Mapping[str, Any]
    own_factors: tuple[tuple[str, str], ...] = ()
    decision_reference: float | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "own_factors": [{"kind": k, "id": i} for k, i in self.own_factors],
            "op": dict(self.op),
            "decision_reference": (
                None if self.decision_reference is None else _bits(self.decision_reference)
            ),
        }


_KINDS = (
    "measurement_mapping",
    "transport",
    "pooling",
    "derived_quantity",
    "decision_utility",
    "preposterior_update",
    "predictive_mapping",
)


def _stage(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    op: Mapping[str, Any],
    own_factors: Sequence[tuple[str, str]],
    decision_reference: float | None,
) -> ChainStage:
    if kind not in _KINDS:
        raise CausalValueError(f"kind must be one of {_KINDS}")
    return ChainStage(
        id=_text(id, "id"),
        kind=kind,
        inputs=tuple(_text(i, "input") for i in inputs),
        op=op,
        own_factors=_factors(own_factors),
        decision_reference=(
            None if decision_reference is None else _real(decision_reference, "decision_reference")
        ),
    )


def affine(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    *,
    scale: Mapping[str, Any] | float = 1.0,
    shift: Mapping[str, Any] | float = 0.0,
    own_factors: Sequence[tuple[str, str]] = (),
    decision_reference: float | None = None,
) -> ChainStage:
    """``input_0 * scale + shift``; a coefficient is a number, :func:`fixed` or
    :func:`input_coefficient` (an uncertain slope or intercept read from another input)."""
    op = {
        "op": "affine",
        "scale": scale if isinstance(scale, Mapping) else fixed(scale),
        "shift": shift if isinstance(shift, Mapping) else fixed(shift),
    }
    return _stage(id, kind, inputs, op, own_factors, decision_reference)


def reweight_stage(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    weights: Sequence[float],
    *,
    own_factors: Sequence[tuple[str, str]] = (),
) -> ChainStage:
    """``sum w_i input_i`` with fixed weights (a transport reweighting over strata)."""
    op = {"op": "reweight", "weights": [_bits(_real(w, "weight")) for w in weights]}
    return _stage(id, kind, inputs, op, own_factors, None)


def pool(
    id: str,  # noqa: A002
    inputs: Sequence[str],
    *,
    dependent: bool,
    own_factors: Sequence[tuple[str, str]] = (),
) -> ChainStage:
    """Pool estimates of one quantity. ``dependent=True`` weights by the joint covariance of the
    aligned draws (shared evidence is counted once); ``dependent=False`` weights by the variances
    alone and is refused when the inputs share evidence."""
    return _stage(
        id, "pooling", inputs, {"op": "pool", "dependent": bool(dependent)}, own_factors, None
    )


def binary(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    combine: str,
    *,
    own_factors: Sequence[tuple[str, str]] = (),
) -> ChainStage:
    """``sum``, ``difference``, ``product`` or ``ratio`` of two inputs."""
    if combine not in ("sum", "difference", "product", "ratio"):
        raise CausalValueError("combine must be sum, difference, product or ratio")
    return _stage(id, kind, inputs, {"op": "binary", "combine": combine}, own_factors, None)


def term_input(position: int) -> dict[str, Any]:
    """An :func:`expression` term that pushes the stage's input at ``position`` (from zero)."""
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise CausalValueError("position must be a non-negative integer")
    return {"term": "input", "position": position}


def term_constant(value: float) -> dict[str, Any]:
    """An :func:`expression` term that pushes a constant."""
    return {"term": "constant", "value": _bits(_real(value, "value"))}


def term_add() -> dict[str, Any]:
    """An :func:`expression` term: pop ``b``, ``a`` and push ``a + b``."""
    return {"term": "add"}


def term_subtract() -> dict[str, Any]:
    """An :func:`expression` term: pop ``b``, ``a`` and push ``a - b``."""
    return {"term": "subtract"}


def term_multiply() -> dict[str, Any]:
    """An :func:`expression` term: pop ``b``, ``a`` and push ``a * b``."""
    return {"term": "multiply"}


def term_divide(min_magnitude: float = 0.0) -> dict[str, Any]:
    """An :func:`expression` term: pop ``b``, ``a`` and push ``a / b``.

    The division is guarded and never silent: it is refused when the draws of ``b`` reach zero (a
    denominator that crosses or touches zero), or when some draw of ``b`` has a magnitude below
    ``min_magnitude``.
    """
    bound = _real(min_magnitude, "min_magnitude")
    if bound < 0:
        raise CausalValueError("min_magnitude must be non-negative")
    return {"term": "divide", "min_magnitude": _bits(bound)}


def term_require_range(low: float, high: float) -> dict[str, Any]:
    """An :func:`expression` term that checks the top column lies in ``[low, high]`` on every row.

    A value outside the range is refused, never clipped.
    """
    lower, upper = _real(low, "low"), _real(high, "high")
    if lower > upper:
        raise CausalValueError("low must not exceed high")
    return {"term": "require_range", "low": _bits(lower), "high": _bits(upper)}


def expression(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    terms: Sequence[Mapping[str, Any]],
    *,
    own_factors: Sequence[tuple[str, str]] = (),
    decision_reference: float | None = None,
) -> ChainStage:
    """A formula over the stage's inputs, evaluated row by row on the aligned draws.

    ``terms`` are in postfix order (:func:`term_input`, :func:`term_constant`, :func:`term_add`,
    :func:`term_subtract`, :func:`term_multiply`, :func:`term_divide`,
    :func:`term_require_range`) and leave exactly one column. A measurement correction, for the
    inputs ``[observed, sensitivity, specificity]``::

        expression("corrected", "measurement_mapping", ["q", "se", "sp"], [
            term_input(0), term_input(2), term_add(), term_constant(-1), term_add(),
            term_input(1), term_input(2), term_add(), term_constant(-1), term_add(),
            term_divide(0.1), term_require_range(0, 1),
        ])

    reads the three columns of one aligned draw table; the ratio is refused when the Youden index
    reaches zero or falls below 0.1 on any row. A predictive line with an independent residual is
    ``[x, slope, intercept, residual]`` with ``x * slope + intercept + residual``.
    """
    wire = [dict(term) for term in terms]
    if not wire:
        raise CausalValueError("an expression holds at least one term")
    op = {"op": "expression", "terms": wire}
    return _stage(id, kind, inputs, op, own_factors, decision_reference)


def divide(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    *,
    min_magnitude: float = 0.0,
    own_factors: Sequence[tuple[str, str]] = (),
) -> ChainStage:
    """``input_0 / input_1`` with a guarded denominator.

    Unlike :func:`binary` with ``"ratio"``, which refuses only a denominator draw that is exactly
    zero, this refuses a denominator whose draws reach zero or come within ``min_magnitude`` of it.
    """
    names = tuple(inputs)
    if len(names) != 2:
        raise CausalValueError("a division reads two inputs")
    terms = [term_input(0), term_input(1), term_divide(min_magnitude)]
    return expression(id, kind, names, terms, own_factors=own_factors)


def unary(
    id: str,  # noqa: A002
    kind: str,
    inputs: Sequence[str],
    function: str,
    *,
    risk_aversion: float | None = None,
    kink: float | None = None,
    below_slope: float | None = None,
    above_slope: float | None = None,
    cutoff: float | None = None,
    below: float | None = None,
    above: float | None = None,
    own_factors: Sequence[tuple[str, str]] = (),
    decision_reference: float | None = None,
) -> ChainStage:
    """A transformation of one input.

    ``function`` is ``exp``, ``log``, ``square``, ``reciprocal``, ``logistic``,
    ``exponential_utility`` (``risk_aversion``), ``piecewise_linear`` (``kink``, ``below_slope``,
    ``above_slope``) or ``threshold`` (``cutoff``, ``below``, ``above``). A stage with a
    ``decision_reference`` reports the verdict of its expected utility against that reference.
    """
    simple = ("exp", "log", "square", "reciprocal", "logistic")
    wire: dict[str, Any]
    if function in simple:
        wire = {"kind": function}
    elif function == "exponential_utility" and risk_aversion is not None:
        wire = {"kind": function, "risk_aversion": _bits(_real(risk_aversion, "risk_aversion"))}
    elif function == "piecewise_linear" and None not in (kink, below_slope, above_slope):
        wire = {
            "kind": function,
            "kink": _bits(_real(kink, "kink")),
            "below_slope": _bits(_real(below_slope, "below_slope")),
            "above_slope": _bits(_real(above_slope, "above_slope")),
        }
    elif function == "threshold" and None not in (cutoff, below, above):
        wire = {
            "kind": function,
            "cutoff": _bits(_real(cutoff, "cutoff")),
            "below": _bits(_real(below, "below")),
            "above": _bits(_real(above, "above")),
        }
    else:
        raise CausalValueError(f"unknown function {function!r} or missing parameters")
    return _stage(
        id, kind, inputs, {"op": "unary", "function": wire}, own_factors, decision_reference
    )


def preposterior(
    id: str,  # noqa: A002
    parameter: str,
    utility: str,
    *,
    noise_sd: float,
    sample_size: float,
    reference_utility: float,
    noise: Sequence[float] | np.ndarray,
) -> ChainStage:
    """A future study of the ``parameter`` column, updating the posterior before the decision
    read from the ``utility`` column is re-evaluated.

    The study observes ``parameter + noise_sd / sqrt(sample_size) * noise`` per row, with one
    standard-normal draw per row in ``noise``. Terminal: no stage may read it.
    """
    op = {
        "op": "preposterior",
        "noise_sd": _bits(_real(noise_sd, "noise_sd")),
        "sample_size": _bits(_real(sample_size, "sample_size")),
        "reference_utility": _bits(_real(reference_utility, "reference_utility")),
        "noise": [_bits(v) for v in _floats(noise, "noise")],
    }
    return _stage(id, "preposterior_update", (parameter, utility), op, (), None)


# ---------------------------------------------------------------------- results


@dataclass(frozen=True)
class ChainLaw:
    """The compiler's replayable output draws; calibration is not measured.

    ``meaning`` distinguishes a parameter posterior from a predictive quantity law.
    ``weights`` is ``None`` for equally weighted rows. A change to a source is
    reflected in the law only after the chain has been recompiled and resealed.
    ``quantity`` is the compiler-checked declared output coordinate, when supplied;
    it does not itself establish a causal intervention law.
    """

    meaning: str
    draws: np.ndarray
    weights: np.ndarray | None
    alignment: str
    chain_identity: str
    program_identity: str | None
    quantity: ScientificQuantity | None
    calibration: str = "not_measured"


@dataclass(frozen=True)
class ChainJointLaw:
    """Aligned output coordinates, one row per declared joint state.

    ``draws`` has shape ``(rows, quantities)``. The order of ``quantities`` is
    the order declared at compile time, including the primary output.
    """

    meaning: str
    quantities: tuple[ScientificQuantity, ...]
    draws: np.ndarray
    weights: np.ndarray | None
    alignment: str
    chain_identity: str
    program_identity: str | None
    calibration: str = "not_measured"


@dataclass(frozen=True)
class ChainAssumptionEnvelope:
    """Rowwise bounds under interval assumptions and an aligned draw law.

    The endpoints have no measured coverage. The interval assumptions carry no
    probability law; each draw row is evaluated over every admissible value.
    ``draw_meaning`` says whether the row axis is sampling, posterior or predictive.
    """

    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray | None
    alignment: str
    chain_identity: str
    program_identity: str | None
    quantity: ScientificQuantity
    draw_meaning: str
    calibration: str = "not_measured"


@dataclass(frozen=True)
class StageResult:
    """The stored result of one stage."""

    stage: str
    status: str
    invalidated_by: tuple[str, ...]
    values: Mapping[str, float | int]

    @property
    def verdict(self) -> str | None:
        """``act``, ``withhold`` or ``unresolved``, for a stage with a decision reference."""
        code = self.values.get("verdict_code")
        return None if code is None else _VERDICTS[int(code)]

    @property
    def robust_verdict(self) -> str | None:
        """Fixed-reference verdict over the assumption range, conditional on supplied draws."""
        code = self.values.get("robust_verdict_code")
        return None if code is None else _VERDICTS[int(code)]

    def value(self, name: str) -> float:
        """A named value (``mean``, ``variance``, ``lower``, ``value_of_information``, ...)."""
        try:
            return float(self.values[name])
        except KeyError as error:
            raise CausalValueError(
                f"no value {name!r} on stage {self.stage!r}; it carries {sorted(self.values)}"
            ) from error


def _results(raw: Sequence[Mapping[str, Any]]) -> tuple[StageResult, ...]:
    out = []
    for item in raw:
        values = {
            name: (int(bits) if name in _COUNTS else _float(bits)) for name, bits in item["values"]
        }
        out.append(
            StageResult(
                stage=item["stage"],
                status=item["status"],
                invalidated_by=tuple(item["invalidated_by"]),
                values=values,
            )
        )
    return tuple(out)


@dataclass(repr=False, frozen=True)
class PropagationChain:
    """A sealed chain: the checked declaration, the draws, the changes and every stage's result."""

    identity: str
    label: str
    output: str
    results: tuple[StageResult, ...]
    #: Identity of the lowered program that produced the chain.
    program_identity: str
    _output_draws: tuple[float, ...] | None = field(repr=False, compare=False)
    _output_weights: tuple[float, ...] | None = field(repr=False, compare=False)
    _output_lower_bounds: tuple[float, ...] | None = field(repr=False, compare=False)
    _output_upper_bounds: tuple[float, ...] | None = field(repr=False, compare=False)
    _alignment: str | None = field(repr=False, compare=False)
    _reading: str = field(repr=False, compare=False)
    _output_quantity: ScientificQuantity | None = field(repr=False, compare=False)
    _draw_meaning: str | None = field(repr=False, compare=False)
    _joint_outputs: tuple[tuple[str, ScientificQuantity], ...] = field(repr=False, compare=False)
    _joint_output_draws: tuple[tuple[float, ...], ...] | None = field(repr=False, compare=False)
    _joint_output_weights: tuple[float, ...] | None = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _bytes: bytes = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            f"PropagationChain(identity={self.identity[:12]!r}, label={self.label!r}, "
            f"stages={[r.stage for r in self.results]})"
        )

    @classmethod
    def _from(cls, parts: tuple[bytes, str, str, str | None, str | None]) -> PropagationChain:
        data, record, program, draw_meaning, alignment = parts
        body = json.loads(record)
        return cls(
            identity=body["identity"],
            label=body["label"],
            output=body["output"],
            results=_results(body["results"]),
            program_identity=program,
            _output_draws=(
                None
                if body.get("output_draws") is None
                else tuple(_float(bits) for bits in body["output_draws"])
            ),
            _output_weights=(
                None
                if body.get("output_weights") is None
                else tuple(_float(bits) for bits in body["output_weights"])
            ),
            _output_lower_bounds=(
                None
                if body.get("output_lower_bounds") is None
                else tuple(_float(bits) for bits in body["output_lower_bounds"])
            ),
            _output_upper_bounds=(
                None
                if body.get("output_upper_bounds") is None
                else tuple(_float(bits) for bits in body["output_upper_bounds"])
            ),
            _alignment=alignment,
            _reading=body["reading"],
            _output_quantity=(
                None
                if body.get("output_quantity") is None
                else ScientificQuantity._from_wire(body["output_quantity"])
            ),
            _draw_meaning=draw_meaning,
            _joint_outputs=tuple(
                (output["stage"], ScientificQuantity._from_wire(output["quantity"]))
                for output in body.get("joint_outputs", ())
            ),
            _joint_output_draws=(
                None
                if body.get("joint_output_draws") is None
                else tuple(
                    tuple(_float(bits) for bits in column) for column in body["joint_output_draws"]
                )
            ),
            _joint_output_weights=(
                None
                if body.get("joint_output_weights") is None
                else tuple(_float(bits) for bits in body["joint_output_weights"])
            ),
            _record=record,
            _bytes=bytes(data),
        )

    def stage(self, id: str) -> StageResult:  # noqa: A002
        """The stored result of a stage."""
        for result in self.results:
            if result.stage == id:
                return result
        raise CausalValueError(f"no stage {id!r}")

    @property
    def answer(self) -> StageResult:
        """The result of the stage whose output is the answer."""
        return self.stage(self.output)

    @property
    def output_law(self) -> ChainLaw:
        """The posterior or predictive output column retained by the compiled chain.

        A point or spread reading has no law carrier. A withdrawn output also
        has none. The returned arrays are copies of the verified record values.
        """
        if self._output_draws is None or self._reading not in ("posterior", "predictive"):
            raise CausalValueError("this chain does not publish an output law")
        assert self._alignment is not None
        return ChainLaw(
            meaning=self._reading,
            draws=np.asarray(self._output_draws, dtype=np.float64),
            weights=(
                None
                if self._output_weights is None
                else np.asarray(self._output_weights, dtype=np.float64)
            ),
            alignment=self._alignment,
            chain_identity=self.identity,
            program_identity=self.program_identity,
            quantity=self._output_quantity,
        )

    @property
    def output_envelope(self) -> ChainAssumptionEnvelope:
        """The draw-indexed assumption envelope retained by the compiled chain."""
        if self._reading != "assumption_envelope" or self._output_lower_bounds is None:
            raise CausalValueError("this chain does not publish an assumption envelope")
        assert self._alignment is not None and self._output_quantity is not None
        assert self._draw_meaning in ("sampling", "posterior")
        assert self._output_upper_bounds is not None
        return ChainAssumptionEnvelope(
            lower=np.asarray(self._output_lower_bounds, dtype=np.float64),
            upper=np.asarray(self._output_upper_bounds, dtype=np.float64),
            weights=(
                None
                if self._output_weights is None
                else np.asarray(self._output_weights, dtype=np.float64)
            ),
            alignment=self._alignment,
            chain_identity=self.identity,
            program_identity=self.program_identity,
            quantity=self._output_quantity,
            draw_meaning=self._draw_meaning,
        )

    @property
    def joint_output_law(self) -> ChainJointLaw:
        """The compiler-retained joint law after checking every declared stage."""
        if self._joint_output_draws is None or self._reading not in ("posterior", "predictive"):
            raise CausalValueError("this chain does not publish a joint output law")
        assert self._alignment is not None
        return ChainJointLaw(
            meaning=self._reading,
            quantities=tuple(quantity for _, quantity in self._joint_outputs),
            draws=np.asarray(self._joint_output_draws, dtype=np.float64).T.copy(),
            weights=(
                None
                if self._joint_output_weights is None
                else np.asarray(self._joint_output_weights, dtype=np.float64)
            ),
            alignment=self._alignment,
            chain_identity=self.identity,
            program_identity=self.program_identity,
        )

    def dependents_of(self, source: str) -> list[str]:
        """The stages that depend on ``source``: the ones a change to it reaches."""
        stages, refusal = _native.propagation_chain_dependents(
            self._record, _text(source, "source")
        )
        _refuse(refusal)
        assert stages is not None
        return list(stages)

    @outcome_operation
    def withdraw(self, source: str, *, execution: Execution | None = None) -> PropagationChain:
        """Withdraw ``source``: stages that depend on it are invalidated and carry no number."""
        execution = resolve_execution(execution)
        change = json.dumps({"change": "withdraw", "source": _text(source, "source")})
        return _chain(_native.propagation_chain_change(self._record, change, execution.cancel))

    @outcome_operation
    def reweight(
        self,
        source: str,
        log_weights: Sequence[float] | np.ndarray,
        *,
        execution: Execution | None = None,
    ) -> PropagationChain:
        """Re-weight the rows of ``source`` (an assumption about it changed).

        Stages that depend on it are summarized under the new weights; every other stage keeps its
        stored numbers bit for bit. ``log_weights`` is one finite log importance weight per row.
        """
        execution = resolve_execution(execution)
        change = json.dumps(
            {
                "change": "reweight",
                "source": _text(source, "source"),
                "log_weights": [_bits(v) for v in _floats(log_weights, "log_weights")],
            }
        )
        return _chain(_native.propagation_chain_change(self._record, change, execution.cancel))

    def interval(self) -> None:
        """Always raises ``uncertainty_representation_insufficient``: a component interval does not cover the
        composition, and no composed coordinate has a measured coverage record."""
        _refuse(
            json.dumps(
                {
                    "code": "uncertainty_representation_insufficient",
                    "stage": "propagate_chain",
                    "detail": _reason_details.propagation_chain.INTERVAL_NOT_MEASURED,
                    "message": "an interval of a composed quantity needs a measured coverage "
                    "record at its composed coordinate",
                    "offending": self.output,
                    "next_attempt": "measure_composed_coverage",
                    "remedy": "measure coverage for the composed coordinate and request the "
                    "interval through the composed-interval check; read the spread meanwhile",
                }
            )
        )

    @artifact_export
    def export(self) -> ExportedArtifact:
        """The sealed, versioned propagation-chain record."""
        return artifact_from_receipt(self, self._bytes)


@dataclass(repr=False, frozen=True)
class ConsumedPropagationChain:
    """A replayed record; every value equals the stored bits."""

    identity: str
    label: str
    output: str
    results: tuple[StageResult, ...]
    reading: str
    alignment: str | None
    output_quantity: ScientificQuantity | None
    output_draw_meaning: str | None
    output_draws: tuple[float, ...] | None = None
    output_weights: tuple[float, ...] | None = None
    output_lower_bounds: tuple[float, ...] | None = None
    output_upper_bounds: tuple[float, ...] | None = None
    joint_outputs: tuple[tuple[str, ScientificQuantity], ...] = ()
    joint_output_draws: tuple[tuple[float, ...], ...] | None = None
    joint_output_weights: tuple[float, ...] | None = None

    def stage(self, id: str) -> StageResult:  # noqa: A002
        """The replayed result of a stage."""
        for result in self.results:
            if result.stage == id:
                return result
        raise CausalValueError(f"no stage {id!r}")

    @property
    def answer(self) -> StageResult:
        """The replayed result of the stage whose output is the answer."""
        return self.stage(self.output)

    @property
    def output_law(self) -> ChainLaw:
        """The outcome or parameter law after independent replay of the record."""
        if self.output_draws is None or self.reading not in ("posterior", "predictive"):
            raise CausalValueError("this chain does not publish an output law")
        assert self.alignment is not None
        return ChainLaw(
            meaning=self.reading,
            draws=np.asarray(self.output_draws, dtype=np.float64),
            weights=(
                None
                if self.output_weights is None
                else np.asarray(self.output_weights, dtype=np.float64)
            ),
            alignment=self.alignment,
            chain_identity=self.identity,
            program_identity=None,
            quantity=self.output_quantity,
        )

    @property
    def output_envelope(self) -> ChainAssumptionEnvelope:
        """The assumption envelope after independent record replay."""
        if self.reading != "assumption_envelope" or self.output_lower_bounds is None:
            raise CausalValueError("this chain does not publish an assumption envelope")
        assert self.alignment is not None and self.output_quantity is not None
        assert self.output_draw_meaning in ("sampling", "posterior")
        assert self.output_upper_bounds is not None
        return ChainAssumptionEnvelope(
            lower=np.asarray(self.output_lower_bounds, dtype=np.float64),
            upper=np.asarray(self.output_upper_bounds, dtype=np.float64),
            weights=(
                None
                if self.output_weights is None
                else np.asarray(self.output_weights, dtype=np.float64)
            ),
            alignment=self.alignment,
            chain_identity=self.identity,
            program_identity=None,
            quantity=self.output_quantity,
            draw_meaning=self.output_draw_meaning,
        )

    @property
    def joint_output_law(self) -> ChainJointLaw:
        """The joint law after independent replay of all output columns."""
        if self.joint_output_draws is None or self.reading not in ("posterior", "predictive"):
            raise CausalValueError("this chain does not publish a joint output law")
        assert self.alignment is not None
        return ChainJointLaw(
            meaning=self.reading,
            quantities=tuple(quantity for _, quantity in self.joint_outputs),
            draws=np.asarray(self.joint_output_draws, dtype=np.float64).T.copy(),
            weights=(
                None
                if self.joint_output_weights is None
                else np.asarray(self.joint_output_weights, dtype=np.float64)
            ),
            alignment=self.alignment,
            chain_identity=self.identity,
            program_identity=None,
        )


def _chain(
    result: tuple[tuple[bytes, str, str, str | None, str | None] | None, str | None],
) -> PropagationChain:
    parts, refusal = result
    _refuse(refusal)
    assert parts is not None
    return PropagationChain._from(parts)


def _declaration(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    output: str,
    reading: str,
    construction: str,
    output_quantity: ScientificQuantity | None,
    joint_outputs: Sequence[tuple[str, ScientificQuantity]],
) -> dict[str, Any]:
    if reading not in (
        "point",
        "spread",
        "posterior",
        "predictive",
        "enclosure",
        "interval",
        "assumption_envelope",
    ):
        raise CausalValueError(
            "reading must be point, spread, posterior, predictive, enclosure, interval or assumption_envelope"
        )
    if construction not in ("joint_draws", "independent_product"):
        raise CausalValueError("construction must be joint_draws or independent_product")
    if output_quantity is not None and not isinstance(output_quantity, ScientificQuantity):
        raise CausalTypeError("output_quantity must be a ScientificQuantity")
    if joint_outputs and reading not in ("posterior", "predictive"):
        raise CausalValueError("joint outputs require a posterior or predictive reading")
    declared_joint = []
    for stage, quantity in joint_outputs:
        if not isinstance(quantity, ScientificQuantity):
            raise CausalTypeError("each joint output quantity must be a ScientificQuantity")
        declared_joint.append(
            {"stage": _text(stage, "joint output stage"), "quantity": quantity._wire()}
        )
    return {
        "version": (
            4
            if joint_outputs
            else 3
            if reading == "assumption_envelope"
            else 2
            if reading in ("posterior", "predictive")
            else 1
        ),
        "required_features": [
            (
                "propagation_chain_joint_law_v4"
                if joint_outputs
                else "propagation_chain_assumption_v3"
                if reading == "assumption_envelope"
                else "propagation_chain_law_v2"
                if reading in ("posterior", "predictive")
                else "propagation_chain_v1"
            )
        ],
        "construction": construction,
        "reading": reading,
        "output": _text(output, "output"),
        "output_quantity": None if output_quantity is None else output_quantity._wire(),
        "joint_outputs": declared_joint,
        "sources": [s._wire() for s in sources],
        "stages": [s._wire() for s in stages],
        "changes": [],
        "label": "",
        "results": [],
        "output_draws": None,
        "output_weights": None,
        "output_lower_bounds": None,
        "output_upper_bounds": None,
        "joint_output_draws": None,
        "joint_output_weights": None,
        "identity": "",
    }


@outcome_operation
def propagate(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    *,
    output: str,
    reading: str = "spread",
    construction: str = "joint_draws",
    output_quantity: ScientificQuantity | None = None,
    joint_outputs: Sequence[tuple[str, ScientificQuantity]] = (),
    execution: Execution | None = None,
) -> PropagationChain:
    """Compile a declared chain, compute every stage and seal the record.

    ``reading`` is ``point``, ``spread``, ``posterior``, ``predictive``,
    ``enclosure`` or ``assumption_envelope``.
    ``interval`` is refused until a composed coverage record exists. ``construction`` is
    ``joint_draws`` (the aligned table carries whatever dependence the sources have) or
    ``independent_product`` (the table is the
    product of independent sources, supported only for sources that share no factor).
    A predictive reading requires ``output_quantity`` with the outcome or utility role. Its
    population and regime are declarations; this route does not establish causal
    identification or calibration for them.
    An ``assumption_envelope`` reading requires ``output_quantity``, aligned sampling or
    posterior draws, and an interval source. It yields uncalibrated bounds on the
    declared mean and a rowwise enclosure, not a confidence interval.
    ``joint_outputs`` names two or more stage coordinates from one aligned
    posterior or predictive table; it must include ``(output, output_quantity)``.
    Every column and the common row weights are retained for independent replay.
    """
    execution = resolve_execution(execution)
    record = _declaration(
        sources, stages, output, reading, construction, output_quantity, joint_outputs
    )
    return _chain(_native.propagation_chain_seal(json.dumps(record), execution.cancel))


def plan_view(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    *,
    output: str,
    reading: str = "spread",
    construction: str = "joint_draws",
    output_quantity: ScientificQuantity | None = None,
    joint_outputs: Sequence[tuple[str, ScientificQuantity]] = (),
    execution: Execution | None = None,
) -> dict[str, Any]:
    """Compile a declared chain and view every layer of its lowered program.

    Nothing is computed. The route is ``PropagationChain`` and the family is
    ``propagation_chain``; the view carries the six layers, the dependence contract as the
    identification of overlap, and the program identity :meth:`PropagationChain.program_identity`
    will carry once the chain executes.
    The output quantity, when declared, is bound into that identity.
    Joint output declarations are also bound into the compiled identity.

    ``unit_support`` carries the support decision of the unit the route runs under
    (``{"unit", "status", "supported", "missing_evidence"}``), or ``None`` while no unit is bound
    to the route; reading it changes no program identity.
    """
    execution = resolve_execution(execution)
    record = _declaration(
        sources, stages, output, reading, construction, output_quantity, joint_outputs
    )
    view, refusal = _native.propagation_chain_plan_view(json.dumps(record), execution.cancel)
    _refuse(refusal)
    assert view is not None
    return dict(json.loads(view))


@outcome_operation
def consume_propagation_chain(
    data: bytes,
    *,
    expected_identity: str | None = None,
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> ConsumedPropagationChain:
    """Independently consume an exported record: the rule is checked again and every stage is
    replayed to identical bits. Any edit is refused, resealed or not."""
    _retained.admit(
        data,
        retained_identity,
        "antecedent.propagation_chain.consume_propagation_chain",
        "antecedent.propagation_chain.consume_propagation_chain(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
        bound=expected_identity is not None,
    )
    execution = resolve_execution(execution)
    data = container_bytes(data)
    replayed, refusal = _native.propagation_chain_consume(
        bytes(data), expected_identity, execution.cancel
    )
    _refuse(refusal)
    assert replayed is not None
    body = json.loads(replayed)
    return ConsumedPropagationChain(
        identity=body["identity"],
        label=body["label"],
        output=body["output"],
        results=_results(body["results"]),
        reading=body["reading"],
        alignment=body["alignment"],
        output_quantity=(
            None
            if body.get("output_quantity") is None
            else ScientificQuantity._from_wire(body["output_quantity"])
        ),
        output_draw_meaning=body.get("output_draw_meaning"),
        output_draws=(
            None
            if body.get("output_draws") is None
            else tuple(_float(bits) for bits in body["output_draws"])
        ),
        output_weights=(
            None
            if body.get("output_weights") is None
            else tuple(_float(bits) for bits in body["output_weights"])
        ),
        output_lower_bounds=(
            None
            if body.get("output_lower_bounds") is None
            else tuple(_float(bits) for bits in body["output_lower_bounds"])
        ),
        output_upper_bounds=(
            None
            if body.get("output_upper_bounds") is None
            else tuple(_float(bits) for bits in body["output_upper_bounds"])
        ),
        joint_outputs=tuple(
            (output["stage"], ScientificQuantity._from_wire(output["quantity"]))
            for output in body.get("joint_outputs", ())
        ),
        joint_output_draws=(
            None
            if body.get("joint_output_draws") is None
            else tuple(
                tuple(_float(bits) for bits in column) for column in body["joint_output_draws"]
            )
        ),
        joint_output_weights=(
            None
            if body.get("joint_output_weights") is None
            else tuple(_float(bits) for bits in body["joint_output_weights"])
        ),
    )


def independent_product_table(
    columns: Sequence[Sequence[float] | np.ndarray], *, max_rows: int = 1_000_000
) -> list[np.ndarray]:
    """The independent product of separately drawn columns as one aligned table.

    Every combination of the columns' values appears exactly once, so the table is the joint law
    of independent variables and nothing is paired by index. Larger than ``max_rows`` is refused
    (``raise_budget_or_thin_draws``).
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise CausalValueError("max_rows must be a positive integer")
    table, refusal = _native.propagation_chain_product_table(
        [list(_floats(c, "column")) for c in columns], max_rows
    )
    _refuse(refusal)
    assert table is not None
    return [np.asarray(column, dtype=np.float64) for column in table]


# ------------------------------------------------------------------ family tables


@dataclass(frozen=True)
class FamilyEstimate:
    """One estimate a method family produced, with the evidence it used.

    ``meaning`` is ``"sampling"`` (the standard error is a sampling or bootstrap spread) or
    ``"posterior"``. ``factors`` are ``(kind, id)`` pairs as for :func:`draws`;
    ``unknown_overlap`` says the list may be partial. ``authority`` is the hex identity of the
    evidence-ledger source that authorizes the estimate (needed for ``ledger_overlap``).
    """

    id: str
    estimate: float
    standard_error: float
    meaning: str = "sampling"
    factors: tuple[tuple[str, str], ...] = ()
    unknown_overlap: bool = False
    authority: str | None = None

    def _wire(self) -> tuple[str, float, float, str, list[tuple[str, str]], bool, str | None]:
        if self.meaning not in ("posterior", "sampling"):
            raise CausalValueError("meaning must be 'posterior' or 'sampling'")
        return (
            _text(self.id, "id"),
            _real(self.estimate, "estimate"),
            _real(self.standard_error, "standard_error"),
            self.meaning,
            list(_factors(self.factors)),
            bool(self.unknown_overlap),
            None if self.authority is None else _text(self.authority, "authority"),
        )


@dataclass(frozen=True)
class FamilyTable:
    """The aligned joint-draw table of a method family.

    ``sources`` are the draw columns (each carries its declared factors, plus the row factor
    a ledger overlap derived), ready for :func:`propagate`; ``covariance`` is the row-major
    joint covariance the draws carry exactly; ``authority`` pairs each chain source with the
    ledger source that authorizes it.
    """

    sources: tuple[ChainSource, ...]
    covariance: np.ndarray
    authority: tuple[tuple[str, str], ...]


def _source_of_wire(wire: Mapping[str, Any]) -> ChainSource:
    return ChainSource(
        id=str(wire["id"]),
        representation=str(wire["representation"]),
        values=tuple(_float(bits) for bits in wire["values"]),
        meaning=wire.get("meaning"),
        alignment=wire.get("alignment"),
        complete_disclosure=bool(wire["complete_disclosure"]),
        factors=tuple((str(f["kind"]), str(f["id"])) for f in wire["factors"]),
    )


def family_table(
    estimates: Sequence[FamilyEstimate],
    *,
    alignment: str,
    dependence: str = "independent",
    covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    ledger: Any = None,
    rows: int = 1597,
    execution: Execution | None = None,
) -> FamilyTable:
    """Turn family estimates into one aligned joint-draw table carrying their dependence.

    Each estimate becomes a normal draw column with its mean and standard error. The columns
    carry the declared joint covariance exactly, so a pool, a derived quantity or a nonlinear
    utility downstream sees shared evidence once, in the joint law, and the chain rule sees the
    same evidence as declared factors.

    ``dependence`` is ``"independent"`` (refused when the declared factors share evidence or
    leave the overlap unknown), ``"covariance"`` (a ``k`` by ``k`` ``covariance`` whose diagonal
    is the squared standard errors; a nonzero off-diagonal term needs a shared declared factor
    and a shared factor needs a nonzero term) or ``"ledger_overlap"`` (the correlation is read
    from the rows the ``ledger`` sources of two estimates share).

    Raises:
        PropagationChainRefusal: a dependence the declarations do not support
            (``evidence_double_counted``, ``evidence_overlap_unknown``) or a table the
            arithmetic refuses.
    """
    from .evidence_ledger import EvidenceLedger

    if dependence not in ("independent", "covariance", "ledger_overlap"):
        raise CausalValueError("dependence must be independent, covariance or ledger_overlap")
    flat: list[float] | None = None
    if dependence == "covariance":
        if covariance is None:
            raise CausalValueError("dependence='covariance' needs a covariance matrix")
        flat = list(_floats(np.asarray(covariance, dtype=np.float64).reshape(-1), "covariance"))
    elif covariance is not None:
        raise CausalValueError("a covariance matrix applies only to dependence='covariance'")
    if ledger is not None and not isinstance(ledger, EvidenceLedger):
        raise CausalTypeError("ledger must be an EvidenceLedger")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise CausalValueError("rows must be a positive integer")
    policy = resolve_execution(execution)
    parts, refusal = policy._call(
        _native.propagation_chain_family_table,
        _text(alignment, "alignment"),
        [e._wire() for e in estimates],
        dependence,
        flat,
        None if ledger is None else ledger._handle,
        rows,
        policy.seed,
    )
    _refuse(refusal)
    assert parts is not None
    wires, matrix, bindings = parts
    k = len(wires)
    return FamilyTable(
        sources=tuple(_source_of_wire(json.loads(w)) for w in wires),
        covariance=np.asarray(matrix, dtype=np.float64).reshape(k, k),
        authority=tuple((str(a), str(b)) for a, b in bindings),
    )


# ---------------------------------------------------------------------- decisions

_DECISION_KIND = "chain_decision_v1"
_ROBUST_DECISION_KIND = "chain_robust_decision_v1"


class ChainDecisionRefusal(PropagationChainRefusal):
    """A typed refusal of a decision compiled together with its chain.

    A :class:`PropagationChainRefusal`. A chain refusal keeps its ``propagation_chain.*`` detail
    and :attr:`next_attempt`, and a contract refusal its ``decision_contract.*`` detail. A law the
    contract cannot be answered from (posterior parameter draws read as outcome risk, a coordinate
    the chain does not retain, action worlds without a shared declared factor, a transported law
    below the support policy) is ``decision_contract_unsatisfied`` with detail
    ``chain_decision.refused``. An artifact whose replay or program differs from the one the
    consumer retained is ``external_binding_mismatch`` with ``chain_decision.replay_refused``.
    ``expected`` and ``supplied`` carry the compared semantics when the refusal names them.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


def _decided(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, ChainDecisionRefusal)
    assert value is not None
    return value


def _controls(execution: Execution) -> dict[str, Any]:
    return native_controls(
        execution, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


def _number(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _verdict(wire: Any) -> Verdict:
    if wire == "no_admissible_action":
        return Verdict("no_admissible_action", ())
    ((kind, value),) = wire.items()
    if kind == "uniquely_optimal":
        return Verdict("uniquely_optimal", (str(value),))
    return Verdict("indistinguishable", tuple(str(action) for action in value))


def _outcomes(wire: Sequence[Mapping[str, Any]]) -> tuple[ActionOutcome, ...]:
    return tuple(
        ActionOutcome(
            id=str(action["id"]),
            admissible=bool(action["admissible"]),
            exclusions=tuple(
                ConstraintExclusion(
                    str(exclusion["constraint_id"]),
                    float(exclusion["probability"]),
                    float(exclusion["required"]),
                )
                for exclusion in action["exclusions"]
            ),
            expected_utility=_number(action["expected_utility"]),
            value=_number(action["value"]),
            standard_error=action["standard_error"],
            expected_regret=action["expected_regret"],
            max_regret=action["max_regret"],
        )
        for action in wire
    )


def _exported(data: bytes, kind: str, identity: str) -> ExportedArtifact[Any]:
    return ExportedArtifact(bytes(data), kind, ArtifactExpectation(kind, identity))


def _forwarded(result: ScientificResult, kind: str, identity: str) -> ExportedArtifact[Any]:
    data = getattr(result, "_artifact_bytes", None)
    if data is None:
        raise CausalTypeError("this decision retains neither its compiled program nor its bytes")
    return _exported(data, kind, identity)


@dataclass(repr=False, frozen=True, slots=True)
class ChainDecision(ScientificResult):
    """A multi-action decision scored on the law its compiled chain retained.

    ``chain`` is the executed :class:`PropagationChain`; its ``program_identity`` is the joint
    chain-and-decision program. ``actions`` are the per-action
    :class:`~antecedent.decision.ActionOutcome` values in declaration order and ``verdict`` the
    :class:`~antecedent.decision.Verdict`; a sampled-law ranking carries no statistical guarantee.
    ``assumptions`` state what the decision rests on: the chain law is model-conditional with no
    measured coverage, and a transported law is a declared input whose fitting is not replayed.
    ``program_identity`` is the identity a consumer retains for :func:`consume_chain_decision`;
    ``replayed`` is ``True`` for a decision an independent consumer recomputed.
    """

    chain: PropagationChain
    contract_identity: str
    criterion_label: str
    actions: tuple[ActionOutcome, ...]
    verdict: Verdict
    evpi: float | None
    n_draws: int
    effective_draws: float
    assumptions: tuple[str, ...]
    program_identity: str
    replayed: bool = False
    _handle: Any = field(repr=False, compare=False, default=None)

    @property
    def selected(self) -> tuple[str, ...]:
        """The leader and any action it cannot be told apart from; empty if none is admissible."""
        return self.verdict.actions

    @artifact_export
    def export(self, *, artifact_id: str = "chain-decision") -> ExportedArtifact[ChainDecision]:
        """The versioned artifact :func:`consume_chain_decision` recompiles and re-executes.

        It carries the chain record, the contract and the decision fingerprint; its expectation
        is :attr:`program_identity`. A replayed decision forwards the bytes it was read from.

        Raises:
            ChainDecisionRefusal: the artifact exceeds its byte bound or cannot be encoded.
        """
        if self._handle is None:
            return _forwarded(self, _DECISION_KIND, self.program_identity)
        data = _decided(self._handle.export(_text(artifact_id, "artifact_id")))
        return _exported(data, _DECISION_KIND, self.program_identity)


def _decision_of(run: Any, *, replayed: bool) -> ChainDecision:
    body = json.loads(run.decision_json())
    return ChainDecision(
        chain=_chain(run.chain()),
        contract_identity=str(body["contract_identity"]),
        criterion_label=str(body["criterion_label"]),
        actions=_outcomes(body["actions"]),
        verdict=_verdict(body["verdict"]),
        evpi=None if body["evpi"] is None else float(body["evpi"]),
        n_draws=int(body["n_draws"]),
        effective_draws=float(body["effective_draws"]),
        assumptions=tuple(str(assumption) for assumption in body["assumptions"]),
        program_identity=str(run.program_identity),
        replayed=replayed,
        _handle=None if replayed else run,
    )


class CompiledChainDecision:
    """A chain law and a decision contract compiled into one six-layer program.

    Compilation evaluates neither a chain stage nor a utility. :attr:`program_identity` is the
    identity of the joint program, the one an executed decision carries and a consumer retains;
    :meth:`plan_view` shows every layer and :meth:`execute` runs the chain and scores the
    contract on its retained output law.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def program_identity(self) -> str:
        """Identity of the joint chain-and-decision program (lowercase hex)."""
        return str(self._handle.program_identity)

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program, with nothing executed."""
        return dict(json.loads(self._handle.plan_view_json()))

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> ChainDecision:
        """Execute the retained chain and score the contract on its aligned output rows.

        The scoring runs under the execution's memory admission and cancellation; a cancelled run
        raises :class:`~antecedent.errors.CausalCancelledError`.

        Raises:
            ChainDecisionRefusal: the output law was withdrawn or the evaluation refuses.
        """
        policy = resolve_execution(execution)
        run = _decided(policy._call(self._handle.execute, **_controls(policy)))
        return _decision_of(run, replayed=False)

    def __repr__(self) -> str:
        return f"CompiledChainDecision(program_identity={self.program_identity[:12]!r})"


def _contract(contract: object) -> Contract:
    contract = require_success(contract)
    if not isinstance(contract, Contract):
        raise CausalTypeError(
            f"contract must be a decision.Contract, not {type(contract).__name__}"
        )
    return contract


@outcome_operation
def compile_chain_decision(
    source: PropagationChain | TransportedResponse,
    contract: Contract,
    *,
    action_worlds: bool = False,
    support_policy: SupportPolicy | None = None,
    execution: Execution | None = None,
) -> CompiledChainDecision:
    """Compile a chain and a multi-action decision over its retained output law into one program.

    ``source`` is a :class:`PropagationChain` with a ``posterior`` or ``predictive`` reading and a
    declared ``output_quantity`` (several aligned coordinates through ``joint_outputs``). Every
    action input must name one of those exact coordinates, and an input with the ``outcome``
    functional needs a ``predictive`` reading: posterior parameter draws are not outcome risk.
    Utilities may be nonlinear, the criterion any :class:`~antecedent.decision.Criterion`, with
    hard chance constraints and regret; the law is read from the chain's replayable output rows,
    never rebuilt from a mean or variance.

    ``action_worlds=True`` declares that each action reads its own intervention, regime or policy
    world of one jointly aligned, fully disclosed draw table whose sources share a declared
    ``model``, ``parameter`` or ``disturbance`` factor. The cross-world coupling regret and
    nonlinear utilities read is that declared one; it is not identified from the marginals.

    ``source`` may instead be a :class:`~antecedent.transported_response.TransportedResponse`
    carrying a jointly aligned law: its columns become chain sources bound to the transport
    receipt, positive row weights become the chain's reweighting and zero-weight rows are dropped.
    ``support_policy`` (a :class:`~antecedent.composition.SupportPolicy`, by default
    ``SupportPolicy.require_all("supported")``) must accept the receipt's support label, so an
    ``extrapolative`` transport needs ``SupportPolicy.require_all("extrapolative")``. The transport
    is a declared input: it is neither refitted nor recalibrated.

    Raises:
        CausalTypeError: ``source``, ``contract`` or ``support_policy`` has the wrong type.
        CausalValueError: ``action_worlds`` or ``support_policy`` does not apply to ``source``.
        ChainDecisionRefusal: the chain, the contract or the law cannot support the decision.
    """
    from .composition import SupportPolicy
    from .transported_response import TransportedResponse

    source = require_success(source)
    declaration = json.dumps(_contract(contract)._wire())
    if not isinstance(action_worlds, bool):
        raise CausalTypeError("action_worlds must be True or False")
    policy = resolve_execution(execution)
    if isinstance(source, PropagationChain):
        if support_policy is not None:
            raise CausalValueError("support_policy applies to a transported law, not to a chain")
        handle = _decided(
            policy._call(
                _native.chain_decision_compile,
                source._record,
                declaration,
                action_worlds=action_worlds,
                cancel=policy.cancel,
            )
        )
        return CompiledChainDecision(handle)
    if isinstance(source, TransportedResponse):
        if action_worlds:
            raise CausalValueError(
                "action worlds are declared on a chain's joint draw table, not on a transported law"
            )
        accepted = (
            SupportPolicy.require_all("supported") if support_policy is None else support_policy
        )
        if not isinstance(accepted, SupportPolicy):
            raise CausalTypeError("support_policy must be a composition.SupportPolicy")
        handle = _decided(
            policy._call(
                _native.chain_decision_compile_transported,
                source._handle,
                declaration,
                accepted.unsupported,
                accepted.weakest_support,
                cancel=policy.cancel,
            )
        )
        return CompiledChainDecision(handle)
    raise CausalTypeError(
        "source must be a PropagationChain or a transported_response.TransportedResponse, "
        f"not {type(source).__name__}"
    )


@outcome_operation
@artifact_consumer(
    proof="propagation_chain.consume_chain_decision: consumes the chain record and the contract, "
    "recompiles their joint program under the retained identity, re-executes the chain law and "
    "the decision and compares the complete decision fingerprint",
    replayable=True,
)
def consume_chain_decision(
    artifact: bytes,
    *,
    expected_identity: str | None = None,
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> ChainDecision:
    """Independently consume a chain decision artifact (plain, action-world or transported law).

    The chain record and the contract are consumed and recompiled into their joint program, the
    chain re-executed and the decision rescored; the complete decision must match the stored
    fingerprint bit for bit. ``expected_identity`` is the :attr:`ChainDecision.program_identity`
    the consumer retained independently of the bytes.

    Raises:
        CausalValueError: ``expected_identity`` is not a 64-hex identity.
        ChainDecisionRefusal: a foreign version, a changed declaration or result, or a program
            other than the expected one (``external_binding_mismatch``).
    """
    _retained.admit(
        artifact,
        retained_identity,
        "antecedent.propagation_chain.consume_chain_decision",
        "antecedent.propagation_chain.consume_chain_decision(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
        bound=expected_identity is not None,
    )
    data = container_bytes(artifact)
    policy = resolve_execution(execution)
    run = _decided(
        policy._call(_native.chain_decision_consume, data, expected_identity, **_controls(policy))
    )
    return _decision_of(run, replayed=True)


@dataclass(frozen=True, slots=True)
class RobustConstraintBound:
    """Probability bounds for one hard constraint under every admissible assumption.

    ``probability_lower`` is the weight of rows that satisfy the constraint for every assumption,
    ``probability_upper`` of rows that satisfy it for at least one; ``required`` is the contract's
    minimum probability.
    """

    id: str
    probability_lower: float
    probability_upper: float
    required: float


@dataclass(frozen=True, slots=True)
class RobustActionBound:
    """Conservative expected-utility bounds and feasibility of one action.

    ``utility_lower`` and ``utility_upper`` enclose the expected utility under every fixed
    admissible assumption. ``certainly_admissible`` holds when every hard constraint is certified
    for every assumption and ``certainly_excluded`` when one fails under every assumption.
    """

    id: str
    utility_lower: float
    utility_upper: float
    constraints: tuple[RobustConstraintBound, ...]
    certainly_admissible: bool
    certainly_excluded: bool


@dataclass(repr=False, frozen=True, slots=True)
class ChainRobustDecision(ScientificResult):
    """A conservative dominance decision over a chain's draw-indexed assumption envelope.

    Each chain row fixes its draw and encloses every admissible interval assumption; utilities and
    constraints are evaluated by interval arithmetic. ``dominant_action`` is an action only when
    its lower utility exceeds the upper utility of every competitor that may be admissible, and
    ``None`` otherwise. It is conditional on the declared draws and assumption boxes, with no
    coverage or exact-minimax claim. ``program_identity`` is the identity a consumer retains for
    :func:`consume_chain_robust_decision`.
    """

    chain: PropagationChain
    contract_identity: str
    actions: tuple[RobustActionBound, ...]
    dominant_action: str | None
    program_identity: str
    replayed: bool = False
    _handle: Any = field(repr=False, compare=False, default=None)

    @artifact_export
    def export(
        self, *, artifact_id: str = "chain-robust-decision"
    ) -> ExportedArtifact[ChainRobustDecision]:
        """The versioned artifact :func:`consume_chain_robust_decision` replays.

        Its expectation is :attr:`program_identity`. A replayed decision forwards the bytes it was
        read from.

        Raises:
            ChainDecisionRefusal: the artifact exceeds its byte bound or cannot be encoded.
        """
        if self._handle is None:
            return _forwarded(self, _ROBUST_DECISION_KIND, self.program_identity)
        data = _decided(self._handle.export(_text(artifact_id, "artifact_id")))
        return _exported(data, _ROBUST_DECISION_KIND, self.program_identity)


def _robust_of(run: Any, *, replayed: bool) -> ChainRobustDecision:
    body = json.loads(run.decision_json())
    return ChainRobustDecision(
        chain=_chain(run.chain()),
        contract_identity=str(body["contract_identity"]),
        actions=tuple(
            RobustActionBound(
                id=str(action["id"]),
                utility_lower=float(action["utility_lower"]),
                utility_upper=float(action["utility_upper"]),
                constraints=tuple(
                    RobustConstraintBound(
                        id=str(bound["id"]),
                        probability_lower=float(bound["probability_lower"]),
                        probability_upper=float(bound["probability_upper"]),
                        required=float(bound["required"]),
                    )
                    for bound in action["constraints"]
                ),
                certainly_admissible=bool(action["certainly_admissible"]),
                certainly_excluded=bool(action["certainly_excluded"]),
            )
            for action in body["actions"]
        ),
        dominant_action=body["dominant_action"],
        program_identity=str(run.program_identity),
        replayed=replayed,
        _handle=None if replayed else run,
    )


class CompiledChainRobustDecision:
    """An assumption-envelope chain and a maximin contract compiled into one program.

    :attr:`program_identity` is the identity an executed decision carries and a consumer retains;
    :meth:`plan_view` shows every layer and :meth:`execute` bounds every action.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def program_identity(self) -> str:
        """Identity of the joint chain-and-decision program (lowercase hex)."""
        return str(self._handle.program_identity)

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program, with nothing executed."""
        return dict(json.loads(self._handle.plan_view_json()))

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> ChainRobustDecision:
        """Replay the rowwise assumption bounds and bound every action's utility and constraints.

        Raises:
            ChainDecisionRefusal: the envelope was withdrawn or a utility enclosure is not finite.
        """
        policy = resolve_execution(execution)
        run = _decided(policy._call(self._handle.execute, **_controls(policy)))
        return _robust_of(run, replayed=False)

    def __repr__(self) -> str:
        return f"CompiledChainRobustDecision(program_identity={self.program_identity[:12]!r})"


@outcome_operation
def compile_chain_robust_decision(
    chain: PropagationChain,
    contract: Contract,
    *,
    execution: Execution | None = None,
) -> CompiledChainRobustDecision:
    """Compile a conservative dominance decision over a chain's assumption envelope.

    ``chain`` has the ``assumption_envelope`` reading and a declared ``output_quantity``; every
    action input must be that coordinate and the contract's criterion
    :meth:`~antecedent.decision.Criterion.maximin_over_structures`. The chain and the contract
    occupy one six-layer program.

    Raises:
        CausalTypeError: ``chain`` or ``contract`` has the wrong type.
        ChainDecisionRefusal: the reading, the criterion, a coordinate or the contract refuses.
    """
    chain = require_success(chain)
    if not isinstance(chain, PropagationChain):
        raise CausalTypeError(f"chain must be a PropagationChain, not {type(chain).__name__}")
    declaration = json.dumps(_contract(contract)._wire())
    policy = resolve_execution(execution)
    handle = _decided(
        policy._call(
            _native.chain_robust_decision_compile, chain._record, declaration, cancel=policy.cancel
        )
    )
    return CompiledChainRobustDecision(handle)


@outcome_operation
@artifact_consumer(
    proof="propagation_chain.consume_chain_robust_decision: consumes the chain record and the "
    "contract, recompiles their joint program under the retained identity, replays the rowwise "
    "assumption bounds and compares the complete result fingerprint",
    replayable=True,
)
def consume_chain_robust_decision(
    artifact: bytes,
    *,
    expected_identity: str | None = None,
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> ChainRobustDecision:
    """Independently replay a robust chain decision under a retained program identity.

    ``expected_identity`` is the :attr:`ChainRobustDecision.program_identity` the consumer
    retained independently of the bytes.

    Raises:
        CausalValueError: ``expected_identity`` is not a 64-hex identity.
        ChainDecisionRefusal: a changed declaration, enclosure, contract, program or result.
    """
    _retained.admit(
        artifact,
        retained_identity,
        "antecedent.propagation_chain.consume_chain_robust_decision",
        "antecedent.propagation_chain.consume_chain_robust_decision(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
        bound=expected_identity is not None,
    )
    data = container_bytes(artifact)
    policy = resolve_execution(execution)
    run = _decided(
        policy._call(
            _native.chain_robust_decision_consume, data, expected_identity, **_controls(policy)
        )
    )
    return _robust_of(run, replayed=True)
