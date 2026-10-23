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

The record is retained (``propagation_chain_v1``) and :func:`consume_propagation_chain` checks the
rule again and replays every stage to identical bits. Every rule, number, identity and refusal is
computed in Rust; this module declares chains and decodes results.

A value under a licensed chain is labelled ``point_only``, its spread ``uncalibrated_spread``,
posterior draws ``posterior_propagation`` and an enclosure ``propagation_enclosure``. No label is
a coverage statement: :meth:`PropagationChain.interval` raises ``interval_level_not_measured``.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "ChainSource",
    "ChainStage",
    "ConsumedPropagationChain",
    "PropagationChain",
    "PropagationChainRefusal",
    "StageResult",
    "affine",
    "binary",
    "consume_propagation_chain",
    "draws",
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
    "unary",
]

_COUNTS = frozenset({"n_rows", "verdict_code"})
_VERDICTS = {0: "act", 1: "withhold", 2: "unresolved"}


class PropagationChainRefusal(StructuredRefusal):
    """A typed refusal of a chain.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``evidence_double_counted``, ``evidence_overlap_unknown``, ``joint_law_required``,
    ``representation_mapping_required``, ``distribution_meaning_mismatch``,
    ``interval_level_not_measured``, ``posterior_draws_below_supported``, ``invalid_argument``) and
    a ``propagation_chain.*`` ``detail``. :attr:`next_attempt` is the stable name of the attempt
    that would carry the uncertainty and ``remedy`` is its sentence.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: Stable name of the attempt the refusal names.
        self.next_attempt: str | None = refusal.get("next_attempt")


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, PropagationChainRefusal)


def _bits(value: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def _float(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", int(bits)))[0])


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not hasattr(value, "__float__"):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


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

    ``meaning`` is ``"posterior"`` (a posterior under a declared prior) or ``"sampling"`` (a
    sampling or bootstrap law). ``alignment`` is the identity of the table: columns that share it
    are rows of one joint law. ``factors`` are ``(kind, id)`` pairs, kind one of ``row``,
    ``cluster``, ``model``, ``parameter``, ``prior``, ``disturbance``, ``period``.
    """
    if meaning not in ("posterior", "sampling"):
        raise CausalValueError("meaning must be 'posterior' or 'sampling'")
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
    return _stage(id, "pooling", inputs, {"op": "pool", "dependent": bool(dependent)}, own_factors, None)


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
    _record: str = field(repr=False, compare=False)
    _bytes: bytes = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            f"PropagationChain(identity={self.identity[:12]!r}, label={self.label!r}, "
            f"stages={[r.stage for r in self.results]})"
        )

    @classmethod
    def _from(cls, parts: tuple[bytes, str, str]) -> PropagationChain:
        data, record, program = parts
        body = json.loads(record)
        return cls(
            identity=body["identity"],
            label=body["label"],
            output=body["output"],
            results=_results(body["results"]),
            program_identity=program,
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

    def dependents_of(self, source: str) -> list[str]:
        """The stages that depend on ``source``: the ones a change to it reaches."""
        stages, refusal = _native.propagation_chain_dependents(self._record, _text(source, "source"))
        _refuse(refusal)
        assert stages is not None
        return list(stages)

    def withdraw(self, source: str, *, cancel: Any = None) -> PropagationChain:
        """Withdraw ``source``: stages that depend on it are invalidated and carry no number."""
        change = json.dumps({"change": "withdraw", "source": _text(source, "source")})
        return _chain(_native.propagation_chain_change(self._record, change, cancel))

    def reweight(
        self, source: str, log_weights: Sequence[float] | np.ndarray, *, cancel: Any = None
    ) -> PropagationChain:
        """Re-weight the rows of ``source`` (an assumption about it changed).

        Stages that depend on it are summarized under the new weights; every other stage keeps its
        stored numbers bit for bit. ``log_weights`` is one finite log importance weight per row.
        """
        change = json.dumps(
            {
                "change": "reweight",
                "source": _text(source, "source"),
                "log_weights": [_bits(v) for v in _floats(log_weights, "log_weights")],
            }
        )
        return _chain(_native.propagation_chain_change(self._record, change, cancel))

    def interval(self) -> None:
        """Always raises ``interval_level_not_measured``: a component interval does not cover the
        composition, and no composed coordinate has a measured coverage record."""
        _refuse(
            json.dumps(
                {
                    "code": "interval_level_not_measured",
                    "stage": "propagate_chain",
                    "detail": "propagation_chain.interval_not_measured",
                    "message": "an interval of a composed quantity needs a measured coverage "
                    "record at its composed coordinate",
                    "offending": self.output,
                    "next_attempt": "measure_composed_coverage",
                    "remedy": "measure coverage for the composed coordinate and request the "
                    "interval through the composed-interval check; read the spread meanwhile",
                }
            )
        )

    def export(self) -> bytes:
        """The sealed ``propagation_chain_v1`` record."""
        return self._bytes


@dataclass(repr=False, frozen=True)
class ConsumedPropagationChain:
    """A replayed record; every value equals the stored bits."""

    identity: str
    label: str
    output: str
    results: tuple[StageResult, ...]

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


def _chain(result: tuple[tuple[bytes, str, str] | None, str | None]) -> PropagationChain:
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
) -> dict[str, Any]:
    if reading not in ("point", "spread", "posterior", "enclosure", "interval"):
        raise CausalValueError("reading must be point, spread, posterior, enclosure or interval")
    if construction not in ("joint_draws", "independent_product"):
        raise CausalValueError("construction must be joint_draws or independent_product")
    return {
        "version": 1,
        "required_features": ["propagation_chain_v1"],
        "construction": construction,
        "reading": reading,
        "output": _text(output, "output"),
        "sources": [s._wire() for s in sources],
        "stages": [s._wire() for s in stages],
        "changes": [],
        "label": "",
        "results": [],
        "identity": "",
    }


def propagate(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    *,
    output: str,
    reading: str = "spread",
    construction: str = "joint_draws",
    cancel: Any = None,
) -> PropagationChain:
    """Check a declared chain, compute every stage and seal the record.

    ``reading`` is ``point``, ``spread``, ``posterior`` or ``enclosure`` (``interval`` is refused
    until a coverage record is measured). ``construction`` is ``joint_draws`` (the aligned table
    carries whatever dependence the sources have) or ``independent_product`` (the table is the
    product of independent sources, licensed only for sources that share no factor).
    """
    record = _declaration(sources, stages, output, reading, construction)
    return _chain(_native.propagation_chain_seal(json.dumps(record), cancel))


def plan_view(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    *,
    output: str,
    reading: str = "spread",
    construction: str = "joint_draws",
    cancel: Any = None,
) -> dict[str, Any]:
    """Compile a declared chain and view every layer of its lowered program.

    Nothing is computed. The route is ``PropagationChain`` and the family is
    ``propagation_chain``; the view carries the six layers, the dependence contract as the
    identification of overlap, and the program identity :meth:`PropagationChain.program_identity`
    will carry once the chain executes.
    """
    record = _declaration(sources, stages, output, reading, construction)
    view, refusal = _native.propagation_chain_plan_view(json.dumps(record), cancel)
    _refuse(refusal)
    assert view is not None
    return dict(json.loads(view))


def consume_propagation_chain(
    data: bytes, *, expected_identity: str | None = None, cancel: Any = None
) -> ConsumedPropagationChain:
    """Independently consume an exported record: the rule is checked again and every stage is
    replayed to identical bits. Any edit is refused, resealed or not."""
    replayed, refusal = _native.propagation_chain_consume(bytes(data), expected_identity, cancel)
    _refuse(refusal)
    assert replayed is not None
    body = json.loads(replayed)
    return ConsumedPropagationChain(
        identity=body["identity"],
        label=body["label"],
        output=body["output"],
        results=_results(body["results"]),
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
