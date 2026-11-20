"""Whole-chain bootstrap union of an assumption envelope.

An ``assumption_envelope`` chain (:func:`antecedent.propagation_chain.propagate`) carries an
interval assumption and aligned sampling draws into a rowwise enclosure of its output. The two
endpoint means of that enclosure still move with the sample. This module reruns the complete
chain on resamples of the sampling rows, holds every interval assumption fixed, and joins the
lower and upper endpoint bounds into one interval at the final composed coordinate::

    from antecedent import Execution
    from antecedent import propagation_chain as pc
    from antecedent import sampling_assumption_union as sau

    union = sau.sampling_assumption_union(
        [
            pc.draws("estimate", rows, meaning="sampling", alignment="survey-units"),
            pc.interval("bias", 0.0, 1.0),
        ],
        [pc.binary("answer", "derived_quantity", ["estimate", "bias"], "sum")],
        output="answer",
        output_quantity=quantity,
        scheme="cluster",
        unit_ids=units,
        cluster_ids=households,
        replicates=200,
        execution=Execution(seed=7),
    )
    union.union_lower, union.union_upper      # basic bootstrap bounds, alpha split in two
    artifact = union.export()
    replayed = sau.consume_sampling_assumption_union(
        artifact, expected_identity=union.program_identity, execution=Execution(seed=7)
    )

``scheme`` names how the sampling rows are resampled; every sampling column of the chain is a
column of one table of raw units and its rows are resampled together:

* ``"iid"`` -- independent rows, drawn with replacement;
* ``"cluster"`` -- complete clusters (``cluster_ids``), drawn with replacement;
* ``"stratified_cluster"`` -- complete clusters within fixed strata (``stratum_ids``);
* ``"moving_block"`` -- overlapping blocks of ``block_length`` consecutive rows of one series;
* ``"stratified_moving_block"`` -- such blocks within each series named by ``stratum_ids``;
* ``"panel_block"`` -- whole panels (``panel_ids``), then blocks within each drawn panel;
* ``"declared"`` -- reruns the caller produced: ``bootstrap`` holds one source list per
  replicate (the same sources with resampled draws), ``replicate_ids`` names them and
  ``resampling_design`` identifies the design, all fixed before the result is seen.

A raw-unit scheme takes ``unit_ids`` (one distinct id per sampling row, in row order) and
``replicates`` (from 100 to 4096). Cluster independence, stationarity and the panel regime of a
scheme are declarations, not checked against the rows. The compiler binds the chain, the
schedule and ``execution.seed`` into one program before any row is drawn, and every replicate
chain is recompiled when it runs.

What the answer is **not**: ``level`` is nominal and ``calibration`` is ``"not_measured"``, so
the union is a bootstrap candidate for the assumption-identified set, not a confidence claim. A
refusal raises :class:`SamplingAssumptionUnionRefusal` with its registered ``reason_code``; a
refusal of a chain carries the chain's own ``propagation_chain.*`` detail and ``next_attempt``.
Every number, schedule and refusal is computed in Rust.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactAcceptance,
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    container_bytes,
)
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _count
from ._validate import finite_float_protocol as _real
from ._validate import name as _name
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .joint_distribution import ScientificQuantity
from .outcomes import outcome_operation
from .propagation_chain import ChainSource, ChainStage, PropagationChainRefusal, _declaration

__all__ = [
    "SamplingAssumptionUnion",
    "SamplingAssumptionUnionRefusal",
    "consume_sampling_assumption_union",
    "sampling_assumption_union",
]

SchemeName = Literal[
    "declared",
    "iid",
    "cluster",
    "stratified_cluster",
    "moving_block",
    "stratified_moving_block",
    "panel_block",
]

# The keywords each scheme takes, all of them required.
_SCHEMES: dict[str, tuple[str, ...]] = {
    "declared": ("bootstrap", "replicate_ids", "resampling_design"),
    "iid": ("unit_ids", "replicates"),
    "cluster": ("unit_ids", "replicates", "cluster_ids"),
    "stratified_cluster": ("unit_ids", "replicates", "cluster_ids", "stratum_ids"),
    "moving_block": ("unit_ids", "replicates", "block_length"),
    "stratified_moving_block": ("unit_ids", "replicates", "stratum_ids", "block_length"),
    "panel_block": ("unit_ids", "replicates", "panel_ids", "block_length"),
}


class SamplingAssumptionUnionRefusal(PropagationChainRefusal):
    """A refused union: its resampling declaration, one of its chains, or a replay.

    A :class:`~antecedent.propagation_chain.PropagationChainRefusal` with a registered
    ``reason_code``. A malformed scheme declaration is ``invalid_argument`` with detail
    ``sampling_assumption_union.invalid_declaration``; bytes that do not replay to the retained
    program identity, seed or result are ``external_binding_mismatch`` with detail
    ``sampling_assumption_union.replay_mismatch``; a chain refusal keeps its own detail and
    ``next_attempt``.
    """


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, SamplingAssumptionUnionRefusal)


@dataclass(repr=False, frozen=True)
class SamplingAssumptionUnion(ScientificResult):
    """A whole-chain bootstrap union of an assumption envelope at the final coordinate.

    ``original_lower`` and ``original_upper`` are the plug-in endpoints: the weighted means of
    the original chain's rowwise lower and upper enclosure. ``union_lower`` and
    ``union_upper`` are their basic-bootstrap bounds at ``(1 - level) / 2`` each.
    ``chain_identities`` lists the original chain and every replicate chain in run order, and
    ``program_identity`` is the compiled program a consumer retains to replay it.
    ``calibration`` is always ``"not_measured"``. ``acceptance`` is set on a union replayed by
    :func:`consume_sampling_assumption_union`.
    """

    scheme: str
    original_lower: float
    original_upper: float
    union_lower: float
    union_upper: float
    level: float
    calibration: str
    chain_identities: tuple[str, ...]
    program_identity: str
    acceptance: ArtifactAcceptance | None = field(default=None, compare=False)
    _kind: str = field(default="", repr=False, compare=False)
    _native: Any = field(default=None, repr=False, compare=False)
    _artifact_bytes: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def route(self) -> str | None:
        """The compiled route of a union computed here; ``None`` for a replayed one."""
        return None if self._native is None else str(self._native.route)

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[SamplingAssumptionUnion]:
        """The sealed declaration, seed and result for :func:`consume_sampling_assumption_union`.

        The program is executed again under the seed it was compiled with and must reproduce
        this result before it is sealed; ``execution`` supplies only threads, limits and
        cancellation. A union replayed from an artifact exports the bytes it was consumed from.
        The expectation carries :attr:`program_identity`.
        """
        expectation = ArtifactExpectation(self._kind, self.program_identity)
        if self._native is None:
            if self._artifact_bytes is None:
                raise CausalValueError("this union holds neither a program nor its artifact")
            return ExportedArtifact(self._artifact_bytes, self._kind, expectation)
        execution = resolve_execution(execution)
        data, refusal = execution._call(self._native.export, execution.cancel)
        _refuse(refusal)
        assert data is not None
        return ExportedArtifact(bytes(data), self._kind, expectation)


def _union(wire: dict[str, Any], native: Any = None) -> SamplingAssumptionUnion:
    return SamplingAssumptionUnion(
        scheme=wire["scheme"],
        original_lower=float(wire["original_lower"]),
        original_upper=float(wire["original_upper"]),
        union_lower=float(wire["union_lower"]),
        union_upper=float(wire["union_upper"]),
        level=float(wire["level"]),
        calibration=wire["calibration"],
        chain_identities=tuple(wire["chain_identities"]),
        program_identity=wire["program_identity"],
        _kind=wire["feature"],
        _native=native,
    )


def _labels(values: object, what: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of strings")
    labels = list(values)
    if any(not isinstance(label, str) for label in labels):
        raise CausalTypeError(f"{what} must be a sequence of strings")
    return labels


def _draft(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    output: str,
    quantity: ScientificQuantity,
) -> dict[str, Any]:
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise CausalTypeError("sources must be a sequence of ChainSource declarations")
    if any(not isinstance(source, ChainSource) for source in sources):
        raise CausalTypeError("sources must be ChainSource declarations")
    if any(not isinstance(stage, ChainStage) for stage in stages):
        raise CausalTypeError("stages must be ChainStage declarations")
    return _declaration(sources, stages, output, "assumption_envelope", "joint_draws", quantity, ())


def _request(
    scheme: str,
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    output: str,
    quantity: ScientificQuantity,
    level: float,
    given: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {"original": _draft(sources, stages, output, quantity)}
    if scheme == "declared":
        reruns = given["bootstrap"]
        if isinstance(reruns, (str, bytes)) or not isinstance(reruns, Sequence):
            raise CausalTypeError("bootstrap must be a sequence of source lists, one per replicate")
        request["bootstrap"] = [_draft(rerun, stages, output, quantity) for rerun in reruns]
        request["replicate_ids"] = _labels(given["replicate_ids"], "replicate_ids")
        request["resampling_design_identity"] = _name(
            given["resampling_design"], "resampling_design"
        )
    else:
        request["unit_ids"] = _labels(given["unit_ids"], "unit_ids")
        request["bootstrap_replicates"] = _count(given["replicates"], "replicates", minimum=0)
        for name in ("cluster_ids", "stratum_ids", "panel_ids"):
            if name in _SCHEMES[scheme]:
                request[name] = _labels(given[name], name)
        if "block_length" in _SCHEMES[scheme]:
            request["block_length"] = _count(given["block_length"], "block_length", minimum=0)
    request["level"] = level
    return request


@outcome_operation
def sampling_assumption_union(
    sources: Sequence[ChainSource],
    stages: Sequence[ChainStage],
    *,
    output: str,
    output_quantity: ScientificQuantity,
    scheme: SchemeName,
    level: float = 0.95,
    unit_ids: Sequence[str] | None = None,
    replicates: int | None = None,
    cluster_ids: Sequence[str] | None = None,
    stratum_ids: Sequence[str] | None = None,
    panel_ids: Sequence[str] | None = None,
    block_length: int | None = None,
    bootstrap: Sequence[Sequence[ChainSource]] | None = None,
    replicate_ids: Sequence[str] | None = None,
    resampling_design: str | None = None,
    execution: Execution | None = None,
) -> SamplingAssumptionUnion:
    """Compile and run the whole-chain bootstrap union of an assumption-envelope chain.

    ``sources``, ``stages``, ``output`` and ``output_quantity`` declare the chain as for
    :func:`antecedent.propagation_chain.propagate` with ``reading="assumption_envelope"``: at
    least one interval source and aligned ``"sampling"`` draws. ``scheme`` selects the
    resampling (see the module docstring) and takes exactly its own keywords; ``level`` is the
    nominal two-sided level in ``(0, 1)``. ``execution.seed`` seeds the resampling schedule and
    is bound into :attr:`SamplingAssumptionUnion.program_identity`; a consumer replays under the
    same seed.

    Raises:
        CausalTypeError: a declaration of the wrong type.
        CausalValueError: an unknown scheme, a keyword the scheme does not take or a missing
            one.
        SamplingAssumptionUnionRefusal: a declaration the compiler refuses (too few units or
            replicates, unaligned or factor-carrying sampling columns, a group or block layout
            the scheme cannot resample, incompatible reruns) or a chain refusal.
    """
    if scheme not in _SCHEMES:
        raise CausalValueError(f"scheme must be one of {tuple(_SCHEMES)}; got {scheme!r}")
    if not isinstance(output_quantity, ScientificQuantity):
        raise CausalTypeError("output_quantity must be a ScientificQuantity")
    given: dict[str, Any] = {
        "unit_ids": unit_ids,
        "replicates": replicates,
        "cluster_ids": cluster_ids,
        "stratum_ids": stratum_ids,
        "panel_ids": panel_ids,
        "block_length": block_length,
        "bootstrap": bootstrap,
        "replicate_ids": replicate_ids,
        "resampling_design": resampling_design,
    }
    needed = _SCHEMES[scheme]
    extra = [name for name, value in given.items() if value is not None and name not in needed]
    if extra:
        raise CausalValueError(f"scheme {scheme!r} does not take {', '.join(extra)}")
    missing = [name for name in needed if given[name] is None]
    if missing:
        raise CausalValueError(f"scheme {scheme!r} needs {', '.join(missing)}")
    execution = resolve_execution(execution)
    request = _request(
        scheme,
        sources,
        stages,
        _name(output, "output"),
        output_quantity,
        _real(level, "level"),
        given,
    )
    handle, refusal = execution._call(
        _native.sampling_assumption_union_compile,
        scheme,
        json.dumps(request, allow_nan=False, separators=(",", ":")),
        execution.seed,
        execution.cancel,
    )
    _refuse(refusal)
    assert handle is not None
    return _union(json.loads(handle.result_json), native=handle)


@outcome_operation
@artifact_consumer(
    proof="sampling_assumption_union.consume_sampling_assumption_union: the native consumer "
    "recompiles the declared schedule and every chain under the retained program identity and "
    "seed, reruns them and compares the result",
    replayable=True,
)
def consume_sampling_assumption_union(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> SamplingAssumptionUnion:
    """Independently replay an exported union under the identity retained beside the bytes.

    The artifact's scheme selects its consumer, which recompiles the schedule and every chain,
    reruns them under ``execution.seed`` (the seed the union was produced with) and compares
    the result with the sealed one. ``expected_identity`` is the
    :attr:`SamplingAssumptionUnion.program_identity` kept independently of the bytes.

    Raises:
        CausalTypeError: ``artifact`` is not bytes or an exported artifact.
        CausalValueError: ``expected_identity`` is not 64 hexadecimal characters.
        SamplingAssumptionUnionRefusal: another program identity, seed, schedule or result
            (``external_binding_mismatch``), or a chain refusal on replay.
    """
    execution = resolve_execution(execution)
    data = container_bytes(artifact)
    wire, refusal = execution._call(
        _native.sampling_assumption_union_consume,
        data,
        _name(expected_identity, "expected_identity"),
        execution.seed,
        execution.cancel,
    )
    _refuse(refusal)
    assert wire is not None
    # The consumer decorator records the acceptance and the consumed bytes on the result.
    return _union(json.loads(wire))
