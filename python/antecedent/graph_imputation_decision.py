"""Decisions on a compiled graph-imputation mixture, at the source or in a target population.

A :class:`~antecedent.missing.PreparedGraphImputation` retains a graph-selection imputation
mixture of one binary treatment's effect on one outcome.
:func:`compile_graph_imputation_decision` binds a decision
:class:`~antecedent.decision.Contract` to that mixture before anything is fitted, and the
coordinates the contract's actions read say what is decided on:

* actions that read the binary average effect (functional ``"average_effect"``, regime
  ``"do(t=1)-do(t=0)"``) decide on the mixture's effect posterior, at the source or, with
  ``target=``, standardized to an independent target frame;
* actions that read one coordinate per treatment world (regimes ``"do(t=0)"`` and
  ``"do(t=1)"``) decide on paired worlds drawn from one graph, imputation and coefficient
  draw: world means (functional ``"mean"``) at the source or in the target, or individual
  predictive outcomes (functional ``"outcome"``) in the target, Bernoulli for a binary
  outcome or under a declared :class:`GaussianResidual` for a continuous one.

A :class:`TransportTarget` declares the target frame by source column name. Binary covariates
are standardized by strata; ``bandwidths`` for the continuous covariates switch to a Gaussian
kernel density ratio; ``missing_at_random`` declares a raw frame with missing cells, sampling
weights and complete-vector observation probabilities. The producer's ``graph_support`` is the
graph premise: ``"treatment_root_pre_treatment_covariates"`` transports under a parentless
treatment, ``"pre_treatment_covariates"`` under pretreatment confounding, where every retained
graph must pass the augmented back-door check and every target stratum must contain both
arms. A :class:`TargetBootstrap` resamples the target, whole clusters when the target declares
``clusters`` and rows otherwise; every replicate refits the whole schedule.

Every result is model-conditional and reports ``calibration == "not_measured"``: no interval,
coverage or decision reliability is claimed. :meth:`CompiledGraphImputationDecision.export`
seals the original source, the target, the contract and the decision into portable bytes, and
:func:`consume_graph_imputation_decision` replays them independently under the program identity
the consumer retained. A transported effect decision composes into a
:class:`~antecedent.composition_bundle.Bundle` through
:meth:`~antecedent.composition_bundle.BundleBuilder.add_graph_imputation_transport_fit` and the
chain it decided on (:meth:`CompiledGraphImputationDecision.chain_artifact`);
:func:`compile_graph_imputation_transport_bundle` refits the fit and replays the chain, and
:func:`consume_graph_imputation_fit_bundle` consumes a bundle refitting every such fit.

Refusals raise :class:`GraphImputationDecisionRefusal`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ._refusal import raise_refusal
from ._unit_support import read_unit_support
from .decision import ActionOutcome, ConstraintExclusion, Contract, Verdict
from .errors import CausalTypeError, CausalValueError, StructuredRefusal

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact
    from .composition_bundle import Bundle, ConsumedBundle, SuppliedSources
    from .execution import Execution
    from .missing import PreparedGraphImputation

#: The route family a compiled decision belongs to.
GraphImputationRoute = Literal["effect", "world", "transport", "kernel_transport", "mar_transport"]


class GraphImputationDecisionRefusal(StructuredRefusal):
    """A refused graph-imputation decision, transported fit or bundle recomputation.

    A :class:`~antecedent.errors.StructuredRefusal` carrying the registered ``code`` and the
    namespaced ``detail`` of the refusing compiler, replay or decision stage.
    """


def _cell(value: Any) -> float | None:
    from ._data import _is_missing_scalar

    if _is_missing_scalar(value):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _texts(values: Sequence[str] | None, what: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of strings, not one string")
    return [str(value) for value in values]


@dataclass(frozen=True, slots=True)
class MissingAtRandomTarget:
    """The missing-at-random design of a raw target frame.

    ``observation_probabilities`` are the declared probabilities, one per target row, that the
    row's whole covariate vector is observed, conditional on the ``always_observed`` columns;
    ``weights`` are sampling weights (one for every row by default). Complete rows are weighted
    by weight over observation probability. ``observation_mechanism`` and ``sampling_design``
    name the two declarations; both are assumptions the target data cannot test.
    """

    observation_mechanism: str
    sampling_design: str
    observation_probabilities: Sequence[float]
    weights: Sequence[float] | None = None
    always_observed: Sequence[str] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "observation_mechanism": str(self.observation_mechanism),
            "sampling_design": str(self.sampling_design),
            "observation_probabilities": [float(p) for p in self.observation_probabilities],
            "weights": None if self.weights is None else [float(w) for w in self.weights],
            "always_observed": _texts(self.always_observed, "always_observed") or [],
        }


@dataclass(frozen=True, slots=True)
class TransportTarget:
    """An independent target frame of pretreatment covariates.

    ``columns`` maps source column names to the target rows' values; the treatment and the
    outcome are never target columns. ``selection`` names the covariates whose population
    mechanisms may differ between source and target. ``target_population`` is the population
    every decision coordinate names, and ``source_population`` the one the source represents.
    ``target_sample`` identifies the target sample; the source sample identity is the
    producer's. ``source_units`` and ``target_units`` identify the rows (``source-<row>`` and
    ``target-<row>`` by default) and must be distinct. ``clusters`` names each target row's
    independent sampling unit: a :class:`TargetBootstrap` then resamples whole clusters.

    Binary columns are standardized by strata. ``bandwidths`` gives one positive Gaussian
    kernel bandwidth for every continuous column, which switches to kernel density-ratio
    standardization with ``min_kernel_overlap`` as the floor of the mean source kernel support
    at each target row. ``missing_at_random`` declares a raw frame: missing cells (``None`` or
    ``NaN``) are allowed and complete rows are reweighted by the declared design. The target
    frame, the bandwidths and the selection premise are assumptions; the target frame's own
    sampling uncertainty is carried only by a :class:`TargetBootstrap`.
    """

    columns: Mapping[str, Sequence[float | None]]
    selection: Sequence[str]
    target_population: str
    target_sample: str
    source_population: str = "source"
    source_units: Sequence[str] | None = None
    target_units: Sequence[str] | None = None
    clusters: Sequence[str] | None = None
    bandwidths: Mapping[str, float] = field(default_factory=dict)
    min_kernel_overlap: float = 0.0
    missing_at_random: MissingAtRandomTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.columns, Mapping) or not self.columns:
            raise CausalTypeError("TransportTarget.columns must be a non-empty mapping")
        if not isinstance(self.bandwidths, Mapping):
            raise CausalTypeError("TransportTarget.bandwidths must map column names to numbers")
        if self.missing_at_random is not None and not isinstance(
            self.missing_at_random, MissingAtRandomTarget
        ):
            raise CausalTypeError(
                "TransportTarget.missing_at_random must be a MissingAtRandomTarget"
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "columns": [
                [str(name), [_cell(value) for value in values]]
                for name, values in self.columns.items()
            ],
            "selection": _texts(self.selection, "selection") or [],
            "source_population": str(self.source_population),
            "target_population": str(self.target_population),
            "target_sample": str(self.target_sample),
            "source_units": _texts(self.source_units, "source_units"),
            "target_units": _texts(self.target_units, "target_units"),
            "clusters": _texts(self.clusters, "clusters"),
            "bandwidths": [[str(name), float(value)] for name, value in self.bandwidths.items()],
            "min_kernel_overlap": float(self.min_kernel_overlap),
            "missing_at_random": None
            if self.missing_at_random is None
            else self.missing_at_random._wire(),
        }


@dataclass(frozen=True, slots=True)
class TargetBootstrap:
    """Resampling of the target frame, from 40 to 256 replicates.

    Whole clusters are resampled when the :class:`TransportTarget` declares ``clusters``,
    independent rows otherwise (a missing-at-random target always resamples its clusters, each
    row its own cluster unless declared). Every replicate reimputes the source, reselects the
    graphs, standardizes and refits, and contributes one draw: source-only uncertainty under the
    declared resampling unit, with no coverage earned.
    """

    replicates: int = 40
    seed: int = 1

    def _wire(self) -> dict[str, Any]:
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int):
            raise CausalTypeError("TargetBootstrap.replicates must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise CausalValueError(
                "TargetBootstrap.seed must be a non-negative integer",
                reason_code="invalid_argument",
            )
        return {"replicates": self.replicates, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class GaussianResidual:
    """The declared residual law of two continuous predictive treatment worlds.

    ``variance`` is the positive residual variance in each world and
    ``cross_world_correlation`` (from -1 to 1) the conditional correlation of the two worlds'
    residuals, which no data identify; ``seed`` seeds the residual draws.
    """

    variance: float
    cross_world_correlation: float = 0.0
    seed: int = 1

    def _wire(self) -> dict[str, Any]:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise CausalValueError(
                "GaussianResidual.seed must be a non-negative integer",
                reason_code="invalid_argument",
            )
        return {
            "variance": float(self.variance),
            "cross_world_correlation": float(self.cross_world_correlation),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GraphImputationDecisionResult:
    """A decision on the graph-imputation mixture and the draws it was made on.

    ``route`` is the family the decision ran on and ``program`` its compiled program identity
    (lowercase hex); ``receipt`` identifies the executed schedule. For an effect route,
    ``draws`` are the effect posterior draws and ``mean`` and ``diagnostic_sd`` summarize them
    (model-conditional diagnostics, not a standard error); for a treatment-world route,
    ``worlds`` holds one ``(control, active)`` pair per draw and ``mean`` and ``diagnostic_sd``
    are ``None``. ``verdict``, ``actions`` and ``evpi`` are the contract's evaluation over those
    draws, and ``chain_identity`` the propagation chain it read. ``calibration`` is
    ``"not_measured"``: no interval, coverage or decision reliability is claimed.
    """

    route: GraphImputationRoute
    program: str
    calibration: str
    receipt: str
    verdict: Verdict
    actions: tuple[ActionOutcome, ...]
    evpi: float | None
    contract_identity: str
    criterion_label: str
    chain_identity: str
    mean: float | None
    diagnostic_sd: float | None
    draws: tuple[float, ...]
    worlds: tuple[tuple[float, float], ...]

    @property
    def chosen(self) -> tuple[str, ...]:
        """The leading action, then those it ties; empty when no action is admissible."""
        return self.verdict.actions


def _result(text: str) -> GraphImputationDecisionResult:
    wire = json.loads(text)
    decision = wire["decision"]
    actions = tuple(
        ActionOutcome(
            id=action["id"],
            admissible=action["admissible"],
            exclusions=tuple(
                ConstraintExclusion(
                    exclusion["constraint_id"], exclusion["probability"], exclusion["required"]
                )
                for exclusion in action["exclusions"]
            ),
            expected_utility=action["expected_utility"],
            value=action["value"],
            standard_error=action["standard_error"],
            expected_regret=action["expected_regret"],
            max_regret=action["max_regret"],
        )
        for action in decision["actions"]
    )
    return GraphImputationDecisionResult(
        route=wire["route"],
        program=wire["program"],
        calibration=wire["calibration"],
        receipt=wire["receipt"],
        verdict=Verdict(decision["verdict"], tuple(decision["chosen"])),
        actions=actions,
        evpi=decision["evpi"],
        contract_identity=decision["contract_identity"],
        criterion_label=decision["criterion_label"],
        chain_identity=decision["chain_identity"],
        mean=wire["mean"],
        diagnostic_sd=wire["diagnostic_sd"],
        draws=tuple(wire["draws"]),
        worlds=tuple((float(pair[0]), float(pair[1])) for pair in wire["worlds"]),
    )


@dataclass(frozen=True, slots=True)
class GraphImputationDecisionArtifact:
    """Portable decision bytes with the identities a consumer retains separately.

    ``program`` is the compiled program identity to hand
    :func:`consume_graph_imputation_decision`; ``route`` names the family the bytes replay.
    """

    data: bytes
    program: str
    route: GraphImputationRoute


class CompiledGraphImputationDecision:
    """A decision contract bound to a graph-imputation mixture before anything is fitted.

    Build one with :func:`compile_graph_imputation_decision`. It runs under the producer's
    :class:`~antecedent.Execution` (its seed, limits and cancellation).
    """

    def __init__(self, handle: Any, execution: Execution) -> None:
        self._handle = handle
        self._execution = execution

    @property
    def program(self) -> str:
        """The compiled program identity (lowercase hex) a consumer retains."""
        return str(self._handle.program())

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this compiled program runs under.

        ``{"unit", "supported", "missing_evidence"}``, read from the program's route; ``None``
        while no unit is bound to it.
        """
        return read_unit_support(self._handle)

    @property
    def route(self) -> GraphImputationRoute:
        """The route family: ``effect``, ``world``, ``transport``, ``kernel_transport`` or
        ``mar_transport``."""
        route: GraphImputationRoute = self._handle.route()
        return route

    @property
    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit the compiled program runs under.

        A mapping with ``unit`` (the capability record id), ``status``, ``supported`` and
        ``missing_evidence``; ``None`` while no unit is bound to the route. It is read from the
        compiled route and never enters the program identity.
        """
        text = self._handle.unit_support()
        return None if text is None else dict(json.loads(text))

    def execute(self) -> GraphImputationDecisionResult:
        """Reimpute the source, reselect the graphs, refit and evaluate the contract.

        Raises:
            GraphImputationDecisionRefusal: a fit, the target standardization or the decision
                refuses, or the execution is cancelled or out of resources.
        """
        text, refusal = self._execution._call(self._handle.execute)
        raise_refusal(refusal, GraphImputationDecisionRefusal)
        if text is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native decision returned neither a result nor a refusal")
        return _result(text)

    def export(
        self, *, artifact_id: str = "graph-imputation-decision"
    ) -> GraphImputationDecisionArtifact:
        """Replay the last execution and seal it, with its program identity kept apart.

        Raises:
            GraphImputationDecisionRefusal: nothing has been executed, or the replay differs.
        """
        data, refusal = self._execution._call(self._handle.export, artifact_id)
        raise_refusal(refusal, GraphImputationDecisionRefusal)
        if data is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native export returned neither bytes nor a refusal")
        return GraphImputationDecisionArtifact(bytes(data), self.program, self.route)

    def chain_artifact(self) -> bytes:
        """The propagation-chain artifact the last execution decided on.

        It is the terminal a transported fit feeds in a composition bundle: add it with
        ``BundleBuilder.add_artifact("propagation_chain", ...)`` and connect the fit to it.

        Raises:
            GraphImputationDecisionRefusal: nothing has been executed.
        """
        data, refusal = self._handle.chain_artifact()
        raise_refusal(refusal, GraphImputationDecisionRefusal)
        if data is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native chain returned neither bytes nor a refusal")
        return bytes(data)

    def __repr__(self) -> str:
        return (
            f"CompiledGraphImputationDecision(route={self.route!r}, program={self.program[:12]!r})"
        )


def compile_graph_imputation_decision(
    producer: PreparedGraphImputation,
    contract: Contract,
    *,
    target: TransportTarget | None = None,
    bootstrap: TargetBootstrap | None = None,
    residual: GaussianResidual | None = None,
) -> CompiledGraphImputationDecision:
    """Bind ``contract`` to the graph-imputation mixture ``producer`` retains.

    The contract's coordinates choose the family: actions reading the binary average effect
    decide on the effect posterior, and actions reading the two treatment worlds decide on
    paired worlds (means, or predictive outcomes when their functional is ``"outcome"``).
    ``target`` standardizes either to an independent target frame (binary strata, kernel
    density ratio when it declares ``bandwidths``, or a missing-at-random raw frame), under the
    producer's graph support. ``bootstrap`` resamples the target: rows, or whole clusters when
    the target declares them (predictive worlds resample clusters only, world means rows only).
    ``residual`` is the Gaussian residual law of continuous predictive worlds; binary outcomes
    take Bernoulli worlds without one. The producer's mixture is recompiled for the decision,
    with the same program identity; nothing is fitted until :meth:`~CompiledGraphImputationDecision.execute`.

    Raises:
        CausalTypeError: an argument is not of its declared type.
        CausalValueError: a declaration cannot be read.
        GraphImputationDecisionRefusal: the contract, the target, the resampling or the residual
            law does not fit the producer's mixture.
    """
    from . import _native
    from .missing import PreparedGraphImputation

    if not isinstance(producer, PreparedGraphImputation):
        raise CausalTypeError("producer must be an antecedent.missing.PreparedGraphImputation")
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be an antecedent.decision.Contract")
    if target is not None and not isinstance(target, TransportTarget):
        raise CausalTypeError("target must be a TransportTarget")
    if bootstrap is not None and not isinstance(bootstrap, TargetBootstrap):
        raise CausalTypeError("bootstrap must be a TargetBootstrap")
    if residual is not None and not isinstance(residual, GaussianResidual):
        raise CausalTypeError("residual must be a GaussianResidual")
    execution = producer._execution
    handle, refusal = execution._call(
        _native.graph_imputation_decision_compile,
        producer._handle,
        json.dumps(contract._wire()),
        target_json=None if target is None else json.dumps(target._wire()),
        bootstrap_json=None if bootstrap is None else json.dumps(bootstrap._wire()),
        residual_json=None if residual is None else json.dumps(residual._wire()),
        seed=execution.seed,
    )
    raise_refusal(refusal, GraphImputationDecisionRefusal)
    if handle is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native compiler returned neither a plan nor a refusal")
    return CompiledGraphImputationDecision(handle, execution)


def _artifact_bytes(data: object) -> bytes:
    if isinstance(data, GraphImputationDecisionArtifact):
        return data.data
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise CausalTypeError(
        f"expected the exported decision bytes or artifact, not {type(data).__name__}"
    )


def consume_graph_imputation_decision(
    data: bytes | GraphImputationDecisionArtifact,
    *,
    program: str,
    execution: Execution | None = None,
) -> GraphImputationDecisionResult:
    """Independently replay an exported decision under the retained program identity.

    The original source is recompiled and checked against its retained operations, the
    target, imputation, graph selection, fits and decision are replayed, and the outer program
    is rebuilt; ``program`` (the :attr:`CompiledGraphImputationDecision.program` the consumer
    kept apart from the bytes) must equal it.

    Raises:
        CausalTypeError: ``data`` is not bytes or an exported artifact.
        CausalValueError: ``program`` is not 64 hexadecimal characters.
        GraphImputationDecisionRefusal: the bytes, the identity or any replayed stage differs.
    """
    from . import _native
    from .execution import resolve_execution

    execution = resolve_execution(execution)
    text, refusal = execution._call(
        _native.graph_imputation_decision_consume,
        _artifact_bytes(data),
        program,
        execution.seed,
    )
    raise_refusal(refusal, GraphImputationDecisionRefusal)
    if text is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(text)


# --------------------------------------------------------------------------- bundles


@dataclass(frozen=True, slots=True)
class BundleDependency:
    """One replayed bundle dependency: its node, artifact identity and Merkle digest."""

    node_id: str
    artifact_identity: str
    chain_digest: str


@dataclass(frozen=True, slots=True)
class GraphImputationTransportBundleReceipt:
    """The receipt of a refitted transported fit and its replayed chain terminal.

    ``producer_program`` is the transported decision's program, ``chain_program`` the chain
    terminal's and ``program`` the bundle recomputation's own. No sampling coverage or decision
    reliability is earned.
    """

    bundle_identity: str
    terminal_node: str
    terminal_identity: str
    chain_program: str
    producer_program: str
    program: str
    dependencies: tuple[BundleDependency, ...]


def _receipt(text: str) -> GraphImputationTransportBundleReceipt:
    wire = json.loads(text)
    return GraphImputationTransportBundleReceipt(
        bundle_identity=wire["bundle_identity"],
        terminal_node=wire["terminal_node"],
        terminal_identity=wire["terminal_identity"],
        chain_program=wire["chain_program"],
        producer_program=wire["producer_program"],
        program=wire["program"],
        dependencies=tuple(
            BundleDependency(item["node_id"], item["artifact_identity"], item["chain_digest"])
            for item in wire["dependencies"]
        ),
    )


@dataclass(frozen=True, slots=True)
class GraphImputationTransportBundleArtifact:
    """Portable bundle-recomputation bytes with the identities a consumer retains separately."""

    data: bytes
    bundle: str
    program: str


class CompiledGraphImputationTransportBundle:
    """A bundle's transported graph-imputation fit and the chain terminal it feeds.

    Build one with :func:`compile_graph_imputation_transport_bundle`.
    """

    def __init__(self, handle: Any, bundle: str, execution: Execution) -> None:
        self._handle = handle
        self._bundle = bundle
        self._execution = execution

    @property
    def program(self) -> str:
        """The compiled program identity (lowercase hex) a consumer retains."""
        return str(self._handle.program())

    def execute(self) -> GraphImputationTransportBundleReceipt:
        """Refit the transported fit through the bundle consumer and replay its chain terminal.

        Raises:
            GraphImputationDecisionRefusal: a dependency does not replay or the chain differs.
        """
        text, refusal = self._execution._call(self._handle.execute)
        raise_refusal(refusal, GraphImputationDecisionRefusal)
        if text is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native bundle returned neither a receipt nor a refusal")
        return _receipt(text)

    def export(self) -> GraphImputationTransportBundleArtifact:
        """Replay the last receipt and seal the bundle with it.

        Raises:
            GraphImputationDecisionRefusal: nothing has been executed, or the replay differs.
        """
        data, refusal = self._execution._call(self._handle.export)
        raise_refusal(refusal, GraphImputationDecisionRefusal)
        if data is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native export returned neither bytes nor a refusal")
        return GraphImputationTransportBundleArtifact(bytes(data), self._bundle, self.program)


def compile_graph_imputation_transport_bundle(
    bundle: Bundle,
    *,
    fit: str,
    terminal: str,
    execution: Execution | None = None,
) -> CompiledGraphImputationTransportBundle:
    """Compile the recomputation of a bundle's transported fit and its direct chain terminal.

    ``fit`` names a node added with
    :meth:`~antecedent.composition_bundle.BundleBuilder.add_graph_imputation_transport_fit` and
    ``terminal`` the ``propagation_chain`` node it is connected to. The graph, the fit bytes and
    the producer's program enter a distinct program; nothing is refitted until
    :meth:`~CompiledGraphImputationTransportBundle.execute`.

    Raises:
        CausalTypeError: ``bundle`` is not a built composition bundle.
        GraphImputationDecisionRefusal: a node or the edge is missing, or the route is not
            supported in this build (``not_supported``).
    """
    from . import _native
    from .composition_bundle import Bundle
    from .execution import resolve_execution

    if not isinstance(bundle, Bundle):
        raise CausalTypeError("bundle must be an antecedent.composition_bundle.Bundle")
    execution = resolve_execution(execution)
    handle, refusal = execution._call(
        _native.graph_imputation_transport_bundle_compile,
        bundle._native,
        fit,
        terminal,
        execution.seed,
    )
    raise_refusal(refusal, GraphImputationDecisionRefusal)
    if handle is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native compiler returned neither a plan nor a refusal")
    return CompiledGraphImputationTransportBundle(handle, bundle.identity, execution)


def consume_graph_imputation_transport_bundle(
    data: bytes | GraphImputationTransportBundleArtifact,
    *,
    bundle: str,
    program: str,
    execution: Execution | None = None,
) -> GraphImputationTransportBundleReceipt:
    """Recompile and replay an exported bundle recomputation under the retained identities.

    ``bundle`` is the bundle identity and ``program`` the recomputation's program identity, both
    kept apart from the bytes; the fit is refitted, the chain replayed and the receipt required
    to equal the sealed one.

    Raises:
        CausalTypeError: ``data`` is not bytes or an exported artifact.
        CausalValueError: ``program`` is not 64 hexadecimal characters.
        GraphImputationDecisionRefusal: an identity, the fit, the chain or the receipt differs,
            or the route is not supported in this build.
    """
    from . import _native
    from .execution import resolve_execution

    if isinstance(data, GraphImputationTransportBundleArtifact):
        data = data.data
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError(f"expected the exported bundle bytes, not {type(data).__name__}")
    execution = resolve_execution(execution)
    text, refusal = execution._call(
        _native.graph_imputation_transport_bundle_consume,
        bytes(data),
        bundle,
        program,
        execution.seed,
    )
    raise_refusal(refusal, GraphImputationDecisionRefusal)
    if text is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a receipt nor a refusal")
    return _receipt(text)


def consume_graph_imputation_fit_bundle(
    data: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    supplied: SuppliedSources | None = None,
    execution: Execution | None = None,
) -> ConsumedBundle:
    """Consume a composition bundle, refitting every embedded graph-imputation transport fit.

    As :func:`~antecedent.composition_bundle.consume_bundle`, except that each
    ``graph_imputation_transport_fit`` node is reidentified and refitted from its raw source
    under the seed it recorded, rather than reported as an unknown node kind. A build that does
    not support the route reports those nodes as failed. ``expected_identity`` is the bundle
    identity the consumer retained apart from the bytes.

    Raises:
        CausalTypeError: ``data`` is not bytes or an exported bundle.
        CompositionBundleRefusal: the container, its version or the identity does not verify.
    """
    from . import _native
    from ._artifact_protocol import container_bytes
    from .composition_bundle import ConsumedBundle, _refusal_type
    from .execution import resolve_execution

    execution = resolve_execution(execution)
    wire, refusal = execution._call(
        _native.graph_imputation_fit_bundle_consume,
        container_bytes(data),
        expected_identity,
        None if supplied is None else supplied._json(),
        execution.seed,
    )
    raise_refusal(refusal, _refusal_type)
    if wire is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a bundle nor a refusal")
    return ConsumedBundle(json.loads(wire))


__all__ = [
    "BundleDependency",
    "CompiledGraphImputationDecision",
    "CompiledGraphImputationTransportBundle",
    "GaussianResidual",
    "GraphImputationDecisionArtifact",
    "GraphImputationDecisionRefusal",
    "GraphImputationDecisionResult",
    "GraphImputationRoute",
    "GraphImputationTransportBundleArtifact",
    "GraphImputationTransportBundleReceipt",
    "MissingAtRandomTarget",
    "TargetBootstrap",
    "TransportTarget",
    "compile_graph_imputation_decision",
    "compile_graph_imputation_transport_bundle",
    "consume_graph_imputation_decision",
    "consume_graph_imputation_fit_bundle",
    "consume_graph_imputation_transport_bundle",
]
