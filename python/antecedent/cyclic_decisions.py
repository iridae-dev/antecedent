# SPDX-License-Identifier: MIT OR Apache-2.0
"""Decisions on cyclic equilibrium models: paired action worlds, fitted feedback and bundles.

Every route here decides between hard-clamp action worlds of one equilibrium model. Each action
reads the same disturbance row and, when the model is fitted, the same posterior parameter
vector, so the actions are compared on one paired joint law:

* :func:`compile_action_world_decision` reads the worlds and the explicit shared disturbance
  rows of a specified :class:`~antecedent.cyclic.PreparedCyclicSystem`;
* :func:`compile_posterior_decision` samples the parameters of a system prepared with
  ``parameters=`` and ``rows=`` jointly from those training rows, pairs each retained parameter
  vector with one disturbance drawn from the model covariance and, with ``target=``, with one
  row of a :class:`TargetFrame` of exogenous baselines;
* :func:`compile_component_decision` fits components whose ports and couplings
  :func:`prepare_component_feedback` checked, from rows shared by the whole connected system;
  with ``logistic=`` the actions read the logistic transform of each fitted output.

A compiled decision binds the seed of the execution it was compiled under. ``execute`` fits,
solves every world with a uniqueness certificate and evaluates the contract; ``export``
re-executes before sealing a portable artifact, and :func:`consume_decision` recompiles,
refits, re-solves and re-decides it independently under the program identity the consumer
retained. :func:`compile_bundle_recomputation` checks that the propagation-chain terminal of a
composition bundle is the chain a fitted posterior regenerates from its raw training rows, and
:func:`consume_bundle_refitting` consumes a bundle while refitting every native fit node it
embeds.

Answers are conditional on the specified equations, the disturbance covariance, the likelihood
and the prior. No interval, coverage or decision reliability is claimed, and ``calibration`` is
``not_measured``. Refusals keep the shared causal error classes: a refused declaration or run is
a :class:`~antecedent.errors.CausalValueError`, a replay that does not reproduce under the
retained identity a :class:`~antecedent.errors.CausalSerializationError`, and cancellation and
resource refusals keep their own classes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    container_bytes,
)
from ._cyclic_typed import _limits, _with_execution
from ._freeze import freeze
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._unit_support import read_unit_support
from .decision import ActionOutcome, ConstraintExclusion, Contract, Verdict
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .interface_contract import InterfaceContract
from .joint_distribution import ScientificQuantity

if TYPE_CHECKING:
    from .composition_bundle import Bundle, BundleBuilder, ConsumedBundle, SuppliedSources
    from .cyclic import Equation, PreparedCyclicSystem, SolvePolicy
    from .evidence_ledger import EvidenceLedger

__all__ = [
    "BundleDependency",
    "BundleRecomputationReceipt",
    "CompiledBundleRecomputation",
    "CompiledCyclicDecision",
    "CyclicDecision",
    "CyclicDecisionResult",
    "DecisionOutput",
    "FeedbackComponent",
    "FeedbackConnection",
    "FeedbackOutput",
    "LogisticOutputs",
    "MultiOutputComponent",
    "PreparedComponentFeedback",
    "TargetFrame",
    "add_posterior_fit",
    "compile_action_world_decision",
    "compile_bundle_recomputation",
    "compile_component_decision",
    "compile_posterior_decision",
    "consume_bundle_recomputation",
    "consume_bundle_refitting",
    "consume_decision",
    "prepare_component_feedback",
]


# --------------------------------------------------------------------------- helpers


def read_unit_support(handle: Any) -> dict[str, Any] | None:
    """The support decision of the unit a compiled cyclic program runs under.

    Returns ``{"unit", "status", "supported", "missing_evidence"}``: ``unit`` is the capability
    record id, ``supported`` whether that unit is supported and ``missing_evidence`` what an
    unsupported unit still lacks. ``None`` while no unit is bound to the program's route, or
    while the handle holds no compiled program. Reading it changes no program identity.
    """
    if handle is None:
        return None
    raw = handle.unit_support()
    return None if raw is None else dict(json.loads(raw))


def _dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False)


def _controls(execution: Execution) -> dict[str, Any]:
    """The seed, byte limit and cancellation token a native cyclic decision entry takes."""
    return {"seed": execution.seed, **_limits(execution)}


def _index(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalValueError(f"{what} must be a non-negative integer")
    return value


def _real(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{what} must be a real number")
    if not math.isfinite(value):
        raise CausalValueError(f"{what} must be finite")
    return float(value)


def _reals(values: Sequence[Any], what: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of real numbers")
    return [_real(value, what) for value in values]


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a nonempty string")
    return value


def _equation_wire(equation: Equation, what: str) -> dict[str, Any]:
    from .cyclic import Equation

    if not isinstance(equation, Equation):
        raise CausalTypeError(f"{what} must be an antecedent.cyclic.Equation")
    return asdict(equation)


def _system_handle(system: PreparedCyclicSystem) -> Any:
    from .cyclic import PreparedCyclicSystem

    if not isinstance(system, PreparedCyclicSystem):
        raise CausalTypeError("system must be an antecedent.cyclic.PreparedCyclicSystem")
    return system._handle


def _contract_json(contract: Contract) -> str:
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be an antecedent.decision.Contract")
    return _dumps(contract._wire())


def _outputs_json(outputs: Sequence[DecisionOutput]) -> str:
    if isinstance(outputs, (str, bytes)):
        raise CausalTypeError("outputs must be a sequence of DecisionOutput")
    items = tuple(outputs)
    if any(not isinstance(item, DecisionOutput) for item in items):
        raise CausalTypeError("outputs must hold only DecisionOutput values")
    return _dumps([item._wire() for item in items])


def _producer_identity(producer: Any, identity: str | None) -> str:
    if identity is not None:
        return _text(identity, "producer_identity")
    expectation = getattr(producer, "expectation", None)
    if isinstance(producer, ExportedArtifact) and isinstance(expectation, ArtifactExpectation):
        return str(expectation.identity)
    raise CausalValueError(
        "producer_identity is required for producer bytes: the program identity of the "
        "posterior decision that exported them"
    )


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class DecisionOutput:
    """One action world and one model coordinate, published as a scientific quantity.

    ``world`` indexes the worlds of the prepared system (or of the connected components),
    ``equation`` the dense equation (or sorted component output, see
    :attr:`PreparedComponentFeedback.component_order`) whose solved value the action reads, and
    ``quantity`` is the exact outcome coordinate: its regime names the world, and its population
    and horizon are the decision contract's. Each action of the contract reads exactly one
    output, and every world is read by exactly one action.
    """

    world: int
    equation: int
    quantity: ScientificQuantity

    def __post_init__(self) -> None:
        _index(self.world, "DecisionOutput.world")
        _index(self.equation, "DecisionOutput.equation")
        if not isinstance(self.quantity, ScientificQuantity):
            raise CausalTypeError("DecisionOutput.quantity must be a ScientificQuantity")

    def _wire(self) -> dict[str, Any]:
        return {"world": self.world, "equation": self.equation, "quantity": self.quantity._wire()}


@dataclass(frozen=True, slots=True)
class TargetFrame:
    """Independent target units with complete values of selected exogenous baselines.

    ``baseline_indices`` are the ascending dense indices of the only mechanisms whose
    distribution may differ between the source and the target population; each must be an
    exogenous, unfitted mechanism clamped to its observed value in every training row, with no
    disturbance covariance with the feedback mechanisms. ``rows`` hold one baseline vector per
    unit of ``unit_ids``, inside the observed source range. The mechanisms and the disturbance law
    are a declared invariance premise across populations, not a transport theorem.
    """

    source_sample: str
    target_sample: str
    target_population: str
    baseline_indices: tuple[int, ...]
    unit_ids: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    def _wire(self) -> dict[str, Any]:
        return {
            "source_sample": _text(self.source_sample, "TargetFrame.source_sample"),
            "target_sample": _text(self.target_sample, "TargetFrame.target_sample"),
            "target_population": _text(self.target_population, "TargetFrame.target_population"),
            "baseline_indices": [
                _index(index, "TargetFrame.baseline_indices") for index in self.baseline_indices
            ],
            "unit_ids": [_text(unit, "TargetFrame.unit_ids") for unit in self.unit_ids],
            "rows": [_reals(row, "TargetFrame.rows") for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class FeedbackComponent:
    """A scalar mechanism with a checked scientific interface.

    ``interface`` produces exactly one port, the component's endogenous value, and declares
    every port it consumes. ``equation`` is the local scalar mechanism: its single weight is the
    self coefficient, and every incoming coefficient comes from a :class:`FeedbackConnection`.
    """

    interface: InterfaceContract
    equation: Equation

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.interface, InterfaceContract):
            raise CausalTypeError("FeedbackComponent.interface must be an InterfaceContract")
        return {
            "interface": self.interface._wire(),
            "equation": _equation_wire(self.equation, "FeedbackComponent.equation"),
        }


@dataclass(frozen=True, slots=True)
class FeedbackOutput:
    """One output equation of a multi-output component.

    ``produced_port`` names the produced port the equation computes, ``equation`` is its local
    scalar mechanism (one self weight) and ``consumed_ports`` the consumed ports it reads.
    """

    produced_port: str
    equation: Equation
    consumed_ports: tuple[str, ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "produced_port": _text(self.produced_port, "FeedbackOutput.produced_port"),
            "equation": _equation_wire(self.equation, "FeedbackOutput.equation"),
            "consumed_ports": [
                _text(port, "FeedbackOutput.consumed_ports") for port in self.consumed_ports
            ],
        }


@dataclass(frozen=True, slots=True)
class MultiOutputComponent:
    """A component with several produced ports, one :class:`FeedbackOutput` per produced port.

    Each output becomes its own coordinate ``component::port`` of the connected system; every
    consumed port is read by at least one output.
    """

    interface: InterfaceContract
    outputs: tuple[FeedbackOutput, ...]

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.interface, InterfaceContract):
            raise CausalTypeError("MultiOutputComponent.interface must be an InterfaceContract")
        if any(not isinstance(output, FeedbackOutput) for output in self.outputs):
            raise CausalTypeError("MultiOutputComponent.outputs must hold FeedbackOutput values")
        return {
            "interface": self.interface._wire(),
            "outputs": [output._wire() for output in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class FeedbackConnection:
    """A checked producer-to-consumer port and its structural coefficient.

    ``coefficient`` is the weight of the producer's value in the consumer's equation; it must be
    finite and nonzero. Between multi-output components, ``consumer_output`` names the produced
    port of the consumer whose output equation reads the connection.
    """

    producer: str
    produced_port: str
    consumer: str
    consumed_port: str
    coefficient: float
    consumer_output: str | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "producer": _text(self.producer, "FeedbackConnection.producer"),
            "produced_port": _text(self.produced_port, "FeedbackConnection.produced_port"),
            "consumer": _text(self.consumer, "FeedbackConnection.consumer"),
            "consumed_port": _text(self.consumed_port, "FeedbackConnection.consumed_port"),
            "coefficient": _real(self.coefficient, "FeedbackConnection.coefficient"),
            "consumer_output": None
            if self.consumer_output is None
            else _text(self.consumer_output, "FeedbackConnection.consumer_output"),
        }


@dataclass(frozen=True, slots=True)
class LogisticOutputs:
    """Decide on the logistic transform of each fitted output of a multi-output decision.

    ``quantities`` holds, in output order, the logistic coordinate of each real output: the
    same variable, regime, population, horizon and functional, with units ``"dimensionless"``
    and transform ``"logistic"``. ``contract`` is the decision read on those coordinates. The
    structural ports and Gaussian disturbances stay real-valued; only the predictive outputs
    are transformed after fitting.
    """

    quantities: tuple[ScientificQuantity, ...]
    contract: Contract

    def _wire(self) -> tuple[str, str]:
        if any(not isinstance(item, ScientificQuantity) for item in self.quantities):
            raise CausalTypeError("LogisticOutputs.quantities must hold ScientificQuantity values")
        return (
            _dumps([quantity._wire() for quantity in self.quantities]),
            _contract_json(self.contract),
        )


# --------------------------------------------------------------------------- feedback components


class PreparedComponentFeedback:
    """Feedback components checked port by port and prepared as one connected cyclic program.

    :attr:`system` is the connected system as an ordinary
    :class:`~antecedent.cyclic.PreparedCyclicSystem`: execute it, read its contract or check it
    in a composition like any prepared system. :attr:`component_order` is the dense coordinate
    order of the system (``component``, or ``component::port`` for multi-output components),
    which the disturbance covariance, noise rows, initial state, training rows and
    :class:`DecisionOutput` equations use. No component calibration or evidence label is
    promoted to the connected equilibrium.
    """

    __slots__ = ("_native", "execution")

    def __init__(self, native: Any, execution: Execution) -> None:
        self._native = native
        self.execution = execution

    def __repr__(self) -> str:
        return (
            f"PreparedComponentFeedback(components={self.component_order!r}, "
            f"identity={self.identity[:12]!r})"
        )

    @property
    def component_order(self) -> tuple[str, ...]:
        """Dense coordinate order of the connected system."""
        return tuple(self._native.component_order())

    @property
    def composition_identity(self) -> str:
        """Identity of the checked interfaces and structural couplings (lowercase hex)."""
        return str(self._native.composition_identity())

    @property
    def identity(self) -> str:
        """Identity of the connected six-layer cyclic program (lowercase hex)."""
        return str(self._native.identity())

    @property
    def multi_output(self) -> bool:
        """Whether the components declare several output equations each."""
        return bool(self._native.multi_output())

    @property
    def system(self) -> PreparedCyclicSystem:
        """The connected system as a prepared cyclic system, under this execution policy."""
        from .cyclic import PreparedCyclicSystem

        result = object.__new__(PreparedCyclicSystem)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(result, "_handle", self._native.system())
        object.__setattr__(result, "execution", self.execution)
        return result


def _feedback_json(
    components: Sequence[FeedbackComponent | MultiOutputComponent],
    connections: Sequence[FeedbackConnection],
    noise_covariance: Sequence[float],
    worlds: Sequence[Mapping[str, float]] | None,
    noise: Sequence[Sequence[float]] | None,
    initial: Sequence[float] | None,
    policy: SolvePolicy | None,
    factual: Sequence[float] | None,
) -> str:
    from .cyclic import SolvePolicy

    if isinstance(components, (str, bytes)) or isinstance(connections, (str, bytes)):
        raise CausalTypeError("components and connections must be sequences")
    parts = tuple(components)
    if any(not isinstance(part, (FeedbackComponent, MultiOutputComponent)) for part in parts):
        raise CausalTypeError(
            "components must hold FeedbackComponent or MultiOutputComponent values"
        )
    couplings = tuple(connections)
    if any(not isinstance(coupling, FeedbackConnection) for coupling in couplings):
        raise CausalTypeError("connections must hold FeedbackConnection values")
    order = sum(
        len(part.outputs) if isinstance(part, MultiOutputComponent) else 1 for part in parts
    )
    if policy is not None and not isinstance(policy, SolvePolicy):
        raise CausalTypeError("policy must be an antecedent.cyclic.SolvePolicy")
    declared_worlds: list[Mapping[str, float]] = [{}] if worlds is None else list(worlds)
    if any(not isinstance(world, Mapping) for world in declared_worlds):
        raise CausalTypeError("each world maps a component coordinate to its clamped value")
    return _dumps(
        {
            "components": [part._wire() for part in parts],
            "connections": [coupling._wire() for coupling in couplings],
            "noise_covariance": _reals(noise_covariance, "noise_covariance"),
            "noise": [[0.0] * order] if noise is None else [_reals(row, "noise") for row in noise],
            "initial": [0.0] * order if initial is None else _reals(initial, "initial"),
            "worlds": [
                [
                    [_text(name, "world coordinate"), _real(value, "world clamp")]
                    for name, value in world.items()
                ]
                for world in declared_worlds
            ],
            "policy": asdict(policy or SolvePolicy()),
            "factual": None if factual is None else _reals(factual, "factual"),
        }
    )


@_with_execution
def prepare_component_feedback(
    components: Sequence[FeedbackComponent | MultiOutputComponent],
    connections: Sequence[FeedbackConnection],
    *,
    noise_covariance: Sequence[float],
    worlds: Sequence[Mapping[str, float]] | None = None,
    noise: Sequence[Sequence[float]] | None = None,
    initial: Sequence[float] | None = None,
    policy: SolvePolicy | None = None,
    factual: Sequence[float] | None = None,
    ledger: EvidenceLedger | None = None,
    execution: Execution | None = None,
) -> PreparedComponentFeedback:
    """Check every port and coupling, then prepare the components as one cyclic program.

    ``components`` are all :class:`FeedbackComponent` (one scalar mechanism each) or all
    :class:`MultiOutputComponent` (one output equation per produced port). Each
    :class:`FeedbackConnection` is checked against both interfaces before the equations are
    joined, and every consumed port needs exactly one connection. The connected system must
    carry a global contraction certificate (``policy.require_contraction``), so feedback that
    crosses component boundaries is solved as a cycle with a unique equilibrium; no feed-forward
    order or unrolling is used.

    ``noise_covariance`` is the full row-major disturbance covariance in
    :attr:`PreparedComponentFeedback.component_order`. ``worlds`` maps a coordinate name to its
    hard clamp, one mapping per world (default: one unclamped world). ``noise`` holds aligned
    disturbance rows shared by every world (default: one zero row, the placeholder a fitted
    :func:`compile_component_decision` requires), ``initial`` the numerical starting state
    (default zeros) and ``factual`` an optional observed state for shared-noise counterfactuals.

    ``ledger`` (an :class:`~antecedent.evidence_ledger.EvidenceLedger`) proves that consumed
    ports requiring disjoint evidence are fed by producers whose declared sources are
    independent; several such ports without a ledger are refused. The ledger state enters the
    program identity.

    Raises:
        CausalTypeError: a component, connection, world or policy has the wrong type.
        CausalValueError: a port check, coupling, shape, contraction or ledger refusal.
    """
    assert execution is not None
    handle = None
    if ledger is not None:
        from .evidence_ledger import EvidenceLedger

        if not isinstance(ledger, EvidenceLedger):
            raise CausalTypeError("ledger must be an antecedent.evidence_ledger.EvidenceLedger")
        handle = ledger._handle
    declaration = _feedback_json(
        components, connections, noise_covariance, worlds, noise, initial, policy, factual
    )
    native = _native.cyclic_component_feedback_prepare(declaration, handle, **_controls(execution))
    return PreparedComponentFeedback(native, execution)


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True, slots=True)
class CyclicDecision:
    """A decision over paired action worlds and the joint law it read.

    ``verdict`` and ``outcomes`` are the :mod:`antecedent.decision` readings of the contract;
    ``draws`` counts the paired rows of the law, ``chain_identity`` names the propagation chain
    that carried them and ``program_identity`` the chain-decision program that evaluated them.
    A sampled ranking carries no statistical optimality guarantee.
    """

    verdict: Verdict
    outcomes: tuple[ActionOutcome, ...]
    criterion: str
    contract_identity: str
    chain_identity: str
    program_identity: str
    evpi: float | None
    draws: int
    effective_draws: float
    assumptions: tuple[str, ...]


def _number(value: Any) -> float:
    return math.nan if value is None else float(value)


def _decision(wire: Mapping[str, Any]) -> CyclicDecision:
    verdict = wire["verdict"]
    return CyclicDecision(
        verdict=Verdict(verdict["kind"], tuple(verdict["actions"])),
        outcomes=tuple(
            ActionOutcome(
                id=outcome["id"],
                admissible=outcome["admissible"],
                exclusions=tuple(
                    ConstraintExclusion(
                        exclusion["constraint_id"],
                        _number(exclusion["probability"]),
                        _number(exclusion["required"]),
                    )
                    for exclusion in outcome["exclusions"]
                ),
                expected_utility=_number(outcome["expected_utility"]),
                value=_number(outcome["value"]),
                standard_error=outcome["standard_error"],
                expected_regret=outcome["expected_regret"],
                max_regret=outcome["max_regret"],
            )
            for outcome in wire["outcomes"]
        ),
        criterion=wire["criterion"],
        contract_identity=wire["contract_identity"],
        chain_identity=wire["chain_identity"],
        program_identity=wire["program_identity"],
        evpi=wire["evpi"],
        draws=int(wire["draws"]),
        effective_draws=_number(wire["effective_draws"]),
        assumptions=tuple(wire["assumptions"]),
    )


@dataclass(frozen=True, eq=False)
class CyclicDecisionResult(ScientificResult):
    """A fitted or specified cyclic decision, executed or independently consumed.

    ``kind`` names the route its artifact declares and ``identity`` the complete compiled
    program. ``decision`` is the decision on the paired worlds; for a logistic-output decision,
    ``real_valued_decision`` is the decision the untransformed fit made. ``posterior`` holds
    the chain counts and convergence diagnostics of the jointly fitted parameters (``None`` for
    a specified model); the draws are :meth:`posterior_samples`. ``calibration`` is always
    ``not_measured``.
    """

    kind: str
    identity: str
    decision: CyclicDecision
    calibration: str
    posterior: Mapping[str, Any] | None = None
    real_valued_decision: CyclicDecision | None = None
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _native: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "posterior", freeze(self.posterior))

    @classmethod
    def _from_native(cls, native: Any, execution: Execution) -> CyclicDecisionResult:
        wire = json.loads(native.summary_json())
        real = wire["real_valued_decision"]
        return cls(
            kind=wire["kind"],
            identity=wire["identity"],
            decision=_decision(wire["decision"]),
            calibration=wire["calibration"],
            posterior=wire["posterior"],
            real_valued_decision=None if real is None else _decision(real),
            execution=execution,
            _native=native,
        )

    def to_dict(self) -> dict[str, Any]:
        """The native summary: kind, identity, decisions, posterior diagnostics, calibration."""
        summary: dict[str, Any] = json.loads(self._native.summary_json())
        return summary

    def draws(self) -> tuple[tuple[float, ...], ...]:
        """The paired outcome columns the decision read, one per action world, row-aligned."""
        return tuple(tuple(column) for column in self._native.draws())

    def posterior_samples(self) -> tuple[tuple[float, ...], ...] | None:
        """The retained parameter vectors, chain-major; ``None`` for a specified model."""
        samples = self._native.posterior_samples()
        return None if samples is None else tuple(tuple(row) for row in samples)

    def chain_artifact(self) -> bytes:
        """The propagation chain the decision read, as artifact bytes a bundle can embed."""
        return bytes(self._native.chain_artifact())

    @_with_execution
    def export(
        self, *, artifact_id: str = "cyclic-decision", execution: Execution | None = None
    ) -> ExportedArtifact[CyclicDecisionResult]:
        """Re-execute, require the same result and seal it as a portable artifact.

        The program re-runs under the seed it was compiled with; a consumed result returns the
        bytes it was consumed from. The artifact's expectation is the program identity a
        consumer retains for :func:`consume_decision`.

        Raises:
            CausalValueError: the result no longer reproduces, or the artifact exceeds its bound.
        """
        assert execution is not None
        data = bytes(self._native.export(_text(artifact_id, "artifact_id"), **_limits(execution)))
        return ExportedArtifact(
            data, self.kind, ArtifactExpectation(self.kind, self.identity), {"kind": self.kind}
        )


class CompiledCyclicDecision:
    """A compiled cyclic decision: one six-layer program and the seed it binds.

    Compilation checks the worlds, the output coordinates, the posterior schedule and the
    contract before anything is fitted or solved. :attr:`identity` is the program identity a
    consumer retains; :attr:`route` the compiled route name. ``execute`` always runs under the
    seed bound at compilation, whatever seed a later execution policy names.
    """

    __slots__ = ("_native", "execution")

    def __init__(self, native: Any, execution: Execution) -> None:
        self._native = native
        self.execution = execution

    def __repr__(self) -> str:
        return f"CompiledCyclicDecision(route={self.route!r}, identity={self.identity[:12]!r})"

    @property
    def identity(self) -> str:
        """Identity of the complete compiled program (lowercase hex)."""
        return str(self._native.identity())

    @property
    def route(self) -> str:
        """The compiled route name, for example ``CyclicPosteriorActionDecision``."""
        return str(self._native.route())

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the capability unit this route runs under.

        ``{"unit", "status", "supported", "missing_evidence"}``, or ``None`` while no unit is
        bound to the route. Reading it never changes the program identity.
        """
        return read_unit_support(self._native)

    @property
    def kind(self) -> str:
        """The feature the portable artifact declares."""
        return str(self._native.kind())

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this compiled program runs under.

        ``{"unit", "supported", "missing_evidence"}``, read from the program's route; ``None``
        while no unit is bound to it.
        """
        return read_unit_support(self._native)

    @_with_execution
    def execute(self, *, execution: Execution | None = None) -> CyclicDecisionResult:
        """Fit, solve every paired world with a uniqueness certificate and decide.

        Raises:
            CausalValueError: a failed posterior diagnostic, an uncertified equilibrium or a
                refused decision.
        """
        assert execution is not None
        native = self._native.execute(**_limits(execution))
        return CyclicDecisionResult._from_native(native, execution)


@_with_execution
def compile_action_world_decision(
    system: PreparedCyclicSystem,
    outputs: Sequence[DecisionOutput],
    contract: Contract,
    *,
    execution: Execution | None = None,
) -> CompiledCyclicDecision:
    """Compile paired hard-clamp action worlds of a specified cyclic system and a contract.

    ``system`` is a :class:`~antecedent.cyclic.PreparedCyclicSystem` without a fit or factual
    state, with at least two disturbance rows and one distinct, nonempty hard-clamp world per
    action. Every world solves the same rows, so the joint law is the empirical disturbance
    scenario law conditional on the model, not a population predictive.

    Raises:
        CausalTypeError: ``system``, ``outputs`` or ``contract`` has the wrong type.
        CausalValueError: unpaired worlds, a repeated regime or clamp set, a mismatched output
            coordinate or an action that does not read exactly one world.
    """
    assert execution is not None
    native = _native.cyclic_action_world_decision_compile(
        _system_handle(system),
        _outputs_json(outputs),
        _contract_json(contract),
        **_controls(execution),
    )
    return CompiledCyclicDecision(native, execution)


@_with_execution
def compile_posterior_decision(
    system: PreparedCyclicSystem,
    outputs: Sequence[DecisionOutput],
    contract: Contract,
    *,
    prior_sd: float,
    chains: int = 4,
    warmup: int = 500,
    draws: int = 1000,
    proposal_scale: float = 0.05,
    target: TargetFrame | None = None,
    execution: Execution | None = None,
) -> CompiledCyclicDecision:
    """Compile a jointly fitted cyclic posterior, its paired action worlds and a contract.

    ``system`` is a :class:`~antecedent.cyclic.PreparedCyclicSystem` prepared with
    ``parameters=`` and ``rows=``: its model, fitted coordinates, intervention-labelled training
    rows, worlds, initial state and solve policy define the program (its aligned noise rows are
    not read). The parameters are sampled by random-walk Metropolis under an independent normal
    prior of scale ``prior_sd`` (``chains`` chains, ``warmup`` discarded and ``draws`` retained
    draws per chain); each retained vector is paired with one disturbance from the model
    covariance and every world solves that same pair. Failed diagnostics refuse the decision.

    With ``target`` (a :class:`TargetFrame`), each pair also reads one target row whose
    baselines are clamped in every world, so the decision is standardized to the target frame.
    The prior, likelihood, seed, schedule and target rows are all bound into the identity.

    Raises:
        CausalTypeError: an argument has the wrong type.
        CausalValueError: a system without a fit, an invalid schedule, worlds or coordinates,
            or an invalid target frame.
    """
    assert execution is not None
    if target is not None and not isinstance(target, TargetFrame):
        raise CausalTypeError("target must be a TargetFrame")
    native = _native.cyclic_posterior_decision_compile(
        _system_handle(system),
        _outputs_json(outputs),
        _contract_json(contract),
        _real(prior_sd, "prior_sd"),
        _index(chains, "chains"),
        _index(warmup, "warmup"),
        _index(draws, "draws"),
        _real(proposal_scale, "proposal_scale"),
        None if target is None else _dumps(target._wire()),
        **_controls(execution),
    )
    return CompiledCyclicDecision(native, execution)


@_with_execution
def compile_component_decision(
    feedback: PreparedComponentFeedback,
    outputs: Sequence[DecisionOutput],
    contract: Contract,
    *,
    parameters: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    prior_sd: float,
    chains: int = 4,
    warmup: int = 500,
    draws: int = 1000,
    proposal_scale: float = 0.05,
    logistic: LogisticOutputs | None = None,
    execution: Execution | None = None,
) -> CompiledCyclicDecision:
    """Compile checked feedback components, one shared joint fit and paired action worlds.

    ``feedback`` was prepared by :func:`prepare_component_feedback` with one zero noise row and
    no factual state; its worlds are the action worlds. ``parameters`` are the jointly fitted
    coordinates in :attr:`PreparedComponentFeedback.component_order` (for example
    ``{"kind": "weight", "equation": 1, "parent": 0}``) and ``rows`` the intervention-labelled
    training rows of the whole system (``{"values": [...], "clamps": [...]}``); the sampler
    schedule is the one of :func:`compile_posterior_decision`. Each output coordinate must be
    its component port's coordinate with the world's regime, and every port must have real
    support. The component declarations are re-checked from their own wire on replay; a
    ledger is not carried into this route.

    With ``logistic`` (a :class:`LogisticOutputs`, multi-output components only) the actions
    read the logistic transform of each fitted output instead.

    Raises:
        CausalTypeError: an argument has the wrong type.
        CausalValueError: a refused component binding, schedule, coordinate or contract.
    """
    assert execution is not None
    if not isinstance(feedback, PreparedComponentFeedback):
        raise CausalTypeError("feedback must be a PreparedComponentFeedback")
    if logistic is not None and not isinstance(logistic, LogisticOutputs):
        raise CausalTypeError("logistic must be a LogisticOutputs")
    if isinstance(parameters, (str, bytes)) or isinstance(rows, (str, bytes)):
        raise CausalTypeError("parameters and rows must be sequences of mappings")
    schedule = {
        "parameters": [dict(parameter) for parameter in parameters],
        "training": [dict(row) for row in rows],
        "prior_sd": _real(prior_sd, "prior_sd"),
        "chains": _index(chains, "chains"),
        "warmup": _index(warmup, "warmup"),
        "draws_per_chain": _index(draws, "draws"),
        "proposal_scale": _real(proposal_scale, "proposal_scale"),
    }
    native = _native.cyclic_component_decision_compile(
        feedback._native,
        _dumps(schedule),
        _outputs_json(outputs),
        _contract_json(contract),
        **_controls(execution),
    )
    if logistic is not None:
        native = native.logistic(*logistic._wire())
    return CompiledCyclicDecision(native, execution)


@_with_execution
@artifact_consumer(
    proof=(
        "cyclic_decisions.consume_decision: independent native recompilation, refit, "
        "certified re-solve and decision replay under the retained program identity"
    ),
    replayable=True,
)
def consume_decision(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> CyclicDecisionResult:
    """Independently recompile, refit, re-solve and re-decide a portable cyclic decision.

    ``expected_identity`` is the program identity the consumer retained apart from the bytes
    (:attr:`CompiledCyclicDecision.identity`). A fitted decision replays only under the seed it
    was compiled with, so ``execution`` must carry the producer's seed.

    Raises:
        CausalSerializationError: a foreign or altered artifact, a different identity or seed,
            or a fit, solve or decision that does not reproduce.
    """
    assert execution is not None
    native = _native.cyclic_decision_consume(
        container_bytes(artifact),
        _text(expected_identity, "expected_identity"),
        **_controls(execution),
    )
    return CyclicDecisionResult._from_native(native, execution)


# --------------------------------------------------------------------------- bundles


@dataclass(frozen=True, slots=True)
class BundleDependency:
    """One embedded bundle dependency, with its replayed identity and Merkle digest."""

    node_id: str
    artifact_identity: str
    chain_digest: str


@dataclass(frozen=True, eq=True)
class BundleRecomputationReceipt(ScientificResult):
    """The receipt of a bundle whose terminal chain a refitted cyclic posterior regenerated.

    ``identity`` is the complete bundle-and-producer program, ``producer_program_identity`` the
    fitted posterior's program and ``chain_program_identity`` the program that executed the
    terminal chain. ``dependencies`` lists every embedded node the bundle consumer replayed.
    Equal receipts mean the same graph, producer, terminal and replays.
    """

    kind: str
    identity: str
    producer_program_identity: str
    bundle_identity: str
    terminal_node: str
    terminal_identity: str
    chain_program_identity: str
    dependencies: tuple[BundleDependency, ...]
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _native: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Every field is immutable as built; the execution policy is retained as given.
        return

    @classmethod
    def _from_native(cls, native: Any, execution: Execution) -> BundleRecomputationReceipt:
        wire = json.loads(native.receipt_json())
        return cls(
            kind=wire["kind"],
            identity=wire["identity"],
            producer_program_identity=wire["producer_program_identity"],
            bundle_identity=wire["bundle_identity"],
            terminal_node=wire["terminal_node"],
            terminal_identity=wire["terminal_identity"],
            chain_program_identity=wire["chain_program_identity"],
            dependencies=tuple(
                BundleDependency(item["node_id"], item["artifact_identity"], item["chain_digest"])
                for item in wire["dependencies"]
            ),
            execution=execution,
            _native=native,
        )

    def to_dict(self) -> dict[str, Any]:
        """The receipt's identities and replayed dependencies."""
        receipt: dict[str, Any] = json.loads(self._native.receipt_json())
        return receipt

    @_with_execution
    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[BundleRecomputationReceipt]:
        """Re-execute, require this receipt and seal the graph, producer and receipt.

        A consumed receipt returns the bytes it was consumed from.

        Raises:
            CausalValueError: the recomputation no longer reproduces this receipt.
        """
        assert execution is not None
        data = bytes(self._native.export(**_limits(execution)))
        return ExportedArtifact(
            data,
            self.kind,
            ArtifactExpectation(self.kind, self.identity),
            {"bundle_identity": self.bundle_identity},
        )


class CompiledBundleRecomputation:
    """A compiled bundle recomputation: the graph, its cyclic producer and the bound seed.

    Compilation decodes and compiles the terminal chain and binds the producer without
    fitting it; ``execute`` refits the producer from its raw training rows and requires its
    fresh chain to be the bundle's terminal.
    """

    __slots__ = ("_native", "execution")

    def __init__(self, native: Any, execution: Execution) -> None:
        self._native = native
        self.execution = execution

    def __repr__(self) -> str:
        return f"CompiledBundleRecomputation(kind={self.kind!r}, identity={self.identity[:12]!r})"

    @property
    def identity(self) -> str:
        """Identity of the complete bundle-and-producer program (lowercase hex)."""
        return str(self._native.identity())

    @property
    def kind(self) -> str:
        """The feature the portable artifact declares."""
        return str(self._native.kind())

    @_with_execution
    def execute(self, *, execution: Execution | None = None) -> BundleRecomputationReceipt:
        """Refit the producer, execute the terminal chain and require both to agree.

        Raises:
            CausalValueError: changed raw evidence, posterior, generated chain or dependency.
        """
        assert execution is not None
        native = self._native.execute(**_limits(execution))
        return BundleRecomputationReceipt._from_native(native, execution)


@_with_execution
def add_posterior_fit(
    builder: BundleBuilder,
    node_id: str,
    producer: bytes | ExportedArtifact[Any],
    *,
    producer_identity: str | None = None,
    execution: Execution | None = None,
) -> BundleBuilder:
    """Embed a fitted cyclic posterior decision as a bundle node that consumers refit.

    ``producer`` is the artifact a posterior :class:`CyclicDecisionResult` exported (from
    :func:`compile_posterior_decision` without a target) and ``producer_identity`` its program
    identity (read from an :class:`~antecedent.artifacts.ExportedArtifact` when omitted). The
    producer is recompiled and refitted under this execution's seed before the node is
    accepted, and the seed is recorded so later consumers refit it identically. Connect the
    node to the propagation-chain terminal it produced.

    Raises:
        CausalTypeError: ``builder`` is not a composition bundle builder.
        CausalValueError: the producer does not replay, or the id is blank or repeated.
    """
    from .composition_bundle import BundleBuilder

    assert execution is not None
    if not isinstance(builder, BundleBuilder):
        raise CausalTypeError("builder must be an antecedent.composition_bundle.BundleBuilder")
    identity = _producer_identity(producer, producer_identity)
    node = _text(node_id, "node_id")
    _native.cyclic_bundle_add_posterior_fit(
        builder._native, node, container_bytes(producer), identity, **_controls(execution)
    )
    builder._ids.append(node)
    builder._last = node
    return builder


@_with_execution
def compile_bundle_recomputation(
    bundle: Bundle,
    terminal: str,
    *,
    producer: bytes | ExportedArtifact[Any] | None = None,
    producer_identity: str | None = None,
    producer_node: str | None = None,
    execution: Execution | None = None,
) -> CompiledBundleRecomputation:
    """Compile a bundle whose propagation-chain terminal a cyclic posterior regenerates.

    ``terminal`` names an embedded ``propagation_chain`` node of ``bundle``. The producer is
    either ``producer`` (a posterior decision artifact) with ``producer_identity`` (read from
    an :class:`~antecedent.artifacts.ExportedArtifact` when omitted), or ``producer_node``, a
    fit node of the bundle (see :func:`add_posterior_fit`) with a direct edge to the terminal,
    which the bundle consumer refits. Compilation does not fit.

    Raises:
        CausalTypeError: ``bundle`` is not a composition bundle.
        CausalValueError: both or neither producer form, a missing or non-chain terminal, a
            missing edge or a refused chain.
    """
    from .composition_bundle import Bundle

    assert execution is not None
    if not isinstance(bundle, Bundle):
        raise CausalTypeError("bundle must be an antecedent.composition_bundle.Bundle")
    node = _text(terminal, "terminal")
    if (producer is None) == (producer_node is None):
        raise CausalValueError("name either the producer artifact or the producer node")
    if producer_node is not None:
        if producer_identity is not None:
            raise CausalValueError("producer_identity belongs to a producer artifact")
        native = _native.cyclic_bundle_compile(
            bundle._native,
            node,
            None,
            None,
            _text(producer_node, "producer_node"),
            **_controls(execution),
        )
    else:
        native = _native.cyclic_bundle_compile(
            bundle._native,
            node,
            container_bytes(producer),
            _producer_identity(producer, producer_identity),
            None,
            **_controls(execution),
        )
    return CompiledBundleRecomputation(native, execution)


@_with_execution
@artifact_consumer(
    proof=(
        "cyclic_decisions.consume_bundle_recomputation: independent bundle reconstruction, "
        "recompilation, producer refit and receipt replay under the retained identities"
    ),
    replayable=True,
    identity=("expected_bundle", "expected_identity"),
)
def consume_bundle_recomputation(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_bundle: str,
    expected_identity: str,
    execution: Execution | None = None,
) -> BundleRecomputationReceipt:
    """Independently reconstruct, recompile and refit a portable bundle recomputation.

    ``expected_bundle`` is the bundle identity and ``expected_identity`` the compiled program
    identity, both retained apart from the bytes; ``execution`` carries the producer's seed.

    Raises:
        CausalSerializationError: a changed graph, program, producer evidence, fitted terminal
            or receipt.
    """
    assert execution is not None
    native = _native.cyclic_bundle_consume(
        container_bytes(artifact),
        _text(expected_bundle, "expected_bundle"),
        _text(expected_identity, "expected_identity"),
        **_controls(execution),
    )
    return BundleRecomputationReceipt._from_native(native, execution)


@_with_execution
def consume_bundle_refitting(
    data: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    supplied: SuppliedSources | None = None,
    route: Literal["native_fit", "cyclic_posterior"] = "native_fit",
    execution: Execution | None = None,
) -> ConsumedBundle:
    """Consume a composition bundle, refitting every native fit node it embeds.

    As :func:`antecedent.composition_bundle.consume_bundle`, except that each embedded native
    producer (a cyclic posterior fit, a transport fit, a predictive fit) is recompiled from its
    portable evidence and refitted under ``execution``; a node bound to its own seed refits under
    that seed. ``route`` names the supported consumer route: ``"cyclic_posterior"`` for bundles of
    cyclic posterior fits, ``"native_fit"`` for any native fit node. Both run the same verifiers.

    Raises:
        CompositionBundleRefusal: the container, its digests or the identity do not verify.
    """
    from .composition_bundle import ConsumedBundle, SuppliedSources, _refusal_type

    assert execution is not None
    if route not in ("native_fit", "cyclic_posterior"):
        raise CausalValueError("route must be 'native_fit' or 'cyclic_posterior'")
    if supplied is not None and not isinstance(supplied, SuppliedSources):
        raise CausalTypeError("supplied must be SuppliedSources")
    wire, refusal = _native.cyclic_bundle_consume_refitting(
        container_bytes(data),
        _text(expected_identity, "expected_identity"),
        None if supplied is None else supplied._json(),
        route == "cyclic_posterior",
        **_controls(execution),
    )
    raise_refusal(refusal, _refusal_type)
    if wire is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return ConsumedBundle(json.loads(wire))
