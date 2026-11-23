"""Decide on a transported effect or outcome, seal the decision and replay it.

A transported decision reads a :class:`antecedent.decision.Contract` on a target
population whose evidence comes from a source trial. The graph and the
:class:`~antecedent.transport.advanced.TransportQuery` are the ones
:func:`~antecedent.transport.advanced.identify` takes; the route identifies the
query again, fits the raw evidence and evaluates the contract on the fitted
posterior or predictive draws::

    from antecedent import decision
    from antecedent.transport import advanced, decisions

    query = advanced.TransportQuery(
        ResponseCurve("a", "y", grid=[0.0, 1.0]),
        advanced.SelectionDiagram("trial", "target", ["s"]),
        source_experiments=["a"],
    )
    effect = decisions.effect_quantity(graph=graph, query=query, units="score")
    contract = decision.Contract(
        actions=(
            decision.Action("treat", (effect,), decision.x(0), kind="external"),
            decision.Action("hold", (effect,), decision.x(0) * 0.0, kind="external"),
        ),
        utility_units="score",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    compiled = decisions.compile_trial_ipw_decision(
        graph=graph, query=query, rows=rows, contract=contract, effect=effect
    )
    result = compiled.run()
    data = result.export()
    replayed = decisions.consume_transport_decision(
        data, route=compiled.route, expected_identity=compiled.identity
    )

Two producers carry the evidence: trial-to-target inverse probability weighting
on independent trial and target units with supplied selection and treatment
probabilities (:func:`compile_trial_ipw_decision`), and the joint source-target
Gaussian transport model (:func:`compile_joint_transport_decision`). Either
decides on the target average effect (``effect=``) or on the untreated and
treated target outcome worlds (``worlds=``). A predictive decision needs a
cross-world premise the source does not identify, so it is declared: the trial
route declares the two potential outcomes independent given the posterior mass,
the joint route declares a target residual variance and the correlation of the
two disturbances.

A sealed decision is consumed in another process from its bytes, its route and
the program identity retained independently of them; the consumer
re-identifies, refits and re-decides, and refuses changed evidence or a foreign
identity with ``external_binding_mismatch``. A sealed effect decision also
enters a composition bundle as a fit node whose propagation chain is a live
terminal (:func:`add_fit_node`, :func:`compile_node_bundle`,
:func:`consume_node_bundle`, :func:`consume_fit_bundle`).

No result here carries a measured interval coverage or decision reliability:
``calibration`` is ``not_measured``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np

from .._artifact_protocol import artifact_consumer, container_bytes
from .._columns import coerce_fields
from .._native import CompiledTransportDecisionHandle, CompiledTransportNodeBundleHandle
from .._native import joint_transport_decision_compile as _joint_compile
from .._native import transport_bundle_add_fit as _add_fit
from .._native import transport_decision_consume as _consume
from .._native import transport_fit_bundle_consume as _consume_fit_bundle
from .._native import transport_node_bundle_compile as _bundle_compile
from .._native import transport_node_bundle_consume as _bundle_consume
from .._native import trial_ipw_decision_compile as _trial_compile
from .._refusal import raise_refusal
from .._scientific import ScientificResult
from ..composition_bundle import (
    Bundle,
    BundleBuilder,
    ConsumedBundle,
    SuppliedSources,
    _bundle_acceptance,
    _refusal_type,
)
from ..decision import ActionOutcome, ConstraintExclusion, Contract, Verdict
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..execution import Execution
from ..graph import Admg
from ..joint_distribution import ScientificQuantity
from ..propagation_chain import PropagationChain, _chain
from ..query import ResponseCurve
from ._controls import call_native, resolve
from ._impl import TransportQuery, identify
from ._impl.identify import _response_args
from ._joint_posterior import (
    JointTransportPriors,
    JointTransportSource,
    JointTransportTarget,
    _request_json,
)

__all__ = [
    "CompiledNodeBundle",
    "CompiledTransportDecision",
    "DecisionRoute",
    "FitRoute",
    "NodeBundleReceipt",
    "TransportDecisionResult",
    "TrialIpwRows",
    "add_fit_node",
    "compile_joint_transport_decision",
    "compile_node_bundle",
    "compile_trial_ipw_decision",
    "consume_fit_bundle",
    "consume_node_bundle",
    "consume_transport_decision",
    "effect_quantity",
    "outcome_worlds",
]

DecisionRoute = Literal[
    "trial_ipw_effect",
    "trial_ipw_predictive",
    "joint_transport_effect",
    "joint_transport_predictive",
]
"""The producer and the coordinate a transported decision reads."""
FitRoute = Literal["trial_ipw", "joint_transport_effect"]
"""The producer a bundle fit node refits."""

#: The bundle node kind each fit route fills, and the decision route it seals.
_FIT_KINDS: dict[str, str] = {
    "trial_ipw": "trial_ipw_transport_fit",
    "joint_transport_effect": "joint_transport_effect_fit",
}
_FIT_OF_DECISION: dict[str, str] = {
    "trial_ipw_effect": "trial_ipw",
    "joint_transport_effect": "joint_transport_effect",
}
_CONTROLS: dict[str, str] = {
    "seed": "seed",
    "threads": "threads",
    "cancel": "cancel",
    "memory": "memory_bytes",
}


def _call(execution: Execution | None, native: Any, /, *args: Any, **keywords: Any) -> Any:
    return call_native(execution, native, *args, **_CONTROLS, **keywords)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty string", reason_code="invalid_argument")
    return value


def _route(value: object, routes: tuple[str, ...], what: str) -> str:
    if value not in routes:
        raise CausalValueError(f"{what} must be one of {routes}", reason_code="invalid_argument")
    return str(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Declarations.
# ---------------------------------------------------------------------------


def _floats(values: Sequence[float], what: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of numbers")
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a sequence of numbers") from error
    if any(not math.isfinite(v) for v in out):
        raise CausalValueError(f"{what} must be finite", reason_code="invalid_argument")
    return out


def _flags(values: Sequence[bool], what: str) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of 0/1 or booleans")
    out = []
    for value in values:
        if value in (0, 1):
            out.append(bool(value))
        else:
            raise CausalValueError(
                f"{what} must hold 0/1 or booleans, not {value!r}", reason_code="invalid_argument"
            )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class TrialIpwRows:
    """Independent source-trial and target units with their supplied design probabilities.

    One entry per unit, in the same order in every column. ``trial`` is true for a
    source-trial unit and false for a target unit; ``selection_probability`` is the
    known or externally supplied trial selection probability of every unit and
    ``treatment_probability`` the treatment probability of every trial unit. A target
    unit's outcome and treatment are not read (they are finite placeholders), but
    every target unit counts toward the target sample size. ``unit_ids`` are distinct.
    """

    unit_ids: Sequence[str]
    outcome: Sequence[float]
    treatment: Sequence[bool]
    trial: Sequence[bool]
    selection_probability: Sequence[float]
    treatment_probability: Sequence[float]

    def __post_init__(self) -> None:
        if isinstance(self.unit_ids, (str, bytes)):
            raise CausalTypeError("unit_ids must be a sequence of unit identities")
        ids = tuple(_text(v, "a unit id") for v in self.unit_ids)
        columns = {
            "outcome": _floats(self.outcome, "outcome"),
            "treatment": _flags(self.treatment, "treatment"),
            "trial": _flags(self.trial, "trial"),
            "selection_probability": _floats(self.selection_probability, "selection_probability"),
            "treatment_probability": _floats(self.treatment_probability, "treatment_probability"),
        }
        if any(len(column) != len(ids) for column in columns.values()):
            raise CausalValueError(
                "every TrialIpwRows column needs one entry per unit id",
                reason_code="invalid_argument",
            )
        coerce_fields(self, unit_ids=ids, **columns)

    def _wire(self, *, draws: int, seed: int) -> dict[str, Any]:
        return {
            "unit_ids": list(self.unit_ids),
            "outcomes": list(self.outcome),
            "treatment": list(self.treatment),
            "trial": list(self.trial),
            "selection_probability": list(self.selection_probability),
            "treatment_probability": list(self.treatment_probability),
            "draws": draws,
            "seed": seed,
        }


def _pair(graph: Admg, query: TransportQuery) -> tuple[str, str, int, int]:
    """``(treatment, outcome, treatment id, outcome id)`` of a binary response query."""
    if not isinstance(graph, Admg):
        raise CausalTypeError("graph must be an Admg")
    if not isinstance(query, TransportQuery):
        raise CausalTypeError("query must be an antecedent.transport.advanced.TransportQuery")
    response = query.query
    if not isinstance(response, ResponseCurve):
        raise CausalTypeError("a transported decision reads a ResponseCurve over levels 0 and 1")
    names = list(graph.nodes())
    for name in (response.treatment, response.outcome):
        if name not in names:
            raise CausalValueError(f"unknown variable {name}", reason_code="unknown_variable")
    return (
        response.treatment,
        response.outcome,
        names.index(response.treatment),
        names.index(response.outcome),
    )


def effect_quantity(
    *, graph: Admg, query: TransportQuery, units: str, name: str = "average effect"
) -> ScientificQuantity:
    """The target average-effect coordinate ``E[Y | do(A=1)] - E[Y | do(A=0)]`` of ``query``.

    A contract decided by an effect route reads exactly this coordinate. It names the
    outcome and treatment by their graph coordinates, so build it here rather than by
    hand; ``units`` are the outcome's declared units and ``name`` a display name.
    """
    _, _, treatment_id, outcome_id = _pair(graph, query)
    return ScientificQuantity(
        str(outcome_id),
        _text(name, "name"),
        "outcome",
        _text(units, "units"),
        query.diagram.target,
        f"do({treatment_id}=1)-do({treatment_id}=0)",
        0,
        "average_effect",
    )


def outcome_worlds(
    *, graph: Admg, query: TransportQuery, units: str, name: str = "outcome"
) -> tuple[ScientificQuantity, ScientificQuantity]:
    """The untreated and the treated target outcome coordinates of ``query``, in that order.

    A contract decided by a predictive route has two actions, each reading one of these
    worlds.
    """
    _, _, treatment_id, outcome_id = _pair(graph, query)
    worlds = tuple(
        ScientificQuantity(
            str(outcome_id),
            _text(name, "name"),
            "outcome",
            _text(units, "units"),
            query.diagram.target,
            f"do({treatment_id}={level})",
            0,
            "outcome",
        )
        for level in (0, 1)
    )
    return worlds[0], worlds[1]


def _structure(query: TransportQuery) -> str:
    if query.target is not None:
        raise CausalUnsupportedError(
            "a transported decision reads the labelled target population; a weighted "
            "target is the estimation route's",
            reason_code="route_not_supported",
        )
    return json.dumps(
        {
            "selections": list(query.diagram.selections),
            "source": query.diagram.source,
            "target": query.diagram.target,
            "source_experiments": list(query.source_experiments),
            "response": _response_args(query.query),
        }
    )


def _contract(contract: Contract) -> str:
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be an antecedent.decision.Contract")
    contract._normalized()  # a malformed contract refuses as a DecisionRefusal here
    return json.dumps(contract._wire())


def _coordinates(
    effect: ScientificQuantity | None,
    worlds: Sequence[ScientificQuantity] | None,
) -> tuple[str | None, str | None]:
    if (effect is None) == (worlds is None):
        raise CausalValueError(
            "declare either effect= (the target average effect) or worlds= (the untreated "
            "and treated target outcomes)",
            reason_code="invalid_argument",
        )
    if effect is not None:
        if not isinstance(effect, ScientificQuantity):
            raise CausalTypeError("effect must be a ScientificQuantity")
        return json.dumps(effect._wire()), None
    assert worlds is not None
    pair = tuple(worlds)
    if len(pair) != 2 or any(not isinstance(world, ScientificQuantity) for world in pair):
        raise CausalTypeError("worlds must be the untreated and the treated ScientificQuantity")
    return None, json.dumps([world._wire() for world in pair])


def _draws(draws: int) -> int:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise CausalValueError("draws must be a positive integer", reason_code="invalid_argument")
    return draws


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------


def _actions(wire: Sequence[Mapping[str, Any]]) -> tuple[ActionOutcome, ...]:
    def number(value: float | None) -> float:
        return float("nan") if value is None else float(value)

    return tuple(
        ActionOutcome(
            id=action["id"],
            admissible=bool(action["admissible"]),
            exclusions=tuple(
                ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
                for e in action["exclusions"]
            ),
            expected_utility=number(action["expected_utility"]),
            value=number(action["value"]),
            standard_error=action["standard_error"],
            expected_regret=action["expected_regret"],
            max_regret=action["max_regret"],
        )
        for action in wire
    )


def _verdict(wire: Any) -> Verdict:
    if wire == "no_admissible_action":
        return Verdict("no_admissible_action", ())
    ((kind, value),) = wire.items()
    if kind == "uniquely_optimal":
        return Verdict("uniquely_optimal", (value,))
    return Verdict("indistinguishable", tuple(value))


@dataclass(frozen=True, slots=True)
class TransportDecisionResult(ScientificResult):
    """A fitted transported decision: the draws it read and the decision on them.

    ``effect_draws`` are the aligned posterior draws of the target average effect (an
    effect route); ``predictive_draws`` are the aligned untreated and treated target
    outcome draws (a predictive route). ``actions``, ``verdict`` and ``evpi`` are the
    contract's evaluation on those draws, exactly as
    :class:`antecedent.decision.Decision` reports them. ``target_effect_mean`` and
    ``target_effect_variance`` are the joint route's exact conjugate moments.
    ``program_identity`` is the identity a consumer retains; ``calibration`` is
    ``not_measured``. ``independently_verified`` is true for a consumed result.
    """

    route: str
    program_identity: str
    decision_program_identity: str
    contract_identity: str
    criterion_label: str
    actions: tuple[ActionOutcome, ...]
    verdict: Verdict
    evpi: float | None
    n_draws: int
    effective_draws: float
    assumptions: tuple[str, ...]
    calibration: str
    effect_draws: np.ndarray | None = None
    predictive_draws: tuple[np.ndarray, np.ndarray] | None = None
    target_effect_mean: float | None = None
    target_effect_variance: float | None = None
    independently_verified: bool = False
    _native: Any = field(default=None, repr=False, compare=False)
    _handle: Any = field(default=None, repr=False, compare=False)

    @property
    def selected(self) -> tuple[str, ...]:
        """The leader, and any actions it cannot be told apart from; empty if none admissible."""
        return self.verdict.actions

    @property
    def chain(self) -> PropagationChain:
        """The propagation chain the decision read; it is a bundle's live terminal."""
        return _chain(self._native.chain())

    def export(
        self,
        *,
        artifact_id: str = "transport-decision",
        execution: Execution | None = None,
    ) -> bytes:
        """The decision sealed as portable bytes.

        The program that produced the result executes again and must reproduce it; the
        bytes carry the raw evidence, the contract, the draws and the decision. A
        consumed result returns the bytes it was consumed from.
        """
        if self._handle is None:
            data = getattr(self, "_artifact_bytes", None)
            if data is None:  # pragma: no cover - every result is compiled or consumed
                raise CausalUnsupportedError(
                    "this result has no program and no artifact", reason_code="not_executed"
                )
            return bytes(data)
        return self._handle.export(self, artifact_id=artifact_id, execution=execution)


def _decision_result(
    native: Any, *, handle: CompiledTransportDecision | None, verified: bool
) -> TransportDecisionResult:
    body = json.loads(native.summary_json())
    decision = body["decision"]
    fit = body["fit"] or {}
    effect = body["effect_draws"]
    predictive = body["predictive_draws"]
    return TransportDecisionResult(
        route=body["route"],
        program_identity=body["program_identity"],
        decision_program_identity=body["decision_program_identity"],
        contract_identity=decision["contract_identity"],
        criterion_label=decision["criterion_label"],
        actions=_actions(decision["actions"]),
        verdict=_verdict(decision["verdict"]),
        evpi=None if decision["evpi"] is None else float(decision["evpi"]),
        n_draws=int(decision["n_draws"]),
        effective_draws=float(decision["effective_draws"]),
        assumptions=tuple(decision["assumptions"]),
        calibration=str(body["calibration"]),
        effect_draws=None if effect is None else np.asarray(effect, dtype=np.float64),
        predictive_draws=(
            None
            if predictive is None
            else (
                np.asarray(predictive[0], dtype=np.float64),
                np.asarray(predictive[1], dtype=np.float64),
            )
        ),
        target_effect_mean=fit.get("target_effect_mean"),
        target_effect_variance=fit.get("target_effect_variance"),
        independently_verified=verified,
        _native=native,
        _handle=handle,
    )


class CompiledTransportDecision:
    """A compiled transported decision: the six-layer program, before any fit runs.

    ``route`` names the producer and coordinate, ``identity`` is the program identity a
    consumer retains independently of the bytes, and ``plan`` every compiler layer.
    :meth:`run` fits the raw evidence and decides; the execution defaults to the one
    the program was compiled under.
    """

    __slots__ = ("_execution", "_native")

    def __init__(self, native: CompiledTransportDecisionHandle, execution: Execution) -> None:
        self._native = native
        self._execution = execution

    def __repr__(self) -> str:
        return f"CompiledTransportDecision(route={self.route!r}, identity={self.identity[:12]!r})"

    @property
    def route(self) -> DecisionRoute:
        """``trial_ipw_effect``, ``trial_ipw_predictive``, ``joint_transport_effect`` or
        ``joint_transport_predictive``."""
        route: DecisionRoute = self._native.route  # type: ignore[assignment]
        return route

    @property
    def identity(self) -> str:
        """The program identity (lowercase hex) a consumer retains."""
        return str(self._native.identity)

    @property
    def plan(self) -> Mapping[str, Any]:
        """Every compiler layer of the program, read without fitting."""
        frozen: Mapping[str, Any] = _freeze_json(json.loads(self._native.plan_json()))
        return frozen

    def run(self, *, execution: Execution | None = None) -> TransportDecisionResult:
        """Fit the raw evidence under the program and evaluate the contract."""
        native = _call(self._execution if execution is None else execution, self._native.execute)
        return _decision_result(native, handle=self, verified=False)

    def export(
        self,
        result: TransportDecisionResult,
        *,
        artifact_id: str = "transport-decision",
        execution: Execution | None = None,
    ) -> bytes:
        """Execute again, require ``result`` and seal it as portable bytes."""
        if not isinstance(result, TransportDecisionResult) or result._native is None:
            raise CausalTypeError("result must be a TransportDecisionResult of this program")
        return bytes(
            _call(
                self._execution if execution is None else execution,
                self._native.export,
                result._native,
                _text(artifact_id, "artifact_id"),
            )
        )


def compile_trial_ipw_decision(
    *,
    graph: Admg,
    query: TransportQuery,
    rows: TrialIpwRows,
    contract: Contract,
    effect: ScientificQuantity | None = None,
    worlds: Sequence[ScientificQuantity] | None = None,
    independent_potential_outcomes: bool = False,
    draws: int = 1000,
    execution: Execution | None = None,
) -> CompiledTransportDecision:
    """Compile a trial-to-target IPW decision on the target effect or outcome worlds.

    The posterior is a Dirichlet law over the independent trial units, conditional on the
    supplied selection and treatment probabilities and the target sample size; ``draws``
    (40 through 100,000) posterior draws are taken under ``execution.seed``. With
    ``effect=`` (:func:`effect_quantity`) the contract reads the target average effect, a
    parameter, never outcome risk. With ``worlds=`` (:func:`outcome_worlds`) it reads the
    untreated and treated target outcomes; the two arms' empirical outcome supports are
    reweighted by the same posterior mass and, conditional on it, the two outcomes are
    independent only by declaration: pass ``independent_potential_outcomes=True``.

    ``query`` reads a :class:`~antecedent.ResponseCurve` at treatment levels ``0`` and
    ``1``; identification runs again on every compile and every replay.
    """
    if not isinstance(rows, TrialIpwRows):
        raise CausalTypeError("rows must be TrialIpwRows")
    _pair(graph, query)
    effect_json, worlds_json = _coordinates(effect, worlds)
    if effect_json is not None and independent_potential_outcomes:
        raise CausalValueError(
            "independent_potential_outcomes declares a cross-world premise of outcome worlds; "
            "an effect decision reads no outcome world",
            reason_code="invalid_argument",
        )
    execution = resolve(execution)
    rows_json = json.dumps(rows._wire(draws=_draws(draws), seed=execution.seed))
    native = _trial_compile(
        graph,
        _structure(query),
        rows_json,
        _contract(contract),
        effect_json,
        worlds_json,
        bool(independent_potential_outcomes),
        catalog=query.catalog,
    )
    return CompiledTransportDecision(native, execution)


def compile_joint_transport_decision(
    *,
    graph: Admg,
    query: TransportQuery,
    sources: Sequence[JointTransportSource],
    target: JointTransportTarget,
    priors: JointTransportPriors,
    contract: Contract,
    effect: ScientificQuantity | None = None,
    worlds: Sequence[ScientificQuantity] | None = None,
    target_noise_variance: float | None = None,
    target_noise_correlation: float | None = None,
    features: Sequence[str] = (),
    varying: Literal["intercept", "intercept_and_covariates"] = "intercept",
    sharing: Literal[
        "independent_varying_blocks", "shared_varying_block"
    ] = "independent_varying_blocks",
    dependence: Literal["independent_samples", "overlapping_units", "unknown"] = (
        "independent_samples"
    ),
    draws: int = 1024,
    max_unsupported_mass: float = 0.0,
    conflict_z_threshold: float = 3.0,
    execution: Execution | None = None,
) -> CompiledTransportDecision:
    """Compile a joint source-target transport decision on the target effect or outcomes.

    The source trials and target frame are the ones
    :func:`~antecedent.transport.advanced.joint_bayesian_transport` fits: a Gaussian
    mechanism with invariant and source-varying blocks under the declared ``priors``, its
    exact conjugate posterior drawn ``draws`` times under ``execution.seed``. With
    ``effect=`` the contract reads the target average effect. With ``worlds=`` it reads
    the untreated and treated target outcomes: predicting an absolute outcome assumes the
    shared varying block (``sharing="shared_varying_block"``) and a known
    ``target_noise_variance`` apply in the target, and pairing the two worlds needs the
    declared ``target_noise_correlation`` in ``[-1, 1]``, which the source does not fix.
    """
    treatment, outcome, _, _ = _pair(graph, query)
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise CausalTypeError("sources must be a sequence of JointTransportSource")
    effect_json, worlds_json = _coordinates(effect, worlds)
    if effect_json is not None and (
        target_noise_variance is not None or target_noise_correlation is not None
    ):
        raise CausalValueError(
            "an effect decision takes no target noise variance or correlation",
            reason_code="invalid_argument",
        )
    if worlds_json is not None and (
        target_noise_variance is None or target_noise_correlation is None
    ):
        raise CausalValueError(
            "outcome worlds need target_noise_variance and target_noise_correlation",
            reason_code="invalid_argument",
        )
    execution = resolve(execution)
    request = _request_json(
        sources=list(sources),
        target=target,
        features=list(features),
        identification=identify(graph=graph, query=query),
        priors=priors,
        treatment=treatment,
        outcome=outcome,
        varying=varying,
        sharing=sharing,
        dependence=dependence,
        basis_degree=1,
        draws=_draws(draws),
        seed=execution.seed,
        learned=False,
        max_unsupported_mass=max_unsupported_mass,
        conflict_z_threshold=conflict_z_threshold,
    )
    native = _joint_compile(
        graph,
        _structure(query),
        request,
        _contract(contract),
        effect_json,
        worlds_json,
        None if target_noise_variance is None else float(target_noise_variance),
        None if target_noise_correlation is None else float(target_noise_correlation),
        catalog=query.catalog,
    )
    return CompiledTransportDecision(native, execution)


@artifact_consumer(
    proof="native transported-decision consumer re-identifies the stored graph, refits the raw evidence and re-decides under the retained program identity",
    replayable=True,
    input="data",
)
def consume_transport_decision(
    data: bytes,
    *,
    route: DecisionRoute,
    expected_identity: str,
    execution: Execution | None = None,
) -> TransportDecisionResult:
    """Independently replay a sealed transported decision from its bytes.

    ``route`` and ``expected_identity`` are retained independently of the bytes (they are
    :attr:`CompiledTransportDecision.route` and :attr:`CompiledTransportDecision.identity`).
    The consumer re-identifies the stored selection diagram, recompiles the program and
    requires its identity, refits the raw evidence and requires the same draws, and
    replays the decision. Changed evidence, a changed decision or a foreign identity
    refuses with ``external_binding_mismatch``. A joint predictive decision draws its
    target units and disturbances from the execution's streams, so it replays under the
    seed it was produced with.
    """
    data = container_bytes(data)
    native = _call(
        execution,
        _consume,
        data,
        _route(route, get_args(DecisionRoute), "route"),
        _text(expected_identity, "expected_identity"),
    )
    return _decision_result(native, handle=None, verified=True)


# ---------------------------------------------------------------------------
# Bundles that replay a transported fit.
# ---------------------------------------------------------------------------


def add_fit_node(
    builder: BundleBuilder,
    result: TransportDecisionResult,
    *,
    node_id: str,
    execution: Execution | None = None,
) -> BundleBuilder:
    """Embed a sealed effect decision in ``builder`` as a transported fit node.

    The builder re-identifies and refits the decision under its program identity before
    it accepts the node, so a node that would not replay never enters a bundle. Add the
    decision's :attr:`TransportDecisionResult.chain` as a ``propagation_chain`` artifact
    and connect the fit to it for :func:`compile_node_bundle`. Returns ``builder``.
    """
    if not isinstance(builder, BundleBuilder):
        raise CausalTypeError("builder must be an antecedent.composition_bundle.BundleBuilder")
    if not isinstance(result, TransportDecisionResult):
        raise CausalTypeError("result must be a TransportDecisionResult")
    fit = _FIT_OF_DECISION.get(result.route)
    if fit is None:
        raise CausalUnsupportedError(
            f"a {result.route} decision is not a transported effect fit node",
            reason_code="route_not_supported",
        )
    node_id = _text(node_id, "node_id")
    _call(
        execution,
        _add_fit,
        builder._native,
        fit,
        node_id,
        result.export(execution=execution),
        result.program_identity,
    )
    builder._ids.append(node_id)
    builder._last = node_id
    return builder


@dataclass(frozen=True, slots=True)
class NodeBundleReceipt(ScientificResult):
    """The receipt of a replayed transported-fit bundle.

    The fit node was re-identified and refitted through the bundle consumer, and the
    chain terminal executed again and matched the fit's chain. ``dependencies`` lists
    every node's verified identity and Merkle digest. No sampling coverage or decision
    reliability is earned. ``independently_verified`` is true for a consumed receipt.
    """

    route: str
    bundle_identity: str
    terminal_node: str
    terminal_identity: str
    chain_program_identity: str
    producer_program_identity: str
    program_identity: str
    dependencies: tuple[Mapping[str, str], ...]
    independently_verified: bool = False
    _native: Any = field(default=None, repr=False, compare=False)


def _receipt(route: str, parts: tuple[str, str], *, verified: bool) -> NodeBundleReceipt:
    view, wire = parts
    body = json.loads(view)
    return NodeBundleReceipt(
        route=route,
        bundle_identity=body["bundle_identity"],
        terminal_node=body["terminal_node"],
        terminal_identity=body["terminal_identity"],
        chain_program_identity=body["chain_program_identity"],
        producer_program_identity=body["producer_program_identity"],
        program_identity=body["program_identity"],
        dependencies=tuple(MappingProxyType(dict(d)) for d in body["dependencies"]),
        independently_verified=verified,
        _native=wire,
    )


class CompiledNodeBundle:
    """A bundle whose transported fit node feeds a live propagation-chain terminal.

    ``identity`` is the outer program identity a consumer retains with the bundle
    identity. :meth:`run` refits the fit node and executes the terminal.
    """

    __slots__ = ("_execution", "_native")

    def __init__(self, native: CompiledTransportNodeBundleHandle, execution: Execution) -> None:
        self._native = native
        self._execution = execution

    def __repr__(self) -> str:
        return f"CompiledNodeBundle(route={self.route!r}, identity={self.identity[:12]!r})"

    @property
    def route(self) -> FitRoute:
        """``trial_ipw`` or ``joint_transport_effect``: the producer the bundle refits."""
        route: FitRoute = self._native.route  # type: ignore[assignment]
        return route

    @property
    def identity(self) -> str:
        """The outer program identity (lowercase hex) a consumer retains."""
        return str(self._native.identity)

    @property
    def plan(self) -> Mapping[str, Any]:
        """Every compiler layer of the outer program, read without fitting."""
        frozen: Mapping[str, Any] = _freeze_json(json.loads(self._native.plan_json()))
        return frozen

    def run(self, *, execution: Execution | None = None) -> NodeBundleReceipt:
        """Refit the fit node through the bundle consumer and execute the terminal."""
        parts = _call(self._execution if execution is None else execution, self._native.execute)
        return _receipt(self.route, parts, verified=False)

    def export(self, receipt: NodeBundleReceipt, *, execution: Execution | None = None) -> bytes:
        """Execute again, require ``receipt`` and seal the bundle with it."""
        if not isinstance(receipt, NodeBundleReceipt) or receipt._native is None:
            raise CausalTypeError("receipt must be a NodeBundleReceipt of this program")
        return bytes(
            _call(
                self._execution if execution is None else execution,
                self._native.export,
                receipt._native,
            )
        )


def compile_node_bundle(
    bundle: Bundle,
    *,
    producer_node: str,
    terminal_node: str,
    execution: Execution | None = None,
) -> CompiledNodeBundle:
    """Compile a bundle whose fit node ``producer_node`` feeds ``terminal_node`` directly.

    ``producer_node`` is a fit node :func:`add_fit_node` made; ``terminal_node`` is the
    fit decision's propagation chain, connected from it. Compilation binds the graph, the
    fit bytes and the producer identity into a distinct program without fitting.
    """
    if not isinstance(bundle, Bundle):
        raise CausalTypeError("bundle must be an antecedent.composition_bundle.Bundle")
    producer_node = _text(producer_node, "producer_node")
    kinds = {node.id: node.kind for node in bundle.nodes}
    route = next(
        (fit for fit, kind in _FIT_KINDS.items() if kinds.get(producer_node) == kind), None
    )
    if route is None:
        raise CausalValueError(
            f"`{producer_node}` is not a transported fit node of this bundle; add one with "
            "add_fit_node",
            reason_code="invalid_argument",
        )
    execution = resolve(execution)
    native = _call(
        execution,
        _bundle_compile,
        bundle._native,
        route,
        producer_node,
        _text(terminal_node, "terminal_node"),
    )
    return CompiledNodeBundle(native, execution)


@artifact_consumer(
    proof="native transported-fit bundle consumer recompiles the outer program, refits the fit node and executes the chain terminal",
    replayable=True,
    identity=("expected_bundle", "expected_identity"),
    input="data",
)
def consume_node_bundle(
    data: bytes,
    *,
    route: FitRoute,
    expected_bundle: str,
    expected_identity: str,
    execution: Execution | None = None,
) -> NodeBundleReceipt:
    """Independently replay a sealed transported-fit bundle from its bytes.

    ``route``, ``expected_bundle`` (the bundle identity) and ``expected_identity`` (the
    outer program identity) are retained independently of the bytes. The consumer
    rebuilds the bundle, recompiles the program and requires its identity, refits the fit
    node, executes the terminal and requires the stored receipt; anything else refuses
    with ``external_binding_mismatch``.
    """
    data = container_bytes(data)
    checked = _route(route, get_args(FitRoute), "route")
    parts = _call(
        execution,
        _bundle_consume,
        data,
        checked,
        _text(expected_bundle, "expected_bundle"),
        _text(expected_identity, "expected_identity"),
    )
    return _receipt(checked, parts, verified=True)


@artifact_consumer(
    proof="BundleConsumer::consume with the transported-fit verifiers refitting every fit node",
    acceptance=_bundle_acceptance,
    input="data",
)
def consume_fit_bundle(
    data: bytes,
    *,
    route: FitRoute,
    expected_identity: str,
    supplied: SuppliedSources | None = None,
    execution: Execution | None = None,
) -> ConsumedBundle:
    """Consume an exported bundle whose transported fit nodes are refitted, not trusted.

    Like :func:`antecedent.composition_bundle.consume_bundle`, every node is reported
    verified, reference-unresolved or failed at a named stage; a ``route`` fit node is
    re-identified and refitted from its embedded evidence under ``execution`` and counts
    as replayed only when it reproduces. ``expected_identity`` is the bundle identity
    retained independently of the bytes.
    """
    data = container_bytes(data)
    if supplied is not None and not isinstance(supplied, SuppliedSources):
        raise CausalTypeError("supplied must be SuppliedSources or None")
    wire, refusal = _call(
        execution,
        _consume_fit_bundle,
        data,
        _route(route, get_args(FitRoute), "route"),
        _text(expected_identity, "expected_identity"),
        None if supplied is None else supplied._json(),
    )
    raise_refusal(refusal, _refusal_type)
    assert wire is not None
    return ConsumedBundle(json.loads(wire))
