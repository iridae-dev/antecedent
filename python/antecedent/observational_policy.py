"""Observational policy value under a declared treatment support.

The value of a treatment policy is the mean outcome if every unit were given the action the
policy picks from its covariates. From observational data it is estimated by a cross-fitted
doubly robust score: a multinomial-logit propensity and one outcome regression per action are
fitted on the other folds, so the estimate is consistent when either side is correct. The
treatment support (the actions the policy may take, each of which must have positive
propensity), the objective and the evaluation have no default and are refused when absent::

    from antecedent import observational_policy

    # A policy fixed before seeing these rows, scored on every row.
    value = observational_policy.policy_value(
        outcome, treatment,
        support=[0.0, 1.0], objective="maximize",
        evaluation="prespecified", policy_actions=treat_if_age_over_60,
        policy_origin="prespecified",
        adjust={"age": age},
    )

    # A policy learned on part of the data and scored on independent held-out rows.
    held_out = observational_policy.policy_value(
        outcome, treatment, support=[0.0, 1.0], objective="maximize",
        evaluation="held_out", evaluation_fraction=0.4, adjust={"age": age},
    )

    # A policy learned on the same data: selection-aware cross-fitted learning and scoring.
    honest = observational_policy.policy_value(
        outcome, treatment, support=[0.0, 1.0], objective="maximize",
        evaluation="cross_fitted_learning", adjust={"age": age},
    )

A policy declared ``policy_origin="learned_on_evaluation_rows"`` is refused
(``policy_learned_on_evaluation_rows``): its value on the rows it was fitted to is estimated
too high. ``held_out`` learns the greedy policy (the action with the best predicted outcome)
on a seeded learning part and reports that one policy's value on the held-out rows;
``cross_fitted_learning`` is the value of the learning procedure fed ``(K - 1) / K`` of the
data, not of one final policy.

A causally sufficient DAG over the treatment, the outcome and the ``adjust`` columns may be
declared (``graph=``, with ``treatment_name`` and ``outcome_name`` naming the treatment and
outcome nodes). The compiler then checks that the adjustment set blocks every backdoor path and
holds no descendant of the treatment, conditionally on the graph being true. ``information``
names the adjustment columns the policy may act on; the nuisance models still use them all::

    graph = [("z", "a"), ("z", "y"), ("w", "y"), ("a", "y")]
    value = observational_policy.policy_value(
        y, a, support=[0.0, 1.0], objective="maximize", evaluation="held_out",
        evaluation_fraction=0.4, adjust={"z": z, "w": w},
        graph=graph, treatment_name="a", outcome_name="y", information=["z"],
        execution=Execution(seed=2),
    )

Every value is the output of one compiled program and can be sealed for independent replay:
:meth:`ObservationalPolicyValue.export` recompiles the request under its split seed, checks
that the program reproduces the value and seals the request with it;
:func:`consume_compiled_observational_policy` recompiles and re-estimates it under the same
seed. For a graph-checked cross-fitted learner whose policy acts on a strict subset of the
adjustment columns, :func:`compile_policy_subsampling` refits the whole learner on
``replicates`` subsamples of ``subsample_rows`` distinct rows and reports a basic m-out-of-n
interval for the fold-policy average (:class:`CrossFittedPolicyValueCandidate`), with the
calibration slot of that exact construction; :func:`consume_policy_subsampling` replays every
refit.

What the answer is **not**:

* Without ``graph=`` no graph is checked: the caller owns that ``adjust`` is a valid
  adjustment set. With it, the check is conditional on the declared graph.
* The standard error is an influence-function standard error. Its calibration is
  **not_measured** (``calibration == "not_measured"``): no interval is produced and no coverage
  claim is made. The subsampling candidate's interval carries the calibration slot of its own
  construction; it is ``not_measured`` until a coverage record measures that construction.
* An action outside the declared support, or one whose propensity falls below the floor,
  refuses instead of being extrapolated.

Everything numerical, and every refusal, is computed in Rust.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from . import _native
from ._artifact_protocol import ArtifactExpectation, ExportedArtifact, container_bytes
from ._native import evaluate_observational_policy as _evaluate
from ._scientific import ScientificResult
from ._validate import count as _count
from ._validate import name as _name
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, native_controls, resolve_execution

__all__ = [
    "ActionValue",
    "CrossFittedPolicyValueCandidate",
    "ObservationalPolicyValue",
    "ReferenceComparison",
    "compile_policy_subsampling",
    "consume_compiled_observational_policy",
    "consume_policy_subsampling",
    "policy_value",
]

EvaluationName = Literal["prespecified", "held_out", "cross_fitted_learning"]
ObjectiveName = Literal["maximize", "minimize"]
PolicyOriginName = Literal["prespecified", "learned_on_evaluation_rows"]

_POLICY_KIND = "observational_policy_compiled"
_SUBSAMPLING_KIND = "cross_fitted_policy_subsampling_v1"


@dataclass(frozen=True, slots=True)
class ActionValue:
    """The value of one action given to everyone, on the scored rows."""

    level: float
    value: float
    se: float
    #: Fraction of the scored rows to which the policy gives this action.
    policy_share: float
    #: Smallest out-of-sample propensity of the action over the scored rows.
    propensity_min: float
    #: Complete-case rows observed at the action.
    rows: int


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """The policy against a single declared action given to everyone."""

    level: float
    incremental_value: float
    incremental_se: float


@dataclass(frozen=True, slots=True)
class _PolicyRequest:
    """The columns, declaration and report a value was computed from, kept for its export."""

    outcome: list[float]
    treatment: list[float]
    names: list[str]
    columns: list[list[float]]
    declaration: str
    report: str


@dataclass(frozen=True, slots=True)
class ObservationalPolicyValue:
    """The value of a policy with the evidence needed to read it."""

    evaluation: str
    estimand: str
    assumptions: tuple[str, ...]
    objective: str
    support: tuple[float, ...]
    #: ``"estimated"`` (multinomial logit) or ``"known"`` (declared assignment probabilities).
    propensity_source: str
    value: float
    #: Influence-function standard error (calibration not_measured).
    se: float
    action_values: tuple[ActionValue, ...]
    reference: ReferenceComparison | None
    evaluation_rows: int
    learning_rows: int
    rows_dropped_missing: int
    #: Original row index, policy action and score of each scored row.
    row_index: tuple[int, ...]
    policy_actions: tuple[float, ...]
    scores: tuple[float, ...]
    calibration: str
    _request: _PolicyRequest | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable view."""
        return {
            "evaluation": self.evaluation,
            "estimand": self.estimand,
            "assumptions": list(self.assumptions),
            "objective": self.objective,
            "support": list(self.support),
            "propensity_source": self.propensity_source,
            "value": self.value,
            "se": self.se,
            "action_values": [
                {
                    "level": a.level,
                    "value": a.value,
                    "se": a.se,
                    "policy_share": a.policy_share,
                    "propensity_min": a.propensity_min,
                    "rows": a.rows,
                }
                for a in self.action_values
            ],
            "reference": None
            if self.reference is None
            else {
                "level": self.reference.level,
                "incremental_value": self.reference.incremental_value,
                "incremental_se": self.reference.incremental_se,
            },
            "evaluation_rows": self.evaluation_rows,
            "learning_rows": self.learning_rows,
            "rows_dropped_missing": self.rows_dropped_missing,
            "row_index": list(self.row_index),
            "policy_actions": list(self.policy_actions),
            "scores": list(self.scores),
            "calibration": self.calibration,
        }

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[ObservationalPolicyValue]:
        """Seal the compiled policy program with this value for independent replay.

        The request (rows, adjustment columns, declaration and split seed) is compiled into one
        program, executed again and checked to reproduce this value before both are sealed.
        ``execution`` supplies only the memory limit and cancellation: the split seed is the
        one the value was computed under. The expectation carries the program identity
        :func:`consume_compiled_observational_policy` checks.

        Raises:
            CausalValueError: the value was replayed from an artifact; export that artifact.
            CausalError: the compiled program does not reproduce the value, or refuses.
        """
        request = self._request
        if request is None:
            raise CausalValueError(
                "a replayed policy value has no request to compile; export the artifact it was "
                "consumed from"
            )
        execution = resolve_execution(execution)
        data, identity = execution._call(
            _native.observational_policy_export,
            request.outcome,
            request.treatment,
            request.names,
            request.columns,
            request.declaration,
            request.report,
            **native_controls(execution, memory="memory_bytes", cancel="cancel"),
        )
        return ExportedArtifact(
            bytes(data), _POLICY_KIND, ArtifactExpectation(_POLICY_KIND, identity)
        )


@dataclass(repr=False, frozen=True)
class CrossFittedPolicyValueCandidate(ScientificResult):
    """A full-refit subsampling interval for a graph-checked cross-fitted policy learner.

    ``value`` is the ordinary cross-fitted report on all rows. ``lower`` and ``upper`` are the
    basic m-out-of-n bounds at ``level``: the learner, its folds and every nuisance model are
    refitted on each subsample and the deviations are scaled by ``sqrt(m / n)``. The interval
    targets the average value of the fold policies on these rows, not the value of one final
    policy, and rests on an appropriate subsampling limit; near decision ties or for an
    unstable learner it may fail. ``calibration`` is the status of this exact construction in
    the coverage records, ``basis`` the key it was matched under and ``slot`` the full
    calibration slot. ``program_identity`` identifies the compiled subsampling schedule.
    """

    value: ObservationalPolicyValue
    lower: float
    upper: float
    level: float
    calibration: str
    basis: Mapping[str, Any]
    slot: Mapping[str, Any]
    program_identity: str
    _native: Any = field(default=None, repr=False, compare=False)
    _artifact_bytes: bytes | None = field(default=None, repr=False, compare=False)

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[CrossFittedPolicyValueCandidate]:
        """Seal the source rows, the schedule and this candidate for independent replay.

        Every refit is executed again and must reproduce the candidate before it is sealed.
        A candidate replayed from an artifact exports the bytes it was consumed from.
        """
        expectation = ArtifactExpectation(_SUBSAMPLING_KIND, self.program_identity)
        if self._native is None:
            if self._artifact_bytes is None:
                raise CausalValueError("this candidate holds neither a program nor its artifact")
            return ExportedArtifact(self._artifact_bytes, _SUBSAMPLING_KIND, expectation)
        execution = resolve_execution(execution)
        data = execution._call(
            self._native.export,
            **native_controls(execution, memory="memory_bytes", cancel="cancel"),
        )
        return ExportedArtifact(bytes(data), _SUBSAMPLING_KIND, expectation)


def _column(values: object, what: str) -> list[float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    return [float(v) for v in array]


def _causal_graph(
    graph: object,
    treatment_name: str | None,
    outcome_name: str | None,
    information: Sequence[str] | None,
    adjustment: Sequence[str],
) -> dict[str, Any]:
    if treatment_name is None or outcome_name is None:
        raise CausalValueError("a policy graph needs treatment_name and outcome_name")
    treatment = _name(treatment_name, "treatment_name")
    outcome = _name(outcome_name, "outcome_name")
    if isinstance(graph, _native.Dag):
        edges = [(str(a), str(b)) for a, b in graph.edges()]
        nodes = [str(node) for node in graph.nodes()]
    elif isinstance(graph, Sequence) and not isinstance(graph, (str, bytes)):
        edges = []
        for edge in graph:
            if isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence) or len(edge) != 2:
                raise CausalTypeError("graph edges must be (cause, effect) pairs of names")
            edges.append((_name(edge[0], "graph node"), _name(edge[1], "graph node")))
        # Every declared column is a node, connected or not.
        nodes = []
        for node in (*[name for edge in edges for name in edge], treatment, outcome, *adjustment):
            if node not in nodes:
                nodes.append(node)
    else:
        raise CausalTypeError("graph must be a Dag or a list of (cause, effect) edges")
    if information is None:
        information_set = list(adjustment)
    elif isinstance(information, (str, bytes)) or not isinstance(information, Sequence):
        raise CausalTypeError("information must be a sequence of adjustment column names")
    else:
        information_set = [_name(item, "information") for item in information]
    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in edges],
        "treatment": treatment,
        "outcome": outcome,
        "information_set": information_set,
    }


def _declaration(
    *,
    seed: int,
    support: Sequence[float] | None,
    objective: str | None,
    evaluation: str | None,
    policy_actions: Sequence[float] | None,
    policy_origin: str | None,
    evaluation_fraction: float | None,
    reference: float | None,
    known_propensity: Sequence[float] | Sequence[Sequence[float]] | None,
    folds: int | None,
    min_propensity: float | None,
    causal_graph: dict[str, Any] | None,
) -> str:
    declaration: dict[str, Any] = {"seed": seed}
    if support is not None:
        declaration["support"] = _column(support, "support")
    if objective is not None:
        declaration["objective"] = str(objective)
    if evaluation is not None:
        declaration["evaluation"] = str(evaluation)
    if policy_actions is not None:
        declaration["policy_actions"] = _column(policy_actions, "policy_actions")
    if policy_origin is not None:
        declaration["policy_origin"] = str(policy_origin)
    if evaluation_fraction is not None:
        declaration["evaluation_fraction"] = float(evaluation_fraction)
    if reference is not None:
        declaration["reference"] = float(reference)
    if known_propensity is not None:
        known = np.asarray(known_propensity, dtype=np.float64)
        if known.ndim == 1:
            declaration["known_propensity"] = [float(v) for v in known]
        elif known.ndim == 2:
            declaration["known_propensity_columns"] = [[float(v) for v in col] for col in known]
        else:
            raise CausalValueError("known_propensity must be one-dimensional or two-dimensional")
    if folds is not None:
        declaration["folds"] = int(folds)
    if min_propensity is not None:
        declaration["min_propensity"] = float(min_propensity)
    if causal_graph is not None:
        declaration["causal_graph"] = causal_graph
    return json.dumps(declaration)


def _value(report: Mapping[str, Any], request: _PolicyRequest | None) -> ObservationalPolicyValue:
    ref = report["reference"]
    return ObservationalPolicyValue(
        evaluation=report["evaluation"],
        estimand=report["estimand"],
        assumptions=tuple(report["assumptions"]),
        objective=report["objective"],
        support=tuple(report["support"]),
        propensity_source=report["propensity_source"],
        value=report["value"],
        se=report["se"],
        action_values=tuple(ActionValue(**a) for a in report["action_values"]),
        reference=None if ref is None else ReferenceComparison(**ref),
        evaluation_rows=report["evaluation_rows"],
        learning_rows=report["learning_rows"],
        rows_dropped_missing=report["rows_dropped_missing"],
        row_index=tuple(report["row_index"]),
        policy_actions=tuple(report["policy_actions"]),
        scores=tuple(report["scores"]),
        calibration=report["calibration"],
        _request=request,
    )


def policy_value(
    outcome: Sequence[float],
    treatment: Sequence[float],
    *,
    support: Sequence[float] | None = None,
    objective: ObjectiveName | None = None,
    evaluation: EvaluationName | None = None,
    policy_actions: Sequence[float] | None = None,
    policy_origin: PolicyOriginName | None = None,
    evaluation_fraction: float | None = None,
    reference: float | None = None,
    known_propensity: Sequence[float] | Sequence[Sequence[float]] | None = None,
    adjust: Mapping[str, Sequence[float]] | None = None,
    folds: int | None = None,
    min_propensity: float | None = None,
    graph: Any = None,
    treatment_name: str | None = None,
    outcome_name: str | None = None,
    information: Sequence[str] | None = None,
    execution: Execution | None = None,
) -> ObservationalPolicyValue:
    """Estimate the value of a treatment policy from observational data.

    ``treatment`` holds the observed action of each row; ``support`` declares the actions the
    policy may take, and every observed treatment must be one of them. ``objective`` says
    whether a larger or smaller outcome is better (it directs learned policies).
    ``evaluation`` chooses how the policy is obtained and scored (see the module docstring);
    ``"prespecified"`` needs ``policy_actions`` (one action per row) and ``policy_origin``,
    ``"held_out"`` needs ``evaluation_fraction`` in ``(0, 1)``. ``reference`` names an action
    to compare the policy against. ``adjust`` maps column names to the covariates the caller
    declares sufficient for adjustment. ``known_propensity`` declares the assignment
    probabilities of a randomized design and replaces the fitted propensity model: one
    probability per action in ``support`` order (e.g. ``[0.5, 0.5]``), or one column per action
    with a value per row. ``folds`` is the cross-fit fold count (default 5),
    ``min_propensity`` the positivity floor of every declared action and ``execution.seed``
    seeds the folds or the held-out split. A ``NaN`` is a missing cell: its row is dropped from the
    complete cases and counted in ``rows_dropped_missing``.

    ``graph`` (a :class:`~antecedent.Dag` or a list of ``(cause, effect)`` edges) declares a
    causally sufficient DAG; ``treatment_name`` and ``outcome_name`` name the treatment and
    outcome nodes, and ``information`` the adjustment columns the policy may act on (default:
    all of them). The value is then compiled with the backdoor check of the declared graph.
    The result can be sealed with :meth:`ObservationalPolicyValue.export`.

    Raises:
        CausalValueError: A malformed argument.
        CausalUnsupportedError: A refusal with a registered ``reason_code`` such as
            ``required_option_missing``, ``policy_learned_on_evaluation_rows``,
            ``positivity_violated`` or ``treatment_level_undeclared``.
    """
    execution = resolve_execution(execution)
    adjustment = dict(adjust or {})
    names = list(adjustment)
    declaration = _declaration(
        seed=execution.seed,
        support=support,
        objective=objective,
        evaluation=evaluation,
        policy_actions=policy_actions,
        policy_origin=policy_origin,
        evaluation_fraction=evaluation_fraction,
        reference=reference,
        known_propensity=known_propensity,
        folds=folds,
        min_propensity=min_propensity,
        causal_graph=None
        if graph is None
        else _causal_graph(graph, treatment_name, outcome_name, information, names),
    )
    outcome_column = _column(outcome, "outcome")
    treatment_column = _column(treatment, "treatment")
    columns = [_column(values, f"adjust[{name!r}]") for name, values in adjustment.items()]
    text = execution._call(
        _evaluate,
        outcome_column,
        treatment_column,
        names,
        columns,
        declaration,
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    request = _PolicyRequest(outcome_column, treatment_column, names, columns, declaration, text)
    return _value(json.loads(text), request)


def consume_compiled_observational_policy(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str | None = None,
    execution: Execution | None = None,
) -> ObservationalPolicyValue:
    """Independently recompile and re-estimate a sealed policy value.

    The request carried by ``artifact`` is compiled again and executed under
    ``execution.seed``, which must be the split seed the value was computed under; every
    reported number is recomputed and compared with the sealed report. With
    ``expected_identity`` (the program identity retained beside the bytes, for example the
    exported artifact's expectation) the recompiled program must also have that identity.

    Raises:
        CausalTypeError: ``artifact`` is not bytes or an exported artifact.
        CausalValueError: ``expected_identity`` is not 64 hexadecimal characters.
        CausalError: another split seed, program identity or report, or an estimator refusal.
    """
    execution = resolve_execution(execution)
    text = execution._call(
        _native.observational_policy_consume,
        container_bytes(artifact),
        None if expected_identity is None else _name(expected_identity, "expected_identity"),
        execution.seed,
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    return _value(json.loads(text), None)


def _candidate(
    wire: Mapping[str, Any], native: Any = None, artifact: bytes | None = None
) -> CrossFittedPolicyValueCandidate:
    return CrossFittedPolicyValueCandidate(
        value=_value(wire["report"], None),
        lower=float(wire["lower"]),
        upper=float(wire["upper"]),
        level=float(wire["level"]),
        calibration=wire["calibration"],
        basis=wire["basis"],
        slot=wire["slot"],
        program_identity=wire["program_identity"],
        _native=native,
        _artifact_bytes=artifact,
    )


def compile_policy_subsampling(
    outcome: Sequence[float],
    treatment: Sequence[float],
    *,
    support: Sequence[float],
    objective: ObjectiveName,
    adjust: Mapping[str, Sequence[float]],
    graph: Any,
    treatment_name: str,
    outcome_name: str,
    information: Sequence[str],
    subsample_rows: int,
    replicates: int,
    reference: float | None = None,
    known_propensity: Sequence[float] | Sequence[Sequence[float]] | None = None,
    folds: int | None = None,
    min_propensity: float | None = None,
    execution: Execution | None = None,
) -> CrossFittedPolicyValueCandidate:
    """Compile and run full-refit subsampling of a graph-checked cross-fitted policy learner.

    The policy is learned by ``cross_fitted_learning`` on the declared ``graph`` (as for
    :func:`policy_value`), acting on ``information``, a strict subset of the ``adjust``
    columns. The compiler binds the rows, the learner, ``subsample_rows`` (at least 20 and
    fewer than the rows), ``replicates`` (100 to 4096) and ``execution.seed`` into one program
    before any row is drawn; each replicate draws ``subsample_rows`` distinct rows and refits
    the learner, its folds and every nuisance model. A failed refit refuses; none is dropped.
    Every row must be complete.

    Raises:
        CausalValueError: A malformed argument.
        CausalError: an invalid schedule, a graph whose adjustment set leaves a backdoor path
            open, an information set that is not a strict subset of the adjustment columns,
            missing cells, or any refit's refusal.
    """
    execution = resolve_execution(execution)
    adjustment = dict(adjust)
    names = list(adjustment)
    declaration = _declaration(
        seed=execution.seed,
        support=support,
        objective=objective,
        evaluation="cross_fitted_learning",
        policy_actions=None,
        policy_origin=None,
        evaluation_fraction=None,
        reference=reference,
        known_propensity=known_propensity,
        folds=folds,
        min_propensity=min_propensity,
        causal_graph=_causal_graph(graph, treatment_name, outcome_name, information, names),
    )
    handle = execution._call(
        _native.policy_subsampling_compile,
        _column(outcome, "outcome"),
        _column(treatment, "treatment"),
        names,
        [_column(values, f"adjust[{name!r}]") for name, values in adjustment.items()],
        declaration,
        _count(subsample_rows, "subsample_rows", minimum=0),
        _count(replicates, "replicates", minimum=0),
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    return _candidate(json.loads(handle.candidate_json), native=handle)


def consume_policy_subsampling(
    artifact: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> CrossFittedPolicyValueCandidate:
    """Independently recompile and rerun a sealed policy subsampling schedule.

    ``expected_identity`` is the :attr:`CrossFittedPolicyValueCandidate.program_identity`
    retained beside the bytes and ``execution.seed`` must be the split seed of the original
    run; the learner and every refit are recomputed and the candidate is compared with the
    sealed one.

    Raises:
        CausalTypeError: ``artifact`` is not bytes or an exported artifact.
        CausalValueError: ``expected_identity`` is not 64 hexadecimal characters.
        CausalError: another seed, program identity, source or result, or a refit's refusal.
    """
    execution = resolve_execution(execution)
    data = container_bytes(artifact)
    text = execution._call(
        _native.policy_subsampling_consume,
        data,
        _name(expected_identity, "expected_identity"),
        execution.seed,
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    return _candidate(json.loads(text), artifact=data)
