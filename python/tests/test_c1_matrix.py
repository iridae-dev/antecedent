"""C1: the producer/consumer matrix, executed (``docs/2_3-producer-consumer-matrix.md``).

Rows are the producers of a scientific object, columns the consumers that could take it.
Each cell is ``DIRECT`` (a licensed route takes the object), ``ADAPTER`` (a named,
caller-written step or a different entry point takes it) or ``REFUSED`` (the consumer must
not take it: a joint law, a posterior or a causal response is never inferred from a
neighbouring object). This file is the executable source of truth: the table below is
asserted cell by cell, and the document is checked against it.

Every ``DIRECT`` and ``ADAPTER`` cell runs a small producer through the consumer and asserts a
hand-derived value; every ``REFUSED`` cell feeds the producer to every entry point of the
consumer and asserts that it is not accepted, with the exact exception type where the
consumer types its input. The hand values mirror the single-route tests they reuse:

* native response ``E[y | do(1)] = 3`` and ``E[y | do(2)] = 5`` with utilities ``m1`` and
  ``m2 - 1.5``, so ``A = 3`` and ``B = 3.5``;
* external claim ``E[Y | do(a)] = 1 + 2a`` with utility ``2 m - 1``, so ``wait = 1`` and
  ``treat = 9``;
* joint draws ``p = [1, 3, 2, 0]``, ``q = [4, 0, 2, 6]``, ``safe = 3``, so
  ``E[P Q] = 2``, ``E[safe] = 3`` and ``P(P Q >= 4) = 1/2``;
* frozen guess decision with accuracies 3/4 and 5/8, cost 1/10: EVSI ``1/4`` and ``1/8``, net
  ``3/20`` and ``1/40``;
* frozen sensitivity surface ``A = 2 - gamma`` against ``B = 1``: an exact tie at ``gamma = 1``.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import antecedent as ac
import numpy as np
import pytest
from antecedent import (
    Admg,
    Dag,
    decision,
    decision_robust,
    program_claims,
    recalc,
    repair,
    scenario_decision,
)
from antecedent import composition as comp
from antecedent import composition_bundle as cb
from antecedent import design_ranking as dr
from antecedent import inverse_query as iq
from antecedent import sensitivity_decision as sd
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Power, Quantity
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.estimation import PreparedBatch, RetargetClaim
from antecedent.external import ExternalRefusal
from antecedent.inference import PosteriorArtifact, encode_posterior_artifact
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.recalc import RecalcRequest, RecalcSession, Stage, TargetWeights, Utility
from antecedent.results.response import (
    CausalResponseView,
)
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

PRODUCERS: tuple[str, ...] = (
    "native_response",
    "fitted_effect_model",
    "retargeted_batch",
    "external_claim",
    "joint_draws",
    "bayesian_prior",
    "identified_scenarios",
    "sensitivity_artifact",
    "decision",
    "study_ranking",
    "failed_contract",
)
CONSUMERS: tuple[str, ...] = (
    "decision",
    "inverse_query",
    "design_ranking",
    "sensitivity_decision",
    "scenario_decision",
    "repair",
    "bundle_node",
    "recalc_plan",
)
TYPED = (CausalTypeError, CausalValueError, CausalUnsupportedError, CausalSerializationError)

# --------------------------------------------------------------------------------- the table
# One row per producer, one status per consumer, in CONSUMERS order. The checks registered
# below must agree with it; the document must agree with it.
D, A, R = "DIRECT", "ADAPTER", "REFUSED"
STATUS: dict[str, tuple[str, ...]] = {
    "native_response": (D, A, R, R, R, R, R, A),
    "fitted_effect_model": (R, R, R, R, R, R, R, A),
    "retargeted_batch": (R, R, R, R, R, R, R, D),
    "external_claim": (D, D, R, R, R, R, D, A),
    "joint_draws": (D, D, R, R, R, R, D, A),
    "bayesian_prior": (R, R, A, R, R, R, R, A),
    "identified_scenarios": (A, A, R, R, D, R, R, A),
    "sensitivity_artifact": (R, R, R, D, R, R, D, A),
    "decision": (R, R, A, R, R, R, D, A),
    "study_ranking": (R, R, D, R, R, R, D, A),
    "failed_contract": (R, R, R, R, R, D, R, A),
}


def _status(producer: str, consumer: str) -> str:
    return STATUS[producer][CONSUMERS.index(consumer)]


# ------------------------------------------------------------------------------- fixtures: native


def _native_program() -> program_claims.ProgramBinding:
    return program_claims.ProgramBinding.from_response(
        _native_view(), outcome_units="mmHg", dose_units="mg"
    )


@lru_cache(maxsize=1)
def _executed_native_view() -> CausalResponseView:
    # Y=1+2A+X/2 with crossed symmetric X, so the causal means are 3 and 5.
    levels = np.linspace(1.0, 2.0, 101)
    treatment = np.repeat(levels, 21)
    covariate = np.tile(np.linspace(-1.0, 1.0, 21), len(levels))
    return ac.analyze(
        {"x": covariate, "a": treatment, "y": 1 + 2 * treatment + covariate / 2},
        graph=gen.EDGES,
        query=ac.ResponseCurve("a", "y", grid=[1.0, 2.0]),
        estimator_config={
            "bandwidth": 2.1,
            "nuisance_lambda": 0.0,
            "nuisance_basis": 4,
            "folds": 2,
        },
        bootstrap=0,
        refute="none",
    )


def _native_view(estimand: Any = None) -> CausalResponseView:
    view = _executed_native_view()
    return view if estimand is None else view.model_copy(update={"estimand": estimand})


def _native_q(dose: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id=f"do(a={dose})",
        horizon=0,
        functional_id="mean",
        conditioning=(),
        transform_id="identity",
    )


def _native_contract(criterion: decision.Criterion) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("A", inputs=(_native_q("1"),), utility=decision.x(0)),
            decision.Action("B", inputs=(_native_q("2"),), utility=decision.x(0) - 1.5),
        ),
        utility_units="util",
        criterion=criterion,
        target_population="target",
    )


# ------------------------------------------------------------------- fixtures: claim, law, bundle


def _mean_contract(claim: Any) -> decision.Contract:
    """``wait`` reads ``a = 0`` and ``treat`` reads ``a = 2``; utility ``2 * mean - 1``."""
    q = claim.quantities
    return decision.Contract(
        actions=(
            decision.Action("wait", inputs=(q[0],), utility=decision.x(0) * 2.0 - 1.0),
            decision.Action("treat", inputs=(q[2],), utility=decision.x(0) * 2.0 - 1.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _joint_bundle(contract: decision.Contract, law: JointDistributionArtifact) -> cb.Bundle:
    builder = cb.Bundle.builder()
    builder.add_artifact("decision_contract", contract.export(artifact_id="contract"), node_id="c")
    builder.add_artifact("distribution", law.export("law"), node_id="law")
    builder.add_artifact(
        "decision_result", contract.evaluate(law).export(artifact_id="result"), node_id="result"
    )
    return builder.connect("law", "result").connect("c", "result").build()


# ------------------------------------------------------------------------ fixtures: design


def _state(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


def _signal_spec() -> dr.SignalSpec:
    return dr.SignalSpec(
        prior_id="prior-1",
        state=_state("state"),
        observation=_state("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=3,
    )


GUESS_ACTIONS = (dr.ActionUtility("guess0", 1.0, -1.0), dr.ActionUtility("guess1", 0.0, 1.0))
UTILITY_MAP = dr.CostMap("utility", "utility", 1.0)


def _guess() -> dr.Decision:
    return dr.Decision(
        contract="contract-1",
        actions=GUESS_ACTIONS,
        prior=dr.Prior.draws([0.0, 1.0]),
        utility_units="utility",
    )


def _guess_candidate(cid: str, accuracy: float) -> dr.Candidate:
    law = dr.ExternalLaw.posterior(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        predictive=[0.5, 0.5],
        posterior=[[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]],
    )
    provider = dr.ExternalSignal("lab", "signal-object", "v1", "snap", "lab-qa", law)
    return dr.Candidate(cid, 1, provider, cost=0.1, cost_unit="utility")


def _ranking() -> dr.DesignRankingResult:
    return dr.rank_designs(
        _guess(),
        [_guess_candidate("cand-1", 0.75), _guess_candidate("cand-2", 0.625)],
        signal=_signal_spec(),
        cost_map=UTILITY_MAP,
    )


# ------------------------------------------------------------------- fixtures: sensitivity


def _surface_q(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=name,
        variable_name=name,
        role="outcome",
        units="utils",
        population_id="target",
        regime_id="do(a=1)",
        horizon=0,
        functional_id="sensitivity_surface",
    )


def _frozen_sensitivity() -> sd.SensitivityArtifact:
    return sd.SensitivityArtifact.from_surface(
        coordinate=sd.AssumptionCoordinate(
            "gamma", "sensitivity_parameter", "dimensionless", 0.0, 2.0
        ),
        grid=[0.0, 1.0, 2.0],
        quantities=[
            sd.SurfaceQuantity(_surface_q("ua"), (2.0, 1.0, 0.0)),
            sd.SurfaceQuantity(_surface_q("ub"), (1.0, 1.0, 1.0)),
        ],
        actions=[sd.Action("A", sd.quantity("ua")), sd.Action("B", sd.quantity("ub"))],
        provenance=sd.SurfaceProvenance(
            source_kind="supplied_surface",
            query_binding="c1-matrix",
            provider_snapshot="snapshot-1",
            source_regime="regime:1",
            method="hand-derived surface",
            causal_contract_id="checked-contract",
        ),
    )


# -------------------------------------------------------------------- fixtures: scenarios


def _scenario_stage() -> Any:
    names = ["z", "x", "y"]
    graph = Admg.from_edges(names, [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])
    coordinates = [transport.VariableCoordinate(n, "binary") for n in names]
    scenarios = transport.TransportScenarioSet(
        [
            transport.TransportScenario("standardize", graph, ["z"], None),
            transport.TransportScenario("direct", graph, [], None),
        ],
        coordinates,
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]),
        ]
    )
    laws = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "trial",
                (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
                (0.32, 0.08, 0.12, 0.48),
                "trial",
                interventions=(("x", 1.0),),
            ),
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("z", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
                (0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1),
                "target",
            ),
        )
    )
    stage = transport.prepare_transport_scenarios(
        scenarios,
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=catalog,
        laws=laws,
        at={"x": 1.0},
    )
    stage.estimate()
    return stage


def _scenario_quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="units",
        population_id="target",
        regime_id="do(x=1)",
        horizon=0,
        functional_id="outcome",
    )


def _scenario_contract(cost: float, policy: Any) -> decision.Contract:
    y = _scenario_quantity()
    return decision.Contract(
        actions=(
            decision.Action("treat", inputs=(y,), utility=decision.x(0) - cost),
            decision.Action("hold", inputs=(y,), utility=0.5 * decision.x(0), kind="regime"),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy=policy,
    )


def _decide_scenarios(stage: Any, cost: float, policy: str) -> scenario_decision.ScenarioDecision:
    return scenario_decision.decide_from_scenarios(
        _scenario_contract(cost, policy),
        stage,
        policy,  # type: ignore[arg-type]
        outcomes=[scenario_decision.OutcomeBinding("y", _scenario_quantity())],
        causal_contract_id="causal-identification-1",
        default_support=decision_robust.Support("supported"),
    )


# ---------------------------------------------------------------------------- fixtures: robust


def _rob_quantity(variable: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


ROB_COLUMNS = (_rob_quantity("a", "do(a=1)"), _rob_quantity("b", "do(a=0)"))


def _rob_law(a: float, b: float) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=ROB_COLUMNS,
        alignment="joint",
        source_id="structure",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration",
        causal_contract_id="checked",
    )
    draws = np.array([[a, b], [a, b]], dtype=np.float64)
    return JointDistributionArtifact(identity, draws, calibration="exact")


def _rob_contract(
    policy: Any = "require_invariant_best_action",
    rules: decision_robust.AdmissibilityRules | None = None,
) -> decision_robust.AdmissibleContract:
    base = decision.Contract(
        actions=(
            decision.Action("A", inputs=(ROB_COLUMNS[0],), utility=decision.x(0)),
            decision.Action("B", inputs=(ROB_COLUMNS[1],), utility=decision.x(0)),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy=policy,
    )
    return decision_robust.admissible_contract(base, rules)


# ----------------------------------------------------------------------------- fixtures: repair

_REPAIR_NAMES = ["t", "y", "z1", "z2", "w1", "w2", "w3"]
_REPAIR_EDGES = [("z1", "t"), ("z1", "y"), ("z2", "t"), ("z2", "y"), ("t", "y")]


def _failed_contract() -> repair.BackdoorContract:
    return repair.BackdoorContract(
        graph=Dag.from_edges(_REPAIR_NAMES, _REPAIR_EDGES),
        treatment="t",
        outcome="y",
        population="clinic",
        observed=["t", "y"],
    )


def _study(label: str, measured: list[str], cost: int) -> repair.StudyCandidate:
    return repair.StudyCandidate.observation(
        label,
        population="clinic",
        measured=measured,
        cost=cost,
        sample_size=500,
        recruitment="consecutive patients",
        timing="baseline",
        unit="patient",
        cost_unit="USD",
    )


# ------------------------------------------------------------------------ fixtures: batch, recalc

BATCH_GRAPH = [
    ("z", "t1"),
    ("z", "t2"),
    ("z", "y1"),
    ("z", "y2"),
    ("t1", "y1"),
    ("t1", "y2"),
    ("t2", "y1"),
    ("t2", "y2"),
]
Q_T1_Y1 = ac.AverageEffect("t1", "y1")
Q_T1_Y2 = ac.AverageEffect("t1", "y2")
Q_T2_Y1 = ac.AverageEffect("t2", "y1")


def _batch_retarget() -> Any:
    rng = np.random.default_rng(11)
    n = 500
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.1 - 0.6 * z)))).astype(float)
    y1 = 2.0 * t1 + 0.5 * t2 + z + 0.3 * rng.normal(size=n)
    y2 = -t1 + 1.5 * t2 - 0.5 * z + 0.3 * rng.normal(size=n)
    data = {"t1": t1, "t2": t2, "y1": y1, "y2": y2, "z": z}
    batch = PreparedBatch.prepare(
        data,
        graph=BATCH_GRAPH,
        queries=[Q_T1_Y1, Q_T1_Y2, Q_T2_Y1],
        estimator="aipw",
        refute=False,
        seed=11,
        bootstrap=0,
    )
    rows = batch.retarget_rows()
    assert rows is not None
    zs = np.asarray(data["z"])[list(rows)]
    return batch.retarget(
        [
            RetargetClaim("a", Q_T1_Y1, np.exp(0.4 * zs), depends_on=["z"]),
            RetargetClaim("b", 0, np.exp(-0.4 * zs), depends_on=["z"]),
        ]
    )


RECALC_N = 600
RECALC_SEED = 61
RECALC_STAGES = (
    Stage.GRAPH,
    Stage.QUERY,
    Stage.REGIME,
    Stage.EVIDENCE,
    Stage.SOURCE_POPULATION,
    Stage.TARGET_POPULATION,
    Stage.DATA_SNAPSHOT,
    Stage.ROW_DESIGN,
    Stage.TREATMENT_GRID,
    Stage.LEARNER_FOLDS_RNG,
    Stage.UTILITY,
    Stage.IDENTIFICATION,
    Stage.SCORE_ARTIFACT,
    Stage.LAW,
    Stage.DECISION,
    Stage.external_study(0),
    Stage.provider_request(0),
    Stage.prior(0),
)


def _recalc_request() -> RecalcRequest:
    rng = np.random.default_rng(RECALC_SEED)
    n = RECALC_N
    z = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))
    w = rng.standard_normal(n)
    t = (rng.random(n) < p).astype(np.float64)
    y = (2.0 + 0.8 * z) * t + z + 0.3 * rng.standard_normal(n)
    y2 = -t + 0.5 * z + 0.3 * rng.standard_normal(n)
    return RecalcRequest(
        data={"t": t, "y": y, "z": z, "w": w, "y2": y2},
        edges=(("z", "t"), ("z", "y"), ("t", "y"), ("z", "y2"), ("t", "y2")),
        treatment="t",
        outcome="y",
        utility=Utility(2.0, 0.5),
    )


# ------------------------------------------------------------------------------------- objects


@dataclass(frozen=True)
class Objs:
    """One small, hand-checkable producer of each kind."""

    native: program_claims.NativeClaim
    native_program: program_claims.ProgramBinding
    compact: CompactExport
    retarget: Any
    claim: Any
    law: JointDistributionArtifact
    posterior: PosteriorArtifact
    stage: Any
    sensitivity: sd.SensitivityArtifact
    result: decision.Decision
    ranking: dr.DesignRankingResult
    failed: repair.BackdoorContract


def _build_objects() -> Objs:
    program = _native_program()
    return Objs(
        native=program_claims.native_claim(_native_view(), program),
        native_program=program,
        compact=CompactExport.build(
            response=Quantity("y", units="mmHg", role="outcome"),
            inputs=[Numeric("x", 0.0, 10.0, units="mg", role="treatment")],
            terms=[Intercept(), Linear("x"), Power("x", 2)],
            coefficients=[1.0, 2.0, -0.5],
            covariance=[[0.04, 0.0, -0.002], [0.0, 0.01, 0.0], [-0.002, 0.0, 0.0004]],
        ),
        retarget=_batch_retarget(),
        claim=gen.external_claim(),
        law=gen.decision_source(),
        posterior=PosteriorArtifact(
            n_draws=3,
            mean=[0.0, 1.0, 2.0],
            sd=[1.0, 1.0, 0.1],
            q025=[-0.95, 0.05, 1.905],
            q975=[0.95, 1.95, 2.095],
            draws=[-1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.9, 2.0, 2.1],
            backend_id="laplace",
            identification="NonparametricallyIdentified",
            quantity_names=["coef_0", "coef_1", "ate"],
        ),
        stage=_scenario_stage(),
        sensitivity=_frozen_sensitivity(),
        result=gen.decision_contract().evaluate(gen.decision_source()),
        ranking=_ranking(),
        failed=_failed_contract(),
    )


@pytest.fixture(scope="module")
def objs() -> Objs:
    return _build_objects()


def _object(producer: str, o: Objs) -> object:
    return {
        "native_response": o.native,
        "fitted_effect_model": o.compact,
        "retargeted_batch": o.retarget,
        "external_claim": o.claim,
        "joint_draws": o.law,
        "bayesian_prior": o.posterior,
        "identified_scenarios": o.stage,
        "sensitivity_artifact": o.sensitivity,
        "decision": o.result,
        "study_ranking": o.ranking,
        "failed_contract": o.failed,
    }[producer]


def _bundle_payload(producer: str, o: Objs) -> object:
    """What a caller would hand ``add_artifact``: bytes where the producer has an export."""
    if producer == "fitted_effect_model":
        return o.compact.export()
    if producer == "bayesian_prior":
        return bytes(encode_posterior_artifact(o.posterior))
    if producer == "identified_scenarios":
        return bytes(o.stage.export())
    return _object(producer, o)


# --------------------------------------------------------------------------------- feeders
# The slot of each consumer that expects the producer: (label, callable, exception type the
# consumer raises for a neighbour, or None where the consumer does not type its input).


def _feed_evaluate(obj: Any) -> object:
    return gen.decision_contract().evaluate(obj)


def _feed_support(obj: Any) -> object:
    return comp.evaluate_with_support(gen.decision_contract(), [obj])


def _feed_inverse(obj: Any) -> object:
    query = iq.InverseQuery(gen.decision_contract(), ("risky", "safe"), (iq.target_mean(2.5),))
    return query.evaluate(obj)


def _feed_prior(obj: Any) -> object:
    framed = dr.Decision(
        contract="contract-1", actions=GUESS_ACTIONS, prior=obj, utility_units="utility"
    )
    return dr.rank_designs(framed, [_guess_candidate("cand-1", 0.75)], signal=_signal_spec())


def _feed_sensitivity(obj: Any) -> object:
    return sd.decide(gen.decision_contract(), obj)


def _feed_scenarios(obj: Any) -> object:
    return scenario_decision.decide_from_scenarios(
        gen.decision_contract(),
        obj,
        "report_only",
        outcomes=[scenario_decision.OutcomeBinding("p", gen.COLUMNS[0])],
        causal_contract_id="c",
    )


def _feed_cpdag(obj: Any) -> object:
    return scenario_decision.decide_from_cpdag_completions(
        gen.decision_contract(),
        obj,
        "report_only",
        outcomes=[scenario_decision.OutcomeBinding("p", gen.COLUMNS[0])],
        causal_contract_id="c",
    )


def _feed_obligations(obj: Any) -> object:
    return repair.obligations(obj)


def _feed_repair(obj: Any) -> object:
    return repair.repair(obj, [])


def _feed_bundle(obj: Any) -> object:
    return cb.Bundle.builder().add_artifact("auto", obj)


Expected = type[BaseException] | tuple[type[BaseException], ...] | None
Feeder = tuple[str, Callable[[Any], object], Expected]
FEEDERS: dict[str, tuple[Feeder, ...]] = {
    "decision": (
        ("Contract.evaluate", _feed_evaluate, CausalTypeError),
        ("evaluate_with_support", _feed_support, CausalTypeError),
    ),
    "inverse_query": (("InverseQuery.evaluate", _feed_inverse, CausalTypeError),),
    "design_ranking": (("rank_designs(prior=)", _feed_prior, CausalTypeError),),
    "sensitivity_decision": (("sensitivity_decision.decide", _feed_sensitivity, CausalTypeError),),
    "scenario_decision": (
        ("decide_from_scenarios", _feed_scenarios, CausalTypeError),
        (
            "decide_from_cpdag_completions",
            _feed_cpdag,
            scenario_decision.ScenarioDecisionRefusal,
        ),
    ),
    "repair": (
        ("repair.obligations", _feed_obligations, CausalTypeError),
        ("repair.repair", _feed_repair, CausalTypeError),
    ),
    "bundle_node": (
        ("add_artifact('auto')", _feed_bundle, (CausalTypeError, cb.CompositionBundleRefusal)),
    ),
}


def _payload(producer: str, consumer: str, o: Objs) -> object:
    return _bundle_payload(producer, o) if consumer == "bundle_node" else _object(producer, o)


def _check_refused(producer: str, consumer: str, o: Objs) -> None:
    """No entry point of ``consumer`` accepts ``producer``; typed where the consumer types it."""
    payload = _payload(producer, consumer, o)
    for label, feed, expected in FEEDERS[consumer]:
        with pytest.raises(Exception) as caught:  # noqa: PT011
            feed(payload)
        if expected is not None:
            assert isinstance(caught.value, expected), (producer, label, type(caught.value))


# --------------------------------------------------------------------------- recalc adapter


def _workflow(changed: Stage, identity: str) -> dict[Stage, str]:
    declared = {stage: recalc.stage_identity(stage, "v1") for stage in RECALC_STAGES}
    declared[changed] = recalc.stage_identity(changed, identity)
    return declared


def _check_recalc_adapter(stage: Stage, identity: str) -> None:
    """The plan reads only the digest a caller declares: same identity reused, changed recomputed."""
    first = recalc.plan_recalculation(_workflow(stage, identity), _workflow(stage, identity))
    assert first.recomputed == ()
    second = recalc.plan_recalculation(
        _workflow(stage, identity), _workflow(stage, identity + "-changed")
    )
    assert second.status(stage).recomputed
    assert not second.status(stage).reused


# ------------------------------------------------------------------------------------- checks


def _check_native_decision(o: Objs) -> None:
    contract = _native_contract(decision.Criterion.expected_utility())
    source = o.native.as_decision_source(contract)
    decided = contract.evaluate(source.source)
    by_id = {a.id: a for a in decided.actions}
    assert by_id["A"].expected_utility == pytest.approx(3.0, abs=1e-12)
    assert by_id["B"].expected_utility == pytest.approx(3.5, abs=1e-12)
    assert decided.selected == ("B",)

    # The support-aware evaluator takes the same means through a caller-written input.
    adapted = comp.DecisionInput.from_means(
        "native",
        list(o.native.coordinates),
        list(o.native.means),
        provider_id=source.source.provider_id,
        snapshot_id=source.source.snapshot_id,
        causal_contract_id=source.source.causal_contract_id,
        support=list(o.native.support),  # type: ignore[arg-type]
    )
    assert adapted.source == "mean" and not adapted.native
    supported = comp.evaluate_with_support(contract, [adapted])
    assert supported.outcome("B").expected_utility == pytest.approx(3.5, abs=1e-12)
    assert supported.verdict.selected == "B"

    # A mean response never answers a quantile or a probability.
    for criterion in (
        decision.Criterion.quantile(0.5),
        decision.Criterion.threshold_probability(3.0),
    ):
        with pytest.raises(ExternalRefusal) as refused:
            o.native.as_decision_source(_native_contract(criterion))
        assert refused.value.detail == "native_claims.source_not_supplied"
        assert refused.value.reason_code == "decision_contract_unsatisfied"
        assert refused.value.expected == "joint_draws,marginal_draws"
        assert refused.value.supplied == "mean"
        assert_registered_refusal(refused.value)

    # An estimate that is not a response curve is not a native response claim.
    with pytest.raises(ExternalRefusal) as estimate:
        program_claims.native_claim(
            _native_view(estimand=ac.AverageEffect("a", "y")), o.native_program
        )
    assert estimate.value.detail == "native_claims.unsupported_estimand"
    assert estimate.value.reason_code == "route_not_supported"
    assert estimate.value.expected == "mean_curve"

    # The claim itself is not a decision source.
    with pytest.raises(Exception):  # noqa: B017, PT011
        contract.evaluate(o.native)  # type: ignore[arg-type]


def _check_native_inverse(o: Objs) -> None:
    contract = _native_contract(decision.Criterion.expected_utility())
    mean_claim = o.native.as_decision_source(contract).mean_claim()
    answered = iq.InverseQuery(contract, ("A", "B"), (iq.target_mean(3.2),)).evaluate(mean_claim)
    assert answered.feasible_actions == ("B",)
    assert answered.selected == "B"
    for constraint in (
        iq.probability_threshold(3.0, 0.5, tail="lower", direction="at_most"),
        iq.target_quantile(0.5, 3.0),
    ):
        with pytest.raises(iq.InverseQueryRefusal) as refused:
            iq.InverseQuery(contract, ("A", "B"), (constraint,)).evaluate(mean_claim)
        assert refused.value.detail == "decision_evaluation.mean_source_insufficient"
        assert refused.value.reason_code == "decision_contract_unsatisfied"
        assert_registered_refusal(refused.value)
    retained = iq.InverseQuery(contract, ("A", "B"), (iq.target_mean(3.2),)).evaluate(o.native)
    assert retained.selected == "B"
    assert retained.source_evidence[0].coordinates == o.native.coordinates
    assert retained.source_evidence[0].diagnostics == o.native.source_evidence.diagnostics


def _check_claim_decision(o: Objs) -> None:
    contract = _mean_contract(o.claim)
    decided = contract.evaluate(o.claim)
    by_id = {a.id: a for a in decided.actions}
    assert by_id["wait"].expected_utility == pytest.approx(1.0, abs=1e-12)
    assert by_id["treat"].expected_utility == pytest.approx(9.0, abs=1e-12)
    assert decided.selected == ("treat",)
    assert decided.evpi is None, "regret needs state-aligned draws; a mean grid has none"
    with pytest.raises(decision.DecisionRefusal) as export:
        decided.export()
    assert export.value.detail == "decision_evaluation.mean_source_not_replayable"
    assert export.value.reason_code == "route_not_supported"

    attested = comp.DecisionInput.from_claim("claim", o.claim)
    assert attested.source == "mean" and not attested.native
    supported = comp.evaluate_with_support(contract, [attested])
    # The claim marks dose 2 outside empirical support, so the composition boundary reports
    # `treat` as unsupported (with its reason) instead of scoring it, and still scores `wait`.
    assert supported.disposition("treat").status == "unsupported"
    assert supported.outcome("wait").expected_utility == pytest.approx(1.0, abs=1e-12)
    expectation = comp.evaluate_functional(
        contract, "wait", comp.Functional.expected_utility(), attested
    )
    assert expectation.value == pytest.approx(1.0, abs=1e-12)
    assert expectation.standard_error is None
    # A mean never answers a probability or a quantile.
    for functional in (comp.Functional.probability(2.0, "upper"), comp.Functional.quantile(0.5)):
        with pytest.raises(comp.SupportRefusal) as refused:
            comp.evaluate_functional(contract, "wait", functional, attested)
        assert refused.value.detail == "composition_boundary.mean_is_not_a_distribution"
        assert refused.value.reason_code == "decision_contract_unsatisfied"
        assert_registered_refusal(refused.value)


def _check_claim_inverse(o: Objs) -> None:
    contract = _mean_contract(o.claim)
    query = iq.InverseQuery(contract, ("wait", "treat"), (iq.target_mean(5.0),))
    answered = query.evaluate(o.claim)
    assert answered.feasible_actions == ("treat",)
    assert answered.action("wait").point_values[0].value == pytest.approx(1.0, abs=1e-12)
    assert answered.action("treat").point_values[0].value == pytest.approx(9.0, abs=1e-12)
    refused_query = iq.InverseQuery(contract, ("wait", "treat"), (iq.target_quantile(0.5, 5.0),))
    with pytest.raises(iq.InverseQueryRefusal) as refused:
        refused_query.evaluate(o.claim)
    assert refused.value.detail == "decision_evaluation.mean_source_insufficient"


def _check_claim_bundle(o: Objs) -> None:
    contract = _mean_contract(o.claim)
    result = cb.mean_decision(contract, o.claim)
    assert cb.detect_kind(result) == "decision_result"
    builder = cb.Bundle.builder()
    builder.add_artifact("external_claim", o.claim, node_id="claim")
    builder.add_artifact(
        "decision_contract", contract.export(artifact_id="contract"), node_id="contract"
    )
    builder.add_artifact("decision_result", result, node_id="result")
    bundle = builder.connect("claim", "result").connect("contract", "result").build()
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    assert consumed.value("result", "wait.expected_utility") == pytest.approx(1.0, abs=1e-9)
    assert consumed.value("result", "treat.expected_utility") == pytest.approx(9.0, abs=1e-9)
    assert consumed.claim_label == "point_only_attested"
    assert consumed.node("claim").facts["law"] == "mean_only"
    # A functional a mean cannot answer is refused when the result is made.
    squared = replace(
        contract,
        actions=(
            decision.Action(
                "treat", inputs=(o.claim.quantities[2],), utility=decision.x(0) * decision.x(0)
            ),
            contract.actions[0],
        ),
    )
    with pytest.raises(cb.UnsupportedLawRefusal) as insufficient:
        cb.mean_decision(squared, o.claim)
    assert insufficient.value.detail == "composition_bundle.unsupported_law"
    assert_registered_refusal(insufficient.value)


def _check_law_decision(o: Objs) -> None:
    contract = gen.decision_contract()
    decided = contract.evaluate(o.law)
    by_id = {a.id: a for a in decided.actions}
    assert by_id["risky"].expected_utility == pytest.approx(2.0, abs=1e-12)
    assert by_id["safe"].expected_utility == pytest.approx(3.0, abs=1e-12)
    assert decided.evpi == pytest.approx(0.5, abs=1e-12)
    assert decided.selected == ("safe",)
    assert len(decided.export(artifact_id="result")) > 0, "a joint-draw result replays"

    law_input = comp.DecisionInput.from_distribution("law", o.law)
    assert law_input.source == "joint_law"
    supported = comp.evaluate_with_support(contract, [law_input])
    assert supported.outcome("risky").expected_utility == pytest.approx(2.0, abs=1e-12)
    # P(P * Q >= 4) over the rows 4, 0, 4, 0 is 1/2: only an aligned joint law answers it.
    probability = comp.evaluate_functional(
        contract, "risky", comp.Functional.probability(4.0, "upper"), law_input
    )
    assert probability.value == pytest.approx(0.5, abs=1e-12)


def _check_law_inverse(o: Objs) -> None:
    contract = gen.decision_contract()
    query = iq.InverseQuery(contract, ("risky", "safe"), (iq.target_mean(2.5),))
    answered = query.evaluate(o.law)
    assert answered.action("risky").point_values[0].value == pytest.approx(2.0, abs=1e-12)
    assert answered.action("safe").point_values[0].value == pytest.approx(3.0, abs=1e-12)
    assert answered.feasible_actions == ("safe",)
    assert answered.selected == "safe"


def _check_law_bundle(o: Objs) -> None:
    bundle = _joint_bundle(gen.decision_contract(), o.law)
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    assert consumed.value("result", "risky.expected_utility") == pytest.approx(2.0, abs=1e-9)
    assert consumed.value("result", "safe.expected_utility") == pytest.approx(3.0, abs=1e-9)
    assert consumed.claim_label == "joint_draw"
    assert consumed.node("law").facts["law"] == "joint_draw"


def _check_prior_design(o: Objs) -> None:
    # The posterior is summarised by the caller into three equally weighted states
    # (mean - sd, mean, mean + sd) of the quantity `ate`: 1.9, 2.0 and 2.1.
    mean, sd_ = o.posterior.mean[2], o.posterior.sd[2]
    states = [mean - sd_, mean, mean + sd_]
    assert states == pytest.approx([1.9, 2.0, 2.1], abs=1e-12)
    framed = dr.Decision(
        contract="contract-posterior",
        actions=(dr.ActionUtility("hold", 0.0, 0.0), dr.ActionUtility("act", -1.95, 1.0)),
        prior=dr.Prior.draws(states),
        utility_units="utility",
    )
    no_information = dr.ExternalLaw.decision_values(
        branch_probabilities=[1.0], action_ids=["hold", "act"], values=[[0.0, 0.05]]
    )
    candidate = dr.Candidate(
        "uninformative",
        1,
        dr.ExternalSignal("lab", "signal-object", "v1", "snap", "lab-qa", no_information),
        cost=0.0,
        cost_unit="utility",
    )
    ranked = dr.rank_designs(framed, [candidate], signal=_signal_spec())
    # act = state - 1.95 is -0.05, 0.05, 0.15 on the three states: its mean 0.05 beats hold = 0
    # and perfect information would add (0 + 0.05 + 0.15) / 3 - 0.05 = 1/60.
    assert ranked.bayes_action == "act"
    assert ranked.prior_expected_utility == pytest.approx(0.05, abs=1e-12)
    assert ranked.evpi == pytest.approx(1.0 / 60.0, abs=1e-12)
    assert ranked.candidates[0].evsi == pytest.approx(0.0, abs=1e-12)
    assert ranked.candidates[0].provider_trust == "externally_attested"
    assert ranked.candidates[0].claim == "point_only"
    # There is no checked prior-to-signal adapter on the Python surface: the Rust
    # `prior_signal::adapt_prior_to_signal` is not exported, so the step above is the caller's.
    assert not hasattr(dr, "adapt_prior_to_signal")
    assert not hasattr(ac, "adapt_prior_to_signal")


def _check_scenarios_decision(o: Objs) -> None:
    # Independent finite structures are decided by the robustness adapters, over laws the
    # caller supplies for each structure (a transport stage retains no joint law in Python).
    support = decision_robust.Support("supported")
    robust = decision_robust.finite_scenarios(
        _rob_contract(
            rules=decision_robust.AdmissibilityRules(default_weakest_support="supported")
        ),
        [
            decision_robust.Claim.evaluated("s1", _rob_law(5.0, 3.0), support=support),
            decision_robust.Claim.evaluated("s2", _rob_law(6.0, 2.0), support=support),
        ],
    )
    assert robust.verdict == decision_robust.RobustVerdict("structurally_robust", action="A")
    by_id = {a.id: a for a in robust.actions}
    assert by_id["A"].range == (5.0, 6.0)
    assert by_id["B"].range == (2.0, 3.0)

    clear = decision_robust.identified_sets(
        _rob_contract(),
        [
            decision_robust.IdentifiedUtility("A", 4.0, 6.0),
            decision_robust.IdentifiedUtility("B", 1.0, 3.0),
        ],
    )
    assert clear.verdict == decision_robust.IdentifiedVerdict("necessarily_best", action="A")
    # An identified set is never a probability law.
    with pytest.raises(decision.DecisionRefusal) as refused:
        decision_robust.identified_sets(
            _rob_contract("bayes_over_structures"),
            [
                decision_robust.IdentifiedUtility("A", 2.0, 5.0),
                decision_robust.IdentifiedUtility("B", 3.0, 6.0),
            ],
        )
    assert refused.value.detail == "decision_adapters.bayes_over_identified_set"
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    # The transport stage is not itself a decision source.
    with pytest.raises(Exception):  # noqa: B017, PT011
        gen.decision_contract().evaluate(o.stage)


def _check_scenarios_inverse(o: Objs) -> None:
    law = gen.decision_source()
    contract = gen.decision_contract()
    query = iq.InverseQuery(contract, ("risky", "safe"), (iq.target_mean(2.5),))
    declared = query.evaluate(
        identified_set=iq.IdentifiedSet((iq.Scenario.evaluated("s1", law),), True)
    )
    assert declared.action("safe").identified_set == "feasible"
    assert declared.action("risky").identified_set == "infeasible"
    assert declared.action("safe").point is None, "an absent kind of evidence is never feasible"
    undeclared = query.evaluate(
        identified_set=iq.IdentifiedSet((iq.Scenario.evaluated("s1", law),), False)
    )
    assert undeclared.action("safe").identified_set == "unevaluated"
    # The prepared transport stage is not a forward claim.
    with pytest.raises(CausalTypeError):
        query.evaluate(o.stage)  # type: ignore[arg-type]


def _check_scenarios_scenario(o: Objs) -> None:
    decided = _decide_scenarios(o.stage, 0.2, "report_only")
    assert decided.kind == "transport_scenarios"
    # direct: treat 0.56 - 0.2 = 0.36 against hold 0.28; standardize: 0.35 - 0.2 = 0.15 vs 0.175.
    assert dict(decided.leaders) == {"direct": ("treat",), "standardize": ("hold",)}
    assert decided.action("treat").per_atom["direct"] == pytest.approx(0.36, abs=1e-12)
    assert decided.action("hold").per_atom["standardize"] == pytest.approx(0.175, abs=1e-12)
    cheap = _decide_scenarios(o.stage, 0.05, "require_invariant_best_action")
    assert cheap.verdict.kind == "invariant_best" and cheap.selected == "treat"
    # Weights are the set's own; none were declared, so Bayes is refused, not defaulted.
    with pytest.raises(scenario_decision.ScenarioDecisionRefusal) as bayes:
        _decide_scenarios(o.stage, 0.2, "bayes_over_structures")
    assert bayes.value.detail == "decision_claims.probabilities_not_declared"
    assert bayes.value.reason_code == "decision_contract_unsatisfied"


def _check_sensitivity_decision(o: Objs) -> None:
    result = sd.decide(o.sensitivity.contract(), o.sensitivity)
    assert result.kind == "assumption_dependent"
    assert not result.robust
    switch = result.switch
    assert switch is not None
    assert (switch.from_actions, switch.to_actions) == (("A",), ("B",))
    assert switch.exact and switch.tipping_coordinate == 1.0
    assert result.coordinates == (0.0, 1.0, 2.0)
    assert o.sensitivity.outcome == result.outcome
    # The assumption range is never a sampling interval and a sampling interval is never composed.
    with pytest.raises(sd.SensitivityRefusal) as composed:
        sd.decide(o.sensitivity.contract(), o.sensitivity, sampling_composition="merge")
    assert composed.value.reason_code == "cell_not_licensed"


def _check_sensitivity_bundle(o: Objs) -> None:
    data = o.sensitivity.export()
    assert cb.detect_kind(data) == "sensitivity"
    described = cb.describe_artifact("sensitivity", data)
    assert described.identity == o.sensitivity.identity["digest"]
    assert described.values["grid_points"] == pytest.approx(3.0)
    builder = cb.Bundle.builder().add_artifact("sensitivity", data, node_id="sensitivity")
    bundle = builder.build()
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    assert consumed.value("sensitivity", "grid_points") == pytest.approx(3.0)


def _check_decision_design(o: Objs) -> None:
    # The decision's contract (not its result) is what a value-of-information problem binds.
    q = _state("state")
    contract = decision.Contract(
        actions=(
            decision.Action("guess0", inputs=(q,), utility=decision.x(0)),
            decision.Action("guess1", inputs=(q,), utility=decision.x(0)),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    framed = dr.Decision(contract=contract, actions=GUESS_ACTIONS, prior=dr.Prior.draws([0.0, 1.0]))
    ranked = dr.rank_designs(
        framed, [_guess_candidate("cand-1", 0.75)], signal=_signal_spec(), cost_map=UTILITY_MAP
    )
    assert ranked.decision_contract_identity == contract.identity
    assert ranked.candidates[0].evsi == pytest.approx(0.25, abs=1e-12)
    assert ranked.candidates[0].net_value == pytest.approx(0.15, abs=1e-12)
    # The binding to a Contract object checks the action set ...
    mismatched = dr.Decision(
        contract=contract,
        actions=(dr.ActionUtility("other", 1.0, -1.0), dr.ActionUtility("guess1", 0.0, 1.0)),
        prior=dr.Prior.draws([0.0, 1.0]),
    )
    with pytest.raises(CausalValueError):
        dr.rank_designs(mismatched, [_guess_candidate("cand-1", 0.75)], signal=_signal_spec())
    # ... a bare identity string binds only the identity, so the result's identity is accepted.
    by_identity = dr.Decision(
        contract=o.result.contract_identity,
        actions=GUESS_ACTIONS,
        prior=dr.Prior.draws([0.0, 1.0]),
        utility_units="utility",
    )
    again = dr.rank_designs(by_identity, [_guess_candidate("cand-1", 0.75)], signal=_signal_spec())
    assert again.decision_contract_identity == o.result.contract_identity


def _check_decision_bundle(o: Objs) -> None:
    # A joint-draw result exports and replays; it is a bundle node beneath its law and contract.
    exported = o.result.export(artifact_id="result")
    assert cb.detect_kind(exported) == "decision_result"
    bundle = _joint_bundle(gen.decision_contract(), gen.decision_source())
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    assert consumed.node("result").verified
    assert consumed.value("result", "risky.expected_utility") == pytest.approx(2.0, abs=1e-9)
    assert consumed.value("result", "safe.expected_utility") == pytest.approx(3.0, abs=1e-9)
    # Without its source the same result stands on nothing and is refused as a tampered quantity.
    alone = cb.Bundle.builder().add_artifact("decision_result", exported, node_id="result").build()
    orphan = cb.consume_bundle(alone.export(), expected_identity=alone.identity)
    assert isinstance(orphan.node("result").status, cb.Failed)
    assert orphan.value("result", "safe.expected_utility") is None


def _check_ranking_design(o: Objs) -> None:
    consumed = dr.consume(o.ranking.export(), expected=o.ranking.expectation())
    by_id = {e.id: e for e in consumed.entries}
    assert by_id["cand-1"].evsi == pytest.approx(0.25, abs=1e-12)
    assert by_id["cand-2"].evsi == pytest.approx(0.125, abs=1e-12)
    assert by_id["cand-1"].net_value == pytest.approx(0.15, abs=1e-12)
    assert by_id["cand-2"].net_value == pytest.approx(0.025, abs=1e-12)
    assert by_id["cand-1"].rank < by_id["cand-2"].rank
    assert not by_id["cand-1"].natively_replayed, "an external signal is never native"
    assert consumed.identity == o.ranking.identity
    # A changed cost map is refused even though the bytes are untouched.
    with pytest.raises(dr.CostUnitsRefusal) as refused:
        dr.consume(
            o.ranking.export(),
            expected=dr.Expectation(cost_map=dr.CostMap("utility", "utility", 2.0)),
        )
    assert refused.value.reason_code == "design_cost_units_mismatch"
    assert refused.value.detail == "design_ranking.cost_units_mismatch"


def _check_ranking_bundle(o: Objs) -> None:
    data = o.ranking.export()
    assert cb.detect_kind(data) == "study_ranking"
    builder = cb.Bundle.builder().add_artifact("study_ranking", data)
    assert builder.last_node_id.startswith("study_ranking:")
    # A ranking is not a decision result.
    with pytest.raises(cb.CompositionBundleRefusal) as wrong:
        cb.Bundle.builder().add_artifact("decision_result", data)
    assert_registered_refusal(wrong.value)


def _check_failed_repair(o: Objs) -> None:
    (obligation,) = repair.obligations(o.failed)
    assert obligation.kind == "provide_joint_law"
    assert set(obligation.variables) == {"t", "y", "z1", "z2"}
    candidates = [
        _study("registry_z1", ["t", "y", "z1"], 5),
        _study("registry_z2", ["t", "y", "z2"], 5),
        _study("cohort", ["t", "y", "z1", "z2"], 20),
    ]
    result = repair.repair(o.failed, candidates)
    assert result.outcome == "repaired" and result.inference_claim == "none"
    assert result.best is not None and result.best.labels == ("cohort",)
    # Two separate studies never combine into one joint law.
    assert result.classification("registry_z1", "registry_z2") == "insufficient"
    # Neither a decision nor a response is a failed contract.
    with pytest.raises(CausalTypeError):
        repair.obligations(object())


def _recalc_stage(producer: str, o: Objs) -> tuple[Stage, str]:
    """The stage a caller declares each producer's identity into, and that identity."""
    return {
        "native_response": (Stage.QUERY, o.native.program_identity),
        "fitted_effect_model": (Stage.SCORE_ARTIFACT, o.compact.identity),
        "external_claim": (Stage.external_study(0), str(o.claim.identity["provider_fingerprint"])),
        "joint_draws": (Stage.LAW, decision.source_digest(o.law)),
        "bayesian_prior": (Stage.prior(0), "ate:" + ",".join(map(str, o.posterior.mean))),
        "identified_scenarios": (Stage.GRAPH, hashlib.sha256(bytes(o.stage.export())).hexdigest()),
        "sensitivity_artifact": (Stage.EVIDENCE, o.sensitivity.identity["digest"]),
        "decision": (Stage.DECISION, o.result.contract_identity + o.result.source_digest),
        "study_ranking": (Stage.UTILITY, o.ranking.identity),
        "failed_contract": (Stage.QUERY, o.failed.contract_id),
    }[producer]


def _check_recalc_for(producer: str, o: Objs) -> None:
    stage, identity = _recalc_stage(producer, o)
    _check_recalc_adapter(stage, identity)


def _check_retargeted_recalc(o: Objs) -> None:
    """The recalculation executor is itself the retargeting producer: reweight, no refit."""
    base = _recalc_request()
    session = RecalcSession()
    first = session.execute(base, seed=RECALC_SEED)
    z = np.asarray(base.data["z"])
    weights = np.exp(0.4 * z)
    second = session.execute(replace(base, target=TargetWeights(weights, ("z",))), seed=RECALC_SEED)
    assert second.plan.status(Stage.TARGET_POPULATION).recomputed
    assert second.plan.status(Stage.SCORE_ARTIFACT).reused
    assert second.receipt.totals.fold_fits == 0
    assert second.receipt.totals.as_tuple() == (0, 0, 0, 1, 1)
    contrast = session.score_contrast()
    assert contrast is not None
    assert second.law.ate == pytest.approx(
        float(np.sum(weights * contrast) / np.sum(weights)), abs=1e-12
    )
    assert abs(second.law.ate - first.law.ate) > 0.05
    # A BatchRetarget is a different producer: the plan reads digests only.
    family = o.retarget.family_id
    _check_recalc_adapter(Stage.TARGET_POPULATION, family)
    assert {m.name for m in o.retarget.claims} == {"a", "b"}


# ------------------------------------------------------------------------- the check registry

Check = Callable[[Objs], None]
CHECKS: dict[tuple[str, str], Check] = {
    ("native_response", "decision"): _check_native_decision,
    ("native_response", "inverse_query"): _check_native_inverse,
    ("retargeted_batch", "recalc_plan"): _check_retargeted_recalc,
    ("external_claim", "decision"): _check_claim_decision,
    ("external_claim", "inverse_query"): _check_claim_inverse,
    ("external_claim", "bundle_node"): _check_claim_bundle,
    ("joint_draws", "decision"): _check_law_decision,
    ("joint_draws", "inverse_query"): _check_law_inverse,
    ("joint_draws", "bundle_node"): _check_law_bundle,
    ("bayesian_prior", "design_ranking"): _check_prior_design,
    ("identified_scenarios", "decision"): _check_scenarios_decision,
    ("identified_scenarios", "inverse_query"): _check_scenarios_inverse,
    ("identified_scenarios", "scenario_decision"): _check_scenarios_scenario,
    ("sensitivity_artifact", "sensitivity_decision"): _check_sensitivity_decision,
    ("sensitivity_artifact", "bundle_node"): _check_sensitivity_bundle,
    ("decision", "design_ranking"): _check_decision_design,
    ("decision", "bundle_node"): _check_decision_bundle,
    ("study_ranking", "design_ranking"): _check_ranking_design,
    ("study_ranking", "bundle_node"): _check_ranking_bundle,
    ("failed_contract", "repair"): _check_failed_repair,
}


def _run_cell(producer: str, consumer: str, o: Objs) -> None:
    status = _status(producer, consumer)
    if status == "REFUSED":
        _check_refused(producer, consumer, o)
        return
    if (producer, consumer) in CHECKS:
        CHECKS[(producer, consumer)](o)
        return
    assert consumer == "recalc_plan" and status == "ADAPTER", (producer, consumer)
    _check_recalc_for(producer, o)


# ----------------------------------------------------------------------- the table is complete


def test_c1_matrix_declares_every_cell_once_and_counts_them() -> None:
    assert set(STATUS) == set(PRODUCERS)
    assert all(len(row) == len(CONSUMERS) for row in STATUS.values())
    cells = [status for row in STATUS.values() for status in row]
    assert len(cells) == len(PRODUCERS) * len(CONSUMERS) == 88
    assert cells.count("DIRECT") == 15
    assert cells.count("ADAPTER") == 15
    assert cells.count("REFUSED") == 58
    assert set(cells) == {"DIRECT", "ADAPTER", "REFUSED"}
    # Every non-refused cell has a check; the only shared one is the recalc adapter.
    for producer in PRODUCERS:
        for consumer in CONSUMERS:
            status = _status(producer, consumer)
            if status != "REFUSED" and consumer != "recalc_plan":
                assert (producer, consumer) in CHECKS, (producer, consumer)
    for producer, consumer in CHECKS:
        assert _status(producer, consumer) != "REFUSED", (producer, consumer)


def test_c1_matrix_document_agrees_with_the_executable_table() -> None:
    path = Path(__file__).resolve().parents[2] / "docs" / "2_3-producer-consumer-matrix.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[str, tuple[str, ...]] = {}
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 1 + len(CONSUMERS) or not cells[0].startswith("`"):
            continue
        key = cells[0].strip("`")
        if key in STATUS:
            rows[key] = tuple(
                cell.replace("*", "").split(":", 1)[0].split(" ", 1)[0] for cell in cells[1:]
            )
    assert rows == STATUS
    text = path.read_text(encoding="utf-8")
    assert "test_c1_matrix_" in text


# ------------------------------------------------------------------------- one test per row


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_native_response_row(consumer: str, objs: Objs) -> None:
    _run_cell("native_response", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_fitted_effect_model_row(consumer: str, objs: Objs) -> None:
    _run_cell("fitted_effect_model", consumer, objs)
    # Its own consumer is the only thing that evaluates it, with a model-based SE and no more.
    estimate = objs.compact.evaluate({"x": 2.0})
    assert estimate.point == pytest.approx(3.0, abs=1e-12)
    assert "unmeasured" in estimate.calibration


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_retargeted_batch_row(consumer: str, objs: Objs) -> None:
    _run_cell("retargeted_batch", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_external_claim_row(consumer: str, objs: Objs) -> None:
    _run_cell("external_claim", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_joint_draws_row(consumer: str, objs: Objs) -> None:
    _run_cell("joint_draws", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_bayesian_prior_row(consumer: str, objs: Objs) -> None:
    _run_cell("bayesian_prior", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_identified_scenarios_row(consumer: str, objs: Objs) -> None:
    _run_cell("identified_scenarios", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_sensitivity_artifact_row(consumer: str, objs: Objs) -> None:
    _run_cell("sensitivity_artifact", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_decision_row(consumer: str, objs: Objs) -> None:
    _run_cell("decision", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_study_ranking_row(consumer: str, objs: Objs) -> None:
    _run_cell("study_ranking", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_c1_matrix_failed_contract_row(consumer: str, objs: Objs) -> None:
    _run_cell("failed_contract", consumer, objs)


# ------------------------------------------------------------------------ cross-cutting rules


def test_c1_matrix_every_boundary_rejects_neighbours_with_typed_errors(objs: Objs) -> None:
    """Every refused producer reaches a checked boundary, including exporter protocols."""
    for producer in PRODUCERS:
        for consumer in FEEDERS:
            if _status(producer, consumer) != "REFUSED":
                continue
            payload = _payload(producer, consumer, objs)
            for label, feed, expected in FEEDERS[consumer]:
                assert expected is not None, (producer, label)
                with pytest.raises(expected):
                    feed(payload)


def test_c1_matrix_invalid_export_signatures_are_typed_but_callback_errors_propagate(
    objs: Objs,
) -> None:
    # The fitted export's no-argument callback is outside the bundle exporter
    # protocol. Refuse before invoking it rather than leaking a binding TypeError.
    with pytest.raises(CausalTypeError, match="artifact_id"):
        cb.Bundle.builder().add_artifact("auto", objs.compact)

    class BrokenExporter:
        def export(self, *, artifact_id: str) -> bytes:
            raise TypeError("failure inside an otherwise valid export callback")

    with pytest.raises(TypeError, match="failure inside") as caught:
        cb.Bundle.builder().add_artifact("auto", BrokenExporter())
    assert type(caught.value) is TypeError


def test_c1_matrix_a_mean_never_answers_a_probability_a_quantile_or_a_law(objs: Objs) -> None:
    contract = _mean_contract(objs.claim)
    mean_input = comp.DecisionInput.from_claim("claim", objs.claim)
    law_input = comp.DecisionInput.from_distribution("law", objs.law)
    # The mean is a distribution functional's special case only for the expectation.
    assert comp.evaluate_functional(
        contract, "wait", comp.Functional.expected_utility(), mean_input
    ).value == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(comp.SupportRefusal):
        comp.evaluate_functional(
            contract, "wait", comp.Functional.probability(2.0, "upper"), mean_input
        )
    # The same question on an aligned joint law is answered (1/2, by hand).
    assert comp.evaluate_functional(
        gen.decision_contract(), "risky", comp.Functional.probability(4.0, "upper"), law_input
    ).value == pytest.approx(0.5, abs=1e-12)
    # A mean cannot be paired with draws, and a claim's mean grid is not a joint law.
    with pytest.raises(comp.CompositionRefusal) as paired:
        comp.check_paired_draws([law_input, mean_input], [])
    assert paired.value.reason_code == "joint_law_required"
    assert_registered_refusal(paired.value)
    # A joint law required by the contract is refused over a mean-only claim in a bundle.
    squared = replace(
        contract,
        actions=(
            decision.Action(
                "treat", inputs=(objs.claim.quantities[2],), utility=decision.x(0) * decision.x(0)
            ),
            contract.actions[0],
        ),
    )
    with pytest.raises(cb.UnsupportedLawRefusal):
        cb.mean_decision(squared, objs.claim)
