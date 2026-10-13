"""The producer/consumer matrix, executed (``docs/producer-consumer-matrix.md``).

Rows are the producers of a scientific object, columns the consumers that could take it.
Each cell is ``DIRECT`` (a licensed route takes the object), ``ADAPTER`` (a named,
caller-written step or a different entry point takes it) or ``REFUSED`` (the consumer must
not take it: a joint law, a posterior or a causal response is never inferred from a
neighbouring object). Cell status lives in ``parity/producer_consumer.toml``;
this file executes each cell and does not read the markdown table.

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
* frozen sensitivity surface ``A = 2 - gamma`` against ``B = 1``: an exact tie at ``gamma = 1``;
* binary misclassification with sensitivity 9/10, specificity 4/5 and 410 positives in 1000:
  latent prevalence ``(0.41 + 0.8 - 1) / 0.7 = 0.3``;
* the declared linear-Gaussian chain ``X -> M -> Y`` of ``test_mechanism_composition.py``:
  ``E[Y | do(X=1)] = 7`` and ``E[Y | do(X=0)] = 1/2``, ``Var(Y | do(X=1)) = 37/4``;
* two disjoint studies ``y = (1, 3)``, variances ``(1, 4)``: pooled estimate 7/5, variance 4/5;
* a confounding-free ``X -> Y`` population with joint law ``(1/4, 1/4, 1/8, 3/8)``:
  ``P(Y_{x=1} = 1) = 3/4`` and ``P(Y_{x=0} = 1) = 1/2``.

The four point-only producers (measurement result, composed model, evidence ledger,
counterfactual transport result) reach a mean consumer only through a declaration the caller
writes (``MeanSource`` or ``MeanClaim``: the coordinate, the point, and the producer identity as
provider and snapshot). Nothing here makes that point a law, a posterior or a response: a
quantile or probability contract is refused ``decision_evaluation.mean_source_insufficient`` and
the standard error is not carried.
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
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
from antecedent import design as dr
from antecedent import evidence_ledger as el
from antecedent import inverse_query as iq
from antecedent import measurement_models as mm
from antecedent import mechanism_composition as mc
from antecedent import sensitivity_decision as sd
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Power, Quantity
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.estimation import PreparedBatch, RetargetClaim
from antecedent.evidence_ledger import EvidenceLedgerRefusal
from antecedent.external import ExternalRefusal
from antecedent.inference import PosteriorArtifact, encode_posterior_artifact
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.measurement_models import MeasurementModelRefusal
from antecedent.mechanism_composition import MechanismCompositionRefusal
from antecedent.recalc import RecalcRequest, RecalcSession, Stage, TargetWeights, Utility
from antecedent.results.response import (
    CausalResponseView,
)
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal
from _repo_text import REPO_ROOT

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
    "measurement_result",
    "composed_model",
    "evidence_ledger",
    "counterfactual_transport_result",
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
# One row per producer, one status per consumer, in CONSUMERS order. The checks
# registered below must agree with parity/producer_consumer.toml.
_MATRIX = tomllib.loads(
    (REPO_ROOT / "parity" / "producer_consumer.toml").read_text(encoding="utf-8")
)
if tuple(row["key"] for row in _MATRIX["producer"]) != PRODUCERS:
    raise AssertionError("producer_consumer.toml producers drifted from this file")
if tuple(row["key"] for row in _MATRIX["consumer"]) != CONSUMERS:
    raise AssertionError("producer_consumer.toml consumers drifted from this file")
_BY_PAIR = {(row["producer"], row["consumer"]): row["status"] for row in _MATRIX["cell"]}
STATUS: dict[str, tuple[str, ...]] = {
    producer: tuple(_BY_PAIR[(producer, consumer)] for consumer in CONSUMERS)
    for producer in PRODUCERS
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


def _guess() -> dr.DesignDecision:
    return dr.DesignDecision(
        contract="contract-1",
        actions=GUESS_ACTIONS,
        prior=dr.StatePrior.draws([0.0, 1.0]),
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
        [_guess_candidate("cand-1", 0.75), _guess_candidate("cand-2", 0.625)],
        decision=_guess(),
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
        actions=[
            sd.SensitivityAction("A", sd.quantity("ua")),
            sd.SensitivityAction("B", sd.quantity("ub")),
        ],
        provenance=sd.SurfaceProvenance(
            source_kind="supplied_surface",
            query_binding="producer-consumer-matrix",
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


# ------------------------------------------------------- fixtures: point-only producers
# Measurement result, composed model, evidence ledger and counterfactual transport result each
# return a point (at most with an uncalibrated standard error or a declared-mechanism variance).


def _measurement_analysis() -> mm.AnalysisBinding:
    return mm.AnalysisBinding(
        construct_id="construct:latent", population_id="cohort", regime_id="observational"
    )


def _measurement_prepared() -> mm.PreparedMeasurementModel:
    """Sensitivity 9/10, specificity 4/5, 410 positives of 1000: latent prevalence 0.3."""
    model = mm.MeasurementModel(
        indicator=mm.MeasurementVariable("schema:indicator", "noisy_misclassified", "unit"),
        construct=mm.MeasurementVariable("construct:latent", "latent_scientific", "unit"),
        parameters=mm.BinaryParameters(
            (mm.BinaryRates(0.9, 0.8, 1e-4, 4e-4, arm_id=None),), min_youden=0.1
        ),
        population_id="cohort",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("producer-consumer-matrix"),
        structure="non_differential",
        representation=mm.RepresentationVersion("scalar", 1),
    )
    return mm.prepare_measurement_model(
        model, _measurement_analysis(), mm.BinaryPrevalenceData(410, 1000)
    )


_COMPOSITION_ROLES = {"X": "treatment", "M": "mediator", "Y": "outcome"}


def _mechanism(variable: str, intercept: float, sigma: float, parents: Any = ()) -> mc.Mechanism:
    def quantity(name: str) -> mc.Quantity:
        return mc.Quantity(
            name,
            units="u",
            population_id="pop",
            role=_COMPOSITION_ROLES.get(name, "covariate"),
            variable_name=f"label of {name}",
        )

    declaration = mc.MechanismDeclaration(
        output=quantity(variable),
        parents=tuple(quantity(name) for name, _ in parents),
        artifact_id=f"fit:{variable}",
        evidence_id=f"data:{variable}",
    )
    return mc.Mechanism.declared_linear(
        declaration, intercept=intercept, sigma=sigma, coefficients=dict(parents)
    )


def _composition() -> mc.PreparedMechanismComposition:
    """``X = 1 + 2 e``, ``M = 1/2 + 2 X + e``, ``Y = -1 + X/2 + 3 M + e/2``."""
    return mc.compose_mechanisms(
        [
            _mechanism("X", 1.0, 2.0),
            _mechanism("M", 0.5, 1.0, [("X", 2.0)]),
            _mechanism("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)]),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )


@dataclass(frozen=True)
class Ledgered:
    """One ledger, its point-only pooled outcome and its assumption-range envelope."""

    ledger: el.EvidenceLedger
    pooled: el.SynthesisOutcome
    envelope: el.SynthesisOutcome
    export: el.LedgerExport


def _ledgered() -> Ledgered:
    ledger = el.EvidenceLedger()
    first = ledger.add_source(
        el.SourceDeclaration("s1", "observation_set", observations=el.rows("cohort", range(0, 10)))
    )
    second = ledger.add_source(
        el.SourceDeclaration("s2", "observation_set", observations=el.rows("cohort", range(10, 20)))
    )
    pooled = ledger.pool_summaries(
        "pooled", [el.StudySummary(first, 1.0, 1.0), el.StudySummary(second, 3.0, 4.0)]
    )
    envelope = ledger.dependence_range(
        "range",
        [el.StudyEstimate(first, 1.0, 1.0), el.StudyEstimate(second, 3.0, 2.0)],
        rho_lo=0.0,
        rho_hi=0.9,
    )
    return Ledgered(ledger, pooled, envelope, ledger.export())


@dataclass(frozen=True)
class Transported:
    """The exact results and the sampled plug-ins of ``P(Y_{x=1} = 1)`` and ``P(Y_{x=0} = 1)``."""

    treated: transport.CounterfactualTransportResult
    control: transport.CounterfactualTransportResult
    sampled_treated: transport.SampledCounterfactualTransportEstimate
    sampled_control: transport.SampledCounterfactualTransportEstimate


def _transported() -> Transported:
    """Unconfounded ``X -> Y`` with joint law ``(1/4, 1/4, 1/8, 3/8)`` over ``(x, y)``."""
    graph = Admg.from_edges(["X", "Y"], [("X", "Y")], [])
    catalog = transport.EvidenceCatalog(
        regimes=(
            transport.EvidenceRegime(
                "pop|obs", "pop", kind="observational", interventions=[], measured=["X", "Y"]
            ),
        )
    )
    axes = (("X", (0.0, 1.0)), ("Y", (0.0, 1.0)))
    law = transport.ExactDiscreteLaw(
        "pop", "pop|obs", axes, (0.25, 0.25, 0.125, 0.375), "snap|pop|obs", interventions=()
    )
    table = transport.SampledCountTable(
        "pop", "pop|obs", axes, [5000, 5000, 2500, 7500], "snap|pop|obs"
    )
    results = []
    for level in (1.0, 0.0):
        decided = transport.identify_counterfactual(
            graph=graph,
            query=transport.CounterfactualTransportQuery(
                target="pop",
                levels={"X": (0.0, 1.0), "Y": (0.0, 1.0)},
                worlds=(transport.CounterfactualWorld("do", {"X": level}),),
                event=(transport.CounterfactualEvent("Y", 1.0, "do"),),
            ),
            catalog=catalog,
        )
        assert decided.status == "identified", decided.decision
        derivation = decided.derivation
        results.append(
            (
                derivation.evaluate(evidence=[law]),
                derivation.estimate_sampled(
                    [table], overlap=transport.SamplingOverlap.all_independent()
                ),
            )
        )
    (treated, sampled_treated), (control, sampled_control) = results
    return Transported(treated, control, sampled_treated, sampled_control)


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
    measurement_prepared: mm.PreparedMeasurementModel
    measurement: mm.MeasurementEstimate
    composition: mc.PreparedMechanismComposition
    composition_export: mc.CompositionExport
    ledgered: Ledgered
    transported: Transported


def _build_objects() -> Objs:
    program = _native_program()
    prepared = _measurement_prepared()
    composition = _composition()
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
        measurement_prepared=prepared,
        measurement=prepared.estimate(),
        composition=composition,
        composition_export=composition.export({"X": 1.0}, outcome="Y"),
        ledgered=_ledgered(),
        transported=_transported(),
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
        "measurement_result": o.measurement,
        "composed_model": o.composition,
        "evidence_ledger": o.ledgered.ledger,
        "counterfactual_transport_result": o.transported.treated,
    }[producer]


def _companions(producer: str, o: Objs) -> tuple[object, ...]:
    """The other objects the row stands for, fed to the same entry points as the primary one."""
    return {
        "measurement_result": (o.measurement_prepared,),
        "composed_model": (o.composition_export,),
        "evidence_ledger": (o.ledgered.pooled, o.ledgered.envelope, o.ledgered.export),
        "counterfactual_transport_result": (
            o.transported.control,
            o.transported.sampled_treated,
            o.transported.sampled_control,
        ),
    }.get(producer, ())


def _bundle_payload(producer: str, o: Objs) -> object:
    """What a caller would hand ``add_artifact``: bytes where the producer has an export."""
    if producer == "fitted_effect_model":
        return o.compact.export()
    if producer == "bayesian_prior":
        return bytes(encode_posterior_artifact(o.posterior))
    if producer == "identified_scenarios":
        return bytes(o.stage.export())
    if producer == "measurement_result":
        return o.measurement.export()
    if producer == "composed_model":
        return o.composition_export.artifact
    if producer == "evidence_ledger":
        return o.ledgered.export.artifact
    if producer == "counterfactual_transport_result":
        return o.transported.treated.export()
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
    framed = dr.DesignDecision(
        contract="contract-1", actions=GUESS_ACTIONS, prior=obj, utility_units="utility"
    )
    return dr.rank_designs(
        [_guess_candidate("cand-1", 0.75)], decision=framed, signal=_signal_spec()
    )


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


def _payloads(producer: str, consumer: str, o: Objs) -> tuple[object, ...]:
    """The primary payload and the row's companions, each as ``consumer`` would be handed it.

    A bundle consumer is also handed the exporter objects themselves (the estimate, the ledger
    session, the result) so that an exporter that is not the bundle protocol is typed.
    """
    primary = _payload(producer, consumer, o)
    if consumer == "bundle_node":
        extras = (_object(producer, o), *_companions(producer, o))
        return (primary, *extras)
    return (primary, *_companions(producer, o))


def _check_refused(producer: str, consumer: str, o: Objs) -> None:
    """No entry point of ``consumer`` accepts ``producer``; typed where the consumer types it."""
    for payload in _payloads(producer, consumer, o):
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
    framed = dr.DesignDecision(
        contract="contract-posterior",
        actions=(dr.ActionUtility("hold", 0.0, 0.0), dr.ActionUtility("act", -1.95, 1.0)),
        prior=dr.StatePrior.draws(states),
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
    ranked = dr.rank_designs([candidate], decision=framed, signal=_signal_spec())
    # act = state - 1.95 is -0.05, 0.05, 0.15 on the three states: its mean 0.05 beats hold = 0
    # and perfect information would add (0 + 0.05 + 0.15) / 3 - 0.05 = 1/60.
    assert ranked.bayes_action == "act"
    assert ranked.prior_expected_utility == pytest.approx(0.05, abs=1e-12)
    assert ranked.evpi == pytest.approx(1.0 / 60.0, abs=1e-12)
    assert ranked.candidates[0].evsi == pytest.approx(0.0, abs=1e-12)
    assert ranked.candidates[0].provider_trust == "externally_attested"
    assert ranked.candidates[0].claim == "point_only"
    # This independent caller-declared oracle remains valid; original producer-byte
    # adaptation and reuse/transport refusals are exercised in test_prior_signal_bridge.py.
    assert callable(dr.adapt_prior_to_signal)
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
    framed = dr.DesignDecision(
        contract=contract, actions=GUESS_ACTIONS, prior=dr.StatePrior.draws([0.0, 1.0])
    )
    ranked = dr.rank_designs(
        [_guess_candidate("cand-1", 0.75)],
        decision=framed,
        signal=_signal_spec(),
        cost_map=UTILITY_MAP,
    )
    assert ranked.decision_contract_identity == contract.identity
    assert ranked.candidates[0].evsi == pytest.approx(0.25, abs=1e-12)
    assert ranked.candidates[0].net_value == pytest.approx(0.15, abs=1e-12)
    # The binding to a Contract object checks the action set ...
    mismatched = dr.DesignDecision(
        contract=contract,
        actions=(dr.ActionUtility("other", 1.0, -1.0), dr.ActionUtility("guess1", 0.0, 1.0)),
        prior=dr.StatePrior.draws([0.0, 1.0]),
    )
    with pytest.raises(CausalValueError):
        dr.rank_designs(
            [_guess_candidate("cand-1", 0.75)], decision=mismatched, signal=_signal_spec()
        )
    # ... a bare identity string binds only the identity, so the result's identity is accepted.
    by_identity = dr.DesignDecision(
        contract=o.result.contract_identity,
        actions=GUESS_ACTIONS,
        prior=dr.StatePrior.draws([0.0, 1.0]),
        utility_units="utility",
    )
    again = dr.rank_designs(
        [_guess_candidate("cand-1", 0.75)], decision=by_identity, signal=_signal_spec()
    )
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
    consumed = dr.consume(o.ranking.export(), expected_identity=o.ranking.expectation())
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
            expected_identity=dr.Expectation(cost_map=dr.CostMap("utility", "utility", 2.0)),
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
        "external_claim": (
            Stage.external_study(0),
            str(o.claim.identity_fields["provider_fingerprint"]),
        ),
        "joint_draws": (Stage.LAW, decision.source_digest(o.law)),
        "bayesian_prior": (Stage.prior(0), "ate:" + ",".join(map(str, o.posterior.mean))),
        "identified_scenarios": (Stage.GRAPH, hashlib.sha256(bytes(o.stage.export())).hexdigest()),
        "sensitivity_artifact": (Stage.EVIDENCE, o.sensitivity.identity["digest"]),
        "decision": (Stage.DECISION, o.result.contract_identity + o.result.source_digest),
        "study_ranking": (Stage.UTILITY, o.ranking.identity),
        "failed_contract": (Stage.QUERY, o.failed.contract_id),
        "measurement_result": (
            Stage.EVIDENCE,
            o.measurement.model_identity + o.measurement.data_identity,
        ),
        "composed_model": (Stage.SCORE_ARTIFACT, o.composition.plan_identity),
        "evidence_ledger": (Stage.EVIDENCE, o.ledgered.export.identity),
        "counterfactual_transport_result": (
            Stage.IDENTIFICATION,
            str(o.transported.treated.premises_digest) + str(o.transported.treated.data_digest),
        ),
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


# ------------------------------------------------- declared means of the point-only producers


@dataclass(frozen=True)
class Declared:
    """What the caller writes to carry a producer's point into a mean consumer.

    The coordinates, the points, and the producer's own identity as provider and snapshot. It
    claims nothing about the outcome law: only an affine utility's expectation is answered.
    """

    quantities: tuple[ScientificQuantity, ...]
    means: tuple[float, ...]
    provider_id: str
    snapshot_id: str
    causal_contract_id: str

    def source(self) -> decision.MeanSource:
        return decision.MeanSource(
            coordinates=self.quantities,
            means=self.means,
            provider_id=self.provider_id,
            snapshot_id=self.snapshot_id,
            causal_contract_id=self.causal_contract_id,
        )

    def claim(self) -> iq.MeanClaim:
        return iq.MeanClaim(
            coordinates=self.quantities,
            means=self.means,
            provider_id=self.provider_id,
            snapshot_id=self.snapshot_id,
            causal_contract_id=self.causal_contract_id,
        )

    def contract(self, criterion: decision.Criterion | None = None) -> decision.Contract:
        """``act`` reads the first coordinate (utility ``4 m - 1``), ``hold`` the last (``2 m``)."""
        return decision.Contract(
            actions=(
                decision.Action(
                    "act", inputs=(self.quantities[0],), utility=decision.x(0) * 4.0 - 1.0
                ),
                decision.Action("hold", inputs=(self.quantities[-1],), utility=decision.x(0) * 2.0),
            ),
            utility_units="utility",
            criterion=criterion or decision.Criterion.expected_utility(),
            target_population=self.quantities[0].population_id,
        )


def _declared_quantity(
    variable: str, regime: str, *, units: str, population: str, functional: str = "mean"
) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id=population,
        regime_id=regime,
        horizon=0,
        functional_id=functional,
    )


def _check_declared_decision(declared: Declared, *, act: float, hold: float, selected: str) -> None:
    """``Contract.evaluate`` and ``evaluate_with_support`` over the declared mean source."""
    contract = declared.contract()
    decided = contract.evaluate(declared.source())
    by_id = {a.id: a for a in decided.actions}
    assert by_id["act"].expected_utility == pytest.approx(act, abs=1e-12)
    assert by_id["hold"].expected_utility == pytest.approx(hold, abs=1e-12)
    assert decided.selected == (selected,)
    # A point carries no sampling error and no regret: nothing was upgraded on the way in.
    assert by_id["act"].standard_error is None and decided.evpi is None
    assert any("means only" in line for line in decided.assumptions)
    with pytest.raises(decision.DecisionRefusal) as export:
        decided.export()
    assert export.value.detail == "decision_evaluation.mean_source_not_replayable"
    assert export.value.reason_code == "route_not_supported"

    # The support-aware evaluator takes the same means as a caller-written input.
    adapted = comp.DecisionInput.from_means(
        "declared",
        list(declared.quantities),
        list(declared.means),
        provider_id=declared.provider_id,
        snapshot_id=declared.snapshot_id,
        causal_contract_id=declared.causal_contract_id,
    )
    assert adapted.source == "mean" and not adapted.native
    supported = comp.evaluate_with_support(contract, [adapted])
    assert supported.outcome("act").expected_utility == pytest.approx(act, abs=1e-12)
    assert supported.verdict.selected == selected

    # A mean is not an outcome distribution: a quantile or a probability is refused.
    with pytest.raises(decision.DecisionRefusal) as quantile:
        declared.contract(decision.Criterion.quantile(0.5)).evaluate(declared.source())
    assert quantile.value.detail == "decision_evaluation.mean_source_insufficient"
    assert quantile.value.reason_code == "decision_contract_unsatisfied"
    assert_registered_refusal(quantile.value)
    with pytest.raises(comp.SupportRefusal) as probability:
        comp.evaluate_functional(
            contract, "act", comp.Functional.probability(0.5, "upper"), adapted
        )
    assert probability.value.detail == "composition_boundary.mean_is_not_a_distribution"


def _check_declared_inverse(
    declared: Declared, *, target: float, feasible: tuple[str, ...]
) -> None:
    answered = iq.InverseQuery(
        declared.contract(), ("act", "hold"), (iq.target_mean(target),)
    ).evaluate(declared.claim())
    assert answered.feasible_actions == feasible
    assert answered.selected == feasible[0]
    for constraint in (
        iq.probability_threshold(0.5, 0.5, tail="lower", direction="at_most"),
        iq.target_quantile(0.5, 0.5),
    ):
        with pytest.raises(iq.InverseQueryRefusal) as refused:
            iq.InverseQuery(declared.contract(), ("act", "hold"), (constraint,)).evaluate(
                declared.claim()
            )
        assert refused.value.detail == "decision_evaluation.mean_source_insufficient"
        assert refused.value.reason_code == "decision_contract_unsatisfied"
        assert_registered_refusal(refused.value)


def _check_raw_rejected(*objects: object, feeders: tuple[Callable[[Any], object], ...]) -> None:
    """The producer's own objects are no source: every entry point of the slot types them."""
    for obj in objects:
        for feed in feeders:
            with pytest.raises(CausalTypeError):
                feed(obj)


_DECISION_FEEDERS = (_feed_evaluate, _feed_support)
_INVERSE_FEEDERS = (_feed_inverse,)


# ------------------------------------------------------------------ measurement result


def _declared_measurement(o: Objs) -> Declared:
    estimate = o.measurement
    consumed = mm.consume_measurement_model_artifact(estimate.export(), estimate.model_identity)
    # Oracle: pi = (p* + Sp - 1) / (Se + Sp - 1) with p* = 410 / 1000, Se = 9/10, Sp = 4/5.
    assert estimate.point == pytest.approx(0.3, abs=1e-12)
    assert estimate.naive_point == pytest.approx(0.41, abs=1e-15)
    assert consumed.point == pytest.approx(estimate.point, abs=1e-12)
    assert estimate.operation == "latent_prevalence"
    assert estimate.inference == "uncalibrated_standard_error"
    # The independent consumer re-asks the contract for the analysis the decision is about.
    consumed.check_use(_measurement_analysis(), "latent_prevalence")
    with pytest.raises(MeasurementModelRefusal) as unlicensed:
        o.measurement_prepared.check_operation("condition_on_construct")
    assert unlicensed.value.reason_code == "cell_not_licensed"
    assert unlicensed.value.detail == "measurement_model.operation_not_licensed"
    for closed in (estimate, o.measurement_prepared):
        with pytest.raises(MeasurementModelRefusal) as interval:
            closed.interval()
        assert interval.value.detail == "measurement_model.interval_closed"
    quantity = _declared_quantity(
        "construct:latent", "observational", units="unit", population="cohort"
    )
    return Declared(
        (quantity,),
        (consumed.point,),
        f"measurement_model:{estimate.model_identity}",
        f"data:{estimate.data_identity}",
        "declared:latent_prevalence",
    )


def _check_measurement_decision(o: Objs) -> None:
    declared = _declared_measurement(o)
    _check_declared_decision(declared, act=0.2, hold=0.6, selected="hold")
    _check_raw_rejected(o.measurement, o.measurement_prepared, feeders=_DECISION_FEEDERS)


def _check_measurement_inverse(o: Objs) -> None:
    declared = _declared_measurement(o)
    _check_declared_inverse(declared, target=0.5, feasible=("hold",))
    _check_raw_rejected(o.measurement, o.measurement_prepared, feeders=_INVERSE_FEEDERS)


# ---------------------------------------------------------------------- composed model


def _declared_composition(o: Objs) -> Declared:
    composition = o.composition
    treated = composition.intervention_law({"X": 1.0})
    control = composition.intervention_law({"X": 0.0})
    # Oracle by hand: M = 1/2 + 2 X, Y = -1 + X/2 + 3 M, so E[Y | do(X)] = 1/2 + 13/2 X and
    # Var(Y | do(X)) = 9 * 1 + 1/4.
    assert treated.mean_of("Y") == pytest.approx(7.0, abs=1e-12)
    assert control.mean_of("Y") == pytest.approx(0.5, abs=1e-12)
    assert treated.variance_of("Y") == pytest.approx(9.25, abs=1e-12)
    contrast = composition.mean_contrast(mc.MeanContrast.ate("Y", "X", 1.0, 0.0))
    assert contrast == pytest.approx(6.5, abs=1e-12)
    # The independent consumer replays the premises and the stored query to the same law.
    replayed = mc.consume_mechanism_composition_artifact(
        o.composition_export.artifact, o.composition_export.premises_digest
    )
    assert isinstance(replayed.result, mc.GaussianLawResult)
    assert replayed.result.mean == pytest.approx(7.0, abs=1e-12)
    assert replayed.result.variance == pytest.approx(9.25, abs=1e-12)
    assert o.composition.declared_mechanisms, "a declared mechanism is never labelled fitted"
    with pytest.raises(MechanismCompositionRefusal) as closed:
        composition.interval()
    assert closed.value.reason_code == "cell_not_licensed"
    assert closed.value.details == ("mechanism_composition.interval_route_closed",)
    treated_q = _declared_quantity("Y", "do(X=1)", units="u", population="pop")
    control_q = _declared_quantity("Y", "do(X=0)", units="u", population="pop")
    return Declared(
        (treated_q, control_q),
        (treated.mean_of("Y"), control.mean_of("Y")),
        f"mechanism_composition:{composition.plan_identity}",
        f"premises:{o.composition_export.premises_digest}",
        "declared:mechanism_composition",
    )


def _check_composition_decision(o: Objs) -> None:
    declared = _declared_composition(o)
    _check_declared_decision(declared, act=27.0, hold=1.0, selected="act")
    _check_raw_rejected(o.composition, o.composition_export, feeders=_DECISION_FEEDERS)


def _check_composition_inverse(o: Objs) -> None:
    declared = _declared_composition(o)
    _check_declared_inverse(declared, target=5.0, feasible=("act",))
    _check_raw_rejected(o.composition, o.composition_export, feeders=_INVERSE_FEEDERS)


# ---------------------------------------------------------------------- evidence ledger


def _declared_ledger(o: Objs) -> Declared:
    ledgered = o.ledgered
    pooled = ledgered.pooled
    value = pooled.value
    # Oracle: inverse-variance pooling of y = (1, 3), v = (1, 4): weights (4/5, 1/5), 7/5, 4/5.
    assert isinstance(value, el.FixedEffect)
    assert pooled.claim_class == "point_only"
    assert value.estimate == pytest.approx(7.0 / 5.0, abs=1e-12)
    assert value.variance == pytest.approx(4.0 / 5.0, abs=1e-12)
    # An independent consumer replays the operations and recomputes the pooled values.
    assert pooled.result in ledgered.export.carried
    replayed = el.consume_evidence_ledger_artifact(
        ledgered.export.artifact, ledgered.export.identity
    )
    replay = replayed.synthesis_of(pooled.result)
    assert replay is not None
    assert any(abs(v - 7.0 / 5.0) < 1e-9 for v in replay.values)
    # The dependence range is an assumption range: no point, so no source.
    assert ledgered.envelope.claim_class == "assumption_range"
    assert isinstance(ledgered.envelope.value, el.DependenceEnvelope)
    with pytest.raises(EvidenceLedgerRefusal) as closed:
        ledgered.ledger.pooled_interval(pooled.result)
    assert closed.value.reason_code == "cell_not_licensed"
    assert closed.value.detail == "evidence_ledger.interval_not_licensed"
    # The ledger carries no estimand: the caller declares the pooled quantity.
    quantity = _declared_quantity(
        "pooled_effect",
        "contrast:treated-control",
        units="unit",
        population="target",
        functional="mean_difference",
    )
    return Declared(
        (quantity,),
        (value.estimate,),
        f"evidence_ledger_result:{pooled.result}",
        f"ledger_head:{ledgered.ledger.head}",
        "declared:pooled_effect",
    )


def _ledger_objects(o: Objs) -> tuple[object, ...]:
    ledgered = o.ledgered
    return (
        ledgered.ledger,
        ledgered.pooled,
        ledgered.envelope,
        ledgered.envelope.value,
        ledgered.export,
    )


def _check_ledger_decision(o: Objs) -> None:
    declared = _declared_ledger(o)
    _check_declared_decision(declared, act=4.6, hold=2.8, selected="act")
    _check_raw_rejected(*_ledger_objects(o), feeders=_DECISION_FEEDERS)


def _check_ledger_inverse(o: Objs) -> None:
    declared = _declared_ledger(o)
    _check_declared_inverse(declared, target=3.0, feasible=("act",))
    _check_raw_rejected(*_ledger_objects(o), feeders=_INVERSE_FEEDERS)


# ----------------------------------------------------- counterfactual transport result


def _declared_transport(o: Objs, *, sampled: bool) -> Declared:
    t = o.transported
    # Oracle: P(Y=1 | X=1) = (3/8) / (1/2) = 3/4 and P(Y=1 | X=0) = (1/4) / (1/2) = 1/2.
    assert t.treated.probability == pytest.approx(0.75, abs=1e-12)
    assert t.control.probability == pytest.approx(0.5, abs=1e-12)
    assert t.treated.uncertainty is None and t.treated.scope.endswith("point_only")
    assert t.treated.premises_digest is not None and t.treated.data_digest is not None
    consumed = transport.consume_counterfactual_transport_artifact(
        t.treated.export(),
        expected_premises_digest=t.treated.premises_digest,
        expected_data_digest=t.treated.data_digest,
    )
    assert consumed.independently_verified
    assert consumed.probability == t.treated.probability
    # The sampled plug-in has the same value from counts, four labels and no interval.
    for estimate, truth in ((t.sampled_treated, 0.75), (t.sampled_control, 0.5)):
        assert estimate.probability == pytest.approx(truth, abs=1e-12)
        assert estimate.labels["point_estimate"] == "plug_in_point_estimate"
        assert estimate.labels["statistical_inference"] == "withheld_cell_not_licensed"
        assert estimate.interval["available"] is False
        assert estimate.interval["reason"] == "cell_not_licensed"
        assert estimate.interval["detail"] == (
            "counterfactual_transport_sampled.interval_not_licensed"
        )
        assert estimate.standard_error is not None
        assert estimate.standard_error["label"] == "uncalibrated_standard_error"
    treated_q = _declared_quantity("Y", "do(X=1)", units="probability", population="pop")
    control_q = _declared_quantity("Y", "do(X=0)", units="probability", population="pop")
    if sampled:
        assert t.sampled_treated.probability is not None
        assert t.sampled_control.probability is not None
        return Declared(
            (treated_q, control_q),
            (t.sampled_treated.probability, t.sampled_control.probability),
            f"sampled_counterfactual_transport:{t.sampled_treated.derivation_identity}",
            f"evidence:{t.sampled_treated.evidence_identity}",
            "declared:counterfactual_transport",
        )
    return Declared(
        (treated_q, control_q),
        (t.treated.probability, t.control.probability),
        f"counterfactual_transport:{t.treated.premises_digest}",
        f"data:{t.treated.data_digest}",
        "declared:counterfactual_transport",
    )


def _transport_objects(o: Objs) -> tuple[object, ...]:
    t = o.transported
    return (t.treated, t.control, t.sampled_treated, t.sampled_control)


def _check_transport_decision(o: Objs) -> None:
    for sampled in (False, True):
        declared = _declared_transport(o, sampled=sampled)
        _check_declared_decision(declared, act=2.0, hold=1.0, selected="act")
    _check_raw_rejected(*_transport_objects(o), feeders=_DECISION_FEEDERS)


def _check_transport_inverse(o: Objs) -> None:
    for sampled in (False, True):
        declared = _declared_transport(o, sampled=sampled)
        _check_declared_inverse(declared, target=1.5, feasible=("act",))
    _check_raw_rejected(*_transport_objects(o), feeders=_INVERSE_FEEDERS)


# -------------------------------------------------------------- bundle boundary (new rows)


def _check_no_bundle_node_kind(producer: str, o: Objs) -> None:
    """The producer's artifact is not a bundle node kind, whatever kind it is offered as."""
    payload = _bundle_payload(producer, o)
    assert isinstance(payload, bytes)
    assert cb.detect_kind(payload) is None
    for kind in ("decision_result", "distribution"):
        with pytest.raises(cb.CompositionBundleRefusal) as wrong:
            cb.Bundle.builder().add_artifact(kind, payload)
        assert_registered_refusal(wrong.value)


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
    ("measurement_result", "decision"): _check_measurement_decision,
    ("measurement_result", "inverse_query"): _check_measurement_inverse,
    ("composed_model", "decision"): _check_composition_decision,
    ("composed_model", "inverse_query"): _check_composition_inverse,
    ("evidence_ledger", "decision"): _check_ledger_decision,
    ("evidence_ledger", "inverse_query"): _check_ledger_inverse,
    ("counterfactual_transport_result", "decision"): _check_transport_decision,
    ("counterfactual_transport_result", "inverse_query"): _check_transport_inverse,
}
#: Further checks of a REFUSED cell, beyond feeding every entry point of the consumer.
REFUSED_EXTRAS: dict[tuple[str, str], Check] = {
    (producer, "bundle_node"): (
        lambda o, producer=producer: _check_no_bundle_node_kind(producer, o)
    )
    for producer in (
        "measurement_result",
        "composed_model",
        "evidence_ledger",
        "counterfactual_transport_result",
    )
}


def _run_cell(producer: str, consumer: str, o: Objs) -> None:
    status = _status(producer, consumer)
    if status == "REFUSED":
        _check_refused(producer, consumer, o)
        if (producer, consumer) in REFUSED_EXTRAS:
            REFUSED_EXTRAS[(producer, consumer)](o)
        return
    if (producer, consumer) in CHECKS:
        CHECKS[(producer, consumer)](o)
        return
    assert consumer == "recalc_plan" and status == "ADAPTER", (producer, consumer)
    _check_recalc_for(producer, o)


# ----------------------------------------------------------------------- the table is complete


def test_producer_consumer_matrix_declares_every_cell_once_and_counts_them() -> None:
    assert set(STATUS) == set(PRODUCERS)
    assert all(len(row) == len(CONSUMERS) for row in STATUS.values())
    cells = [status for row in STATUS.values() for status in row]
    assert len(cells) == len(PRODUCERS) * len(CONSUMERS) == 120
    assert cells.count("DIRECT") == 15
    assert cells.count("ADAPTER") == 27
    assert cells.count("REFUSED") == 78
    assert set(cells) == {"DIRECT", "ADAPTER", "REFUSED"}
    # Every non-refused cell has a check; the only shared one is the recalc adapter.
    for producer in PRODUCERS:
        for consumer in CONSUMERS:
            status = _status(producer, consumer)
            if status != "REFUSED" and consumer != "recalc_plan":
                assert (producer, consumer) in CHECKS, (producer, consumer)
    for producer, consumer in CHECKS:
        assert _status(producer, consumer) != "REFUSED", (producer, consumer)


# ------------------------------------------------------------------------- one test per row


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_native_response_row(consumer: str, objs: Objs) -> None:
    _run_cell("native_response", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_fitted_effect_model_row(consumer: str, objs: Objs) -> None:
    _run_cell("fitted_effect_model", consumer, objs)
    # Its own consumer is the only thing that evaluates it, with a model-based SE and no more.
    estimate = objs.compact.evaluate({"x": 2.0})
    assert estimate.point == pytest.approx(3.0, abs=1e-12)
    assert "unmeasured" in estimate.calibration


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_retargeted_batch_row(consumer: str, objs: Objs) -> None:
    _run_cell("retargeted_batch", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_external_claim_row(consumer: str, objs: Objs) -> None:
    _run_cell("external_claim", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_joint_draws_row(consumer: str, objs: Objs) -> None:
    _run_cell("joint_draws", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_bayesian_prior_row(consumer: str, objs: Objs) -> None:
    _run_cell("bayesian_prior", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_identified_scenarios_row(consumer: str, objs: Objs) -> None:
    _run_cell("identified_scenarios", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_sensitivity_artifact_row(consumer: str, objs: Objs) -> None:
    _run_cell("sensitivity_artifact", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_decision_row(consumer: str, objs: Objs) -> None:
    _run_cell("decision", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_study_ranking_row(consumer: str, objs: Objs) -> None:
    _run_cell("study_ranking", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_failed_contract_row(consumer: str, objs: Objs) -> None:
    _run_cell("failed_contract", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_measurement_result_row(consumer: str, objs: Objs) -> None:
    _run_cell("measurement_result", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_composed_model_row(consumer: str, objs: Objs) -> None:
    _run_cell("composed_model", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_evidence_ledger_row(consumer: str, objs: Objs) -> None:
    _run_cell("evidence_ledger", consumer, objs)


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_producer_consumer_matrix_counterfactual_transport_result_row(
    consumer: str, objs: Objs
) -> None:
    _run_cell("counterfactual_transport_result", consumer, objs)


# ------------------------------------------------------------------------ cross-cutting rules


def test_producer_consumer_matrix_every_boundary_rejects_neighbours_with_typed_errors(
    objs: Objs,
) -> None:
    """Every refused producer reaches a checked boundary, including exporter protocols."""
    for producer in PRODUCERS:
        for consumer in FEEDERS:
            if _status(producer, consumer) != "REFUSED":
                continue
            for payload in _payloads(producer, consumer, objs):
                for label, feed, expected in FEEDERS[consumer]:
                    assert expected is not None, (producer, label)
                    with pytest.raises(expected):
                        feed(payload)


def test_producer_consumer_matrix_invalid_export_signatures_are_typed_but_callback_errors_propagate(
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


def test_producer_consumer_matrix_a_mean_never_answers_a_probability_a_quantile_or_a_law(
    objs: Objs,
) -> None:
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
