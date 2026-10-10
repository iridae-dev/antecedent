"""Shared fixtures of the 2.3 C4 multi-source acceptance story.

The story, its hand derivation and every asserted value are written out in the header of
``test_c4_multi_source.py``; this module only builds the objects. Nothing here computes an
expected value: the numbers below are the inputs the header derives its answers from.

* program: outcome ``y`` (mmHg) under ``do(a = d)`` for the dose grid ``d in {0, 1, 2}`` mg in
  the ``target`` population, identified on the graph ``x -> a``, ``x -> y``, ``a -> y``;
* native analysis ``N`` (mean response): ``E[y | do(a)] = 2, 4, 6``, the dose-2 coordinate outside
  empirical support;
* external study ``E1`` (attested point means, ``lab-1`` / ``snap-e1``): ``1, 3, 5.5``;
* external study ``E2`` (``lab-2`` / ``snap-e2``): point means ``2, 3.5, 5`` and two equally likely
  joint draws ``(y0, y1, y2, state) = (2, 6, 3, 1/4)`` and ``(2, 1, 7, 3/4)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import antecedent as ac
import numpy as np
from antecedent import composition as comp
from antecedent import composition_bundle as cb
from antecedent import decision, external, program_claims
from antecedent import design as dr
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.results.response import (
    CausalResponseView,
)

GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]
UNITS = "mmHg"
ASSUMPTION = "ignorability"
SUPPORTED = ("supported", "supported", "supported")
OUTSIDE = "outside_empirical_support"

NATIVE_MEANS = (2.0, 4.0, 6.0)
E1_MEANS = (1.0, 3.0, 5.5)
E2_MEANS = (2.0, 3.5, 5.0)
#: Rows of the E2 joint law: ``(y(do(a=0)), y(do(a=1)), y(do(a=2)), response rate)``.
LAW_ROWS = ((2.0, 6.0, 3.0, 0.25), (2.0, 1.0, 7.0, 0.75))
#: The ranking decision's prior draws: the law's own state column.
STATE_DRAWS = [0.25, 0.75]
#: The node the declared independence of ``e1`` and ``e2claim`` becomes in a bundle.
RELATION = "relation:e1:e2claim"


def near(value: float | None, expected: float, tolerance: float = 1e-12) -> bool:
    return value is not None and abs(value - expected) <= tolerance


# ----------------------------------------------------------------------- program and claims


def identification(
    edges: list[tuple[str, str]] | None = None,
    names: list[str] | None = None,
    grid: list[float] | None = None,
) -> Any:
    return ac.identify(
        graph=EDGES if edges is None else edges,
        names=NAMES if names is None else names,
        query=ac.ResponseCurve("a", "y", grid=GRID if grid is None else grid),
    )


def spec(ident: Any = None, **kwargs: Any) -> external.ExternalSpec:
    return external.response(
        native_view().program_identification if ident is None else ident,
        outcome_units=UNITS,
        dose_units="mg",
        population="target",
        require_assumptions=(ASSUMPTION,),
        **kwargs,
    )


def program(ident: Any = None) -> program_claims.ProgramBinding:
    return program_claims.ProgramBinding.from_identification(
        native_view().program_identification if ident is None else ident,
        outcome_units=UNITS,
        dose_units="mg",
        require_assumptions=(ASSUMPTION,),
    )


def provider_object(
    provider_id: str, snapshot: str, request: str, **kwargs: Any
) -> external.ProviderObject:
    fields: dict[str, Any] = {
        "provider_id": provider_id,
        "object_id": "curve",
        "version": "v1",
        "snapshot": snapshot,
        "request": request,
        "meaning": "interventional_predictive",
        "capabilities": ("mean",),
    }
    return external.ProviderObject(**{**fields, **kwargs})


def provider_response(
    provider: external.ProviderObject,
    values: tuple[float, ...],
    evidence: str,
    support: tuple[str, ...] = SUPPORTED,
    assumptions: tuple[str, ...] = (ASSUMPTION,),
) -> external.Response:
    return external.Response(
        provider=provider,
        values=values,
        evidence=(evidence,),
        assumptions=assumptions,
        attested_by=provider.provider_id,
        support=support,
    )


def bound_claim(
    response: external.Response,
    the_spec: external.ExternalSpec | None = None,
    the_program: program_claims.ProgramBinding | None = None,
) -> external.BoundExternalClaim:
    """Bind ``response`` through the checked program contract, never the bare spec."""
    checked = program_claims.bind_to_program(
        spec() if the_spec is None else the_spec, program() if the_program is None else the_program
    )
    return checked.bind(response)


def e1_claim(
    *,
    support: tuple[str, ...] = SUPPORTED,
    evidence: str = "study:e1",
    snapshot: str = "snap-e1",
    request: str = "req-e1",
) -> external.BoundExternalClaim:
    provider = provider_object("lab-1", snapshot, request)
    return bound_claim(provider_response(provider, E1_MEANS, evidence, support))


def e2_claim(
    *,
    evidence: str = "study:e2",
    snapshot: str = "snap-e2",
    request: str = "req-e2",
) -> external.BoundExternalClaim:
    provider = provider_object("lab-2", snapshot, request)
    return bound_claim(provider_response(provider, E2_MEANS, evidence))


# ------------------------------------------------------------------------- native response


@lru_cache(maxsize=8)
def _executed_native(grid: tuple[float, ...]) -> CausalResponseView:
    """Fit the original checked response engine to a full-rank additive SCM.

    Cross each treatment level with the same symmetric covariate population:
    Y = 2 + 2 A + X/2, E[X]=0, hence E[Y|do(a)]=2+2a.
    Fixed bandwidth 2.1 permits evaluating the full query while the original
    producer still labels doses beyond [0,1] outside empirical support.
    """
    levels = np.linspace(0.0, 1.0, 101)
    treatment = np.repeat(levels, 21)
    covariate = np.tile(np.linspace(-1.0, 1.0, 21), len(levels))
    return ac.analyze(
        {"x": covariate, "a": treatment, "y": 2 + 2 * treatment + covariate / 2},
        graph=EDGES,
        query=ac.ResponseCurve("a", "y", grid=list(grid)),
        estimator_config={
            "bandwidth": 2.1,
            "nuisance_lambda": 0.0,
            "nuisance_basis": 4,
            "folds": 2,
        },
        bootstrap=0,
        refute="none",
    )


def native_view(
    *,
    grid: tuple[float, ...] = (0.0, 1.0, 2.0),
    means: tuple[float, ...] | None = None,
    point_status: tuple[str, ...] | None = None,
    support: str | None = None,
    snapshot: str | None = None,
) -> CausalResponseView:
    """An actual issued response; optional substitutions are negative fixtures."""
    view = _executed_native(grid)
    if means is not None:
        view = view.model_copy(
            update={"response": view.response.model_copy(update={"values": [[m] for m in means]})}
        )
    if support is not None or point_status is not None:
        view = view.model_copy(
            update={
                "support": view.support.model_copy(
                    update={
                        "status": view.support.status if support is None else support,
                        "point_status": view.support.point_status
                        if point_status is None
                        else point_status,
                    }
                )
            }
        )
    if snapshot is not None:
        view = view.model_copy(update={"data_snapshot_id": snapshot})
    return view


#: These are supplied-source assertions in historical bundles, not replay authority.
SNAPSHOT = str(_executed_native((0.0, 1.0, 2.0)).data_snapshot_id)
NATIVE_DIGEST = SNAPSHOT


def native_claim(
    view: CausalResponseView | None = None,
    the_program: program_claims.ProgramBinding | None = None,
) -> program_claims.NativeClaim:
    return program_claims.native_claim(
        native_view() if view is None else view, program() if the_program is None else the_program
    )


# --------------------------------------------------------------------------- coordinates


def coordinate(dose: float, functional: str = "mean") -> ScientificQuantity:
    """The coordinate written out by hand, independent of the production derivation."""
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units=UNITS,
        population_id="target",
        regime_id=f"do(a={dose:g})",
        horizon=0,
        functional_id=functional,
        conditioning=(),
        transform_id="identity",
    )


def state_quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="response_rate",
        variable_name="response_rate",
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


def observation_quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="trial_successes",
        variable_name="trial_successes",
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="count",
    )


# --------------------------------------------------------------------------------- laws


def law_columns() -> tuple[ScientificQuantity, ...]:
    return (
        coordinate(0.0, "outcome"),
        coordinate(1.0, "outcome"),
        coordinate(2.0, "outcome"),
        state_quantity(),
    )


def law(
    *,
    provider_id: str = "lab-2",
    snapshot: str = "snap-e2",
    supported: tuple[bool, ...] | None = None,
    alignment: str = "joint",
    columns: tuple[ScientificQuantity, ...] | None = None,
) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=law_columns() if columns is None else columns,
        alignment=alignment,  # type: ignore[arg-type]
        source_id=f"study-{snapshot}",
        provider_id=provider_id,
        rng_id="deterministic_exact",
        snapshot_id=snapshot,
        causal_contract_id=program().identity,
    )
    draws = np.array(LAW_ROWS, dtype=np.float64)
    return JointDistributionArtifact(identity, draws, supported=supported)


# ------------------------------------------------------------------------------ contracts


def _actions(functional: str) -> tuple[decision.Action, ...]:
    return (
        decision.Action("wait", inputs=(coordinate(0.0, functional),), utility=decision.x(0)),
        decision.Action(
            "treat", inputs=(coordinate(1.0, functional),), utility=decision.x(0) - 1.0
        ),
        decision.Action(
            "extend", inputs=(coordinate(2.0, functional),), utility=decision.x(0) - 3.0
        ),
    )


def mean_contract(criterion: decision.Criterion | None = None) -> decision.Contract:
    """wait: ``m0``; treat: ``m1 - 1``; extend: ``m2 - 3``, over the mean coordinates."""
    return decision.Contract(
        actions=_actions("mean"),
        utility_units="util",
        criterion=criterion or decision.Criterion.expected_utility(),
        target_population="target",
    )


def law_contract() -> decision.Contract:
    """The same three actions over the outcome-law coordinates of the E2 joint law."""
    return decision.Contract(
        actions=_actions("outcome"),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def rollout_contract() -> decision.Contract:
    """The follow-up decision priced in the response rate ``theta``: bet pays ``theta - 1/2``."""
    state = state_quantity()
    return decision.Contract(
        actions=(
            decision.Action("abstain", inputs=(state,), utility=decision.x(0) * 0.0),
            decision.Action("bet", inputs=(state,), utility=decision.x(0) - 0.5),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


# ------------------------------------------------------------------------ decision inputs


def native_input(
    the_claim: program_claims.NativeClaim | None = None, input_id: str = "native"
) -> comp.DecisionInput:
    """The actual issued native claim, retaining its supported coordinates and authority."""
    claim = native_claim() if the_claim is None else the_claim
    return comp.DecisionInput.from_native_claim(input_id, claim, contract=mean_contract())


def e1_input(
    claim: external.BoundExternalClaim | None = None, input_id: str = "e1"
) -> comp.DecisionInput:
    return comp.DecisionInput.from_claim(input_id, e1_claim() if claim is None else claim)


def e2_mean_input(input_id: str = "e2mean") -> comp.DecisionInput:
    return comp.DecisionInput.from_claim(input_id, e2_claim())


def e2_law_input(
    artifact: JointDistributionArtifact | None = None, input_id: str = "e2law"
) -> comp.DecisionInput:
    return comp.DecisionInput.from_distribution(
        input_id,
        law() if artifact is None else artifact,
        evidence=comp.TrustEvidence.attested("lab-2"),
    )


# ---------------------------------------------------------------------------- ranking


def ranking_decision() -> dr.DesignDecision:
    """abstain pays 0 and bet pays ``theta - 1/2`` over the prior draws ``{1/4, 3/4}``."""
    return dr.DesignDecision(
        contract=rollout_contract(),
        actions=(dr.ActionUtility("abstain", 0.0, 0.0), dr.ActionUtility("bet", -0.5, 1.0)),
        prior=dr.StatePrior.draws(STATE_DRAWS),
    )


def signal_spec() -> dr.SignalSpec:
    return dr.SignalSpec(
        prior_id="prior:e2-state",
        state=state_quantity(),
        observation=observation_quantity(),
        evidence_lineage=("snapshot:snap-e2",),
        rng_seed=3,
    )


#: ``P(k successes | theta)`` posterior of two Bernoulli trials, with ``theta in {1/4, 3/4}``.
EQUIVALENT_POSTERIOR = dr.ExternalLaw.posterior(
    states=[0.25, 0.75],
    statistics=[0.0, 1.0, 2.0],
    predictive=[0.3125, 0.375, 0.3125],
    posterior=[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],
)


def candidates(
    *, external_reuses: tuple[str, ...] = (), law_of_signal: dr.ExternalLaw | None = None
) -> list[dr.Candidate]:
    attested = dr.ExternalSignal(
        "lab-2",
        "trial-2",
        "v1",
        "snap-e2",
        "lab-qa",
        EQUIVALENT_POSTERIOR if law_of_signal is None else law_of_signal,
    )
    return [
        dr.Candidate("native-trial", 2, dr.BinomialSignal(), cost=0.02, cost_unit="utility"),
        dr.Candidate(
            "external-trial",
            2,
            attested,
            cost=0.05,
            cost_unit="utility",
            reused_observations=external_reuses,
        ),
    ]


def rank(**kwargs: Any) -> dr.DesignRankingResult:
    options: dict[str, Any] = {
        "signal": signal_spec(),
        "cost_map": dr.CostMap("utility", "utility", 1.0),
        "prior_observations": ("obs:e2-registry",),
        "source_digests": (decision.source_digest(law()),),
        "rng_seed": 5,
    }
    candidate_list = kwargs.pop("candidate_list", None)
    options.update(kwargs)
    return dr.rank_designs(
        candidates() if candidate_list is None else candidate_list,
        decision=ranking_decision(),
        **options,
    )


# --------------------------------------------------------------------------- bundles


def composed_builder(
    *,
    e2_evidence: str = "study:e2",
    embedded_law: JointDistributionArtifact | None = None,
) -> cb.BundleBuilder:
    """Program reference, both claims, their declared relation, the law, decision and ranking.

    ``embedded_law`` replaces only the law bytes a consumer reads; the decision result and the
    ranking stay computed from the original law, as in a tampered export.
    """
    artifact = law()
    contract = law_contract()
    builder = cb.Bundle.builder()
    builder.add_reference(
        "program",
        "causal_contract",
        identity=program().identity,
        requires=cb.DataRequirement(SNAPSHOT, NATIVE_DIGEST),
        inspected={"native.mean.do(a=1)": 4.0},
    )
    builder.add_artifact("external_claim", e1_claim(), node_id="e1")
    builder.add_artifact("external_claim", e2_claim(evidence=e2_evidence), node_id="e2claim")
    builder.relate("e1", "e2claim", "independent")
    builder.add_artifact(
        "decision_contract", contract.export(artifact_id="contract"), node_id="contract_j"
    )
    shown = artifact if embedded_law is None else embedded_law
    builder.add_artifact("distribution", shown.export("law"), node_id="e2law")
    builder.add_artifact(
        "decision_result",
        contract.evaluate(artifact).export(artifact_id="result"),
        node_id="result_j",
    )
    builder.add_artifact("study_ranking", rank().export(), node_id="ranking")
    for upstream, dependent in (
        ("program", "e1"),
        ("program", "e2claim"),
        ("contract_j", "result_j"),
        ("e2law", "result_j"),
        ("e2law", "ranking"),
    ):
        builder.connect(upstream, dependent)
    return builder


def point_only_builder(claim: external.BoundExternalClaim | None = None) -> cb.BundleBuilder:
    """The E1 mean decision, exported as a point-only attested result."""
    bound = e1_claim() if claim is None else claim
    contract = mean_contract()
    builder = cb.Bundle.builder()
    builder.add_artifact("external_claim", bound, node_id="claim")
    builder.add_artifact(
        "decision_contract", contract.export(artifact_id="contract"), node_id="contract_m"
    )
    builder.add_artifact("decision_result", cb.mean_decision(contract, bound), node_id="result_m")
    return builder.connect("claim", "result_m").connect("contract_m", "result_m")
