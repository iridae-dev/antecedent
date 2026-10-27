"""Evidence plans through the Python surface: candidate templates from verified
obligations, every theorem route's own check, ordering by the identification gate and then
by value net of a declared cost map, budget versus impossibility, equal cost units, derived
unlocks and structure flags, and the replayable ``evidence_plan_v1`` artifact.

Expected verdicts follow from the theorems (back-door set {z}, front-door mediator {m},
instrument validity as an assumption no study establishes, a hedge answered by an experiment),
and the values from the two-point binomial bet written out by hand: prior ``theta`` in
{1/4, 3/4}, the bet pays ``theta - 1/2``, so ``EVSI(n=2) = 1/16``, ``EVSI(n=4) = 11/128`` and
``EVPI = 1/8``.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import Admg, Dag, design, repair
from antecedent import sensitivity_decision as sd
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.joint_distribution import ScientificQuantity

from _refusal import assert_registered_refusal


def obs(label, measured, cost, *, n=2, unit="usd", kind="observation", **fields):
    make = getattr(repair.StudyCandidate, kind)
    return make(
        label,
        population="study",
        measured=measured,
        cost=cost,
        sample_size=n,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost_unit=unit,
        **fields,
    )


def backdoor_contract(**fields):
    graph = Dag.from_edges(["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")])
    return repair.FailureContract(
        graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"], **fields
    )


def complete(template, cost, *, n=2, unit="usd", **fields):
    return template.complete(
        sample_size=n,
        recruitment="consecutive recruitment",
        timing="baseline",
        unit="patient",
        cost=cost,
        cost_unit=unit,
        sample_budget=n,
        feasible=True,
        **fields,
    )


def template_measuring(contract, names, kind="observation"):
    found = [
        t
        for t in repair.candidate_templates(contract)
        if t.kind == kind and set(t.measured) == set(names)
    ]
    assert len(found) == 1, (
        kind,
        names,
        [(t.kind, t.measured) for t in repair.candidate_templates(contract)],
    )
    return found[0]


# -- candidate generation ---------------------------------------------------------------------


def test_a_template_fills_only_what_the_obligations_determine():
    contract = backdoor_contract()
    assert "backdoor" in contract.routes
    template = template_measuring(contract, {"t", "y", "z"})
    assert template.joint and template.interventions == ()
    assert template.addresses  # the obligation ids it could address
    # Cost, timing, recruitment, unit rules, feasibility and sample size are never invented.
    with pytest.raises(repair.StudyCandidateRefusal) as caught:
        template.validate()
    assert caught.value.detail == "study_candidate.declarations_required"
    with pytest.raises(TypeError, match="required keyword"):
        template.complete(sample_size=2)  # type: ignore[call-arg]
    with pytest.raises(repair.StudyCandidateRefusal):
        complete(template, 0)  # a zero cost is no declaration
    candidate = complete(template, 3, label="registry with z")
    assert candidate.label == "registry with z" and candidate.kind == "observation"


def test_assumptions_are_returned_unmatched_and_the_instrument_route_proposes_two_kinds():
    graph = Dag.from_edges(["z", "t", "y", "u"], [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")])
    contract = repair.FailureContract(
        graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
    )
    generation = repair.generate_candidates(contract)
    kinds = {t.kind for t in generation.templates}
    assert {"instrument", "natural_experiment"} <= kinds
    assert generation.unmatched
    assert all(u.kind == "establish_assumption" for u in generation.unmatched)


# -- verification and ordering ----------------------------------------------------------------


def test_a_back_door_study_is_verified_and_its_unlocks_are_derived():
    contract = backdoor_contract()
    cohort = complete(template_measuring(contract, {"t", "y", "z"}), 3, label="cohort")
    blind = obs("blind registry", ["t", "y"], 1)
    plan = design.evidence_plan(contract, [cohort, blind])
    assert plan.outcome == "closing" and plan.reason_code is None
    best = plan.best
    assert best is not None and best.route == "backdoor" and best.labels == ("cohort",)
    assert best.gate == "verified_sufficient" and best.derivation.verified
    assert best.resolves and best.unresolved == ()
    assert plan.options[-1].gate == "insufficient"
    assert plan.inference_claim == "none"
    assert (plan.ranking_guarantee, plan.calibration) == ("point_only", "unmeasured")
    # The ranker's unlock list comes from the verified study, not from a hand-written list.
    assert plan.unlocks.variables == ("z",) and plan.unlocks.verified == ("cohort",)
    assert plan.unlocks.unverified == ("blind registry",)
    assert plan.variable_unlocks(query_id=4) == {4: (2,)}
    assert "hypothetical identification" in plan.explain()


def test_several_routes_are_compared_in_one_plan():
    graph = Dag.from_edges(["t", "m", "y", "u"], [("t", "m"), ("m", "y"), ("u", "t"), ("u", "y")])
    contract = repair.FailureContract(
        graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
    )
    mediator = obs("mediator cohort", ["t", "m", "y"], 4)
    assay = obs("confounder assay", ["t", "y", "u"], 2)
    plan = design.evidence_plan(contract, [mediator, assay])
    sufficient = {(o.route, o.labels[0]) for o in plan.closing}
    assert sufficient == {("backdoor", "confounder assay"), ("frontdoor", "mediator cohort")}
    # Without a value the order within the gate is the declared cost.
    assert [o.labels[0] for o in plan.closing] == ["confounder assay", "mediator cohort"]


def test_a_hedge_is_answered_by_an_experiment_and_a_sampling_study_is_rejected():
    graph = Admg.from_edges(["t", "w", "y"], [("t", "w"), ("w", "y")], [("t", "y"), ("w", "y")])
    contract = repair.FailureContract(
        graph=graph, treatment="t", outcome="y", population="study", observed=["t", "w", "y"]
    )
    trial = repair.StudyCandidate.experiment(
        "randomized trial",
        population="study",
        interventions=["t"],
        measured=["y"],
        cost=9,
        sample_size=2,
        recruitment="randomized",
        timing="baseline",
        unit="patient",
        cost_unit="usd",
    )
    rows = repair.StudyCandidate.sample_increase(
        "more rows",
        population="study",
        measured=["t", "w", "y"],
        cost=1,
        sample_size=4,
        recruitment="census",
        timing="baseline",
        unit="patient",
        cost_unit="usd",
    )
    plan = design.evidence_plan(contract, [trial, rows])
    assert plan.best is not None and plan.best.labels == ("randomized trial",)
    assert all(o.gate != "verified_sufficient" for o in plan.options if o.labels == ("more rows",))
    assert plan.unlocks.verified == ("randomized trial",) and plan.unlocks.unverified == (
        "more rows",
    )
    assert plan.unlocks.variables == ("t",)


def test_an_instrument_study_is_not_certified_and_the_assumptions_remain():
    graph = Dag.from_edges(["z", "t", "y", "u"], [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")])
    contract = repair.FailureContract(
        graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
    )
    trial = complete(template_measuring(contract, {"z", "t", "y"}, "instrument"), 3)
    plan = design.evidence_plan(contract, [trial])
    option = next(o for o in plan.options if o.route == "instrument")
    assert option.gate == "not_certified"
    assert "does not establish assumption" in option.reasons[0]
    assert len(option.assumptions_remaining) == 3
    assert plan.outcome == "none_certified" and plan.best is None
    assert plan.reason_code == "transport_missing_evidence"
    assert "no study establishes" in plan.explain()


# -- budget, units, refusals -----------------------------------------------------------------


def test_a_budget_stop_is_a_budget_and_never_an_impossibility():
    contract = backdoor_contract()
    candidates = [obs("a", ["t", "y"], 1), obs("b", ["t", "y"], 1, n=4)]
    full = design.evidence_plan(contract, candidates)
    assert full.outcome == "none_certified"
    assert full.detail == "evidence_plan.none_certified"
    stopped = design.evidence_plan(
        contract, candidates, limits=repair.RepairLimits(max_operations=1, max_depth=2)
    )
    assert stopped.outcome == "budget_exhausted"
    assert stopped.reason_code == "transport_budget_cancel"
    assert any(route.unevaluated_total > 0 for route in stopped.routes)
    assert any(o.gate == "unevaluated" for o in stopped.options)
    assert "not impossible" in stopped.explain()


def test_costs_are_compared_only_in_one_unit():
    contract = backdoor_contract()
    candidates = [
        obs("usd cohort", ["t", "y", "z"], 1),
        obs("eur cohort", ["t", "y", "z"], 1, n=4, unit="eur"),
    ]
    with pytest.raises(CausalUnsupportedError) as caught:
        design.evidence_plan(contract, candidates)
    assert caught.value.reason_code == "design_cost_units_mismatch"
    assert_registered_refusal(caught.value)
    with pytest.raises(CausalUnsupportedError) as caught:
        design.evidence_plan(contract, candidates, objective="minimize_sample_budget")
    assert caught.value.reason_code == "design_cost_units_mismatch"
    assert caught.value.detail == "evidence_plan.cost_units_mismatch"


def test_an_identified_graph_has_no_gap_to_plan_for():
    graph = Dag.from_edges(["t", "y"], [("t", "y")])
    with pytest.raises(CausalUnsupportedError) as caught:
        repair.FailureContract(
            graph=graph, treatment="t", outcome="y", population="study", observed=["t", "y"]
        )
    assert caught.value.detail == "identification_repair.not_a_failure"
    with pytest.raises(CausalTypeError):
        repair.repair(backdoor_contract(), [])  # a failure spans routes: use evidence_plan


# -- value ------------------------------------------------------------------------------------


def _quantity(name: str) -> ScientificQuantity:
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


def _signal() -> design.SignalSpec:
    return design.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=3,
    )


def _bet() -> design.DesignDecision:
    return design.DesignDecision(
        contract="contract-bet",
        actions=(design.ActionUtility("abstain", 0.0, 0.0), design.ActionUtility("bet", -0.5, 1.0)),
        prior=design.StatePrior.draws([0.25, 0.75]),
        utility_units="utility",
    )


def _valued(per_cost: float):
    contract = backdoor_contract()
    template = template_measuring(contract, {"t", "y", "z"})
    cheap = complete(template, 1, n=2, label="cheap")
    large = complete(template, 3, n=4, label="large")
    value = design.EvidenceValue(
        decision=_bet(),
        studies={
            "cheap": design.StudyValue(design.BinomialSignal(), _signal()),
            "large": design.StudyValue(design.BinomialSignal(), _signal()),
        },
        cost_map=design.CostMap("usd", "utility", per_cost),
        rng_seed=5,
    )
    return design.evidence_plan(contract, [cheap, large], value=value)


def test_options_of_one_gate_are_ordered_by_value_net_of_the_declared_cost_map():
    plan = _valued(0.01)
    assert plan.basis == "net_value" and plan.utility_unit == "utility"
    # The plan lists each study under every route; the verified gate holds the back-door pair.
    verified = [o for o in plan.options if o.gate == "verified_sufficient"]
    assert [o.labels[0] for o in verified] == ["large", "cheap"]
    first = plan.options[0].value
    assert first.evsi == pytest.approx(11 / 128, abs=1e-12)
    assert first.net_value == pytest.approx(11 / 128 - 0.03, abs=1e-12)
    assert first.evpi == pytest.approx(1 / 8, abs=1e-12)
    # A dearer cost map reverses the order of the verified pair.
    reversed_plan = _valued(0.02)
    reversed_verified = [o for o in reversed_plan.options if o.gate == "verified_sufficient"]
    assert [o.labels[0] for o in reversed_verified] == ["cheap", "large"]
    assert reversed_verified[0].value.net_value == pytest.approx(1 / 16 - 0.02, abs=1e-12)
    assert plan.cost_map["cost_unit"] == "usd" and plan.value_identity


def test_measuring_the_named_confounder_is_valued_under_the_declared_prior():
    quantity = lambda name: sd.SurfaceQuantity(  # noqa: E731
        ScientificQuantity(
            variable_id=name,
            variable_name=name,
            role="outcome",
            units="utils",
            population_id="target",
            regime_id="do(a=1)",
            horizon=0,
            functional_id="sensitivity_surface",
        ),
        ((0.0, 0.0) if name == "ua" else (-0.25, 0.25)),
    )
    artifact = sd.SensitivityArtifact.from_surface(
        coordinate=sd.AssumptionCoordinate(
            "gamma", "sensitivity_parameter", "dimensionless", 0.0, 0.75
        ),
        grid=[0.25, 0.75],
        quantities=[quantity("ua"), quantity("ub")],
        actions=[
            sd.SensitivityAction("abstain", sd.quantity("ua")),
            sd.SensitivityAction("bet", sd.quantity("ub")),
        ],
        provenance=sd.SurfaceProvenance(
            source_kind="supplied_surface",
            query_binding="sensitivity-test",
            provider_snapshot="snapshot-1",
            source_regime="regime:1",
            method="hand-derived surface",
            causal_contract_id="checked-contract",
        ),
    )
    prior = design.ConfounderPrior("analyst", "pilot assay, 2026", (1, 1))
    decision = design.confounder_decision(artifact, prior)
    assert decision.coordinates == (0.25, 0.75) and decision.prior.states == (0.25, 0.75)
    assert decision.prior_identity.startswith("cp1:")
    finding = repair.SensitivityFinding("e_value", 1.4)
    graph = Dag.from_edges(["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")])
    contract = repair.FailureContract(
        graph=graph,
        treatment="t",
        outcome="y",
        population="study",
        observed=["t", "y", "z"],
        sensitivity=finding,
    )
    assert "unmeasured_confounder" in contract.variable_names
    assay = obs("confounder assay", ["t", "y", "z", "unmeasured_confounder"], 1)
    rows = obs("more rows", ["t", "y", "z"], 1, n=4)
    value = design.EvidenceValue(
        decision=decision,
        studies={"confounder assay": design.StudyValue(design.BinomialSignal(), _signal())},
        cost_map=design.CostMap("usd", "utils", 0.01),
        rng_seed=5,
    )
    plan = design.evidence_plan(contract, [assay, rows], value=value)
    best = plan.best
    assert best is not None and best.route == "sensitivity" and best.labels == ("confounder assay",)
    assert best.value.evsi == pytest.approx(1 / 16, abs=1e-12)
    assert best.value.evpi == pytest.approx(1 / 8, abs=1e-12)
    # The declared prior is part of the value's recorded sources: another prior, another identity.
    other = design.confounder_decision(
        artifact, design.ConfounderPrior("analyst", "pilot assay, 2026", (1, 2))
    )
    assert other.prior_identity != decision.prior_identity
    # An assumption range is never a probability: it needs a declared point surface.
    ranged = sd.SensitivityArtifact.from_surface(
        coordinate=sd.AssumptionCoordinate(
            "gamma", "sensitivity_parameter", "dimensionless", 0.0, 0.75
        ),
        grid=[0.25, 0.75],
        quantities=[
            sd.SurfaceQuantity(quantity("ua").quantity, (0.0, 0.0)),
            sd.SurfaceQuantity(quantity("ub").quantity, (-1.0, -1.0), (1.0, 1.0)),
        ],
        actions=[
            sd.SensitivityAction("abstain", sd.quantity("ua")),
            sd.SensitivityAction("bet", sd.quantity("ub")),
        ],
        provenance=artifact.provenance,
    )
    with pytest.raises(sd.SensitivityRefusal):
        design.confounder_decision(ranged, prior)


# -- structure prior --------------------------------------------------------------------------


def test_structure_flags_are_derived_from_the_identifier_not_declared():
    confounded = Admg.from_edges(["t", "y"], [("t", "y")], [("t", "y")])
    clean = Admg.from_edges(["t", "y"], [("t", "y")])
    prior = design.StructurePrior.from_graphs(
        [confounded, clean],
        treatment="t",
        outcome="y",
        weights=[0.3, 0.7],
        interventions=["t"],
    )
    assert prior.identified == (False, True)
    assert prior.identified_under_intervention == (True, True)
    assert prior.weights == (0.3, 0.7)
    with pytest.raises(CausalTypeError):
        design.StructurePrior.from_graphs([], treatment="t", outcome="y")


# -- artifact ---------------------------------------------------------------------------------


def test_the_plan_exports_and_replays_independently():
    plan = _valued(0.01)
    data = bytes(plan.export())
    replayed = design.consume_plan(data)
    assert replayed.identity == plan.identity
    assert [o.labels for o in replayed.options] == [o.labels for o in plan.options]
    assert replayed.options[0].value.evsi == plan.options[0].value.evsi
    assert replayed.outcome == "closing"
    with pytest.raises(CausalTypeError):
        design.consume_plan("not bytes")  # type: ignore[arg-type]
    damaged = bytearray(data)
    damaged[len(damaged) // 2] ^= 0xFF
    with pytest.raises((CausalSerializationError, CausalUnsupportedError)):
        design.consume_plan(bytes(damaged))
    with pytest.raises(CausalValueError):
        replayed.variable_unlocks()


def test_a_candidate_may_name_only_variables_the_contract_declares():
    contract = backdoor_contract()
    assert set(contract.variable_names) >= {"t", "y", "z"}
    assert "ghost" not in contract.variable_names
    plan = design.evidence_plan(contract, [obs("registry with z", ["t", "y", "z"], 1)])
    assert plan.best is not None and plan.best.labels == ("registry with z",)
    with pytest.raises((CausalValueError, CausalUnsupportedError)):
        design.evidence_plan(contract, [obs("ghost cohort", ["t", "y", "ghost"], 1)])


# -- ranking through a prepared study's contract -------------------------------------------------


def licensed_study():
    n = 120
    z = np.sin(np.arange(n) / 7.0)
    t = (z + np.cos(np.arange(n) / 5.0) > 0).astype(float)
    return ant.prepare(
        {"t": t, "y": 1.0 + 2.0 * t + 0.5 * z, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=3),
    )


def test_only_the_study_the_repair_search_verified_is_ranked_through_the_contract():
    contract = backdoor_contract()
    cohort = complete(template_measuring(contract, {"t", "y", "z"}), 3, label="cohort")
    blind = obs("blind registry", ["t", "y"], 1)
    prior = design.StructurePrior((1.0, 1.0), (True, False), (1, 2))
    ranking = design.rank_studies(
        licensed_study(), contract, [blind, cohort], prior, route="backdoor"
    )
    # The cohort measures z, the back-door set: it is verified, and its measured variable
    # outside the observed base law is the unlock list. The blind registry only names t and y.
    assert ranking.verified == (1,)
    assert ranking.unlock == ("z",)
    assert [(u.index, u.label) for u in ranking.unverified] == [(0, "blind registry")]
    assert ranking.unverified[0].classification in {"insufficient", "not_certified"}
    best = ranking.best
    assert best is not None and (best.label, best.index, best.rank) == ("cohort", 1, 0)
    # Unlocking z identifies the structure that was not: identified mass 1/2 -> 1.
    assert best.score == pytest.approx(0.5, abs=1e-9)
    assert [r.label for r in ranking.ranked] == ["cohort"]
    assert [(v.id, v.index, v.constraint) for v in ranking.violations] == [
        ("blind registry", 0, "unlicensed_candidate")
    ]
    assert ranking.calibration == "unmeasured"


def test_without_a_verified_study_nothing_is_ranked_and_no_unlock_is_invented():
    contract = backdoor_contract()
    prior = design.StructurePrior((1.0, 1.0), (True, False), (1, 2))
    ranking = design.rank_studies(
        licensed_study(), contract, [obs("blind registry", ["t", "y"], 1)], prior, route="backdoor"
    )
    assert ranking.ranked == () and ranking.best is None
    assert ranking.verified == () and ranking.unlock == ()
    assert [u.label for u in ranking.unverified] == ["blind registry"]
    assert [v.constraint for v in ranking.violations] == ["unlicensed_candidate"]


def test_study_ranking_refuses_an_unknown_route_a_missing_study_and_bad_declarations():
    contract = backdoor_contract()
    cohort = complete(template_measuring(contract, {"t", "y", "z"}), 3, label="cohort")
    prior = design.StructurePrior((1.0, 1.0), (True, False), (1, 2))
    study = licensed_study()
    with pytest.raises(CausalValueError, match="no theorem route"):
        design.rank_studies(study, contract, [cohort], prior, route="not_a_route")
    # A real route that the graph cannot satisfy is not an unknown route: the study stays
    # unverified, with the route's own reason, and nothing is ranked.
    on_instrument = design.rank_studies(study, contract, [cohort], prior, route="instrument")
    assert on_instrument.ranked == () and on_instrument.verified == ()
    assert [u.label for u in on_instrument.unverified] == ["cohort"]
    assert any("no instrument" in reason for reason in on_instrument.unverified[0].reasons)
    assert [v.constraint for v in on_instrument.violations] == ["unlicensed_candidate"]
    with pytest.raises(CausalValueError, match="at least one"):
        design.rank_studies(study, contract, [], prior, route="backdoor")
    with pytest.raises(CausalTypeError, match="FailureContract"):
        design.rank_studies(study, object(), [cohort], prior)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match="prepared study"):
        design.rank_studies(object(), contract, [cohort], prior, route="backdoor")
    with pytest.raises(CausalValueError, match="objective"):
        design.rank_studies(study, contract, [cohort], prior, route="backdoor", objective="x")  # type: ignore[arg-type]
