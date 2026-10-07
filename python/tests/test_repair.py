"""B0.1 (F9/F10/F13): evidence obligations, durable study candidates and the
identification-repair facade through the Python surface.

Two distinct contract families are repaired by their own theorem checkers: a
fixed-graph back-door contract (``t <- z1, z2 -> y`` with ``t -> y``; the only
admissible adjustment set is ``{z1, z2}`` and it must be read from one joint
law) and a catalog-aware classical transport contract (``x -> y`` with a
selection mechanism on ``x``; the target query needs the source experiment on
``x``). Expected verdicts follow from those theorems, not from the code under
test.
"""

from __future__ import annotations

import pytest
from antecedent import Admg, Dag, repair
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalUnsupportedError
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

NAMES = ["t", "y", "z1", "z2", "w1", "w2", "w3"]
EDGES = [("z1", "t"), ("z1", "y"), ("z2", "t"), ("z2", "y"), ("t", "y")]


def backdoor(assumptions=()):
    return repair.BackdoorContract(
        graph=Dag.from_edges(NAMES, EDGES),
        treatment="t",
        outcome="y",
        population="clinic",
        observed=["t", "y"],
        assumptions=assumptions,
    )


_DEFAULTS = {
    "sample_size": 500,
    "recruitment": "consecutive patients",
    "timing": "baseline",
    "unit": "patient",
    "cost_unit": "USD",
}


def observation(label, measured, cost, *, population="clinic", **fields):
    return repair.StudyCandidate.observation(
        label,
        population=population,
        measured=measured,
        cost=cost,
        **{**_DEFAULTS, **fields},
    )


def candidates():
    return [
        observation("registry_z1", ["t", "y", "z1"], 5),
        observation("registry_z2", ["t", "y", "z2"], 5),
        observation("cohort", ["t", "y", "z1", "z2"], 20),
    ]


def transport_contract():
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in ("x", "y"))
    catalog = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates, selection_targets=("x",)),
            transport.Environment("target", coordinates),
        )
    )
    return repair.TransportContract(
        graph=Admg.from_edges(["x", "y"], [("x", "y")]),
        selections=["x"],
        source="source",
        target="target",
        outcomes=["y"],
        treatments=["x"],
        catalog=catalog,
    )


def trial(label, population, cost, *, measured=("y",), interventions=("x",), **fields):
    return repair.StudyCandidate.experiment(
        label,
        population=population,
        interventions=list(interventions),
        measured=list(measured),
        cost=cost,
        **{**_DEFAULTS, "sample_size": 300, "timing": "12 weeks", **fields},
    )


# ----------------------------------------------------------------- obligations


def test_f9_failed_backdoor_contract_exposes_a_joint_law_obligation_with_its_proof_step():
    (obligation,) = repair.obligations(backdoor())
    assert obligation.kind == "provide_joint_law"
    assert set(obligation.variables) == {"t", "y", "z1", "z2"}
    assert obligation.population == "clinic"
    assert obligation.joint is True
    assert obligation.interventions == ()
    assert obligation.family == "backdoor"
    assert obligation.proof_step == "backdoor.adjustment_set:0,1,2,3"
    assert obligation.satisfiable_by_study is True
    assert obligation.id.startswith("eo1:provide_joint_law:")


def test_f9_failed_transport_contract_exposes_the_source_experiment_obligation():
    (obligation,) = repair.obligations(transport_contract())
    assert obligation.kind == "intervene"
    assert obligation.population == "source"
    assert obligation.interventions == ("x",)
    assert obligation.variables == ("y",)
    assert obligation.family == "transport"
    assert obligation.proof_step is not None
    assert obligation.proof_step.startswith("searched:")
    assert "solver:" in obligation.reason


def test_f9_obligations_accept_a_failed_identification():
    class Query:
        treatment = "t"
        outcome = "y"

    class Failed:
        graph = Dag.from_edges(NAMES, EDGES)
        names = NAMES
        query = Query()
        status = "not_identified"

    with pytest.raises(ValueError, match="population"):
        repair.obligations(Failed())
    (obligation,) = repair.obligations(Failed(), population="clinic", observed=["t", "y"])
    assert obligation.kind == "provide_joint_law"
    with pytest.raises(CausalTypeError):
        repair.obligations(object())


def test_f9_assumption_obligations_are_never_satisfied_by_a_study():
    contract = backdoor(
        [repair.UnresolvedAssumption("assume:positivity", "positivity given the adjustment set")]
    )
    kinds = {o.kind: o for o in repair.obligations(contract)}
    assert kinds["establish_assumption"].satisfiable_by_study is False
    assert kinds["establish_assumption"].required_slots == ("assume:positivity",)
    result = repair.repair(contract, [candidates()[2]])
    # The back-door checker identifies on the cohort, but a study does not
    # establish an assumption: the subset is not verified sufficient.
    assert result.classification("cohort") == "not_certified"
    assert result.outcome == "none_certified"
    assert result.best is None
    assert result.reason_code == "transport_missing_evidence"
    assert result.detail == "evidence_obligations.wrong_contract"


# --------------------------------------------------------------- study candidate


def test_f10_candidate_needs_unit_and_cost_semantics():
    for field, value in (("unit", " "), ("cost_unit", ""), ("timing", ""), ("recruitment", "")):
        with pytest.raises(repair.StudyCandidateRefusal) as caught:
            observation("bad", ["t", "y"], 5, **{field: value})
        assert caught.value.detail == "study_candidate.wrong_contract"
        assert caught.value.reason_code == "design_signal_invalid"
        assert_registered_refusal(caught.value)
    with pytest.raises(repair.StudyCandidateRefusal):
        observation("free", ["t", "y"], 0)
    with pytest.raises(CausalUnsupportedError):
        observation("free", ["t", "y"], 0)


def test_f10_candidate_kinds_cluster_and_sample_increase_are_declared_not_conducted():
    experiment = trial("exp", "source", 3)
    assert experiment.kind == "experiment" and experiment.interventions == ("x",)
    rows = repair.StudyCandidate.sample_increase(
        "more_rows",
        population="clinic",
        sample_size=100,
        recruitment="extend enrolment",
        timing="6 months",
        unit="site",
        cluster="hospital",
        whole_cluster_sampling=True,
        cost=2,
        cost_unit="USD",
    )
    assert rows.kind == "sample_increase" and rows.cluster == "hospital"
    # A sample increase delivers no new regime: it cannot repair a back-door
    # contract that needs a joint law.
    result = repair.repair(backdoor(), [rows])
    assert result.classification("more_rows") == "insufficient"
    assert result.best is None


def test_f10_candidate_observing_separate_regimes_cannot_claim_the_joint_factor():
    full = ["t", "y", "z1", "z2"]
    marginals = observation("marginals", full, 4, joint=False)
    liar = observation(
        "liar",
        full,
        4,
        joint=False,
        evidence=[repair.ExpectedEvidence("clinic", full, joint=True)],
    )
    elsewhere = observation("elsewhere", full, 4, population="hospital")
    result = repair.repair(backdoor(), [marginals, liar, elsewhere])
    # Separate marginals and another population have every variable name and
    # still do not supply the joint law in this population.
    assert result.classification("marginals") == "insufficient"
    assert result.classification("elsewhere") == "insufficient"
    # Claiming the joint law while observing separate regimes is invalid.
    assert result.classification("liar") == "invalid"
    (row,) = [r for r in result.table if r.labels == ("liar",)]
    assert "study_candidate.wrong_contract" in row.reasons[0]
    assert result.best is None


# ------------------------------------------------------------------- repair


def test_f9_failed_backdoor_contract_is_repaired_by_the_candidate_that_supplies_the_joint_law():
    result = repair.repair(backdoor(), candidates())
    assert result.family == "backdoor"
    assert result.outcome == "repaired"
    assert result.inference_claim == "none"
    assert result.best is not None
    assert result.best.labels == ("cohort",)
    assert result.classification("cohort") == "verified_sufficient"
    assert result.best.derivation is not None
    assert result.best.derivation.checker == "backdoor.adjustment"
    assert result.best.derivation.verified is True
    assert "adjustment_set:[2,3]" in result.best.derivation.steps
    (obligation,) = result.obligations
    assert result.best.addressed == (obligation.id,)
    assert result.reason_code is None
    assert result.sufficient == result.ranked


def test_f9_separate_studies_cannot_meet_a_joint_regime_obligation():
    result = repair.repair(backdoor(), candidates()[:2])
    # Each confounder alone leaves the other back-door path open.
    assert result.classification("registry_z1") == "insufficient"
    assert result.classification("registry_z2") == "insufficient"
    # z1 and z2 measured in two studies are not one joint law over (t, y, z1, z2).
    assert result.classification("registry_z1", "registry_z2") == "insufficient"
    pair = next(r for r in result.table if len(r.labels) == 2)
    assert pair.derivation is None
    assert pair.addressed == () and len(pair.unmet) == 1
    assert any("no admissible set exists" in reason for reason in pair.reasons)
    assert result.outcome == "none_certified"
    assert result.best is None
    assert result.reason_code == "transport_not_certified"
    assert result.detail == "identification_repair.wrong_contract"


def test_f10_transport_family_is_repaired_by_the_source_experiment_only():
    contract = transport_contract()
    result = repair.repair(
        contract,
        [
            trial("source_trial", "source", 3),
            observation("source_obs", ["y"], 1, population="source"),
            observation("mars", ["x", "y"], 1, population="mars"),
        ],
    )
    assert result.family == "transport"
    assert result.classification("source_trial") == "verified_sufficient"
    # Same variable, wrong regime: a source observation is not do(x).
    assert result.classification("source_obs") == "insufficient"
    # A population the catalog does not declare.
    assert result.classification("mars") == "invalid"
    assert result.best is not None and result.best.labels == ("source_trial",)
    assert result.best.derivation is not None
    assert result.best.derivation.checker == "classical_transport.catalog"
    # Supersets of the sufficient experiment are skipped, never re-evaluated.
    assert result.receipt.dominated_skipped >= 2


def test_f13_the_same_candidate_repairs_only_the_family_whose_theorem_it_satisfies():
    cohort = candidates()[2]
    experiment = trial("source_trial", "source", 3)
    assert repair.repair(transport_contract(), [experiment]).outcome == "repaired"
    assert repair.repair(backdoor(), [cohort]).outcome == "repaired"
    # Variable names do not travel across contracts: an experiment on z1 has the
    # right variables but is not an observational joint law, and a joint source
    # observation of (x, y) is not the source experiment on x.
    experiment_z1 = trial("exp_z1", "clinic", 4, measured=("t", "y", "z2"), interventions=("z1",))
    assert repair.repair(backdoor(), [experiment_z1]).best is None
    source_joint = observation("source_joint", ["x", "y"], 5, population="source")
    assert repair.repair(transport_contract(), [source_joint]).best is None


def test_f13_objective_ranks_sufficient_subsets_and_cost_units_must_agree():
    cheap = observation("cheap_wide", ["t", "y", "z1", "z2"], 20)
    dear = observation("dear_narrow", ["t", "y", "z1", "z2", "w1"], 30, sample_budget=10)
    by_cost = repair.repair(backdoor(), [cheap, dear], "minimize_cost")
    assert by_cost.best is not None and by_cost.best.labels == ("cheap_wide",)
    by_samples = repair.repair(backdoor(), [cheap, dear], "minimize_sample_budget")
    assert by_samples.best is not None and by_samples.best.labels == ("dear_narrow",)
    euros = repair.StudyCandidate.observation(
        "euros",
        population="clinic",
        measured=["t", "y", "z1", "z2"],
        sample_size=500,
        recruitment="registry",
        timing="baseline",
        unit="patient",
        cost=9,
        cost_unit="EUR",
    )
    with pytest.raises(repair.RepairRefusal) as caught:
        repair.repair(backdoor(), [cheap, euros], "minimize_cost")
    assert caught.value.detail == "identification_repair.cost_units_mismatch"
    assert caught.value.reason_code == "design_cost_units_mismatch"
    assert_registered_refusal(caught.value)
    with pytest.raises(ValueError, match="objective"):
        repair.repair(backdoor(), [cheap], "minimize_regret")


def test_f13_a_contract_that_already_identifies_is_not_a_repair_target():
    with pytest.raises(repair.RepairRefusal) as caught:
        repair.BackdoorContract(
            graph=Dag.from_edges(["a", "b"], [("a", "b")]),
            treatment="a",
            outcome="b",
            population="clinic",
            observed=["a", "b"],
        )
    assert caught.value.detail == "identification_repair.not_a_failure"
    assert_registered_refusal(caught.value)


def test_f13_budget_truncated_search_is_exhausted_never_impossible():
    many = [
        observation("a", ["t", "y", "z1"], 5),
        observation("b", ["t", "y", "z2"], 5),
        observation("w1", ["t", "y", "w1"], 1),
        observation("w2", ["t", "y", "w2"], 1),
        observation("w3", ["t", "y", "w3"], 1),
    ]
    result = repair.repair(
        backdoor(), many, limits=repair.RepairLimits(max_operations=3, max_depth=2)
    )
    # 5 singles and 10 pairs; three operations decide three of them.
    assert result.outcome == "exhausted"
    assert result.best is None
    assert result.receipt.stop == "search.operations"
    assert result.receipt.operations_consumed == 3
    assert len(result.receipt.explored) == 3
    assert result.receipt.unevaluated_total == 12
    assert len(result.receipt.unevaluated) == 12
    unevaluated = [r for r in result.table if r.classification == "unevaluated"]
    assert len(unevaluated) == 12
    assert result.reason_code == "transport_budget_cancel"
    assert result.detail == "identification_repair.budget"
    assert "impossible" not in repr(result).lower()
    assert "impossible" not in (result.detail or "")


def test_f13_subsets_beyond_the_declared_depth_are_not_examined_and_say_so():
    result = repair.repair(
        backdoor(),
        candidates()[:2],
        limits=repair.RepairLimits(max_operations=100, max_depth=1),
    )
    assert result.receipt.beyond_declared_depth is True
    assert all(len(row.labels) == 1 for row in result.table)
    assert result.receipt.stop is None


def test_f13_request_bounds_are_refusals():
    many = [observation(f"n{k}", ["t", "y", "z1"], k + 1) for k in range(17)]
    with pytest.raises(repair.RepairRefusal) as caught:
        repair.repair(backdoor(), many)
    assert caught.value.detail == "identification_repair.bounds_exceeded"
    assert caught.value.reason_code == "invalid_argument"
    with pytest.raises(repair.RepairRefusal) as caught:
        repair.repair(backdoor(), candidates(), limits=repair.RepairLimits(max_operations=10**9))
    assert caught.value.detail == "identification_repair.bounds_exceeded"


# ------------------------------------------------------------------- artifact


def test_f13_export_consume_round_trip_replays_the_report():
    for contract, studies in (
        (backdoor(), candidates()),
        (transport_contract(), [trial("source_trial", "source", 3)]),
    ):
        result = repair.repair(contract, studies)
        data = result.export()
        replayed = repair.consume(data)
        assert replayed.outcome == result.outcome == "repaired"
        assert replayed.table == result.table
        assert replayed.obligations == result.obligations
        assert replayed.receipt == result.receipt
        assert replayed.best == result.best


def test_f13_truncated_search_survives_the_artifact_as_unevaluated():
    many = [observation(f"s{k}", ["t", "y", "w1"], k + 1) for k in range(4)]
    result = repair.repair(
        backdoor(), many, limits=repair.RepairLimits(max_operations=2, max_depth=2)
    )
    replayed = repair.consume(result.export())
    assert replayed.outcome == "exhausted"
    assert replayed.receipt.unevaluated_total == result.receipt.unevaluated_total > 0
    assert sum(r.classification == "unevaluated" for r in replayed.table) == len(
        result.receipt.unevaluated
    )


def test_f13_consumer_refuses_oversized_limits_and_malformed_bytes():
    data = repair.repair(backdoor(), candidates()).export()
    with pytest.raises(repair.RepairArtifactRefusal) as caught:
        repair.consume(data, max_operations=1)
    assert caught.value.detail == "repair_artifact.bounds_exceeded"
    assert_registered_refusal(caught.value)
    with pytest.raises(CausalTypeError):
        repair.consume("not bytes")  # type: ignore[arg-type]
    with pytest.raises(CausalSerializationError):
        repair.consume(b"not an artifact")
    with pytest.raises((CausalSerializationError, repair.RepairArtifactRefusal)):
        repair.consume(data[: len(data) // 2])
