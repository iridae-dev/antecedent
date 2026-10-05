"""Learned continuous-outcome trial transport with an estimator menu (2.2A cell X4)."""

import dataclasses

import antecedent as ac
import numpy as np
import pytest
from antecedent import transport as tr
from antecedent.errors import (
    CausalResourceError,
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.learners import Linear
from antecedent.transport import advanced

DESIGNS = ("nested_cohort", "independent_samples")


def fixture(design="independent_samples", scenario="good", n_trial=600, n_target=400, seed=3):
    """z ~ N(0, 1) in the trial, target z ~ N(0.5, 1); y = z + a (2 + z) + e; truth 2.5."""
    rng = np.random.default_rng(seed)
    total = n_trial + n_target
    if design == "nested_cohort":
        source = rng.random(total) < n_trial / total
    else:
        source = np.arange(total) < n_trial
    shift = 3.0 if scenario == "weak" else 0.5
    z = np.where(source, rng.normal(size=total), shift + rng.normal(size=total))
    a = rng.random(total) < 0.5
    y = z + a * (2.0 + z) + rng.normal(size=total)
    data = tr.TrialAipwData(
        {"z": z},
        np.where(source, y, 0.0),
        [bool(v) for v in (a & source)],
        [bool(v) for v in source],
        [0.5] * total,
        design,
    )
    graph = ac.Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    query = advanced.TrialAipwQuery(
        graph, advanced.SelectionDiagram("trial", "target", ["z"]), "a", "y"
    )
    return query, data


def options(**kwargs):
    return advanced.LearnedContinuousOptions(outcome=Linear(), folds=3, **kwargs)


@pytest.mark.parametrize("design", DESIGNS)
def test_good_overlap_point_matches_the_known_truth_under_each_design(design):
    query, data = fixture(design)
    result = advanced.prepare_learned_continuous(query, data, options=options(), seed=5).estimate()
    assert result.estimate == pytest.approx(2.5, abs=0.3)
    assert result.interval is None
    assert result.uncertainty["status"] == "point_only"
    assert result.sampling == design
    # Source membership and treatment overlap are reported apart.
    assert result.overlap["selection"]["probability_min"] > 0.05
    assert result.overlap["treatment"]["probability_min"] == pytest.approx(0.5)
    assert len(result.provenance) == 9
    assert all(p["implementation"] and p["version"] for p in result.provenance)
    assert result.folds["scheme"] == "stratified_round_robin_source_arm_and_target"
    assert result.certificate["formula"] == "standardize"


def test_weak_overlap_refuses_rather_than_extrapolating():
    query, data = fixture(scenario="weak")
    study = advanced.prepare_learned_continuous(query, data, options=options())
    with pytest.raises(
        CausalUnsupportedError, match="learned_transport.membership_overlap"
    ) as info:
        study.estimate()
    assert info.value.reason_code == "transport_support_failure"
    query, data = fixture()
    low = dataclasses.replace(data, randomization=[0.01] * len(data.randomization))
    with pytest.raises(CausalUnsupportedError, match="learned_transport.treatment_overlap") as info:
        advanced.prepare_learned_continuous(query, low, options=options())
    assert info.value.reason_code == "transport_support_failure"


def test_declared_bounds_designs_and_targets_refuse_with_their_details():
    query, data = fixture()
    with pytest.raises(CausalUnsupportedError, match="learned_transport.bounds_exceeded") as info:
        advanced.prepare_learned_continuous(query, data, options=options().__class__(folds=21))
    assert info.value.reason_code == "route_not_supported"
    with pytest.raises(CausalUnsupportedError, match="learned_transport.bounds_exceeded"):
        advanced.prepare_learned_continuous(
            query, data, options=advanced.LearnedContinuousOptions(folds=3, bootstrap=2001)
        )
    with pytest.raises(CausalUnsupportedError, match="learned_transport.non_iid_design") as info:
        advanced.prepare_learned_continuous(
            query, dataclasses.replace(data, sampling="clustered"), options=options()
        )
    assert info.value.reason_code == "sampling_dependence_unknown"
    with pytest.raises(CausalUnsupportedError, match="learned_transport.cate_requested") as info:
        advanced.prepare_learned_continuous(query, data, options=options(), target="cate")
    assert info.value.reason_code == "route_not_supported"
    with pytest.raises(CausalValueError):
        advanced.LearnedContinuousOptions(min_membership_probability=0.7)
    # Below the replicate floor the point is kept and the interval withheld.
    below = advanced.prepare_learned_continuous(
        query, data, options=options(bootstrap=50), seed=2
    ).estimate()
    assert below.uncertainty["reason"] == "estimator_inference_mismatch"
    assert below.uncertainty["detail"] == "learned_transport.bootstrap_below_floor"
    assert below.estimate == pytest.approx(2.5, abs=0.3)


def test_estimator_menu_lists_eligibility_requirements_and_refusals():
    query, _ = fixture()
    menu = advanced.estimator_menu(query, outcome=Linear())
    assert menu.selection == "manual"
    assert menu.eligible == ("learned_trial_aipw", "trial_ipw_supplied_probabilities")
    learned = menu["learned_trial_aipw"]
    assert learned.refusal is None
    assert learned.required_laws and learned.required_graph_conditions
    assert learned.nuisance_tasks and learned.support_requirements
    assert set(learned.sampling_design) == set(DESIGNS)
    assert "cell_not_licensed" in learned.uncertainty_status
    # Requirements are computed from the certificate, learners and thresholds in force.
    assert any("X=[v0]" in law for law in learned.required_laws)
    assert any("learner linear" in task for task in learned.nuisance_tasks)
    assert any("at least 0.05" in req for req in learned.support_requirements)
    assert learned.static_fields == ()
    assert "nuisance_tasks" in menu["trial_ipw_supplied_probabilities"].static_fields
    other = advanced.estimator_menu(query, outcome=Linear(), membership=None)
    assert other["learned_trial_aipw"].nuisance_tasks == learned.nuisance_tasks
    ridge = advanced.estimator_menu(query)
    assert ridge["learned_trial_aipw"].nuisance_tasks != learned.nuisance_tasks
    refused = menu["dr_learner_cate"]
    assert not refused.eligible
    assert refused.refusal["detail"] == "learned_transport.cate_requested"
    assert not menu["exact_finite_law_evaluator"].eligible
    assert not any("recommend" in repr(entry).lower() for entry in menu.entries)
    # A graph whose selection acts on the outcome certifies no trial estimator.
    graph = ac.Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    blocked = advanced.TrialAipwQuery(
        graph, advanced.SelectionDiagram("trial", "target", ["y"]), "a", "y"
    )
    menu = advanced.estimator_menu(blocked)
    assert menu.eligible == ()
    assert menu["learned_trial_aipw"].refusal["code"] == "transport_not_certified"
    # The prepared study exposes the same menu without fitting.
    query, data = fixture()
    prepared = advanced.prepare_learned_continuous(query, data, options=options())
    assert prepared.estimator_menu().eligible == (
        "learned_trial_aipw",
        "trial_ipw_supplied_probabilities",
    )


def test_estimator_menu_reads_the_retained_plan_after_builder_disposal():
    builder_query, builder_data = fixture()
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    del builder_query, builder_data
    # The retained plan carries the certificate and learners the menu is derived from.
    plan = study.estimator_menu()
    assert plan.eligible == ("learned_trial_aipw", "trial_ipw_supplied_probabilities")
    assert plan["learned_trial_aipw"].refusal is None
    assert study.estimate().estimate == pytest.approx(2.5, abs=0.3)


def test_prepare_learned_continuous_retains_a_checked_plan_after_builder_disposal():
    builder_query, builder_data = fixture()
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    del builder_query, builder_data
    # The retained plan holds the certificate: estimating never identifies again.
    plan = study.estimate()
    assert plan.certificate["formula"] == "standardize"
    assert plan.certificate["over"] == (0,)
    assert plan.estimate == pytest.approx(2.5, abs=0.3)


@pytest.mark.parametrize("design", DESIGNS)
def test_estimate_executes_the_retained_plan_after_builder_disposal(design):
    builder_query, builder_data = fixture(design)
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    del builder_query, builder_data
    plan = study.estimate()
    assert study.estimate().execution_id == plan.execution_id
    assert plan.sampling == design


def test_refresh_rebinds_rows_and_re_executes_the_retained_plan():
    builder_query, builder_data = fixture()
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    first = study.estimate()
    del builder_query, builder_data
    _, fresh = fixture(seed=9)
    study.refresh(fresh)
    plan = study.estimate()
    assert plan.execution_id != first.execution_id
    assert plan.certificate == first.certificate


@pytest.mark.parametrize("design", DESIGNS)
def test_prepare_once_estimate_export_and_independent_consume_after_builder_disposal(design):
    builder_query, builder_data = fixture(design)
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    del builder_query, builder_data
    result = study.estimate()
    assert study.estimate().execution_id == result.execution_id
    artifact = study.export()
    plan = advanced.consume_learned_continuous(artifact)
    assert plan.estimate == result.estimate
    assert plan.execution_id == result.execution_id
    assert plan.premises_digest == result.premises_digest
    assert plan.evidence_digest == result.evidence_digest
    assert plan.variable_names == ("z", "a", "y")
    assert plan.provenance == result.provenance
    assert plan.overlap == result.overlap


def test_consumer_replays_the_point_after_builder_disposal():
    builder_query, builder_data = fixture()
    study = advanced.prepare_learned_continuous(
        builder_query, builder_data, options=options(), seed=7
    )
    result = study.estimate()
    artifact = study.export()
    del builder_query, builder_data, study
    # The artifact alone (certificate, rows, out-of-fold predictions) replays the plan's point.
    plan = advanced.consume_learned_continuous(artifact)
    assert plan.estimate == result.estimate
    assert plan.folds == result.folds


def test_export_requires_an_execution_and_refresh_keeps_the_certificate():
    query, data = fixture()
    study = advanced.prepare_learned_continuous(query, data, options=options(), seed=7)
    with pytest.raises(CausalUnsupportedError, match="estimate before exporting"):
        study.export()
    first = study.estimate()
    fresh_query, fresh = fixture(seed=9)
    study.refresh(fresh)
    with pytest.raises(CausalUnsupportedError, match="estimate before exporting"):
        study.export()
    assert study.estimate().execution_id != first.execution_id
    with pytest.raises((ValueError, CausalUnsupportedError), match="reprepare_required"):
        study.refresh(dataclasses.replace(fresh, sampling="nested_cohort"))
    with pytest.raises(CausalUnsupportedError, match="learned_transport.treatment_overlap"):
        study.refresh(dataclasses.replace(fresh, randomization=[0.0] * len(fresh.randomization)))
    del fresh_query


def test_a_tampered_artifact_or_relabelled_names_fail_consumption_with_typed_errors():
    query, data = fixture()
    study = advanced.prepare_learned_continuous(query, data, options=options(), seed=7)
    study.estimate()
    artifact = study.export()
    with pytest.raises(CausalSerializationError):
        advanced.consume_learned_continuous(artifact[:-40])
    with pytest.raises(CausalSerializationError):
        advanced.consume_learned_continuous(b"not an artifact")
    # Flip bytes inside the payload: a digest, point or shape check must refuse it.
    tampered = bytearray(artifact)
    for offset in (len(artifact) // 2, len(artifact) // 3, len(artifact) - 200):
        tampered[offset] ^= 0x55
    with pytest.raises(CausalSerializationError):
        advanced.consume_learned_continuous(bytes(tampered))
    with pytest.raises(CausalResourceError, match="row count"):
        advanced.consume_learned_continuous(artifact, max_rows=10)


def test_the_analytic_interval_replays_and_the_legacy_bootstrap_refuses():
    query, data = fixture()
    study = advanced.prepare_learned_continuous(query, data, options=options(bootstrap=199), seed=4)
    with pytest.raises(CausalUnsupportedError, match="learned_transport.bootstrap_not_licensed") as info:
        study.interval()
    assert info.value.reason_code == "estimator_inference_mismatch"
    result = study.estimate()
    assert result.uncertainty["status"] == "withheld"
    assert result.uncertainty["reason"] == "cell_not_licensed"
    assert result.interval is None
    assert result.estimate == pytest.approx(2.5, abs=0.3)
    consumed = advanced.consume_learned_continuous(study.export())
    assert consumed.uncertainty == result.uncertainty
    assert consumed.interval is None
    analytic = advanced.prepare_learned_continuous(query, data, options=options(), seed=4)
    interval = analytic.interval()
    assert interval.uncertainty["status"] == "available"
    assert interval.standard_error > 0
    assert interval.interval[0] < interval.estimate < interval.interval[1]
    replay = advanced.consume_learned_continuous(analytic.export())
    assert replay.interval == interval.interval
    assert replay.standard_error == interval.standard_error
