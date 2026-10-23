"""Experiment planning from Python: typed results, both directions, simulation, refusals.

The request is compiled natively (its lowered program retains every assumption) and solved
analytically; a declared simulation runs the whole procedure with the design's analysis
estimator and is the same at every thread count.
"""

from __future__ import annotations

import pytest
from antecedent import design
from antecedent.design import planning
from antecedent.errors import CausalError, CausalTypeError


def _continuous(**declared: object) -> planning.Assumptions:
    return planning.Assumptions.continuous(1.0, **declared)


def test_two_sample_sizes_match_the_published_table() -> None:
    plan = planning.sample_size(planning.Complete(), _continuous(effect=0.5))
    assert isinstance(plan, design.PlanningResult)
    assert isinstance(plan.analytic, planning.AnalyticCharacteristics)
    assert abs(plan.units - 128) <= 1
    assert plan.analytic.value >= 0.8
    assert plan.target_met
    assert plan.calibration == "unmeasured"
    assert plan.analytic.method == "noncentral_t"


def test_minimum_detectable_effect_inverts_the_sample_size() -> None:
    planned = planning.sample_size(planning.Complete(), _continuous(effect=0.4))
    mde = planning.minimum_detectable_effect(planning.Complete(), _continuous(), planned.units)
    assert mde.effect is not None and mde.effect <= 0.4 + 1e-9
    assert mde.analytic.value == pytest.approx(0.8, abs=1e-6)
    again = planning.operating_characteristics(
        planning.Complete(), _continuous(effect=0.4), planned.units
    )
    assert again.analytic.value >= 0.8


def test_cluster_design_effect_and_attrition_are_retained() -> None:
    plan = planning.plan_experiment(
        planning.Cluster(mean_cluster_size=20, icc=0.05),
        _continuous(effect=0.5),
        planning.Power(0.8),
        planning.SampleSize(),
    )
    assert plan.analytic.design_effect == pytest.approx(1.95)
    assert plan.unit == "cluster"
    with_attrition = planning.plan_experiment(
        planning.Cluster(mean_cluster_size=20, icc=0.05),
        _continuous(effect=0.5, attrition=0.2),
        planning.Power(0.8),
        planning.SampleSize(),
    )
    assert with_attrition.request_identity != plan.request_identity
    assert "attrition=0.2" in with_attrition.assumptions


def test_simulation_agrees_and_does_not_depend_on_threads() -> None:
    def run(threads: int) -> design.PlanningResult:
        return planning.plan_experiment(
            planning.Complete(),
            _continuous(effect=0.3),
            planning.Power(0.8),
            planning.Evaluate(200),
            simulation=planning.Simulation(replicates=800, seed=5),
            threads=threads,
        )

    serial = run(1)
    parallel = run(4)
    assert serial.simulation == parallel.simulation
    simulated = serial.simulation
    assert simulated is not None and simulated.monte_carlo_se is not None
    gap = abs(simulated.value - serial.analytic.value)
    assert gap <= 4 * simulated.monte_carlo_se + 0.02


def test_multi_arm_and_sequential_characteristics_are_typed() -> None:
    family = planning.operating_characteristics(
        planning.MultiArm(3, multiplicity="dunnett", family_power="any"),
        _continuous(effect=0.4),
        240,
    )
    assert family.analytic.family is not None
    assert family.analytic.family.all <= family.analytic.family.any
    sequential = planning.sample_size(
        planning.GroupSequential((1 / 3, 2 / 3, 1.0)),
        _continuous(effect=0.3),
        planning.Power(0.9),
    )
    looks = sequential.analytic.sequential
    assert looks is not None and len(looks.critical) == 3
    assert looks.expected_units_alternative < sequential.units


def test_bayesian_criteria_plan_through_the_same_entry() -> None:
    prior = planning.NormalPrior(0.0, 1.0)
    precision = planning.sample_size(
        planning.Complete(),
        _continuous(),
        planning.PosteriorPrecision(prior, max_posterior_sd=0.12),
    )
    assert precision.units == 274
    assurance = planning.operating_characteristics(
        planning.Complete(),
        _continuous(),
        200,
        planning.Assurance(
            planning.NormalPrior(0.3, 0.15),
            planning.PosteriorSuccess(prior, threshold=0.0, probability=0.95),
        ),
    )
    assert 0.0 < assurance.analytic.value < 1.0


def test_unplannable_designs_are_refused_with_the_next_attempt() -> None:
    with pytest.raises(CausalError) as caught:
        planning.sample_size(planning.Interference(), _continuous(effect=0.3))
    assert caught.value.reason_code == "required_option_missing"
    assert "declare_exposure_model" in str(caught.value)
    with pytest.raises(CausalError) as caught:
        planning.sample_size(planning.OutcomeAdaptive(), _continuous(effect=0.3))
    assert caught.value.reason_code == "adaptive_contract_incomplete"
    with pytest.raises(CausalError) as caught:
        planning.sample_size(
            planning.Switchback(10, 4, autocorrelation=0.3, carryover=0.5),
            _continuous(effect=0.3, control_mean=1.0),
        )
    assert caught.value.reason_code == "effect_not_identified"
    with pytest.raises(CausalTypeError):
        planning.plan_experiment(
            "complete",  # type: ignore[arg-type]
            _continuous(effect=0.3),
            planning.Power(),
            planning.SampleSize(),
        )


def test_a_plan_feeds_design_ranking_with_its_sample_size() -> None:
    plan = planning.sample_size(planning.Complete(), _continuous(effect=0.3))
    sampling = plan.sampling_plan(cost_per_individual=2.0, tag=1)
    assert sampling.additional_samples == plan.sample_size
    assert sampling.sample_budget == plan.sample_size
    assert sampling.cost == pytest.approx(2.0 * plan.sample_size)
    experiment = plan.experiment_plan([0], tag=2)
    assert experiment.sample_budget == plan.sample_size
    ranking = design.rank_designs(
        [sampling, experiment],
        prior=design.StructurePrior.uniform([True, False]),
        query_id=0,
    )
    assert len(ranking.candidates) == 2


def test_program_identity_tracks_every_assumption() -> None:
    base = planning.planning_program_identity(
        planning.Complete(), _continuous(effect=0.3), planning.Power(), planning.SampleSize()
    )
    for assumptions in (
        _continuous(effect=0.3, allocation_ratio=2.0),
        _continuous(effect=0.3, alpha=0.01),
        _continuous(effect=0.3, sidedness="one_sided"),
    ):
        other = planning.planning_program_identity(
            planning.Complete(), assumptions, planning.Power(), planning.SampleSize()
        )
        assert other != base
