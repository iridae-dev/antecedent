"""Joint cells carry their complete scientific declaration into native preparation."""

from dataclasses import replace

import antecedent as ant
import numpy as np
import pytest
from antecedent import observation, population
from antecedent._batch import _joint_cell_batch_specs
from antecedent._freeze import thaw
from antecedent.estimation import prepare_many
from antecedent.outcomes import require_success


def inputs(n=600):
    rng = np.random.default_rng(413)
    z = rng.normal(size=n)
    t1 = (rng.random(n) < 0.5).astype(float)
    t2 = (rng.random(n) < 0.5).astype(float)
    y = 2 * t1 + 3 * t2 + z + rng.normal(0, 0.2, n)
    return {"z": z, "t1": t1, "t2": t2, "y": y, "recorded": y.copy(), "selected": np.ones(n)}


GRAPH = [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")]
OPTIONS = dict(
    graph=GRAPH,
    estimator="cell.aipw",
    inference=ant.Frequentist(bootstrap=0),
    execution=ant.Execution(seed=11, threads=1),
    refute=False,
)


def query(**declarations):
    return ant.InterventionResponse(
        "y",
        intervention=[ant.intervention.Set("t1", 1), ant.intervention.Set("t2", 1)],
        **declarations,
    )


def prepare(data=None, **declarations):
    return prepare_many(
        inputs() if data is None else data, queries=[query(**declarations)], **OPTIONS
    )


def witness(member):
    result = require_success(member)
    assert result.certificate is not None
    return thaw(result.certificate)["witness_query"]["response"], result


def test_default_and_explicit_complete_all_observed_cells_have_the_same_native_identity():
    data = ant.data.table(inputs())
    baseline = require_success(prepare(data).estimate()).member(0)
    declared = require_success(
        prepare(
            data,
            target_population=population.AllRows(),
            observation=observation.Complete(),
        ).estimate()
    ).member(0)
    default_wire, default_result = witness(baseline)
    explicit_wire, explicit_result = witness(declared)
    assert default_wire == explicit_wire
    assert explicit_wire["target_population"] == "all_observed"
    assert explicit_wire["observation"] == "complete"
    assert explicit_wire["observation_assumptions"] == []
    assert default_result.effect == explicit_result.effect
    assert baseline.inspect().claim_id == declared.inspect().claim_id
    assert default_result.program_id == explicit_result.program_id


def test_every_observation_assumption_reaches_the_native_query_and_identification_ledger():
    data = ant.data.table(inputs())
    declarations = [
        observation.IndependentGiven(["z"]),
        observation.OutcomeIndependentGiven(["t1", "t2"]),
        observation.Structural("observation-v1"),
    ]
    initial = prepare(
        data, observation=observation.Complete(), observation_assumptions=declarations
    )
    assert initial._native.materialized_plan_count() == 0
    declarations.clear()  # Preparation owns the original conjunction.
    batch = require_success(initial.estimate())
    wire, result = witness(batch.member(0))
    names = list(result.certificate["names"])
    assert wire["observation_assumptions"] == [
        {"independent_given": [names.index("z")]},
        {"outcome_independent_given": [names.index("t1"), names.index("t2")]},
        {"structural": "observation-v1"},
    ]
    for identifier in (
        "observation.independent_given",
        "observation.outcome_independent_given",
        "observation.structural",
    ):
        assert any(identifier in entry for entry in result.assumptions)
    baseline = require_success(prepare(data).estimate()).member(0)
    assert result.effect == require_success(baseline).effect
    assert batch.member(0).inspect().claim_id != baseline.inspect().claim_id
    rebound = ant.rebind(batch.study, data=inputs(700))
    assert rebound._native.materialized_plan_count() == 0
    assert len(rebound.queries[0].observation_assumptions) == 3
    later_wire, later = witness(require_success(rebound.estimate()).member(0))
    assert later_wire == wire
    assert any("observation.structural" in entry for entry in later.assumptions)


@pytest.mark.parametrize(
    "target", [population.Treated(), population.Untreated(), population.Rows((0, 1, 2))]
)
def test_nondefault_populations_are_refused_by_native_science_instead_of_erased(target):
    study = prepare(target_population=target)
    assert study.queries[0].target_population == target
    outcome = study.estimate()
    assert outcome.refusal is not None
    assert outcome.refusal.code == "population_not_estimable"
    assert "AllObserved" in outcome.refusal.message
    # Reconfiguration explicitly changes the scientific target and can answer it.
    branch = ant.reconfigure(
        study, queries=[replace(study.queries[0], target_population=population.AllRows())]
    )
    accepted = require_success(branch.estimate())
    # Independent reference: the least-squares fit of y on (1, t1, t2, z) predicted at
    # t1 = t2 = 1 and averaged over the observed z; the doubly robust cell estimate differs from
    # it only by cross-fitting variation (observed 0.014).
    data = inputs()
    design = np.column_stack([np.ones(600), data["t1"], data["t2"], data["z"]])
    beta = np.linalg.lstsq(design, data["y"], rcond=None)[0]
    reference = beta[0] + beta[1] + beta[2] + beta[3] * data["z"].mean()
    assert require_success(accepted.member(0)).effect == pytest.approx(reference, abs=0.05)
    assert study.estimate().refusal.code == "population_not_estimable"


def test_selected_observation_is_carried_to_the_native_joint_adjustment_refusal():
    mechanism = observation.Selected("y", "recorded", "selected")
    assumption = observation.IndependentGiven(["z"])
    declared = query(observation=mechanism, observation_assumptions=[assumption])
    spec = _joint_cell_batch_specs([declared])[0]
    assert spec[6] == {
        "observation_kind": "selected",
        "latent": "y",
        "observed": "recorded",
        "indicator": "selected",
    }
    assert spec[7] == [{"assumption_kind": "independent_given", "assumption_variables": ["z"]}]
    study = prepare(observation=mechanism, observation_assumptions=[assumption])
    outcome = study.estimate()
    assert outcome.refusal is not None
    assert "joint adjustment requires complete-observation static data" in outcome.refusal.message
    assert study.queries[0].observation == mechanism
    assert tuple(study.queries[0].observation_assumptions[0].variables) == ("z",)


@pytest.mark.parametrize(
    "mechanism",
    [
        observation.RightCensored("y", "recorded", "selected", "selected"),
        observation.LeftCensored("y", "recorded", "selected", "selected"),
        observation.IntervalCensored("y", "recorded", "selected"),
        observation.Truncated("y", "recorded", lower="selected"),
    ],
)
def test_all_existing_mechanism_codecs_reach_native_observation_validation(mechanism):
    study = prepare(
        observation=mechanism, observation_assumptions=[observation.Structural("model")]
    )
    outcome = study.estimate()
    assert outcome.refusal is not None
    assert "joint adjustment requires complete-observation static data" in outcome.refusal.message
