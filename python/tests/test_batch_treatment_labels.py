"""Shared treatment encodings preserve displayed contrasts and native row scope."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimation import CandidateScreen, analyze_many, prepare_many
from antecedent.outcomes import require_success


def inputs(n=400):
    rng = np.random.default_rng(82)
    z = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(float)
    s = (rng.random(n) < 0.5).astype(float)
    numeric = {"z": z, "t": t, "s": s, "y": 3 * t - 2 * s + z + rng.normal(0, 0.1, n)}
    labelled = {
        **numeric,
        "t": np.where(t == 0, "control", "treated"),
        "s": np.where(s == 0, "off", "on"),
    }
    return numeric, labelled


GRAPH = [("z", "t"), ("z", "s"), ("z", "y"), ("t", "y"), ("s", "y")]
OPTIONS = dict(graph=GRAPH, estimator="aipw", inference=ant.Frequentist(bootstrap=0), refute=False)


def query(treatment="t", control="control", active="treated"):
    return ant.AverageEffect(treatment, "y", control_level=control, active_level=active)


def effects(batch):
    return tuple(require_success(member).effect for member in batch.members)


def test_label_batches_share_native_numeric_storage_and_preserve_contrasts():
    numeric, labelled = inputs()
    # One label column avoids asking scalar numeric ingestion to encode unrelated labels.
    labelled["s"] = numeric["s"]
    source = ant.data.table(labelled)
    queries = [query(), query(control="treated", active="control")]
    study = prepare_many(source, queries=queries, **OPTIONS)
    assert study.queries == tuple(queries)
    assert study._native.materialized_plan_count() == 0
    assert study._source_data is not None
    assert tuple(study._source_data["t"]) == tuple(source["t"])
    assert np.shares_memory(study._data["z"], source["z"])
    assert np.shares_memory(study._data["y"], source["y"])
    assert not study._data["t"].flags.writeable
    actual = require_success(study.estimate())
    expected = require_success(
        analyze_many(
            numeric,
            queries=[
                ant.AverageEffect("t", "y"),
                ant.AverageEffect("t", "y", control_level=1, active_level=0),
            ],
            **OPTIONS,
        )
    )
    assert effects(actual) == effects(expected)
    # y = 3 t - 2 s + z + noise: control -> treated is +3, the reversed contrast -3.
    assert effects(actual) == pytest.approx((3.0, -3.0), abs=0.3)  # sampling deviation 0.11
    assert effects(actual)[0] == pytest.approx(-effects(actual)[1], abs=1e-10)
    assert actual.member(0).study.query == queries[0]
    assert actual.member(1).study.query == queries[1]


def test_different_labelled_treatments_use_independent_shared_mappings():
    numeric, labelled = inputs()
    declared = [query(), query("s", "off", "on")]
    batch = require_success(analyze_many(labelled, queries=declared, **OPTIONS))
    expected = require_success(
        analyze_many(
            numeric, queries=[ant.AverageEffect("t", "y"), ant.AverageEffect("s", "y")], **OPTIONS
        )
    )
    assert effects(batch) == effects(expected)
    # y = 3 t - 2 s + z: each labelled treatment keeps its own coefficient and orientation.
    assert effects(batch) == pytest.approx((3.0, -2.0), abs=0.3)  # sampling deviation 0.19
    # Rebinding a member must encode both its treatment and the other labelled covariate.
    branch = ant.rebind(batch.member(0).study, data=labelled)
    assert require_success(ant.estimate(branch)).effect == effects(batch)[0]


def test_numeric_member_retains_the_other_treatments_label_encoding_on_branches():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    declared = [query(), ant.AverageEffect("s", "y")]
    batch = require_success(analyze_many(labelled, queries=declared, **OPTIONS))
    member = batch.member(1)
    rebound = ant.rebind(member.study, data=labelled)
    assert require_success(ant.estimate(rebound)).effect == effects(batch)[1]
    changed = ant.reconfigure(member.study, estimator="linear.adjustment.ate")
    expected = require_success(
        ant.analyze(numeric, query=declared[1], **{**OPTIONS, "estimator": "linear.adjustment.ate"})
    )
    assert require_success(ant.estimate(changed)).effect == expected.effect


def test_label_inputs_are_owned_and_batch_branches_retain_their_declarations():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    study = prepare_many(labelled, queries=[query()], **OPTIONS)
    labelled["t"][:] = "other"
    initial = require_success(study.estimate())
    assert effects(initial)[0] == pytest.approx(3, abs=0.3)
    fresh_numeric, fresh = inputs(260)
    fresh["s"] = fresh_numeric["s"]
    rebound = ant.rebind(initial.study, data=fresh)
    assert rebound.queries == study.queries
    assert rebound._native.materialized_plan_count() == 0
    changed = require_success(rebound.estimate())
    assert effects(changed)[0] == pytest.approx(3, abs=0.3)
    reversed_study = ant.reconfigure(
        initial.study, queries=[query(control="treated", active="control")]
    )
    assert reversed_study._data is initial.study._data
    execution_branch = ant.reconfigure(initial.study, execution=ant.Execution(seed=93, threads=1))
    assert execution_branch._data is initial.study._data
    reversed_batch = require_success(reversed_study.estimate())
    assert effects(reversed_batch)[0] == pytest.approx(-effects(initial)[0], abs=1e-10)
    assert effects(initial)[0] == pytest.approx(3, abs=0.3)


def test_screened_label_member_reconfiguration_preserves_actual_native_row_scope():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    labelled["y"] = labelled["y"] + 4 * numeric["t"] * (np.arange(400) >= 200)
    screen = CandidateScreen("labels", "unrecorded", tuple(range(200)), tuple(range(200, 400)))
    batch = require_success(
        analyze_many(labelled, queries=[query()], candidate_screen=screen, **OPTIONS)
    )
    member = batch.member(0)
    branch = ant.reconfigure(member.study, estimator="linear.adjustment.ate")
    changed = require_success(ant.estimate(branch))
    expected = require_success(
        ant.analyze(
            {name: values[200:] for name, values in labelled.items()},
            query=query(),
            **{**OPTIONS, "estimator": "linear.adjustment.ate"},
        )
    )
    assert changed.effect == expected.effect
    assert changed.effect == pytest.approx(7, abs=0.3)
    assert batch.retarget_rows() == tuple(range(200, 400))
    assert member.study._data.num_rows == 400
    assert np.shares_memory(branch._data._encoded_data["y"], batch.study._data["y"])


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_label_cells_match_numeric_missingness(missing):
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    labelled["t"] = labelled["t"].astype(object)
    labelled["t"][0] = missing
    numeric["t"][0] = np.nan
    source = ant.data.table(labelled)
    prepared = prepare_many(source, queries=[query()], **OPTIONS)
    assert np.isnan(prepared._data["t"][0])
    assert "nan" not in dict(dict(prepared._source_data._treatment_codes)["t"])
    actual = analyze_many(labelled, queries=[query()], **OPTIONS)
    expected = analyze_many(numeric, queries=[ant.AverageEffect("t", "y")], **OPTIONS)
    assert actual.accepted == expected.accepted
    if actual.refusal is not None:
        assert actual.refusal.code == expected.refusal.code
    else:
        assert effects(require_success(actual)) == effects(require_success(expected))


def test_literal_nan_label_is_not_reinterpreted_as_a_missing_value():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    labelled["t"][0] = "nan"
    study = prepare_many(labelled, queries=[query()], **OPTIONS)
    assert study._data["t"][0] == 2.0
    assert dict(dict(study._source_data._treatment_codes)["t"])["nan"] == 2.0
    outcome = study.estimate()
    assert outcome.refusal is not None
    assert outcome.refusal.code == "treatment_level_undeclared"


def test_undeclared_labels_reach_the_same_native_refusal_as_scalar_preparation():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    labelled["t"][0] = "unexpected"
    scalar = ant.analyze(labelled, query=query(), **OPTIONS)
    batched = analyze_many(labelled, queries=[query()], **OPTIONS)
    assert scalar.refusal is not None and batched.refusal is not None
    assert scalar.refusal.code == batched.refusal.code
    assert "treatment_level_undeclared" in str(batched.refusal)


def test_different_declared_label_pairs_remain_native_scientific_questions():
    numeric, labelled = inputs()
    labelled["s"] = numeric["s"]
    study = prepare_many(labelled, queries=[query(), query(active="other")], **OPTIONS)
    assert study.queries[1].active_level == "other"
    mapping = dict(dict(study._source_data._treatment_codes)["t"])
    assert mapping == {"control": 0.0, "treated": 1.0, "other": 2.0}
    # The second query cannot treat observed 'treated' as 'other': 'other' is never observed, so
    # the batch refuses with the typed undeclared-level code instead of mapping it to a seen level.
    outcome = study.estimate()
    assert outcome.refusal is not None
    assert outcome.refusal.code == "treatment_level_undeclared"


def test_numeric_and_label_contrasts_on_one_column_are_explicitly_ambiguous():
    _, labelled = inputs()
    with pytest.raises(CausalValueError, match="mixes numeric levels"):
        prepare_many(labelled, queries=[query(), ant.AverageEffect("t", "y")], **OPTIONS)
