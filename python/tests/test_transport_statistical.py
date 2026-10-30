"""Empirical-table stages publish licensed pointwise intervals on the common handle."""

import pytest
from antecedent import (
    Admg,
    Execution,
    ResourceLimits,
    estimate,
    load,
    prepare,
    rebind,
    reconfigure,
)
from antecedent.errors import CausalCancelledError, CausalResourceError, CausalUnsupportedError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport


def statistical_study(identified, catalog, data, at, *, seed=1, **settings):
    """Prepare the typed empirical question on the common immutable lifecycle."""
    return prepare(
        data,
        query=transport.StatisticalTransportQuery(identified, catalog, at, **settings),
        execution=Execution(seed=seed),
    )


def estimated(study):
    outcome = estimate(study)
    return outcome, require_success(outcome)


def test_statistical_transport_rejects_nonlaw_payload_entries():
    identification, catalog, data = fixture()
    invalid = transport.StatisticalTransportData(samples=data.samples, laws=(object(),))
    with pytest.raises(ValueError, match="laws must be exact law tables"):
        statistical_study(identification, catalog, invalid, {"x": 1.0})


def fixture():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identification = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment(
                "source",
                [
                    transport.VariableCoordinate("x", "binary"),
                    transport.VariableCoordinate("y", "binary"),
                ],
            ),
            transport.Environment(
                "target",
                [
                    transport.VariableCoordinate("x", "binary"),
                    transport.VariableCoordinate("y", "binary"),
                ],
            ),
        ],
        regimes=[
            transport.EvidenceRegime(
                "trial",
                "source",
                kind="experimental",
                interventions=["x"],
                measured=["y"],
            ),
        ],
        bindings=[
            transport.RegimeBinding(
                "trial",
                "v1",
                sampling="independent",
                dependence="independent_studies",
            ),
        ],
        target_sampling="representative_sample",
    )
    sample = transport.RegimeSample(
        "source",
        "trial",
        "v1",
        {"y": [0.0] * 20 + [1.0] * 80},
        interventions=(("x", 1.0),),
    )
    return identification, catalog, transport.StatisticalTransportData(samples=(sample,))


def test_statistical_prepare_estimate_and_inspect_license():
    identified, catalog, data = fixture()
    study = prepare(
        data,
        query=transport.StatisticalTransportQuery(
            # A nominal 0.95 percentile interval needs B >= 40 (PERCENTILE_95_MIN_REPLICATES);
            # fewer replicates run cleanly but withhold the interval as unlicensed.
            identified,
            catalog,
            {"x": 1.0},
            bootstrap=40,
        ),
        execution=Execution(seed=7),
    )
    inspection = study.inspect()
    assert inspection.identification.available
    # Preparation retains the checked derivation; the interval license is decided by
    # the execution that fits the supplied rows.
    assert not inspection.uncertainty.available
    assert inspection.uncertainty.reason == "not_executed"
    outcome, result = estimated(study)
    assert result.mean("y") == pytest.approx(0.8)
    assert result.uncertainty["available"]
    assert result.inspect().uncertainty.available
    assert result.uncertainty["row"]["method"] == "percentile_bootstrap"
    assert result.uncertainty["row"]["interval_scope"] == "pointwise"
    loaded = require_success(load(bytes(outcome.export())))
    assert loaded.probabilities == result.probabilities
    assert loaded.uncertainty["available"]


def test_prepare_statistical_retains_checked_plan_after_builder_disposal():
    """The prepared statistical study keeps its checked plan, rebinds atomically and
    invalidates its execution identity when the snapshot changes."""
    from dataclasses import replace

    identified, catalog, data = fixture()
    query_builder = transport.StatisticalTransportQuery(
        identified, catalog, {"x": 1.0}, bootstrap=40
    )
    plan = prepare(data, query=query_builder, execution=Execution(seed=7))
    del query_builder, identified
    before = plan.inspect()
    assert before.identification.available
    assert not before.uncertainty.available
    outcome, result = estimated(plan)
    # Independent arithmetic: 80 of 100 trial rows have y = 1.
    assert result.mean("y") == pytest.approx(0.8)
    assert result.uncertainty["row"]["method"] == "percentile_bootstrap"
    assert result.uncertainty["row"]["interval_scope"] == "pointwise"
    original = result.inspect()
    assert original.uncertainty.available
    # A sample outside the declared domain is rejected atomically; the source stands.
    invalid = replace(data.samples[0], interventions=(("x", 2.0),))
    with pytest.raises(ValueError, match="declared finite domain"):
        require_success(
            estimate(rebind(plan, data=transport.StatisticalTransportData(samples=(invalid,))))
        )
    assert result.inspect().execution_id == original.execution_id
    # A different snapshot is another branch with another execution claim.
    larger = replace(data.samples[0], columns={"y": data.samples[0].columns["y"] * 2})
    branch = rebind(plan, data=transport.StatisticalTransportData(samples=(larger,)))
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        branch._engine._native.export()
    changed = require_success(estimate(branch)).inspect()
    assert changed.program_id == original.program_id
    assert changed.identification_id == original.identification_id
    assert changed.data_snapshot_id != original.data_snapshot_id
    assert changed.execution_id != original.execution_id
    again = require_success(estimate(rebind(branch, data=data)))
    assert again.probabilities == pytest.approx(result.probabilities)
    assert again.uncertainty["replicate_ids"] == result.uncertainty["replicate_ids"]
    assert again.inspect().execution_id == original.execution_id
    assert bytes(outcome.export()) == bytes(result.export())


def test_empirical_table_execution_survives_identification_builder_disposal():
    """The empirical joint table is evaluated from retained checked transport semantics."""
    identified, catalog, _ = fixture()
    builder = identified
    program = builder.formula
    assert program
    del builder
    sample = transport.RegimeSample(
        "source",
        "trial",
        "v1",
        {"y": [0.0] * 20 + [1.0] * 80},
        interventions=(("x", 1.0),),
    )
    data = transport.StatisticalTransportData(samples=(sample,))
    prepared = statistical_study(identified, catalog, data, {"x": 1.0}, bootstrap=0)
    plan = prepared.inspect()
    assert plan.identification.available
    outcome, result = estimated(prepared)
    assert result.mean("y") == pytest.approx(0.8)
    consumer = transport.consume_statistical(bytes(outcome.export()))
    assert consumer.inspect().program_id == result.inspect().program_id
    assert consumer.inspect().identification.available
    assert consumer.probabilities == pytest.approx(result.probabilities)
    # Raw empirical rows are an explicit dependency for a new estimate: the consumed
    # read model names them and cannot be re-estimated.
    assert consumer.acceptance.verified
    assert not consumer.acceptance.resumable
    assert any("not embedded" in item for item in consumer.acceptance.unresolved)
    replay = require_success(estimate(rebind(prepared, data=data)))
    assert replay.probabilities == pytest.approx(result.probabilities)
    assert replay.mean("y") == pytest.approx(0.8)


@pytest.mark.parametrize(
    "provider, estimator_id",
    [
        ("empirical_support_bayesian_bootstrap", "transport.empirical_support_bayesian_bootstrap"),
        ("state_space_dirichlet", "transport.state_space_dirichlet"),
    ],
)
def test_bayesian_statistical_providers_keep_posterior_unlicensed(provider, estimator_id):
    identified, catalog, data = fixture()
    study = statistical_study(identified, catalog, data, {"x": 1.0}, estimator=provider, seed=23)

    assert not study.inspect().uncertainty.available
    outcome, result = estimated(study)
    posterior = result.uncertainty["bayesian_posterior"]
    assert posterior["estimator"] == estimator_id
    assert posterior["interval_method"] == "posterior_equal_tail"
    assert posterior["draws_requested"] == posterior["draws_ok"] == 199
    assert posterior["draws_failed"] == 0
    assert posterior["calibration_status"] == "estimator_grid_not_measured"
    assert posterior["mean_intervals"]
    assert result.uncertainty["available"] is False
    assert result.inspect().uncertainty.available

    # The portable transport artifact retains the posterior summaries and provider identity.
    consumed = transport.consume_statistical(bytes(outcome.export()))
    replayed = consumed.uncertainty["bayesian_posterior"]
    assert replayed["estimator"] == estimator_id
    assert replayed["draws_requested"] == replayed["draws_ok"] == 199
    assert replayed["calibration_status"] == "estimator_grid_not_measured"
    assert replayed["probabilities"] == posterior["probabilities"]
    # Re-fitting from the raw rows on a rebound branch reproduces the posterior summaries.
    refit = require_success(estimate(rebind(study, data=data))).uncertainty["bayesian_posterior"]
    assert refit["estimator"] == estimator_id
    assert refit["draws_requested"] == refit["draws_ok"] == 199
    assert refit["calibration_status"] == "estimator_grid_not_measured"


def test_unknown_dependence_keeps_identification_and_withholds_interval():
    identified, catalog, data = fixture()
    catalog = transport.EvidenceCatalog(
        environments=catalog.environments,
        regimes=catalog.regimes,
        bindings=[transport.RegimeBinding("trial", "v1")],
        target_sampling="representative_sample",
    )
    study = prepare(
        data,
        query=transport.StatisticalTransportQuery(identified, catalog, {"x": 1.0}, bootstrap=39),
        execution=Execution(seed=3),
    )
    assert not study.inspect().uncertainty.available
    result = require_success(estimate(study))
    assert result.mean("y") == pytest.approx(0.8)
    assert result.uncertainty["available"] is False
    assert result.uncertainty["reason"] == "transport.unsupported_dependence"
    assert not result.inspect().uncertainty.available


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_invalid_coverage_is_rejected_during_prepare(level):
    identified, catalog, data = fixture()
    with pytest.raises(ValueError, match="coverage_level"):
        statistical_study(identified, catalog, data, {"x": 1.0}, coverage_level=level)


def test_statistical_native_authority_identity_and_load_round_trip():
    from dataclasses import replace

    identified, catalog, data = fixture()
    study = statistical_study(identified, catalog, data, {"x": 1.0}, bootstrap=29, seed=41)
    outcome, result = estimated(study)
    loaded = require_success(load(bytes(outcome.export())))
    assert loaded.uncertainty == result.uncertainty
    assert loaded.inspect().execution_id == result.inspect().execution_id
    assert bytes(loaded.export()) == bytes(result.export())
    assert result.uncertainty["row"]["calibration_status"] == "not_bound_to_this_execution"
    with pytest.raises(TypeError, match="does not support item assignment"):
        data.samples[0].columns["y"] = ()
    larger = replace(data.samples[0], columns={"y": data.samples[0].columns["y"] * 2})
    second = require_success(
        estimate(
            statistical_study(
                identified,
                catalog,
                transport.StatisticalTransportData((larger,)),
                {"x": 1.0},
                bootstrap=29,
                seed=41,
            )
        )
    )
    assert second.inspect().data_snapshot_id != result.inspect().data_snapshot_id
    assert second.inspect().identification_id == result.inspect().identification_id
    changed = replace(result, probabilities=(1.0, 0.0), uncertainty={"available": False})
    assert require_success(load(bytes(changed.export()))).uncertainty == result.uncertainty


def test_consume_statistical_rechecks_the_point_and_rebinds_after_builder_disposal():
    """The consumer rechecks the embedded plug-in point; only raw samples can re-estimate."""
    from dataclasses import replace

    identified, catalog, data = fixture()
    producer_builder = statistical_study(
        identified, catalog, data, {"x": 1.0}, bootstrap=29, seed=41
    )
    plan = producer_builder.inspect()
    assert plan.identification.available
    outcome, result = estimated(producer_builder)
    # Forged display fields are not the claim: the embedded native point is.
    forged = replace(result, probabilities=(1.0, 0.0), uncertainty={"available": False})
    wire = bytes(forged.export())
    assert wire == bytes(result.export())
    del identified
    consumed = transport.consume_statistical(wire)
    assert consumed.inspect().program_id == result.inspect().program_id
    assert consumed.inspect().identification.available
    assert consumed.inspect().execution_id == result.inspect().execution_id
    # Raw samples are an explicit dependency for re-estimation; nothing is fitted or fetched on load.
    assert not consumed.acceptance.resumable
    assert any("not embedded" in item for item in consumed.acceptance.unresolved)
    replay = require_success(estimate(rebind(producer_builder, data=data)))
    # Independent arithmetic: 80 of 100 trial rows have y = 1.
    assert replay.mean("y") == pytest.approx(0.8)
    assert replay.probabilities == pytest.approx(result.probabilities)
    assert replay.uncertainty == result.uncertainty == consumed.uncertainty
    assert replay.uncertainty["row"]["calibration_status"] == "not_bound_to_this_execution"
    assert require_success(load(wire)).probabilities == result.probabilities
    assert require_success(load(wire)).uncertainty == result.uncertainty
    assert bytes(outcome.export()) == wire


def test_statistical_cancel_and_memory_limits_preserve_atomic_rebind():
    from antecedent.state import CancellationToken

    identified, catalog, data = fixture()
    study = statistical_study(identified, catalog, data, {"x": 1.0}, bootstrap=19)
    outcome = estimate(study)
    require_success(outcome)
    before = bytes(outcome.export())
    token = CancellationToken()
    cancelled = reconfigure(study, execution=Execution(cancel=token))
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        rebind(cancelled, data=data)
    with pytest.raises(CausalCancelledError, match="cancel"):
        estimate(cancelled)
    assert bytes(outcome.export()) == before
    assert bytes(estimate(study).export()) == before
    limited = Execution(limits=ResourceLimits(memory_bytes=1024))
    with pytest.raises(CausalResourceError, match="memory|budget|admission"):
        prepare(
            data,
            query=transport.StatisticalTransportQuery(identified, catalog, {"x": 1.0}),
            execution=limited,
        )


def test_statistical_missingness_is_not_silently_a_population_law():
    from dataclasses import replace

    identified, catalog, data = fixture()
    sample = replace(data.samples[0], columns={"y": [None, 0.0, 1.0]})
    with pytest.raises(ValueError, match="missing"):
        estimate(
            statistical_study(
                identified, catalog, transport.StatisticalTransportData((sample,)), {"x": 1.0}
            )
        )


def test_intervention_world_resamples_do_not_collide():
    from dataclasses import replace

    identified, catalog, data = fixture()
    # Different arm sizes exposed the old key collision: indices from the larger
    # arm were silently used to refit the smaller arm, causing discarded draws.
    zero = replace(
        data.samples[0], columns={"y": [0.0] * 5 + [1.0] * 5}, interventions=(("x", 0.0),)
    )
    two_worlds = transport.StatisticalTransportData((zero, data.samples[0]))
    study = statistical_study(identified, catalog, two_worlds, {"x": 0.0}, bootstrap=29)
    result = require_success(estimate(study))
    assert result.mean("y") == pytest.approx(0.5)
    assert result.uncertainty["row"]["replicates_ok"] == 29
    assert result.uncertainty["row"]["replicates_failed"] == 0
    assert result.uncertainty.replicate_ids == tuple(range(29))
    assert result.uncertainty.to_dict()["replicate_ids"] == list(range(29))


def test_joint_grid_and_contrasts_preserve_shared_replicates():
    from dataclasses import replace

    identified, catalog, data = fixture()
    zero = replace(
        data.samples[0], columns={"y": [0.0] * 40 + [1.0] * 10}, interventions=(("x", 0.0),)
    )
    points = transport.evaluate_statistical_grid(
        identified,
        catalog,
        transport.StatisticalTransportData((zero, data.samples[0])),
        at=[{"x": 0.0}, {"x": 1.0}],
        # A nominal 0.95 percentile interval needs B >= 40 (PERCENTILE_95_MIN_REPLICATES);
        # 39 runs cleanly but withholds the interval as unlicensed.
        bootstrap=40,
        execution=Execution(seed=17),
    )
    assert points[0].mean("y") == pytest.approx(0.2)
    assert points[1].mean("y") == pytest.approx(0.8)
    assert points[0].uncertainty["replicate_ids"] == points[1].uncertainty["replicate_ids"]
    contrast = points[1].contrast(points[0], "y")
    assert contrast["estimate"] == pytest.approx(0.6)
    assert contrast["replicates_ok"] == 40
    assert contrast["interval"][0] < 0.6 < contrast["interval"][1]
    assert points[0].contrast(points[0], "y")["interval"] == (0.0, 0.0)
    assert (
        require_success(load(bytes(points[1].export()))).contrast(
            require_success(load(bytes(points[0].export()))), "y"
        )
        == contrast
    )


def test_grid_failure_scope_changes_identity_and_survives_rebind():
    identified, base, _ = fixture()
    catalog = transport.EvidenceCatalog(
        environments=base.environments,
        regimes=[transport.EvidenceRegime("obs", "target", measured=["x", "y"])],
        bindings=[
            transport.RegimeBinding(
                "obs", "v1", sampling="independent", dependence="independent_studies"
            )
        ],
        target_sampling="representative_sample",
    )
    data = transport.StatisticalTransportData(
        (
            transport.RegimeSample(
                "target",
                "obs",
                "v1",
                {"x": [0.0] * 40 + [1.0], "y": [0.0] * 20 + [1.0] * 21},
            ),
        )
    )
    single = require_success(
        estimate(statistical_study(identified, catalog, data, {"x": 0.0}, bootstrap=99, seed=17))
    )
    points = transport.evaluate_statistical_grid(
        identified,
        catalog,
        data,
        at=[{"x": 0.0}, {"x": 1.0}],
        bootstrap=99,
        execution=Execution(seed=17),
    )
    assert single.uncertainty["row"]["replicates_failed"] == 0
    assert points[0].uncertainty["row"]["replicates_failed"] > 0
    assert points[0].uncertainty["replicate_ids"] == points[1].uncertainty["replicate_ids"]
    assert points[0].inspect().inference_binding_id != single.inspect().inference_binding_id
    loaded = transport.consume_statistical(bytes(points[0].export()))
    assert loaded.uncertainty == points[0].uncertainty
    assert loaded.inspect().execution_id == points[0].inspect().execution_id
    assert loaded.inspect().inference_binding_id == points[0].inspect().inference_binding_id


def test_convenience_target_flag_does_not_reject_source_only_estimation():
    from dataclasses import replace

    identified, catalog, data = fixture()
    catalog = replace(catalog, target_sampling="convenience_sample")
    result = require_success(
        estimate(statistical_study(identified, catalog, data, {"x": 1.0}, bootstrap=9))
    )
    assert result.mean("y") == pytest.approx(0.8)


@pytest.mark.parametrize("value", [-1.0, 0.5, 2.0])
def test_sample_intervention_must_belong_to_declared_domain(value):
    from dataclasses import replace

    identified, catalog, data = fixture()
    invalid = replace(data.samples[0], interventions=(("x", value),))
    with pytest.raises(ValueError, match="declared finite domain"):
        estimate(
            statistical_study(
                identified,
                catalog,
                transport.StatisticalTransportData(samples=(invalid,)),
                {"x": value},
                bootstrap=0,
            )
        )


def test_bayesian_grid_reuses_one_draw_sequence():
    from dataclasses import replace

    identified, catalog, data = fixture()
    zero = replace(
        data.samples[0], columns={"y": [0.0] * 40 + [1.0] * 10}, interventions=(("x", 0.0),)
    )
    both = transport.StatisticalTransportData((zero, data.samples[0]))
    points = transport.evaluate_statistical_grid(
        identified,
        catalog,
        both,
        at=[{"x": 0.0}, {"x": 1.0}],
        estimator="state_space_dirichlet",
        bootstrap=0,
        execution=Execution(seed=17),
    )
    alone = require_success(
        estimate(
            statistical_study(
                identified,
                catalog,
                both,
                {"x": 0.0},
                estimator="state_space_dirichlet",
                bootstrap=0,
                seed=17,
            )
        )
    )
    shared = points[0].uncertainty["bayesian_posterior"]
    separate = alone.uncertainty["bayesian_posterior"]
    assert shared["probabilities"] == separate["probabilities"]
    assert shared["calibration_status"] == "estimator_grid_not_measured"
    assert points[1].uncertainty["bayesian_posterior"]["draws_ok"] == shared["draws_ok"]


def test_transport_query_defaults_are_read_from_the_native_table():
    import dataclasses

    from antecedent._defaults import OMITTED
    from antecedent.transport._impl import StatisticalTransportQuery

    fields = {f.name: f.default for f in dataclasses.fields(StatisticalTransportQuery)}
    assert fields["bootstrap"] == OMITTED["transport_bootstrap"] == 199
    assert fields["coverage_level"] == OMITTED["transport_coverage_level"] == 0.95


def test_iid_bootstrap_interval_executes_from_retained_plan_after_builder_disposal():
    """The outer IID bootstrap replays the retained checked derivation on every replicate."""
    identified, catalog, data = fixture()
    builder = identified
    program = builder.formula
    assert program
    del builder
    prepared = statistical_study(identified, catalog, data, {"x": 1.0}, bootstrap=40, seed=7)
    plan = prepared.inspect()
    assert plan.identification.available
    outcome, result = estimated(prepared)
    assert result.mean("y") == pytest.approx(0.8)
    assert result.inspect().uncertainty.available
    assert result.uncertainty["available"]
    row = result.uncertainty["row"]
    assert row["method"] == "percentile_bootstrap"
    assert row["interval_scope"] == "pointwise"
    assert row["replicates_ok"] == 40
    assert row["replicates_failed"] == 0
    consumer = transport.consume_statistical(bytes(outcome.export()))
    assert consumer.inspect().program_id == result.inspect().program_id
    assert any("not embedded" in item for item in consumer.acceptance.unresolved)
    assert consumer.uncertainty["replicate_ids"] == result.uncertainty["replicate_ids"]
    replay = require_success(estimate(rebind(prepared, data=data)))
    assert replay.mean("y") == pytest.approx(0.8)
    assert replay.uncertainty["replicate_ids"] == result.uncertainty["replicate_ids"]
    assert replay.uncertainty["row"]["replicates_ok"] == 40
    again = require_success(estimate(prepared))
    assert again.uncertainty["replicate_ids"] == result.uncertainty["replicate_ids"]
    assert again.probabilities == pytest.approx(result.probabilities)
