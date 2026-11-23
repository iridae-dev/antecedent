"""Certified trial transport enters the canonical immutable study lifecycle."""

import dataclasses

import antecedent as ant
import pytest
from antecedent import _native, transport
from antecedent.learners import Linear
from antecedent.outcomes import require_success
from antecedent.transport import advanced


def fixture(sampling="independent_samples", effect=2.0):
    treatment = [False, True] * 60 + [False] * 80
    data = transport.TrialAipwData(
        {},
        [1.0 + effect * t for t in treatment[:120]] + [0.0] * 80,
        treatment,
        [True] * 120 + [False] * 80,
        [0.5] * 200,
        sampling,
    )
    query = advanced.TrialAipwQuery(
        ant.Admg.from_edges(["a", "y"], [("a", "y")]),
        advanced.SelectionDiagram("trial", "target", []),
        "a",
        "y",
    )
    return query, data


def prepare(query, data, bootstrap=5):
    return ant.prepare(
        data,
        query=query,
        provider=transport.TrialAipw(outcome=Linear(), folds=3),
        inference=transport.TransportInference(bootstrap=bootstrap),
        execution=ant.Execution(seed=8),
    )


@pytest.mark.parametrize("sampling", ["independent_samples", "nested_cohort"])
def test_trial_factory_returns_immutable_canonical_study_and_branches(sampling):
    query, data = fixture(sampling)
    study = prepare(query, data)
    assert isinstance(study, ant.PreparedAnalysis)
    with pytest.raises(dataclasses.FrozenInstanceError):
        study.execution = ant.Execution(seed=9)
    assert study.validator_names == ()
    assert study.inspect().stage == "prepared"
    assert study.inspect().identification.available
    assert not study.inspect().uncertainty.available
    baseline = require_success(ant.estimate(study))
    assert baseline.estimate == pytest.approx(2.0, abs=1e-10)
    assert baseline.interval == pytest.approx((2.0, 2.0), abs=1e-10)
    assert len(baseline.replicates) == 5
    _, changed = fixture(sampling, effect=4.0)
    fresh = ant.rebind(study, data=changed)
    assert fresh is not study
    assert "fits" in fresh.reuse.invalidated
    assert require_success(ant.estimate(fresh)).estimate == pytest.approx(4.0, abs=1e-10)
    assert require_success(ant.estimate(study)).estimate == pytest.approx(2.0, abs=1e-10)
    assert baseline.estimate == pytest.approx(2.0, abs=1e-10)
    loaded = require_success(ant.load(bytes(baseline.export())))
    assert loaded.estimate == baseline.estimate
    assert loaded.replicates == baseline.replicates


def test_scientific_reconfigure_reads_owned_snapshot_without_reingestion(monkeypatch):
    query, data = fixture()
    study = prepare(query, data)
    baseline = require_success(study.estimate())
    # Change the caller's buffers after the single native ownership transfer.
    data.outcome[:] = [100.0] * 200
    data.randomization[:] = [0.0] * 200

    def forbidden(*args, **kwargs):
        raise AssertionError("scientific reconfiguration re-ingested Python rows")

    with monkeypatch.context() as context:
        context.setattr(transport.TrialAipwData, "_buffers", forbidden)
        context.setattr(_native, "prepare_learned_trial", forbidden)
        point_only = ant.reconfigure(study, inference=transport.TransportInference(bootstrap=0))
        assert point_only.inspect().stage == "prepared"
        assert point_only.reuse.retained == ("data",)
        assert "statistical_plan" in point_only.reuse.invalidated
        point = require_success(point_only.estimate())
        replay = require_success(study.estimate())
    assert point.estimate == pytest.approx(2.0, abs=1e-10)
    assert point.interval is None
    assert point.replicates == ()
    assert replay.replicates == baseline.replicates
    assert len(baseline.replicates) == 5


def test_preparation_and_scientific_branching_never_call_estimation(monkeypatch):
    from antecedent.estimation import _PreparedEngine as Engine

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation or branching executed the estimator")

    query, data = fixture()
    with monkeypatch.context() as context:
        context.setattr(Engine, "estimate", forbidden)
        study = prepare(query, data)
        branch = ant.reconfigure(study, inference=transport.TransportInference(bootstrap=0))
        fresh = ant.rebind(study, data=data)
        assert (
            study.inspect().stage == branch.inspect().stage == fresh.inspect().stage == "prepared"
        )
        assert "TrialAipwQuery" in repr(study)
    assert require_success(branch.estimate()).estimate == pytest.approx(2.0, abs=1e-10)


def test_shared_transport_inference_has_one_operational_seed_location():
    with pytest.raises(TypeError, match="seed"):
        transport.TransportInference(bootstrap=5, seed=8)
    assert not hasattr(advanced, "prepare_trial")
    query, data = fixture()
    study = prepare(query, data)
    alternative = ant.reconfigure(study, execution=ant.Execution(seed=19))
    assert study.execution.seed == 8
    assert alternative.execution.seed == 19
    assert require_success(alternative.estimate()).estimate == pytest.approx(2.0, abs=1e-10)


def test_reconfigure_rejects_changed_variable_coordinates_without_mutating_source():
    from antecedent.errors import CausalValueError

    query, data = fixture()
    study = prepare(query, data)
    altered = dataclasses.replace(
        query,
        graph=ant.Admg.from_edges(["y", "a"], [("a", "y")]),
    )
    with pytest.raises(CausalValueError, match="coordinate schema"):
        ant.reconfigure(study, query=altered)
    assert require_success(study.estimate()).estimate == pytest.approx(2.0, abs=1e-10)


def test_canonical_trial_preserves_compiler_numerics_encoding_and_identity():
    from antecedent.transport._impl.trial import prepare_trial as compile_trial

    query, data = fixture()
    canonical = require_success(prepare(query, data).estimate())
    compiler = compile_trial(
        query,
        data,
        provider=transport.TrialAipw(outcome=Linear(), folds=3),
        inference=transport.TransportInference(bootstrap=5),
        seed=8,
    )
    reference = require_success(compiler.estimate())
    assert canonical.estimate == reference.estimate
    assert canonical.replicates == reference.replicates
    assert canonical.inspect().execution_id == reference.inspect().execution_id
    assert bytes(canonical.export()) == bytes(reference.export())
