"""Explicit canonical drift diagnostics and independently consumed baselines."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

import antecedent as ant
import pytest
from antecedent import drift
from antecedent.artifacts import ArtifactExpectation
from antecedent.execution import ResourceLimits
from antecedent.outcomes import require_success

from test_staleness import GRAPH, POLICY, QUERY, sample


def _study(**controls):
    return ant.prepare(
        sample(1),
        graph=GRAPH,
        query=QUERY,
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        execution=ant.Execution(**controls),
    )


def test_staleness_assesses_canonical_preparation_without_estimating_or_mutating(monkeypatch):
    from antecedent.estimation import _PreparedEngine as Engine

    def no_causal_estimation(*args, **kwargs):
        raise AssertionError("drift assessment must not estimate the causal effect")

    monkeypatch.setattr(Engine, "estimate", no_causal_estimation)
    study = _study(seed=19, threads=1)
    before = study.plan_view()
    original = require_success(drift.staleness(study, sample(1), policy=POLICY))
    assert original.recommendation == "reuse"
    assert original.findings == ()
    assert original.changed_keys == original.invalidated == ()
    shifted = drift.staleness(study, sample(2, shift=0.35), policy=POLICY)
    assert shifted.accepted
    assert shifted.answer.kind == "structured"
    report = require_success(shifted)
    assert report.recommendation == "refresh"
    assert "data_snapshot" in report.changed_keys
    assert {"score_artifact", "law", "decision"} <= set(report.invalidated)
    assert "identification" not in report.invalidated
    assert report.old_snapshot == original.old_snapshot
    assert report.new_snapshot != original.new_snapshot
    assert study.plan_view() == before
    assert study.inspect().stage == "prepared"


def test_staleness_schema_refusal_is_a_typed_successful_assessment():
    study = _study()
    wider = {**sample(4), "extra": sample(4)["z"]}
    outcome = drift.staleness(study, wider, policy=POLICY)
    assert outcome.accepted
    report = require_success(outcome)
    assert report.recommendation == "refuse"
    assert any(finding.kind == "schema_change" for finding in report.findings)
    assert "project the new data" in report.next_attempt


def test_baseline_canonical_export_and_load_preserve_summary_identity_without_live_state():
    study = _study(seed=19)
    baseline = require_success(drift.baseline(study, policy=POLICY))
    artifact = baseline.export()
    assert bytes(artifact).startswith(b"ANTSTALEBASE1")
    assert artifact.kind == "staleness_baseline_v1"
    assert artifact.expectation == ArtifactExpectation(artifact.kind, baseline.identity)
    loaded = ant.load(artifact, expected=artifact.expectation)
    assert loaded.accepted
    assert loaded.study is None
    assert loaded.acceptance.status == "sealed"
    assert not loaded.acceptance.verified
    assert loaded.acceptance.identity_bound
    assert not loaded.acceptance.replayable and not loaded.acceptance.resumable
    retained = require_success(loaded)
    assert retained.identity == baseline.identity
    assert retained.plan_identity == baseline.plan_identity
    assert retained.snapshot == baseline.snapshot
    assert retained.policy_identity == POLICY.identity
    assert retained.rows == retained.complete_rows == 800
    assert bytes(retained.export()) == bytes(artifact)
    assert not hasattr(retained, "_prepared") and not hasattr(retained, "_data")
    new = sample(5, shift=0.3)
    direct = require_success(drift.staleness(study, new, policy=POLICY))
    independent = require_success(retained.assess(new, policy=POLICY))
    assert independent.to_dict() == direct.to_dict()
    assert (
        require_success(drift.staleness(bytes(artifact), new, policy=POLICY)).identity
        == direct.identity
    )


def test_drift_inspection_is_bounded_and_never_reprofiles(monkeypatch):
    study = _study()
    baseline = require_success(drift.baseline(study, policy=POLICY))
    report = require_success(drift.staleness(study, sample(2, shift=0.35), policy=POLICY))

    def no_materialization(*args, **kwargs):
        raise AssertionError("inspection must not assess, profile or serialize the baseline")

    monkeypatch.setattr(drift, "baseline", no_materialization)
    monkeypatch.setattr(drift, "staleness", no_materialization)
    monkeypatch.setattr(drift.StalenessReport, "to_dict", no_materialization)
    monkeypatch.setattr(drift.StalenessBaseline, "export", no_materialization)
    assert len(repr(report)) < 400
    assert len(repr(baseline)) < 400
    assert report.inspect().stage == "executed"
    assert baseline.inspect().stage == "consumed"
    with pytest.raises(FrozenInstanceError):
        report.recommendation = "reuse"
    with pytest.raises(TypeError, match="does not support item assignment"):
        report.evaluations[0].auxiliary["fabricated"] = 1.0


def test_baseline_consumption_rejects_tampering_and_wrong_expected_identity():
    artifact = require_success(drift.baseline(_study(), policy=POLICY)).export()
    with pytest.raises(ant.CausalError):
        ant.load(artifact, expected="0" * 64)
    changed = bytearray(bytes(artifact))
    changed[-1] ^= 1
    with pytest.raises(ant.CausalError):
        ant.load(bytes(changed))
    with pytest.raises(ant.CausalError):
        drift.consume_baseline(bytes(artifact)[:20])


def test_diagnostic_execution_controls_propagate_and_leave_source_usable():
    from antecedent import _native
    from antecedent.errors import CausalCancelledError, CausalResourceError

    study = _study()
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        drift.baseline(study, policy=POLICY, execution=ant.Execution(cancel=token))
    with pytest.raises(CausalResourceError):
        drift.staleness(
            study,
            sample(2),
            policy=POLICY,
            execution=ant.Execution(limits=ResourceLimits(memory_bytes=1)),
        )
    assert (
        require_success(drift.staleness(study, sample(1), policy=POLICY)).recommendation == "reuse"
    )


def test_staleness_baseline_assessment_is_independent_in_a_fresh_process(tmp_path):
    artifact = require_success(drift.baseline(_study(), policy=POLICY)).export()
    expected = require_success(drift.staleness(artifact, sample(5, shift=0.3), policy=POLICY))
    path = tmp_path / "baseline.ant"
    path.write_bytes(bytes(artifact))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
import antecedent as ant
import numpy as np
from antecedent import drift
from antecedent.outcomes import require_success
rng = np.random.default_rng(5)
z = rng.normal(size=800) + 0.3
t = (rng.uniform(size=800) < 1 / (1 + np.exp(-0.5*z))).astype(float)
y = 1 + 2*t + z + rng.normal(size=800)
loaded = ant.load(Path(sys.argv[1]).read_bytes(), expected=sys.argv[2])
assert loaded.study is None
assert loaded.acceptance.status == 'sealed' and loaded.acceptance.identity_bound
assert not loaded.acceptance.replayable and not loaded.acceptance.resumable
baseline = require_success(loaded)
assert not hasattr(baseline, '_prepared') and not hasattr(baseline, '_data')
report = require_success(drift.staleness(baseline, {'t':t, 'y':y, 'z':z},
    policy=drift.StalenessPolicy(subsample=60, permutations=99)))
print(json.dumps(report.to_dict()))
""",
            str(path),
            artifact.expectation.identity,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == expected.to_dict()


def test_baseline_outcomes_compose_directly_with_staleness_assessment():
    study = _study(seed=19)
    created = drift.baseline(study, policy=POLICY)
    artifact = created.export()
    loaded = ant.load(artifact, expected=artifact.expectation)
    new = sample(5, shift=0.3)
    expected = require_success(drift.staleness(require_success(created), new, policy=POLICY))
    for source in (created, loaded):
        assessed = drift.staleness(source, new, policy=POLICY)
        assert assessed.accepted
        assert require_success(assessed).to_dict() == expected.to_dict()


def test_refused_baseline_outcome_preserves_original_scientific_refusal():
    data = sample(1)
    graph = ant.Admg.from_edges(["t", "y"], [("t", "y")], bidirected=[("t", "y")])
    study = ant.prepare(data, graph=graph, query=QUERY, refute=False)
    created = drift.baseline(study, policy=POLICY)
    assert created.refusal is not None
    assessed = drift.staleness(created, sample(5), policy=POLICY)
    assert assessed.refusal is not None
    assert assessed.refusal.code == created.refusal.code
    assert assessed.refusal._error is created.refusal._error
    with pytest.raises(type(created.refusal._error)) as caught:
        require_success(assessed)
    assert caught.value is created.refusal._error


def test_screened_member_drift_uses_active_rows_and_keeps_original_masked_identity():
    import numpy as np
    from antecedent.estimation import CandidateScreen

    from test_prepared_batch_lifecycle import _inputs, _prepare

    data = _inputs(400)
    data["y"] += 4 * data["t"] * (np.arange(400) >= 200)
    screen = CandidateScreen("split", "unrecorded", tuple(range(200)), tuple(range(200, 400)))
    batch = require_success(_prepare(data, candidate_screen=screen).estimate())
    member = batch.member(0)
    assert require_success(member).effect == pytest.approx(6, abs=0.3)
    original = require_success(drift.baseline(member.study, policy=POLICY))
    assert member.study._data.num_rows == 400
    assert original.rows == original.complete_rows == 200
    unchanged = require_success(drift.staleness(member.study, member.study._data, policy=POLICY))
    assert unchanged.recommendation == "reuse"
    assert unchanged.new_rows == 200
    assert unchanged.old_snapshot == unchanged.new_snapshot == original.snapshot

    # Change only excluded screening observations. Their values still belong to
    # the original stored source identity, while active-row diagnostics stay equal.
    changed = {name: column.copy() for name, column in data.items()}
    changed["z"][:200] += 1000
    changed["y"][:200] -= 1000
    new_batch = require_success(_prepare(changed, candidate_screen=screen).estimate())
    report = require_success(
        drift.staleness(original, new_batch.member(0).study._data, policy=POLICY)
    )
    assert report.recommendation == "reuse"
    assert report.findings == ()
    assert report.new_rows == 200
    assert report.old_snapshot == original.snapshot
    assert report.new_snapshot != original.snapshot


def test_declared_sampling_weighted_population_refuses_unweighted_drift_diagnostics():
    from antecedent.errors import CausalUnsupportedError
    from antecedent.estimators import LinearAdjustment

    from test_sampling_weights import EDGES, _survey
    from test_sampling_weights import QUERY as WEIGHTED_QUERY

    study = ant.prepare(
        _survey(),
        graph=EDGES,
        query=WEIGHTED_QUERY,
        estimator=LinearAdjustment(weights="w"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    with pytest.raises(CausalUnsupportedError, match="weighted staleness diagnostic") as caught:
        require_success(drift.baseline(study, policy=POLICY))
    assert caught.value.reason_code == "route_not_supported"
    # The unsupported diagnostic does not prevent executing the declared weighted estimator.
    result = require_success(ant.estimate(study))
    assert result.estimate.sampling_weights is not None
    assert result.estimate.sampling_weights.effective_sample_size > 0


def test_independent_baseline_load_honors_execution_limits_and_releases_decode_leases():
    from antecedent import _native
    from antecedent.errors import CausalCancelledError, CausalResourceError

    artifact = require_success(drift.baseline(_study(), policy=POLICY)).export()
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.load(artifact, execution=ant.Execution(cancel=token))
    tiny = ant.Execution(limits=ResourceLimits(memory_bytes=1))
    before = tiny._handle.available_memory_bytes
    with pytest.raises(CausalResourceError):
        ant.load(artifact, execution=tiny)
    assert tiny._handle.available_memory_bytes == before
    roomy = ant.Execution(limits=ResourceLimits(memory_bytes=64_000_000))
    before = roomy._handle.available_memory_bytes
    loaded = ant.load(artifact, expected=artifact.expectation, execution=roomy)
    assert loaded.accepted and loaded.acceptance.status == "sealed"
    assert roomy._handle.available_memory_bytes == before
    assert require_success(loaded).identity == artifact.expectation.identity


@pytest.mark.parametrize("source_kind", ["bytes", "artifact", "loaded", "outcome"])
def test_independent_assessment_honors_resource_and_cancel_controls_and_releases_leases(
    source_kind,
):
    from antecedent import _native
    from antecedent.errors import CausalCancelledError, CausalResourceError

    artifact = require_success(drift.baseline(_study(), policy=POLICY)).export()
    loaded = ant.load(artifact, expected=artifact.expectation)
    source = {
        "bytes": bytes(artifact),
        "artifact": artifact,
        "loaded": require_success(loaded),
        "outcome": loaded,
    }[source_kind]
    data = sample(5, shift=0.3)
    tiny = ant.Execution(limits=ResourceLimits(memory_bytes=1))
    before = tiny._handle.available_memory_bytes
    with pytest.raises(CausalResourceError):
        drift.staleness(source, data, policy=POLICY, execution=tiny)
    assert tiny._handle.available_memory_bytes == before

    token = _native.CancellationToken()
    token.cancel()
    cancelled = ant.Execution(cancel=token, limits=ResourceLimits(memory_bytes=64_000_000))
    before = cancelled._handle.available_memory_bytes
    with pytest.raises(CausalCancelledError):
        drift.staleness(source, data, policy=POLICY, execution=cancelled)
    assert cancelled._handle.available_memory_bytes == before

    roomy = ant.Execution(limits=ResourceLimits(memory_bytes=64_000_000))
    before = roomy._handle.available_memory_bytes
    report = require_success(drift.staleness(source, data, policy=POLICY, execution=roomy))
    assert report.recommendation == "refresh"
    assert report.baseline_identity == artifact.expectation.identity
    assert roomy._handle.available_memory_bytes == before
    expected = require_success(drift.staleness(_study(), data, policy=POLICY, execution=roomy))
    assert report.to_dict() == expected.to_dict()
    assert roomy._handle.available_memory_bytes == before
