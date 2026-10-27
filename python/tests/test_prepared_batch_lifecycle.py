"""Immutable native batch execution, ownership, and completed score snapshots."""

from dataclasses import FrozenInstanceError

import antecedent as ant
import numpy as np
import pytest
from antecedent.estimation import estimate_many, prepare_many
from antecedent.outcomes import require_success


def _inputs(n=260, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-0.3 * z))).astype(float)
    y = 2 * t + z + rng.normal(scale=0.2, size=n)
    return {"z": z, "t": t, "y": y, "y2": 0.5 * y - z}


def _prepare(data=None, **declarations):
    return prepare_many(
        _inputs() if data is None else data,
        graph=[("z", "t"), ("z", "y"), ("z", "y2"), ("t", "y"), ("t", "y2")],
        queries=[ant.AverageEffect("t", "y"), ant.AverageEffect("t", "y2")],
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        **declarations,
    )


def _effects(batch):
    return tuple(require_success(member).effect for member in batch.members)


def test_preparation_branch_and_inspection_do_not_materialize_plans():
    study = _prepare()
    assert study._native.materialized_plan_count() == 0
    assert study._native.retained_score_table_count() == 0
    assert study.inspect().stage == "prepared"
    assert "members=2" in repr(study)
    study._repr_html_()
    branch = ant.rebind(study, data=_inputs(seed=5))
    assert branch._native.materialized_plan_count() == 0
    assert study._native.materialized_plan_count() == 0
    assert "fits" in branch.reuse.invalidated
    with pytest.raises(FrozenInstanceError):
        study.execution = ant.Execution(seed=7)


def test_owned_inputs_and_completed_runs_are_independent():
    data = _inputs()
    study = _prepare(data)
    data["y"][:] = 1000
    first = require_success(ant.estimate(study))
    assert _effects(first)[0] == pytest.approx(2, abs=0.3)
    assert study._native.materialized_plan_count() == 0
    assert first.study._native.materialized_plan_count() == 2
    assert first.study._native.retained_score_table_count() == 2
    before = _effects(first)
    second = require_success(ant.estimate(ant.rebind(first.study, data=_inputs(190, seed=9))))
    assert len(first.retarget_rows()) == 260
    assert len(second.retarget_rows()) == 190
    assert first.retarget_snapshot() != second.retarget_snapshot()
    assert _effects(first) == before
    assert first.member(0) is first.members[0]
    assert first.members[0].study is not None
    assert first.members[0].inspect().identification.available


def test_completed_native_fits_are_reused_on_repeated_execution():
    first = require_success(_prepare().estimate())
    second = require_success(estimate_many(first.study))
    assert _effects(second) == _effects(first)
    assert second.retarget_snapshot() == first.retarget_snapshot()
    assert second.study._data is first.study._data
    assert second.study._native.materialized_plan_count() == 2
    for member in second.members:
        result = require_success(member)
        assert any("propensity shared" in note for note in result.diagnostics)


def test_batch_stage_observers_name_each_query_and_preserve_callback_errors():
    events = []
    policy = ant.Execution(on_stage=lambda name, payload: events.append((name, payload)))
    result = require_success(_prepare(execution=policy).estimate())
    assert len(result.members) == 2
    assert {payload["query_index"] for _, payload in events} == {0, 1}
    assert all(isinstance(name, str) and isinstance(payload, dict) for name, payload in events)
    expected = RuntimeError("batch observer failed")

    def explode(_name, _payload):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _prepare(execution=ant.Execution(on_stage=explode)).estimate()
    assert caught.value is expected
    assert any(frame.name == "explode" for frame in caught.traceback)


def test_query_reconfiguration_reprepares_and_retains_owned_table():
    initial = require_success(_prepare().estimate())
    branch = ant.reconfigure(initial.study, queries=[ant.AverageEffect("t", "y2")])
    assert branch._data is initial.study._data
    assert branch._native.materialized_plan_count() == 0
    assert "programs" in branch.reuse.invalidated
    alternative = require_success(branch.estimate())
    assert len(alternative.members) == 1
    assert _effects(alternative)[0] == _effects(initial)[1]
    assert len(initial.members) == 2


def test_completed_member_branches_are_scalar_studies_with_fresh_row_designs():
    batch = require_success(_prepare().estimate())
    member = batch.member(0)
    branch = ant.rebind(member.study, data=_inputs(190, seed=9))
    changed = require_success(ant.estimate(branch))
    solo = require_success(
        ant.analyze(
            _inputs(190, seed=9),
            graph=batch.study._declarations["graph"],
            query=ant.AverageEffect("t", "y"),
            estimator="aipw",
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
        )
    )
    assert changed.effect == solo.effect
    assert "family_inference" in branch.reuse.invalidated
    assert require_success(member).effect == _effects(batch)[0]


def _assert_portable_family_fields(member):
    live = require_success(member)
    artifact = member.export()
    loaded = ant.load(artifact, expected=artifact.expectation)
    restored = require_success(loaded)
    assert loaded.study is None
    assert loaded.acceptance.status == "sealed"
    assert not loaded.acceptance.replayable
    assert "dependencies.checked_batch_family_operation" in loaded.acceptance.unresolved
    assert loaded.inspect().claim_id == member.inspect().claim_id
    assert loaded.inspect().contract == member.inspect().contract
    payload = restored.decoded_artifact.payload
    covariance = payload["joint_covariance"]
    expected_covariance = np.asarray(live.estimate.joint_covariance)
    assert covariance["dim"] == len(expected_covariance)
    np.testing.assert_array_equal(
        np.asarray(covariance["values"]).reshape(expected_covariance.shape), expected_covariance
    )
    assert tuple(payload["simultaneous_interval"]) == live.estimate.simultaneous_interval
    return payload


def test_member_artifacts_keep_native_identity_without_reconstructing_a_live_study():
    batch = require_success(_prepare().estimate())
    _assert_portable_family_fields(batch.member(0))
    with pytest.raises(ant.errors.CausalValueError, match="select member"):
        batch.export()


def test_joint_cell_member_artifacts_preserve_family_contrasts_and_bands():
    data = _inputs(700)
    rng = np.random.default_rng(17)
    data["t2"] = (rng.random(700) < 0.5).astype(float)
    data["y"] += 0.8 * data["t2"] + 0.5 * data["t"] * data["t2"]
    graph = [("z", "t"), ("z", "y"), ("t", "y"), ("t2", "y")]
    queries = [
        ant.InterventionResponse(
            "y", intervention=[ant.intervention.Set("t", 1.0), ant.intervention.Set("t2", level)]
        )
        for level in (0.0, 1.0)
    ]
    batch = require_success(
        prepare_many(
            {name: data[name] for name in ("z", "t", "t2", "y")},
            graph=graph,
            queries=queries,
            estimator="cell.aipw",
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
            family=ant.estimation.CellFamily(posterior_draws=80),
        ).estimate()
    )
    for member in batch.members:
        live = require_success(member)
        payload = _assert_portable_family_fields(member)
        assert tuple(payload["family_contrast"]) == live.estimate.family_contrast
        assert tuple(payload["family_contrast_interval"]) == live.estimate.family_contrast_interval
        assert live.estimate.family_contrast_posterior is not None
        # The supported encoding contains contrast summaries, not the live aligned draws.
        assert "family_contrast_posterior" not in payload


def test_native_identification_refusal_is_retained_before_execution():
    from antecedent.graph import TieredBackground, WithinTier

    data = _inputs()
    data["t2"] = np.roll(data["t"], 1)
    graph = TieredBackground(tiers=[["z"], ["t", "t2"], ["y"]], within_tier=WithinTier.UNKNOWN)
    query = ant.InterventionResponse(
        "y", intervention=[ant.intervention.Set("t", 1.0), ant.intervention.Set("t2", 1.0)]
    )
    study = prepare_many(data, graph=graph, queries=[query], estimator="cell.aipw", refute=False)
    assert study.refusal is not None
    assert study.inspect().stage == "prepared"
    assert study.shared_design is None
    assert study._native is None
    outcome = ant.estimate(study)
    assert outcome.refusal is study.refusal
    assert outcome.study is study


def test_rebinding_reorders_native_views_without_copying_an_owned_table():
    study = _prepare()
    fresh = ant.data.table(_inputs(seed=8))
    reversed_columns = fresh.select(tuple(reversed(fresh.column_names)))
    branch = study.rebind(data=reversed_columns)
    assert branch._data.bytes_copied == fresh.bytes_copied
    assert branch._data.column_names == study._data.column_names
    expected = require_success(study.rebind(data=fresh).estimate())
    assert _effects(require_success(branch.estimate())) == _effects(expected)


def test_batch_retarget_guide_executes_its_declared_scientific_journey():
    import re
    from pathlib import Path

    guide = Path(__file__).resolve().parents[2] / "docs/guides/batch-retarget.md"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), flags=re.DOTALL)
    assert len(blocks) == 2
    namespace = {}
    for index, block in enumerate(blocks):
        exec(compile(block, f"{guide}:python-block-{index}", "exec"), namespace)
    report = namespace["report"]
    assert report.complete
    assert all(abs(claim.value - 2) < 0.3 for claim in report.claims)
    assert report.contrasts[0].std_error is not None
    assert report.simultaneous_interval().level == 0.95


def test_average_effect_batches_reject_every_explicit_cell_family_declaration():
    from antecedent.errors import CausalTypeError, CausalValueError
    from antecedent.estimation import CellFamily, analyze_many

    for operation in (prepare_many, analyze_many):
        for family in (
            CellFamily(),
            CellFamily(contrast="interaction"),
            CellFamily(posterior_draws=80),
        ):
            with pytest.raises(CausalValueError, match="joint InterventionResponse"):
                operation(object(), graph=[], queries=[ant.AverageEffect("t", "y")], family=family)
        with pytest.raises(CausalTypeError, match="CellFamily"):
            operation(
                object(),
                graph=[],
                queries=[ant.AverageEffect("t", "y")],
                family={"contrast": "interaction"},
            )
        for removed in ("family_contrast", "family_posterior_draws"):
            with pytest.raises(TypeError, match="unexpected keyword"):
                operation(
                    object(), graph=[], queries=[ant.AverageEffect("t", "y")], **{removed: 80}
                )
    with pytest.raises(CausalTypeError, match="posterior_draws"):
        CellFamily(posterior_draws=True)
    with pytest.raises(CausalValueError, match="positive"):
        CellFamily(posterior_draws=0)


def test_reconfigure_batch_rejects_data_changes_without_mutating_its_preparation():
    from antecedent.errors import CausalValueError

    study = _prepare()
    with pytest.raises(CausalValueError, match="rebind"):
        ant.reconfigure(study, data=_inputs(seed=4))
    assert study._native.materialized_plan_count() == 0
    assert study._data.num_rows == 260


def test_batch_retarget_returns_a_typed_outcome_and_preserves_argument_errors():
    from antecedent.errors import CausalUnsupportedError
    from antecedent.estimation import RetargetClaim
    from antecedent.outcomes import Outcome

    batch = require_success(_prepare().estimate())
    rows = batch.retarget_rows()
    claim = RetargetClaim("effect", 0, np.ones(len(rows)))
    accepted = batch.retarget([claim])
    assert isinstance(accepted, Outcome)
    assert accepted.accepted
    assert require_success(accepted).complete
    refused = batch.retarget([claim], expected_snapshot="another-dataset")
    assert refused.refusal is not None
    assert refused.refusal.code == "row_weights_bound_to_snapshot"
    with pytest.raises(CausalUnsupportedError) as invalid:
        batch.retarget([claim, claim])
    assert invalid.value.reason_code == "invalid_argument"


def test_screened_member_scientific_reconfiguration_uses_its_actual_native_rows():
    from antecedent.estimation import CandidateScreen

    data = _inputs(400)
    data["y"] += 4 * data["t"] * (np.arange(400) >= 200)
    screen = CandidateScreen("split", "unrecorded", tuple(range(200)), tuple(range(200, 400)))
    batch = require_success(_prepare(data, candidate_screen=screen).estimate())
    member = batch.member(0)
    assert batch.study._data.num_rows == 400
    assert member.study._data.num_rows == 400
    assert batch.retarget_rows() == tuple(range(200, 400))
    assert member.study._data.bytes_copied == 0
    assert member.study._declarations["participant"] is None
    assert member.study._engine._snapshot_data is member.study._data
    alternative = ant.reconfigure(member.study, estimator="linear.adjustment.ate")
    assert alternative._data is member.study._data
    assert alternative._data.num_rows == 400
    assert "family_inference" in alternative.reuse.invalidated
    changed = require_success(ant.estimate(alternative))
    expected = require_success(
        ant.analyze(
            {name: column[200:] for name, column in data.items()},
            query=member.study.query,
            graph=batch.study._declarations["graph"],
            estimator="linear.adjustment.ate",
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
        )
    )
    assert changed.effect == expected.effect
    assert changed.effect == pytest.approx(6, abs=0.3)
    assert batch.study._data.num_rows == 400
    assert member.study._data.num_rows == 400


def test_retarget_outcome_inspects_native_standing_without_reexecuting_or_encoding(monkeypatch):
    from antecedent.estimation import RetargetClaim

    batch = require_success(_prepare().estimate())
    outcome = batch.retarget([RetargetClaim("effect", 0, np.ones(len(batch.retarget_rows())))])
    report = require_success(outcome)

    def unexpected(*_args, **_kwargs):
        pytest.fail("inspection must not execute native work or materialize a report")

    monkeypatch.setattr(type(report), "simultaneous_interval", unexpected)
    monkeypatch.setattr(type(report), "complete_family", unexpected)
    monkeypatch.setattr(type(report), "to_rows", unexpected)
    inspection = outcome.inspect()
    assert outcome.answer.kind == "structured"
    assert inspection.family == "batch_retarget" and inspection.stage == "executed"
    assert outcome.identities["family_id"] == report.family_id
    assert outcome.identities["snapshot_id"] == report.snapshot_id
    assert "data_snapshot_id" not in outcome.identities
    assert outcome.identities["estimator_fingerprint"] == report.estimator_fingerprint
    assert outcome.provenance["scores_source"] == "estimated"
    assert outcome.uncertainty.available
    assert outcome.uncertainty.payload["inference_claim"] == report.inference_claim == "point_only"
    assert outcome.uncertainty.payload["complete_family"]
    assert outcome.uncertainty.payload["simultaneous_interval"]["status"] == "published"
    assert "matrix" not in outcome.uncertainty.payload
    assert outcome.support.available and outcome.support.payload["standing"] == "supported"
    assert outcome.assumptions.available
    assert outcome.assumptions.summary == report.scope_note
    assert outcome.assumptions.payload["standing"] == "declared"
    assert not inspection.identification.available
    assert "simultaneous_interval" in inspection.operations
    assert len(repr(report)) < 100
    report._repr_html_()


def test_partial_retarget_outcome_keeps_surviving_covariance_and_native_failure_standing():
    from antecedent.estimation import RetargetClaim

    batch = require_success(_prepare().estimate())
    outcome = batch.retarget(
        [
            RetargetClaim("good", 0, np.ones(len(batch.retarget_rows()))),
            RetargetClaim("short", 0, np.ones(5)),
        ]
    )
    report = require_success(outcome)
    assert outcome.answer.kind == "partial"
    assert outcome.answer.detail == "batch_retarget.partial_family"
    assert outcome.uncertainty.available  # the surviving claim still has covariance
    assert outcome.uncertainty.reason == "batch_retarget.partial_family"
    assert outcome.uncertainty.payload["covariance_members"] == ("good",)
    assert outcome.uncertainty.payload["failed_members"] == ("short",)
    assert outcome.uncertainty.payload["simultaneous_interval"]["status"] == "unavailable"
    assert outcome.support.payload["standing"] == "partial"
    assert "simultaneous_interval" not in outcome.inspect().operations
    assert "complete_family" not in outcome.inspect().operations
    assert report.claims[1].refusal_code == "invalid_argument"
    assert outcome.assumptions.summary == report.scope_note


def test_point_only_native_report_shape_never_acquires_inferred_covariance():
    """Presentation unit fixture; this does not establish a scientific capability."""
    from antecedent.estimation import BatchRetarget, RetargetMember
    from antecedent.outcomes import Outcome

    refusal = {
        "reason_code": "cell_not_licensed",
        "detail": "batch_retarget.point_only_member",
        "message": "native provenance withholds covariance",
    }
    report = BatchRetarget(
        family_id="family-fixture",
        snapshot_id="row-fixture",
        scores_source="estimated",
        estimator_fingerprint="estimator-fixture",
        claims=(
            RetargetMember(
                kind="claim",
                name="point",
                estimand="fixture",
                status="point_only",
                value=2.0,
                std_error=None,
                uncertainty_kind="none",
                support_status="supported",
            ),
        ),
        contrasts=(),
        covariance_names=(),
        covariance=None,
        complete=False,
        failed_members=(),
        point_only_members=("point",),
        inference_claim="point_only",
        scope_note="Native fixture scope.",
        _closed={"status": "unavailable", **refusal},
        _complete_refusal=refusal,
        _rows=(),
    )
    outcome = Outcome(value=report)
    assert outcome.answer.kind == "structured"  # the point exists; its interval does not
    assert outcome.answer.detail == "batch_retarget.point_only_member"
    assert not outcome.uncertainty.available
    assert outcome.uncertainty.reason == "batch_retarget.point_only_member"
    assert outcome.uncertainty.payload["point_only_members"] == ("point",)
    assert outcome.support.payload["standing"] == "supported"
    assert "covariance_between" not in outcome.inspect().operations
    assert "simultaneous_interval" not in outcome.inspect().operations
    assert outcome.assumptions.summary == report.scope_note
