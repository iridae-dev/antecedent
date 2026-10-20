"""Python parity for the contract refusals bound on ``antecedent._native``.

The cases match ``python/src/contract_refusals_api.rs``: retention refusals, a
rewrite check that does not execute the rewrite, summary adaptation that does
not form a posterior, and feature-version gating.
"""

from __future__ import annotations

import pytest
from antecedent import _native
from antecedent.errors import CausalStateError, CausalUnsupportedError, CausalValueError


def _frame(population: str = "population") -> _native.PreservedIdentities:
    return _native.PreservedIdentities(
        "sample",
        "folds",
        population,
        "history",
        "dependence",
        "uncertainty",
    )


def test_require_refuses_what_the_mode_cannot_serve():
    rows = _native.RetainedOperation.row_reconstruction()
    with pytest.raises(
        CausalStateError, match="summary-only cannot serve row reconstruction"
    ) as caught:
        _native.RetentionMode.summary_only().require(rows)
    assert caught.value.mode == "summary-only"
    assert caught.value.operation == "row reconstruction"

    policy = _native.RetentionPolicy.sufficient_statistics()
    assert policy.result_mode().as_str() == "summary-only"
    policy.require(_native.RetainedOperation.reuse_summary())
    with pytest.raises(CausalStateError):
        policy.require(_native.RetainedOperation.refit_from_retained_rows())

    assert _native.RetentionPolicy.bounded_window(0).result_mode().as_str() == "summary-only"
    raw = _native.RetentionPolicy.raw_history()
    assert raw.result_mode().as_str() == "full reusable state"
    assert raw.max_retained_rows() is None

    _native.RetentionMode.disk_backed().require(_native.RetainedOperation.read_disk_artifact())
    _native.RetentionMode.default().require(_native.RetainedOperation.reuse_full_state())
    with pytest.raises(CausalStateError):
        _native.RetentionMode.default().require(_native.RetainedOperation.read_disk_artifact())


def test_rewrite_that_changes_the_frame_is_refused():
    frame = _frame()
    moved = _frame("other")
    cut = _native.FrameRewrite.column_filter_pushdown(frame, frame, "undeclared_population_cut")
    with pytest.raises(CausalValueError) as caught:
        cut.check()
    assert caught.value.axis == "population"
    assert caught.value.kind == "column_filter_pushdown"
    assert caught.value.reason_code == "invalid_argument"

    identity = _native.FrameRewrite.column_filter_pushdown(frame, moved, "declared")
    with pytest.raises(CausalValueError) as caught:
        identity.check()
    assert caught.value.axis == "population"
    _native.FrameRewrite.column_filter_pushdown(frame, frame, "declared").check()
    with pytest.raises(CausalValueError, match="unknown pushdown effect"):
        _native.FrameRewrite.column_filter_pushdown(frame, frame, "not_an_effect")

    mixes = _native.FrameRewrite.shared_row_design_preparation(
        frame, frame, frame, "mixes_populations"
    )
    with pytest.raises(CausalValueError) as caught:
        mixes.check()
    assert caught.value.axis == "population"
    _native.FrameRewrite.shared_row_design_preparation(frame, frame, frame, "shared").check()

    other_history = _native.FrameRewrite.nuisance_reuse(frame, frame, "fit_on_other_history")
    with pytest.raises(CausalValueError) as caught:
        other_history.check()
    assert caught.value.axis == "history"
    _native.FrameRewrite.nuisance_reuse(frame, frame, "certified").check()

    dropped = _native.FrameRewrite.sufficient_statistics(frame, frame, "drops_sampling_law")
    with pytest.raises(CausalValueError) as caught:
        dropped.check()
    assert caught.value.axis == "uncertainty"
    _native.FrameRewrite.sufficient_statistics(frame, frame, "sufficient").check()

    draws = _native.FrameRewrite.tiled_evaluation(frame, frame, "drops_failed_draws")
    with pytest.raises(CausalValueError) as caught:
        draws.check()
    assert caught.value.axis == "uncertainty"
    _native.FrameRewrite.tiled_evaluation(frame, frame, "faithful").check()


def test_interval_is_not_a_bayesian_law_and_covariance_needs_the_four_fields():
    interval = _native.FrequentistSummary.interval("risk_difference")
    with pytest.raises(
        CausalUnsupportedError, match="summary_observation.interval_stays_frequentist"
    ) as caught:
        _native.SummaryObservationModel.try_adapt(
            interval,
            "normal_approximation",
            "complete_cases",
            "cluster_dependence",
            "trial_estimator",
        )
    assert interval.kind() == "interval"
    assert caught.value.reason_code == "distribution_meaning_mismatch"
    with pytest.raises(CausalUnsupportedError, match="not a Bayesian law"):
        interval.as_bayesian_likelihood()

    with pytest.raises(CausalValueError):
        _native.FrequentistSummary.estimate_covariance(" ")
    summary = _native.FrequentistSummary.estimate_covariance("ate")
    with pytest.raises(CausalValueError):
        _native.SummaryObservationModel.try_adapt(
            summary,
            " ",
            "complete_cases",
            "cluster_dependence",
            "trial_estimator",
        )
    model = _native.SummaryObservationModel.try_adapt(
        summary,
        "normal_approximation",
        "complete_cases",
        "cluster_dependence",
        "trial_estimator",
    )
    assert model.summary_kind() == "estimate_covariance"
    assert model.sampling_approximation() == "normal_approximation"
    assert model.selection() == "complete_cases"
    assert model.dependence() == "cluster_dependence"
    assert model.origin() == "trial_estimator"
    admitted = model.validate_use("likelihood_update")
    assert admitted.summary_kind() == "estimate_covariance"
    with pytest.raises(CausalUnsupportedError, match="summary_observation.not_bayesian_law") as law:
        model.validate_use("bayesian_law")
    assert law.value.reason_code == "distribution_meaning_mismatch"
    assert model.summary_kind() == "estimate_covariance"


def test_container_0_5_is_compatible_and_unknown_or_semver_is_refused():
    accepted = [_native.FeatureVersion("family_payload", 1)]
    read = _native.FeatureRead.classify(0, 5, "family_payload", 1, accepted)
    assert read.kind() == "compatible"
    assert read.name() == "family_payload"
    assert read.version() == 1
    with pytest.raises(
        CausalUnsupportedError, match="unsupported feature `family_payload` version 2"
    ):
        _native.FeatureRead.classify(0, 5, "family_payload", 2, accepted)
    with pytest.raises(
        CausalUnsupportedError, match="package version 1.1.4 is not a feature version"
    ):
        _native.FeatureRead.classify_package(0, 5, 1, 1, 4, accepted)
