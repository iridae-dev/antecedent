"""Rust-compiled joint treatment-selection and MNAR sensitivity through Python."""

from __future__ import annotations

import itertools
from dataclasses import replace
from random import Random

import pytest
from antecedent import Execution, load
from antecedent import _reason_details as reason_details
from antecedent import composition_bundle as cb
from antecedent.longitudinal_joint_sensitivity import (
    LongitudinalSensitivityRefusal,
    LongitudinalSensitivitySpec,
    LongitudinalSubject,
    compile_longitudinal_joint_sensitivity,
    consume_longitudinal_joint_sensitivity,
)
from antecedent.outcomes import require_success


def _rows() -> list[LongitudinalSubject]:
    rows = []
    for w, a0, l1, a1 in itertools.product((0, 1), repeat=4):
        for observed in range(40):
            rows.append(
                LongitudinalSubject(
                    len(rows),
                    w,
                    (a0, a1),
                    l1,
                    0.30 + 0.20 * a0 + 0.04 * l1 + 0.02 * w if observed % 4 else None,
                )
            )
    return rows


def _spec() -> LongitudinalSensitivitySpec:
    return LongitudinalSensitivitySpec(
        "independent_baseline_population",
        "reaction",
        "proportion",
        ((0.0, 0.055), (0.0, 0.055)),
        (0.0, 0.10),
        outcome_support=(0.0, 1.0),
        level=0.9,
        decision_threshold=0.075,
    )


def test_typed_joint_sensitivity_executes_and_replays_with_bound_decision() -> None:
    plan = require_success(compile_longitudinal_joint_sensitivity(_spec(), _rows()))
    assert plan.plan_view()["route"] == "LongitudinalJointSensitivity"
    fit = require_success(plan.execute(execution=Execution(seed=33)))
    assert fit.calibration_status == "not_measured"
    response = fit.report
    assert response.reference_contrast == pytest.approx(0.2)
    for corner in response.corners:
        assert corner.contrast == pytest.approx(
            0.2 - sum(corner.assignment) - 0.5 * corner.missingness
        )
    assert response.assumption_range == pytest.approx((0.04, 0.20))
    assert response.decision is not None and response.decision.assumption_choice == "undetermined"
    assert response.decision.sampling_choice == "undetermined"
    assert response.sampling_radius > 0
    replay = require_success(
        consume_longitudinal_joint_sensitivity(
            fit.export(), expected=fit.identity, expected_program=plan.identity
        )
    )
    assert replay.acceptance.replayed
    assert replay.report == fit.report
    exported = fit.export()
    generic = require_success(load(exported, expected=exported.expectation))
    assert generic.report == response
    bundle = cb.Bundle.builder().add_artifact(exported, node_id="sensitivity").build()
    verified = require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )
    assert verified.all_replayed
    assert verified.value("sensitivity", "assumption_lower") == pytest.approx(0.04)
    assert verified.value("sensitivity", "sampling_lower") == response.extrema_intervals[0][0]
    assert verified.value("sensitivity", "sampling_upper") == response.extrema_intervals[1][1]
    with pytest.raises(LongitudinalSensitivityRefusal) as wrong:
        consume_longitudinal_joint_sensitivity(
            fit.export(), expected="a" * 64, expected_program=plan.identity
        )
    assert wrong.value.reason_code == "invalid_argument"
    assert wrong.value.detail == reason_details.longitudinal_joint_sensitivity.IDENTITY

    incoherent = require_success(
        compile_longitudinal_joint_sensitivity(replace(_spec(), mnar_limits=(0, 2.0)), _rows())
    )
    with pytest.raises(LongitudinalSensitivityRefusal) as impossible:
        incoherent.execute()
    assert impossible.value.reason_code == "invalid_argument"
    assert impossible.value.detail == reason_details.longitudinal_joint_sensitivity.INCOHERENT_BOX


def test_declared_bernoulli_box_has_separate_surface_and_decision_keys() -> None:
    rng = Random(5)
    rows = []
    for subject in range(1600):
        w, a0, l1, a1 = (rng.randrange(2) for _ in range(4))
        probability = 0.30 + 0.20 * a0 + 0.04 * l1 + 0.02 * w
        outcome = float(rng.random() < probability) if rng.random() < 0.75 else None
        rows.append(LongitudinalSubject(subject, w, (a0, a1), l1, outcome))
    spec = LongitudinalSensitivitySpec(
        "balanced-independent-subject-population",
        "binary-outcome",
        "proportion",
        ((0.0, 0.055), (0.0, 0.055)),
        (0.0, 0.10),
        outcome_support=(0.0, 1.0),
        replicates=199,
        level=0.95,
        decision_threshold=0.075,
        coverage_design="balanced_binary_bernoulli_v1",
    )
    fit = require_success(
        require_success(compile_longitudinal_joint_sensitivity(spec, rows)).execute()
    )
    surface = fit.calibration
    [extrema] = surface["secondary"]
    for slot, functional in (
        (surface, "balanced_binary_bernoulli_v1:full_box_simultaneous_band"),
        (extrema, "balanced_binary_bernoulli_v1:joint_decision_extrema"),
    ):
        basis = slot["basis"]
        assert basis["key"]["functional"] == functional
        assert basis["key"]["interval_method"] == "simultaneous_band"
        assert basis["key"]["dependence"] == "shared_subject_refits"
        assert basis["scope"]["row_count"] == 1600
        assert basis["scope"]["replicates_ok"] == 199
    artifact = fit.export()
    replayed = require_success(load(artifact, expected=artifact.expectation))
    assert replayed.calibration == surface

    with pytest.raises(LongitudinalSensitivityRefusal) as refused:
        compile_longitudinal_joint_sensitivity(replace(spec, decision_threshold=0.076), rows)
    assert refused.value.detail == reason_details.longitudinal_joint_sensitivity.INVALID_DECLARATION
