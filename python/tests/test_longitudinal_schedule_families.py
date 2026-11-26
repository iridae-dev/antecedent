"""Public schedule class comparison, honest learning and artifact replay."""

from dataclasses import replace
from random import Random

import antecedent as ant
import pytest
from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success
from antecedent.regimes import FixedScheduleContrasts, LearnedScheduleClass, LongitudinalRegime


SCHEDULES = ((False, False), (False, True), (True, False), (True, True))


def _study(means, specification, *, remove_matches=False):
    subjects = 800
    training = isinstance(specification, LearnedScheduleClass)
    original_rows = [
        i // 8 * 8 + (i % 2) * 4 + i // 2 % 4 if training else i
        for i in range(subjects)
    ]
    query = LongitudinalRegime(
        outcome="outcome",
        method="learn_schedule" if training else "compare_schedules",
        schedule_family=specification,
        treatment_history=[
            SCHEDULES[2] if remove_matches and i % 4 == 3 else SCHEDULES[i % 4]
            for i in original_rows
        ],
        actions=SCHEDULES[0],
        treatment_probabilities=[[0.5, 0.5]] * subjects,
        subject_ids=[f"schedule-{i}" for i in range(subjects)],
        fold_ids=[i % 2 if training else i // 4 % 2 for i in range(subjects)],
    )
    data = {"outcome": [means[i % 4] + 0.125 * (i // 8 % 3 - 1) for i in original_rows]}
    return data, query


def test_paired_longitudinal_comparisons_replay_and_bind_the_class():
    data, query = _study((1.0, 2.0, 3.0, 10.0), FixedScheduleContrasts(
        SCHEDULES, SCHEDULES[0], (SCHEDULES[1], SCHEDULES[3]), draws=199,
    ))
    prepared = ant.prepare(data, query=query)
    result = require_success(prepared.estimate())
    family = result.longitudinal_regime.schedule_family
    assert [contrast.difference for contrast in family.contrasts] == pytest.approx([1.0, 9.0])
    for contrast in family.contrasts:
        span = family.critical_value * contrast.standard_error
        assert contrast.interval == pytest.approx((contrast.difference - span, contrast.difference + span))
    sourced = bytes(result.export(source_data=data))
    artifact = require_success(ant.load(sourced)).replay()
    assert artifact.acceptance.replayed
    bundle = cb.BundleBuilder().add_artifact(sourced, node_id="regimes").build()
    consumed = require_success(cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
    assert consumed.value("regimes", "schedule.3.versus.0.difference") == pytest.approx(9.0)
    assert result.calibration.functional is None
    assert result.calibration.status == "not_measured"

    full_data, full_query = _study((1.0, 2.0, 3.0, 10.0), FixedScheduleContrasts(
        SCHEDULES, SCHEDULES[0], SCHEDULES[1:], draws=199,
    ))
    measured = require_success(ant.analyze(full_data, query=full_query))
    assert measured.calibration.functional is None
    assert measured.calibration.status == "not_measured"
    assert not measured.calibration.secondary

    revised_data = {
        "outcome": [y + 4.0 * (i % 4 == 3) for i, y in enumerate(data["outcome"])]
    }
    revised = require_success(ant.analyze(revised_data, query=query))
    revised_bundle = cb.BundleBuilder().add_artifact(
        bytes(revised.export(source_data=revised_data)), node_id="regimes",
    ).build()
    revised_bytes = bytes(revised_bundle.export())
    with pytest.raises(cb.ExpectedIdentityMismatchRefusal):
        require_success(cb.consume_bundle(revised_bytes, expected_identity=bundle.identity))
    verified = require_success(cb.consume_bundle(revised_bytes, expected_identity=revised_bundle.identity))
    assert verified.value("regimes", "schedule.3.versus.0.difference") == pytest.approx(13.0)
    _, smaller = _study((1.0, 2.0, 3.0, 10.0), FixedScheduleContrasts(
        (SCHEDULES[0], SCHEDULES[1], SCHEDULES[3]), SCHEDULES[0],
        (SCHEDULES[1], SCHEDULES[3]), draws=199,
    ))
    assert ant.prepare(data, query=smaller).plan_view().identity != prepared.plan_view().identity


@pytest.mark.parametrize("means,selected,optima", [
    ((1.0, 2.0, 3.0, 10.0), 3, (3,)),
    ((1.0, 2.0, 10.0, 10.0), 2, (2, 3)),
])
def test_learned_longitudinal_schedule_keeps_unique_and_tied_optima(means, selected, optima):
    declaration = LearnedScheduleClass(SCHEDULES, training_fold=0, draws=199)
    data, query = _study(means, declaration)
    result = require_success(ant.analyze(data, query=query))
    fit = result.longitudinal_regime
    selected_fit = fit.schedule_family.learning
    assert selected_fit.selected_schedule == selected
    assert selected_fit.optimum_set == optima
    assert sum(count for _, count in selected_fit.refit_selections) == 199
    assert fit.value_interval_95[0] <= means[selected] <= fit.value_interval_95[1]
    assert result.calibration.functional is None
    assert result.calibration.status == "not_measured"
    assert not result.calibration.secondary

    source = bytes(result.export(source_data=data))
    artifact = require_success(ant.load(source)).replay()
    assert artifact.acceptance.replayed
    replayed_learning = artifact.decoded_artifact.payload["longitudinal_regime"]["schedule_family"]["learning"]
    assert replayed_learning["optimum_set"] == optima
    bundle = cb.BundleBuilder().add_artifact(source, node_id="learned").build()
    verified = require_success(cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
    assert verified.node("learned").facts["optimum_set.calibration"] == "not_measured"
    assert "optimum_set.calibration_functional" not in verified.node("learned").facts
    assert verified.node("learned").facts["optimum_set"] == ",".join(map(str, optima))

    changed_data = {
        "outcome": [y + 20.0 * (i % 2 == 0 and i // 2 % 4 == 0) for i, y in enumerate(data["outcome"])]
    }
    changed = require_success(ant.analyze(changed_data, query=query))
    changed_result = changed.longitudinal_regime.schedule_family.learning
    assert changed_result.selected_schedule == 0
    assert changed_result.optimum_set == optima
    replacement = cb.BundleBuilder().add_artifact(
        bytes(changed.export(source_data=changed_data)), node_id="learned",
    ).build()
    new_bytes = bytes(replacement.export())
    with pytest.raises(cb.ExpectedIdentityMismatchRefusal):
        require_success(cb.consume_bundle(new_bytes, expected_identity=bundle.identity))
    replayed = require_success(cb.consume_bundle(new_bytes, expected_identity=replacement.identity))
    assert replayed.value("learned", "selected_schedule") == 0.0
    assert replayed.node("learned").facts["optimum_set"] == ",".join(map(str, optima))

@pytest.mark.parametrize(
    "design,means,suffix",
    [
        ("learned_schedule_unique_optimum", (1.0, 2.0, 3.0, 5.0), "unique_gaussian_v1"),
        ("learned_schedule_tied_optima", (1.0, 2.0, 5.0, 5.0), "tied_gaussian_v1"),
    ],
)
def test_learned_schedule_empirical_receipts_bind_the_declared_law(design, means, suffix):
    _, query = _study(means, LearnedScheduleClass(SCHEDULES, training_fold=0, draws=199))
    rng = Random(913)
    codes = [rng.randrange(4) for _ in query.subject_ids]
    query = replace(query, treatment_history=[SCHEDULES[code] for code in codes])
    data = {"outcome": [rng.gauss(means[code], 1.0) for code in codes]}
    nominal = require_success(ant.analyze(data, query=query))
    scoped = require_success(ant.analyze(data, query=replace(query, calibration_design=design)))
    assert nominal.calibration.functional is None
    assert scoped.calibration.functional == f"learned_h2_four_schedules_selected_value_{suffix}"
    assert scoped.calibration.secondary[0].functional == (
        f"learned_h2_four_schedules_all_true_optima_{suffix}"
    )
    assert scoped.program_id != nominal.program_id
    archived = require_success(ant.load(bytes(scoped.export(source_data=data)))).replay()
    assert archived.acceptance.replayed
    assert archived.calibration.functional == scoped.calibration.functional
