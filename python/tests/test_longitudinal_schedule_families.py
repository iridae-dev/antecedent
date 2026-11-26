"""Public schedule class comparison, honest learning and artifact replay."""

import antecedent as ant
import pytest
from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success
from antecedent.regimes import FixedScheduleContrasts, LearnedScheduleClass, LongitudinalRegime


SCHEDULES = ((False, False), (False, True), (True, False), (True, True))


def _study(means, specification, *, remove_matches=False):
    subjects = 800
    training = isinstance(specification, LearnedScheduleClass)
    query = LongitudinalRegime(
        outcome="outcome",
        method="learn_schedule" if training else "compare_schedules",
        schedule_family=specification,
        treatment_history=[
            SCHEDULES[2] if remove_matches and i % 4 == 3 else SCHEDULES[i % 4]
            for i in range(subjects)
        ],
        actions=SCHEDULES[0],
        treatment_probabilities=[[0.5, 0.5]] * subjects,
        subject_ids=[f"schedule-{i}" for i in range(subjects)],
        fold_ids=[i // 4 % 2 for i in range(subjects)],
    )
    data = {"outcome": [means[i % 4] + 0.125 * (i // 8 % 3 - 1) for i in range(subjects)]}
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
    _, smaller = _study((1.0, 2.0, 3.0, 10.0), FixedScheduleContrasts(
        (SCHEDULES[0], SCHEDULES[1], SCHEDULES[3]), SCHEDULES[0],
        (SCHEDULES[1], SCHEDULES[3]), draws=199,
    ))
    assert ant.prepare(data, query=smaller).plan_view().identity != prepared.plan_view().identity


