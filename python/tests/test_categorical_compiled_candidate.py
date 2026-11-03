"""Compiler-owned categorical candidate: bounded shapes, replay, and licensing by record."""

import pytest
from antecedent.categorical_treatment import PreparedCategoricalPosteriorCandidate
from antecedent.errors import CausalUnsupportedError, CausalValueError


def _request(counts=(20, 20)):
    declared = [f"level-{i}" for i in range(len(counts))]
    levels = [declared[i] for i, count in enumerate(counts) for _ in range(count)]
    rows = len(levels)
    return {
        "outcome": [float(i) for i in range(rows)],
        "levels": levels,
        "declared_levels": declared,
        "subject_ids": [f"subject-{i}" for i in range(rows)],
        "snapshot": "source-a",
        "sampling_design": "independent_subject_trajectories",
    }


def test_compiled_candidate_replays_source_and_refuses_without_a_covering_record():
    prepared = PreparedCategoricalPosteriorCandidate(**_request())
    inspection = prepared.inspect()
    assert inspection["route"] == "CategoricalPosteriorCandidate"
    assert inspection["licensed"] is False
    # Licensing is a record lookup at measurement: with no covering record the route
    # refuses, and once one exists the same call returns a calibrated result.
    for call in (prepared.interval, prepared.measured_interval):
        try:
            call()
        except CausalUnsupportedError as refused:
            assert refused.reason_code == "cell_not_licensed"
    artifact = prepared.export_source()
    del prepared
    replayed = PreparedCategoricalPosteriorCandidate.consume_source(
        artifact.source, expected=artifact.expectation
    )
    assert replayed.inspect() == inspection
    changed = bytearray(artifact.source)
    changed[20] ^= 1
    with pytest.raises(ValueError, match="categorical_posterior"):
        PreparedCategoricalPosteriorCandidate.consume_source(
            bytes(changed), expected=artifact.expectation
        )
    with pytest.raises(CausalValueError, match="invalid candidate source or expectation size"):
        PreparedCategoricalPosteriorCandidate.consume_source(artifact.source, expected=b"bad")


@pytest.mark.parametrize("counts", [(15, 15), (10, 10, 10), (2, 2, 26), (4,) * 8, (3, 2, 5, 40)])
def test_general_shapes_compile_replay_and_differ_by_design(counts):
    prepared = PreparedCategoricalPosteriorCandidate(**_request(counts))
    artifact = prepared.export_source()
    replayed = PreparedCategoricalPosteriorCandidate.consume_source(
        artifact.source, expected=artifact.expectation
    )
    assert replayed.inspect() == prepared.inspect()


def test_compiled_candidate_requires_distinct_subjects_and_explicit_design():
    request = _request()
    request["subject_ids"][1] = request["subject_ids"][0]
    with pytest.raises(ValueError, match="subject identities"):
        PreparedCategoricalPosteriorCandidate(**request)
    request = _request()
    request["sampling_design"] = "dependent_time_series"
    with pytest.raises(CausalValueError, match="declare independent_subject_trajectories"):
        PreparedCategoricalPosteriorCandidate(**request)


@pytest.mark.parametrize(
    "counts",
    [(14, 15), (29, 1), (30,), (4,) * 9, (4_095, 2)],
)
def test_compiled_candidate_refuses_out_of_bound_shapes(counts):
    with pytest.raises(ValueError, match="categorical_posterior"):
        PreparedCategoricalPosteriorCandidate(**_request(counts))


def test_compiled_candidate_refuses_nonfinite_outcomes_and_undeclared_levels():
    request = _request()
    request["outcome"][3] = float("nan")
    with pytest.raises(ValueError, match="categorical_posterior"):
        PreparedCategoricalPosteriorCandidate(**request)
    request = _request()
    request["levels"][0] = "other"
    with pytest.raises(ValueError, match="categorical_posterior"):
        PreparedCategoricalPosteriorCandidate(**request)
