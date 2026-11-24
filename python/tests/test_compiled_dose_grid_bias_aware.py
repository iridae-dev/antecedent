"""A bias-aware dose grid runs through its compiled program with the direct row's numbers.

``dose_functional(..., curvature_bound=...)`` binds the request into the ``DoseGridBiasAware``
program family and executes it under the band's seed. The direct row (the native entry that
seals the artifact without the compiler) is reached here by swapping the native the module calls;
both give the same report, the same artifact bytes and the same refusals, and only the compiled
row names its program: the route, the lowered program identity and ``dose_grid_bias_aware``, the
estimator key the measured coverage of the intervals and band is recorded under.

The oracle of the known-truth check is the quadratic response ``m(d) = 1 + 0.5 d + 0.25 d**2``
(``m''' = 0``), so every bias-aware interval and the band contain it.
"""

from __future__ import annotations

import re
import struct

import numpy as np
import pytest
from antecedent import dose_grid
from antecedent.dose_grid import (
    DoseClaims,
    DoseGridProgram,
    consume_dose_grid_artifact,
    dose_functional,
)

N = 201
GRID = [0.5, 1.0, 2.0, 3.5]


def truth_level(d):
    return 1.0 + 0.5 * d + 0.25 * d * d


def rows():
    dose = np.linspace(0.0, 4.0, N)
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    return dose, truth_level(dose) + 0.1 * sign


CASES = {
    "level_band": {
        "functional": "level",
        "grid": GRID,
        "claims": DoseClaims(pointwise_level=True, simultaneous_band=True),
        "curvature_bound": 0.2,
        "band_replicates": 400,
        "band_seed": 7,
    },
    "derivative_band": {
        "functional": "derivative",
        "grid": GRID,
        "claims": DoseClaims(pointwise_level=False, derivative=True, simultaneous_band=True),
        "curvature_bound": 0.3,
        "band_replicates": 300,
        "band_seed": 11,
    },
    "contrast": {
        "functional": "contrast",
        "contrast": (1.0, 3.0),
        "curvature_bound": 0.2,
    },
    "declared_rows": {
        "functional": "level",
        "grid": GRID,
        "claims": DoseClaims(pointwise_level=True, simultaneous_band=True),
        "curvature_bound": 0.2,
        "band_replicates": 300,
        "band_seed": 3,
        "sampling_weights": 1.0 + (np.arange(N) % 3) * 0.5,
        "cluster": np.arange(N) // 4,
    },
}


def run(**kwargs):
    dose, outcome = rows()
    kwargs.setdefault("bandwidth", 0.4)
    kwargs.setdefault("bandwidth_range", (0.1, 1.0))
    return dose_functional(dose, outcome, **kwargs)


def direct(monkeypatch, **kwargs):
    """The same call on the direct row: the native entry that seals without the compiler."""
    with monkeypatch.context() as patch:
        patch.setattr(dose_grid, "_evaluate_bias_aware", dose_grid._evaluate)
        return run(**kwargs)


def bits(values):
    return [struct.pack("<d", float(value)) for value in values]


def assert_same_numbers(compiled, reference):
    assert compiled.to_dict() == reference.to_dict()
    assert compiled == reference
    assert bytes(compiled.artifact) == bytes(reference.artifact)
    assert compiled.premises_digest == reference.premises_digest
    assert compiled.data_digest == reference.data_digest
    for mine, theirs in zip(
        compiled.levels + compiled.derivatives,
        reference.levels + reference.derivatives,
        strict=True,
    ):
        assert bits([mine.value, mine.standard_error, mine.lower, mine.upper]) == bits(
            [theirs.value, theirs.standard_error, theirs.lower, theirs.upper]
        )
    section, other = compiled.bias_aware, reference.bias_aware
    assert section is not None and other is not None
    assert bits(section.smoothing_bias_bound) == bits(other.smoothing_bias_bound)
    assert bits(section.lower) == bits(other.lower)
    assert bits(section.upper) == bits(other.upper)
    assert (section.band_lower is None) == (other.band_lower is None)
    if section.band_lower is not None:
        assert bits(section.band_lower) == bits(other.band_lower)
        assert bits(section.band_upper) == bits(other.band_upper)
        assert bits([section.band_critical_value]) == bits([other.band_critical_value])


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_bias_aware_row_runs_through_its_compiled_program_with_the_direct_numbers(
    case, monkeypatch
):
    compiled = run(**CASES[case])
    reference = direct(monkeypatch, **CASES[case])
    assert_same_numbers(compiled, reference)
    program = compiled.program
    assert isinstance(program, DoseGridProgram)
    assert program.route == "DoseGridBiasAware"
    assert program.calibration_estimator == "dose_grid_bias_aware"
    assert re.fullmatch(r"[0-9a-f]{64}", program.lowered_program_identity)
    assert reference.program is None


def test_the_compiled_intervals_and_band_contain_the_quadratic_truth():
    result = run(**CASES["level_band"])
    section = result.bias_aware
    assert section is not None and section.band_lower is not None
    for j, point in enumerate(result.levels):
        truth = truth_level(point.dose)
        assert section.lower[j] <= truth <= section.upper[j]
        assert section.band_lower[j] <= truth <= section.band_upper[j]


def test_the_compiled_artifact_replays_and_the_replay_names_no_program():
    result = run(**CASES["level_band"])
    fresh = consume_dose_grid_artifact(bytes(result.export()))
    assert fresh.bias_aware == result.bias_aware
    assert fresh == result
    assert fresh.program is None


def test_the_program_identity_binds_the_declared_bound_and_the_rows():
    first = run(**CASES["level_band"]).program
    again = run(**CASES["level_band"]).program
    wider = run(**{**CASES["level_band"], "curvature_bound": 0.25}).program
    assert first is not None and again is not None and wider is not None
    assert first.lowered_program_identity == again.lowered_program_identity
    assert first.lowered_program_identity != wider.lowered_program_identity
    dose, outcome = rows()
    moved = dose_functional(
        dose,
        outcome + np.where(np.arange(N) == 7, 1e-9, 0.0),
        bandwidth=0.4,
        bandwidth_range=(0.1, 1.0),
        **CASES["level_band"],
    ).program
    assert moved is not None
    assert moved.lowered_program_identity != first.lowered_program_identity


def outcome_of(call):
    try:
        call()
    except Exception as error:  # noqa: BLE001 - the refusal itself is compared
        return type(error), str(error)
    return None


@pytest.mark.parametrize(
    "change",
    [
        {"curvature_bound": -1.0},
        {"bandwidth": 2.0},
        {"grid": [0.5, 9.0]},
        {"band_replicates": 1},
    ],
)
def test_refusals_are_the_direct_rows_refusals(change, monkeypatch):
    kwargs = {**CASES["level_band"], **change}
    compiled = outcome_of(lambda: run(**kwargs))
    reference = outcome_of(lambda: direct(monkeypatch, **kwargs))
    assert compiled is not None
    assert compiled == reference


def test_rows_without_a_bound_do_not_compile():
    result = run(functional="level", grid=GRID)
    assert result.bias_aware is None
    assert result.program is None
