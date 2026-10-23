"""Pooled differential measurement error carried to a target population, through Python.

Three validated source populations with heterogeneous error rates widen the interval of the
target's corrected prevalence relative to invariant sources; a target validation sample
narrows it; a single source is refused.
"""

from __future__ import annotations

import pytest

from antecedent.errors import CausalError
from antecedent.measurement_models import (
    PopulationValidationCounts,
    transport_pooled_differential,
)


def _counts(population: str, se: float, sp: float, size: int) -> PopulationValidationCounts:
    positives = size // 2
    negatives = size - positives
    tp = round(se * positives)
    tn = round(sp * negatives)
    return PopulationValidationCounts(population, tp, positives - tp, tn, negatives - tn)


def _recorded() -> int:
    return round((0.3 * 0.85 + 0.7 * 0.1) * 4000)


def test_heterogeneity_widens_and_a_target_sample_narrows_the_interval() -> None:
    invariant = [_counts(name, 0.85, 0.9, 2000) for name in ("a", "b", "c")]
    heterogeneous = [
        _counts("a", 0.75, 0.94, 2000),
        _counts("b", 0.85, 0.9, 2000),
        _counts("c", 0.93, 0.84, 2000),
    ]
    tight = transport_pooled_differential(invariant, "target", _recorded(), 4000)
    wide = transport_pooled_differential(heterogeneous, "target", _recorded(), 4000)
    assert tight.calibration == "unmeasured"
    assert abs(tight.prevalence - 0.3) < 0.03
    assert (wide.interval[1] - wide.interval[0]) > (tight.interval[1] - tight.interval[0])
    informed = transport_pooled_differential(
        heterogeneous,
        "target",
        _recorded(),
        4000,
        target_validation=_counts("target", 0.85, 0.9, 4000),
    )
    assert (informed.interval[1] - informed.interval[0]) < (wide.interval[1] - wide.interval[0])
    with pytest.raises(CausalError, match="Next attempt"):
        transport_pooled_differential(invariant[:1], "target", _recorded(), 4000)
