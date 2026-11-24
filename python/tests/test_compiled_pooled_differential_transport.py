"""The pooled-differential measurement transport is one computation, direct or compiled.

``measurement_models.transport_pooled_differential`` warns that the compiled
``PooledDifferentialTransport`` route replaces it and returns the correction the compiled
program returns under the same execution seed, bit for bit; the compiled route recovers the
planted target prevalence.
"""

from __future__ import annotations

import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalError
from antecedent.execution import Execution
from antecedent.measurement_models import (
    PopulationValidationCounts,
    compile_pooled_differential_transport,
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


SOURCES = [
    _counts("a", 0.75, 0.94, 2000),
    _counts("b", 0.85, 0.9, 2000),
    _counts("c", 0.93, 0.84, 2000),
]


def test_the_direct_call_warns_and_names_the_compiled_route() -> None:
    with pytest.warns(AntecedentDeprecationWarning, match="transport_pooled_differential") as w:
        transport_pooled_differential(SOURCES, "target", _recorded(), 4000)
    assert "compile_pooled_differential_transport" in str(w[0].message)


@pytest.mark.parametrize("seed", [1, 7])
def test_the_compiled_route_returns_the_direct_correction_bit_for_bit(seed: int) -> None:
    execution = Execution(seed=seed)
    target = _counts("target", 0.85, 0.9, 4000)
    with pytest.warns(AntecedentDeprecationWarning):
        direct = transport_pooled_differential(
            SOURCES, "target", _recorded(), 4000, target_validation=target, execution=execution
        )
    compiled = compile_pooled_differential_transport(
        SOURCES, "target", _recorded(), 4000, target_validation=target
    )
    assert compiled.route == "PooledDifferentialTransport"
    assert len(compiled.program_identity) == 64
    replacement = compiled.execute(execution=execution)
    assert replacement == direct
    assert replacement.prevalence.hex() == direct.prevalence.hex()
    assert replacement.calibration == "not_measured"


def test_the_compiled_route_recovers_the_planted_prevalence() -> None:
    invariant = [_counts(name, 0.85, 0.9, 2000) for name in ("a", "b", "c")]
    result = compile_pooled_differential_transport(invariant, "target", _recorded(), 4000).execute()
    assert abs(result.prevalence - 0.3) < 0.03
    assert result.interval[0] <= result.prevalence <= result.interval[1]


def test_the_compiled_route_refuses_a_single_source() -> None:
    compiled = compile_pooled_differential_transport(SOURCES[:1], "target", _recorded(), 4000)
    with pytest.raises(CausalError, match="Next attempt"):
        compiled.execute()
