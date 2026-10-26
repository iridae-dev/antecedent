"""Tier overlap and E-value diagnostics through the Python surface.

The diagnostics are a view over an executed tiered result: nothing is refit, the interval E-value
is never reported, and a result that is not a licensed single-treatment tiered average-effect
result is refused with its registered code.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import tier_diagnostics as td
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.graph import TieredBackground, WithinTier
from antecedent.outcomes import require_success


def _siblings(n: int = 800, seed: int = 214):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.3 + 0.9 * z + 0.5 * u)))).astype(float)
    y = (1.0 + z) * t + 0.4 * z + 0.3 * u + 0.35 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z, "u": u}


def _analyze(graph, data):
    return require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect("t", "y"),
            refute="none",
            inference=antecedent.Frequentist(bootstrap=0),
        )
    )


def test_a_codetermined_result_reports_its_closure_overlap_and_point_evalue() -> None:
    data = _siblings()
    background = TieredBackground(
        tiers=[["z", "u"], ["t"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    report = td.tier_diagnostics(_analyze(background, data))
    assert report.design.kind == "co_determined_closure"
    assert set(report.design.adjustment) == {"z", "u"}
    assert report.design.propensity_scored is True
    assert report.overlap.available is True
    assert 0.0 <= report.overlap.value["propensity_min"] <= report.overlap.value["propensity_max"]
    assert report.evalue.point.available is True
    assert report.evalue.point.value["value"] >= 1.0
    # The interval E-value is a typed unavailable field on every result.
    assert report.evalue.interval.available is False
    assert report.evalue.interval.code == "cell_not_licensed"
    assert report.evalue.interval.detail == "tier_diagnostics.evalue_interval_withheld"


def test_the_prepared_study_and_its_result_give_the_same_diagnostics() -> None:
    data = _siblings()
    background = TieredBackground(
        tiers=[["z", "u"], ["t"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    result = _analyze(background, data)
    assert td.tier_diagnostics(result).to_dict() == td.tier_diagnostics(result.study).to_dict()


def test_a_non_tiered_result_is_refused_and_nothing_is_recomputed() -> None:
    data = _siblings()
    graph = antecedent.Dag.from_edges(
        ["t", "y", "z", "u"], [("z", "t"), ("u", "t"), ("z", "y"), ("u", "y"), ("t", "y")]
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        td.tier_diagnostics(_analyze(graph, data))
    assert raised.value.reason_code == "invalid_argument"
    assert "tier_diagnostics.not_a_tiered_average_result" in str(raised.value)


def test_an_object_without_a_study_is_a_type_error() -> None:
    with pytest.raises(CausalTypeError):
        td.tier_diagnostics(object())
