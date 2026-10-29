"""analyze() returns Pydantic result models; to_dict() stays JSON-safe."""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent import AverageEffect, Frequentist, ResponseCurve, analyze, prepare
from antecedent.outcomes import require_success
from antecedent.results import AnalysisResult, CausalResponseView, InspectionReport
from pydantic import BaseModel


def _data():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 80
    t = rng.integers(0, 2, n)
    y = 1.0 + 0.5 * t + rng.normal(0, 0.1, n)
    return pd.DataFrame({"t": t, "y": y, "x": rng.normal(0, 1, n)})


def _adjusted_slope(data) -> float:
    """Coefficient of t in the least-squares fit y ~ 1 + t + x (backdoor adjustment on x)."""
    design = np.column_stack([np.ones(len(data)), data["t"], data["x"]])
    return float(np.linalg.lstsq(design, data["y"].to_numpy(), rcond=None)[0][1])


def test_analyze_result_is_pydantic_and_to_dict_is_json_safe() -> None:
    result = require_success(
        analyze(
            _data(),
            graph=[("t", "y"), ("x", "y"), ("x", "t")],
            query=AverageEffect("t", "y"),
            inference=Frequentist(bootstrap=0),
            refute="none",
        )
    )
    assert isinstance(result, AnalysisResult)
    assert isinstance(result, BaseModel)
    assert result.estimate.ate == pytest.approx(_adjusted_slope(_data()), abs=1e-9)
    dumped = result.to_dict()
    json.dumps(dumped, allow_nan=False)
    assert dumped["identification"]["status"] == result.identification.status
    assert dumped["estimate"]["ate"] == result.estimate.ate
    assert "_prepared" not in dumped
    assert "_execution" not in dumped
    assert result.study is not None
    report = result.inspect()
    assert isinstance(report, InspectionReport)
    json.dumps(report.to_dict(), allow_nan=False)


def test_prepared_estimate_is_the_same_pydantic_result() -> None:
    prepared = prepare(
        _data(),
        graph=[("t", "y"), ("x", "y"), ("x", "t")],
        query=AverageEffect("t", "y"),
        inference=Frequentist(bootstrap=0),
        refute="none",
    )
    result = require_success(prepared.estimate())
    assert isinstance(result, AnalysisResult)
    assert result.to_dict()["program_id"] == result.program_id
    assert result.estimate.ate == pytest.approx(_adjusted_slope(_data()), abs=1e-9)


def test_response_analyze_is_pydantic() -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(17)
    confounder = rng.normal(size=200)
    treatment = 0.7 * confounder + rng.normal(size=200)
    outcome = 2.0 * treatment + confounder + rng.normal(scale=0.2, size=200)
    data = pd.DataFrame({"t": treatment, "y": outcome, "x": confounder})
    result = require_success(
        analyze(
            data,
            graph=[("t", "y"), ("x", "y"), ("x", "t")],
            query=ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5]),
            inference=Frequentist(bootstrap=0),
            refute="none",
        )
    )
    assert isinstance(result, CausalResponseView)
    assert isinstance(result, BaseModel)
    dumped = result.to_dict()
    # The curve tracks the least-squares fit y ~ 1 + t + x with x averaged over the sample to
    # about twice the observed 2.4e-2; the unadjusted slope (about 2.5) misses by 0.2 at the end.
    design = np.column_stack([np.ones(200), treatment, confounder])
    intercept, slope, effect_x = np.linalg.lstsq(design, outcome, rcond=None)[0]
    grid = np.array([-0.5, 0.0, 0.5])
    want = intercept + slope * grid + effect_x * confounder.mean()
    assert np.asarray(dumped["response"]["points"]).ravel().tolist() == grid.tolist()
    assert np.asarray(dumped["response"]["values"]).ravel() == pytest.approx(want, abs=0.05)
    json.dumps(dumped, allow_nan=False)
    assert dumped["identification"]["status"] == result.identification.status
    assert "_prepared" not in dumped
    assert result.study is not None
