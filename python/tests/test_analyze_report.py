"""analyze() returns Pydantic result models; to_dict() stays JSON-safe."""

from __future__ import annotations

import json

from antecedent import AverageEffect, Frequentist, ResponseCurve, analyze, prepare
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


def test_analyze_result_is_pydantic_and_to_dict_is_json_safe() -> None:
    result = analyze(
        _data(),
        graph=[("t", "y"), ("x", "y"), ("x", "t")],
        query=AverageEffect("t", "y"),
        inference=Frequentist(),
        refute="none",
        bootstrap=0,
    )
    assert isinstance(result, AnalysisResult)
    assert isinstance(result, BaseModel)
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
        inference=Frequentist(),
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate()
    assert isinstance(result, AnalysisResult)
    assert result.to_dict()["program_id"] == result.program_id


def test_response_analyze_is_pydantic() -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(17)
    confounder = rng.normal(size=200)
    treatment = 0.7 * confounder + rng.normal(size=200)
    outcome = 2.0 * treatment + confounder + rng.normal(scale=0.2, size=200)
    data = pd.DataFrame({"t": treatment, "y": outcome, "x": confounder})
    result = analyze(
        data,
        graph=[("t", "y"), ("x", "y"), ("x", "t")],
        query=ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5]),
        inference=Frequentist(),
        refute="none",
        bootstrap=0,
    )
    assert isinstance(result, CausalResponseView)
    assert isinstance(result, BaseModel)
    dumped = result.to_dict()
    json.dumps(dumped, allow_nan=False)
    assert dumped["identification"]["status"] == result.identification.status
    assert "_prepared" not in dumped
    assert result.study is not None
