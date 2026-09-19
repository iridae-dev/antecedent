"""inspect() returns a Pydantic report; to_dict() stays JSON-safe."""

from __future__ import annotations

import json

from antecedent import AverageEffect, Frequentist, prepare
from antecedent.results import InspectionReport


def _data():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 80
    t = rng.integers(0, 2, n)
    y = 1.0 + 0.5 * t + rng.normal(0, 0.1, n)
    return pd.DataFrame({"t": t, "y": y, "x": rng.normal(0, 1, n)})


def test_inspect_is_pydantic_and_to_dict_matches_dump() -> None:
    prepared = prepare(
        _data(),
        graph=[("t", "y"), ("x", "y"), ("x", "t")],
        query=AverageEffect("t", "y"),
        inference=Frequentist(),
        refute="none",
        bootstrap=0,
    )
    report = prepared.inspect()
    assert isinstance(report, InspectionReport)
    dumped = report.to_dict()
    assert dumped == report.model_dump(mode="json")
    json.dumps(dumped, allow_nan=False)
    assert report.calibration.status == "unavailable"
    reconstructed = InspectionReport.model_validate(dumped)
    assert reconstructed.program_id == report.program_id
    live = prepared.estimate(_data())
    live_report = live.inspect()
    assert isinstance(live_report, InspectionReport)
    assert live_report.to_dict()["contract"] == live.inspect().contract
