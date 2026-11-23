"""The direct mediation sensitivity warns, and the compiled target refutation reproduces it."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import validation as v
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success


def _table():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.integers(0, 2, size=n).astype(float)
    m = 0.7 * x + rng.normal(size=n)
    y = 0.2 * x + 0.9 * m + rng.normal(size=n)
    return {"x": x, "m": m, "y": y}


def test_mediation_sensitivity_warns_and_the_compiled_refutation_reads_the_same_working_model():
    data = _table()
    with pytest.warns(AntecedentDeprecationWarning, match="mediation_sensitivity"):
        direct = v.mediation_sensitivity(data, treatment="x", mediator="m", outcome="y")
    study = ant.prepare(
        data,
        graph=[("x", "m"), ("x", "y"), ("m", "y")],
        query=ant.MediationEffect("x", "y", mediators=["m"]),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )
    checks = v.TargetRefutationSpec(suite="cheap", replicates=8, mediator="m")
    report = require_success(v.refute_target(study, checks=checks))
    check = {c.refuter: c for c in report.reports}["target.sequential_ignorability"]
    assert direct.range is not None
    centre = next(p for p in direct.range.tested if p.parameter == 0.0)
    assert check.value == abs(direct.summary["rho_tilde"])
    assert check.refuted[-1] == centre.target
