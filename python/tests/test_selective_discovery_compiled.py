"""The direct ``SelectiveDiscovery.resume`` warns, and ``resume_compiled`` resumes the same search
through the compiled structure-discovery route with the same outcome and the same run report."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success

NAMES = ("a", "b", "c", "d")


def _data(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = 0.8 * a + 0.8 * b + 0.6 * rng.normal(size=n)
    d = 0.9 * c + 0.5 * rng.normal(size=n)
    return {"a": a, "b": b, "c": c, "d": d}


def test_resume_warns_and_the_compiled_resume_matches_it():
    data = _data(800, 3)
    contract = sd.StaticContract(NAMES, algorithm="pc", alpha=0.01, seed=0)
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract, data)
    exported = adapter.export_state("run-compiled")

    with pytest.warns(AntecedentDeprecationWarning, match="SelectiveDiscovery.resume"):
        resumed, report = require_success(
            sd.SelectiveDiscovery.resume(contract, data, exported.artifact, exported.identity)
        )
    compiled = sd.resume_compiled(contract, data, exported.artifact, exported.identity)

    assert resumed.outcome is not None
    assert compiled.outcome.result_identity == resumed.outcome.result_identity
    assert compiled.outcome.structure_identity == resumed.outcome.structure_identity
    assert compiled.report == report
    assert compiled.report.work.ci_tests_attempted == 0
    assert compiled.program is not None and len(compiled.program) == 64


def test_the_compiled_resume_refuses_an_unusable_artifact():
    data = _data(200, 5)
    contract = sd.StaticContract(NAMES, algorithm="pc", alpha=0.01, seed=0)
    with pytest.raises(sd.SelectiveDiscoveryRefusal):
        sd.resume_compiled(contract, data, b"not an artifact")
