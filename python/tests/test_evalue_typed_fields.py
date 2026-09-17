"""The E-value is a typed field, not a number to scrape out of report prose.

``EstimateView.evalue`` carries what the ``sensitivity.evalue`` refuter reported and
``EstimateView.evalue_threshold`` the value it judged that against, so a consumer can
reproduce the pass/fail without parsing ``RefutationReport`` strings. Both stay
``None`` when the refuter did not run.
"""

import antecedent as ac
import numpy as np
import pytest

EVALUE = "sensitivity.evalue"


def confounded_fixture():
    """``y = 0.8a + 0.5b + 0.6ab + z + e`` with ``b`` left out of the analysis, so the
    backdoor ATE of ``a`` on ``y`` adjusts for ``z`` alone."""
    rng = np.random.default_rng(1)
    n = 2500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    b = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.8 * a + 0.5 * b + 0.6 * a * b + z + rng.normal(size=n)
    return {"z": z, "a": a, "y": y}, [("z", "a"), ("z", "y"), ("a", "y")]


def ate(data, graph, refute):
    return ac.analyze(
        data,
        graph=graph,
        query=ac.AverageEffect("a", "y"),
        refute=refute,
        seed=1,
    )


def test_evalue_is_a_typed_field_not_a_report_string():
    data, graph = confounded_fixture()
    result = ate(data, graph, "full")
    report = next(r for r in result.validation.reports if r.refuter == EVALUE)

    assert result.estimate.evalue == report.comparison
    assert result.estimate.evalue == pytest.approx(3.134, abs=5e-4)
    threshold = result.estimate.evalue_threshold
    assert threshold == 2.0
    # The verdict is now reproducible from the typed pair alone.
    assert report.passed == (result.estimate.evalue >= threshold)


def test_evalue_stays_none_when_the_refuter_did_not_run():
    data, graph = confounded_fixture()
    result = ate(data, graph, "none")
    assert not [r for r in result.validation.reports if r.refuter == EVALUE]
    assert result.estimate.evalue is None
    assert result.estimate.evalue_threshold is None
