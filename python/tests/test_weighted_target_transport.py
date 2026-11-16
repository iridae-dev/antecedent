"""Trial-to-target transport toward a weighted target population.

Trial rows have ``z ~ N(0, 1)``, target rows ``z ~ N(1, 1)``, treatment is randomized in the
trial and the planted effect is ``tau(z) = 1 + z``. The selection probability is exactly
``1 / (1 + exp(z - 0.5))``. A Gaussian kernel target of centre ``c`` and bandwidth ``h`` inside
the target population has the closed-form effect ``1 + (1 + c / h**2) / (1 + 1 / h**2)``.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import transport
from antecedent.errors import CausalUnsupportedError
from antecedent.population import target_kernel, target_row_weights
from antecedent.priors import TransportKernel


def _identification() -> transport.TransportIdentification:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
    )
    return transport.identify(graph=graph, query=query)


def _rows(n: int = 12000, seed: int = 3):
    rng = np.random.default_rng(seed)
    trial = np.arange(n) % 2 == 0
    z = rng.normal(size=n) + np.where(trial, 0.0, 1.0)
    selection = 1.0 / (1.0 + np.exp(z - 0.5))
    treatment = rng.random(n) < 0.5
    outcome = (1.0 + z) * treatment + z + 0.3 * rng.normal(size=n)
    return z, selection, treatment, outcome, trial


def _estimate(target, rows, **kwargs):
    z, selection, treatment, outcome, trial = rows
    return transport.estimate_trial_effect(
        _identification(),
        list(treatment),
        outcome,
        list(trial),
        selection,
        np.full(len(z), 0.5),
        target=target,
        covariates={"z": z},
        **kwargs,
    )


def _closed_form(centre: float, bandwidth: float) -> float:
    h2 = bandwidth**2
    return 1.0 + (1.0 + centre / h2) / (1.0 + 1.0 / h2)


@pytest.mark.parametrize(("centre", "bandwidth"), [(1.0, 0.8), (1.5, 1.2), (0.5, 1.0)])
def test_a_kernel_target_effect_equals_the_closed_form_weighted_effect(centre, bandwidth):
    rows = _rows()
    target = target_kernel(TransportKernel("gaussian", bandwidth), center={"z": centre})
    result = _estimate(target, rows)
    assert abs(result.ipw - _closed_form(centre, bandwidth)) < 4.0 * result.ipw_se
    assert result.weighted_target.declaration["family"] == "gaussian"
    assert result.weighted_target.overlap.kish_ess > 20.0
    assert result.trial_overlap.kish_ess > 20.0


def test_constant_weights_equal_the_unweighted_contrast():
    rows = _rows(8000, 4)
    z, selection, treatment, outcome, trial = rows
    uniform = target_row_weights(np.ones(len(z)), depends_on=["z"])
    weighted = _estimate(uniform, rows)
    plain = transport.estimate_trial_effect(
        _identification(),
        list(treatment),
        outcome,
        list(trial),
        selection,
        np.full(len(z), 0.5),
    )
    assert weighted.ipw == pytest.approx(plain.ipw, abs=1e-9)


def test_a_target_without_overlap_refuses_with_a_next_attempt():
    rows = _rows(6000, 5)
    far = target_kernel(TransportKernel("gaussian", 0.2), center={"z": 7.0})
    with pytest.raises(CausalUnsupportedError, match="Next attempt") as caught:
        _estimate(far, rows)
    assert caught.value.reason_code == "transport_target_overlap_failed"


def test_an_unweighted_estimate_carries_no_weighted_target():
    z, selection, treatment, outcome, trial = _rows(4000, 6)
    plain = transport.estimate_trial_effect(
        _identification(),
        list(treatment),
        outcome,
        list(trial),
        selection,
        np.full(len(z), 0.5),
    )
    assert plain.weighted_target is None
    assert plain.ipw_se is None
