"""Weighted target populations through analyze(): kernel and row-weight declarations.

The planted effect is heterogeneous, ``tau(z) = 1 + z``, so the effect for a population is the
weighted mean of ``tau`` over that population's weights. Every expected value below is that
closed form on the planted rows, never a call into the code under test.
"""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.population import (
    AllRows,
    KernelTarget,
    coerce_target_population,
    target_kernel,
    target_row_weights,
)
from antecedent.priors import TransportKernel

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _heterogeneous(n: int = 3000, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.9 * z))).astype(float)
    y = (1.0 + z) * t + z + 0.3 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _gaussian(z: np.ndarray, centre: float, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * ((z - centre) / bandwidth) ** 2)


def _analyze(data, population=None, **kwargs):
    return antecedent.analyze(
        data,
        graph=GRAPH,
        query=antecedent.AverageEffect(treatment="t", outcome="y", target_population=population),
        estimator="aipw",
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0),
        execution=antecedent.Execution(seed=11),
        **kwargs,
    )


def test_kernel_neighbourhood_effect_equals_the_closed_form_weighted_effect():
    data = _heterogeneous()
    weights = _gaussian(data["z"], 1.0, 0.5)
    truth = float(np.sum(weights * (1.0 + data["z"])) / np.sum(weights))
    target = target_kernel(TransportKernel("gaussian", 0.5), center={"z": 1.0})
    result = require_success(_analyze(data, target))
    se = result.estimate.se_analytic
    assert abs(result.ate - truth) < 4.0 * se
    # The whole-population effect is a different number: the target moved the answer.
    everyone = require_success(_analyze(data))
    assert abs(everyone.ate - 1.0) < 4.0 * everyone.estimate.se_analytic
    assert abs(result.ate - everyone.ate) > 0.5


def test_overlap_diagnostics_are_the_hand_computed_kish_quantities():
    data = _heterogeneous()
    weights = _gaussian(data["z"], 1.0, 0.5)
    target = target_kernel(TransportKernel("gaussian", 0.5), center={"z": 1.0})
    result = require_success(_analyze(data, target))
    report = result.weighted_target
    assert report is not None
    overlap = report.overlap
    assert overlap.n_rows == len(weights)
    assert overlap.n_support == int(np.sum(weights > 0.0))
    assert overlap.kish_ess == pytest.approx(float(weights.sum() ** 2 / np.sum(weights**2)))
    assert overlap.max_weight_share == pytest.approx(float(weights.max() / weights.sum()))
    assert report.declaration["family"] == "gaussian"
    assert report.declaration["bandwidth"] == 0.5
    assert report.declaration["center"] == [1.0]
    assert result.provenance["weighted_target"]["identity"] == report.identity
    assert any("weighted target" in text for text in result.assumptions)
    assert any(report.identity in text for text in result.assumptions)


def test_kernel_and_explicit_weights_with_the_same_weights_give_the_same_effect():
    data = _heterogeneous()
    weights = _gaussian(data["z"], 1.0, 0.5)
    kernel = require_success(
        _analyze(data, target_kernel(TransportKernel("gaussian", 0.5), center={"z": 1.0}))
    )
    explicit = require_success(_analyze(data, target_row_weights(weights, depends_on=["z"])))
    assert explicit.ate == kernel.ate
    # The declaration is part of the target's identity: a kernel is not its own weights.
    assert explicit.weighted_target.identity != kernel.weighted_target.identity
    assert explicit.weighted_target.declaration["kind"] == "row_weights"


def test_the_kernel_identity_changes_the_claim_identity():
    data = _heterogeneous()
    narrow = require_success(
        _analyze(data, target_kernel(TransportKernel("gaussian", 0.5), center={"z": 1.0}))
    )
    wide = require_success(
        _analyze(data, target_kernel(TransportKernel("gaussian", 0.8), center={"z": 1.0}))
    )
    assert narrow.weighted_target.identity != wide.weighted_target.identity
    if narrow.claim_id is not None:
        assert narrow.claim_id != wide.claim_id
    assert narrow.ate != wide.ate


def test_effects_move_with_the_bandwidth_toward_the_closed_form_of_each_neighbourhood():
    data = _heterogeneous(n=4000, seed=9)
    for bandwidth in (0.4, 0.7, 1.2):
        weights = _gaussian(data["z"], 0.5, bandwidth)
        truth = float(np.sum(weights * (1.0 + data["z"])) / np.sum(weights))
        target = target_kernel(TransportKernel("gaussian", bandwidth), center={"z": 0.5})
        result = require_success(_analyze(data, target))
        assert abs(result.ate - truth) < 4.0 * result.estimate.se_analytic


def test_a_target_without_overlap_refuses_with_a_next_attempt():
    data = _heterogeneous()
    far = target_kernel(TransportKernel("gaussian", 0.1), center={"z": 8.0})
    outcome = _analyze(data, far)
    assert outcome.refusal is not None
    assert outcome.refusal.code == "transport_target_overlap_failed"
    assert "Next attempt" in outcome.refusal.message


def test_a_thin_target_below_the_kish_floor_refuses_and_a_wider_one_runs():
    data = _heterogeneous()
    thin = target_kernel(TransportKernel("gaussian", 0.3), center={"z": 3.0})
    weights = _gaussian(data["z"], 3.0, 0.3)
    assert weights.sum() ** 2 / np.sum(weights**2) < 20.0
    refused = _analyze(data, thin)
    assert refused.refusal is not None
    assert refused.refusal.code == "transport_target_overlap_failed"
    wide = target_kernel(TransportKernel("gaussian", 1.5), center={"z": 1.0})
    assert math.isfinite(require_success(_analyze(data, wide)).ate)


def test_the_declared_ess_fraction_floor_is_applied():
    data = _heterogeneous()
    strict = target_kernel(
        TransportKernel("gaussian", 0.5), center={"z": 1.0}, min_ess_fraction=0.9
    )
    outcome = _analyze(data, strict)
    assert outcome.refusal is not None
    assert outcome.refusal.code == "transport_target_overlap_failed"


def test_tag_and_default_targets_are_unchanged():
    data = _heterogeneous()
    default = require_success(_analyze(data))
    all_rows = require_success(_analyze(data, AllRows()))
    treated = require_success(_analyze(data, "treated"))
    assert default.weighted_target is None
    assert all_rows.weighted_target is None
    assert treated.weighted_target is None
    assert "weighted_target" not in default.provenance
    assert all("weighted target" not in text for text in (default.assumptions or ()))
    assert default.ate == all_rows.ate
    assert default.claim_id == all_rows.claim_id


def test_a_weighted_target_is_resolved_only_where_a_table_exists():
    target = target_kernel(TransportKernel("gaussian", 0.5), center={"z": 1.0})
    assert isinstance(target, KernelTarget)
    with pytest.raises(CausalUnsupportedError):
        coerce_target_population(target)


def test_a_missing_covariate_column_is_refused_by_name():
    data = _heterogeneous()
    target = target_kernel(TransportKernel("gaussian", 0.5), center={"age": 1.0})
    with pytest.raises(CausalValueError, match="age"):
        require_success(_analyze(data, target))
