"""Declared sampling weights, clusters and serial lags on the dose-grid, categorical and smoothed
dose routes: Python and Rust agree with the frozen exact references.

The dose-grid cases are the 50-digit multiprecision cases of
``crates/antecedent-estimate/tests/oracle_data/dose_grid_declared_reference.json`` (weighted
Gaussian local quadratic, one-way cluster-robust and Newey-West standard errors); the categorical
cases are the exact ``Fraction`` cases of ``categorical_declared_reference.json`` (weighted
dummy-coded regression with HC, cluster-robust and Newey-West sandwiches). The smoothed-dose cases
check the declarations through the Python data class, the artifact and its replay.
"""

from __future__ import annotations

from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent import categorical_treatment as ct
from antecedent.dose_grid import (
    DoseClaims,
    DoseGridRefusal,
    consume_dose_grid_artifact,
    dose_functional,
)
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.transport import advanced

from _repo_text import load_json

ORACLE = Path(__file__).resolve().parents[2] / "crates" / "antecedent-estimate" / "tests" / "oracle_data"


def oracle(name: str):
    return load_json(ORACLE / name)


def close(got, want, rel, what=""):
    assert abs(got - want) <= rel * max(abs(want), 1e-300), f"{what}: {got} vs {want}"


# ---------------------------------------------------------------------------------------------
# Dose grid
# ---------------------------------------------------------------------------------------------


def dose_call(case, functional, **extra):
    h = case["bandwidth"]
    kwargs = dict(
        bandwidth=h,
        bandwidth_range=(h * 0.5, h * 2.0),
        minimum_local_ess=3.0,
        functional=functional,
        claims=DoseClaims(pointwise_level=True, derivative=True),
        sampling_weights=case["weights"],
        cluster=case["groups"],
        serial_lag=case["lag"],
    )
    if functional == "contrast":
        kwargs["contrast"] = (case["contrast"]["from"], case["contrast"]["to"])
    else:
        kwargs["grid"] = case["grid"]
    kwargs.update(extra)
    return dose_functional(case["dose"], case["outcome"], **kwargs)


@pytest.mark.parametrize("case", oracle("dose_grid_declared_reference.json"), ids=lambda c: c["name"])
def test_dose_grid_declared_rows_match_the_multiprecision_reference(case):
    levels = dose_call(case, "level")
    slopes = dose_call(case, "derivative")
    assert levels.n_rows == case["rows_used"]
    for point, expected, slope in zip(levels.levels, case["points"], slopes.derivatives):
        close(point.value, expected["level"], 1e-8, "level")
        close(point.standard_error, expected["level_se"], 1e-8, "level se")
        close(point.local_ess, expected["ess"], 1e-9, "ess")
        close(slope.value, expected["derivative"], 1e-7, "derivative")
        close(slope.standard_error, expected["derivative_se"], 1e-8, "derivative se")
    declared = levels.declared_rows
    assert declared is not None and declared.rows_used == case["rows_used"]
    assert declared.zero_weight_rows == case["zero_weight_rows"]
    assert declared.clusters == case.get("clusters")
    assert declared.serial_lag == case["lag"]
    if case["weights"] is not None:
        close(declared.weight_effective_sample_size, case["kish_weights"], 1e-9, "kish")
        close(declared.weight_sum, case["weight_sum_used"], 1e-12, "weight sum")
    if case["contrast"] is not None:
        contrast = dose_call(case, "contrast").contrast
        close(contrast.estimate, case["contrast"]["estimate"], 1e-8, "contrast")
        close(contrast.standard_error, case["contrast"]["se"], 1e-8, "contrast se")
    # The sealed artifact replays from its own bytes, weights and clusters included.
    again = consume_dose_grid_artifact(bytes(levels.export()))
    assert again.levels == levels.levels and again.declared_rows == declared


def test_dose_grid_declared_rows_bias_aware_intervals_follow_the_declared_dependence():
    case = next(c for c in oracle("dose_grid_declared_reference.json") if c["name"] == "clusters_unbalanced_singletons")
    fit = dose_call(case, "level", curvature_bound=case["curvature_bound"], band_replicates=2000)
    assert fit.bias_aware is not None
    for j, expected in enumerate(case["points"]):
        close(fit.bias_aware.smoothing_bias_bound[j], expected["level_bias"], 1e-7, "bias")
        half = (fit.bias_aware.upper[j] - fit.bias_aware.lower[j]) / 2
        close(half, expected["level_half"], 1e-7, "half width")


def test_dose_grid_declarations_the_rows_cannot_honour_are_refused_by_name():
    dose = np.linspace(0.0, 4.0, 201)
    outcome = 1.0 + 0.5 * dose + 0.25 * dose**2 + 0.01 * np.where(np.arange(201) % 2 == 0, 1.0, -1.0)

    def call(**extra):
        return dose_functional(
            dose, outcome, bandwidth=0.4, bandwidth_range=(0.1, 1.0), grid=[1.0, 2.0],
            minimum_local_ess=10.0, **extra
        )

    with pytest.raises(DoseGridRefusal, match="not offered together"):
        call(cluster=[0] * 100 + [1] * 101, serial_lag=2)
    with pytest.raises(DoseGridRefusal) as info:
        call(sampling_weights=[1.0] * 200)
    assert info.value.detail == "dose_grid.invalid_request"
    with pytest.raises(DoseGridRefusal) as info:
        call(cluster=[5] * 201)
    assert info.value.detail == "dose_grid.too_few_clusters"
    for lag in (201, 500):
        with pytest.raises(DoseGridRefusal) as info:
            call(serial_lag=lag)
        assert info.value.detail == "dose_grid.invalid_lag"
    with pytest.raises(DoseGridRefusal) as info:
        call(serial_lag=0)
    assert info.value.detail == "dose_grid.invalid_request"
    with pytest.raises(CausalUnsupportedError) as info:
        call(sampling_weights=[-1.0] + [1.0] * 200)
    assert info.value.reason_code == "sampling_weights_invalid"
    with pytest.raises(CausalTypeError):
        call(serial_lag=2.5)


# ---------------------------------------------------------------------------------------------
# Categorical treatment
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", oracle("categorical_declared_reference.json"), ids=lambda c: c["name"])
def test_categorical_declared_rows_match_the_exact_reference(case):
    expected = case["expected"]
    kwargs = dict(
        categories=case["levels"],
        reference=case["reference"],
        pairs=[tuple(p) for p in case["pairs"]],
        min_level_rows=1,
        covariance=case["covariance"],
        sampling_weights=case["weights"],
        cluster=case["groups"],
        serial_lag=case["lag"],
    )
    if case["adjustment"] is not None:
        kwargs["adjust"] = {"g": case["adjustment"]}
    if expected["omnibus"] is None:
        with pytest.raises(ct.CategoricalTreatmentRefusal, match="degenerate_covariance"):
            ct.categorical_effects(case["outcomes"], case["row_levels"], **kwargs)
        return
    fit = ct.categorical_effects(case["outcomes"], case["row_levels"], **kwargs)
    scale = max(np.diag(fit.covariance))
    tol = 3e-6 if "offset" in case["name"] else 1e-9  # the 1e8 outcome offset costs digits
    for a, row in enumerate(expected["covariance"]):
        for b, value in enumerate(row):
            assert abs(fit.covariance[a, b] - value) <= tol * scale, (a, b)
    for effect, want in zip(fit.effects, expected["levels"]):
        assert abs(effect.estimate - want["estimate"]) <= tol * max(1.0, abs(want["estimate"]))
        close(effect.standard_error**2, want["variance"], max(tol, 1e-9), "level variance")
    close(fit.omnibus.statistic, expected["omnibus"], 1e-7, "omnibus")
    assert dict(fit.counts) == expected["counts"]
    declared = fit.declared_rows
    assert declared is not None
    assert declared.rows_used == expected["rows_used"]
    assert declared.clusters == expected.get("clusters")
    assert declared.zero_weight_rows == expected.get("zero_weight_rows", 0)
    # The artifact replays; a retained identity pins the weighted design.
    again = ct.consume_categorical_effects(fit.export(), expected_identity=fit.identity)
    assert again.declared_rows == declared and again.identity == fit.identity


def test_categorical_declarations_without_their_covariance_kind_are_refused_by_name():
    case = next(c for c in oracle("categorical_declared_reference.json") if c["name"] == "weights_hc1")
    base = dict(categories=case["levels"], reference="a", min_level_rows=1, adjust={"g": case["adjustment"]})
    y, labels = case["outcomes"], case["row_levels"]
    n = len(y)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="cluster_unread"):
        ct.categorical_effects(y, labels, covariance="hc1", cluster=[i % 4 for i in range(n)], **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="cluster_ids_required"):
        ct.categorical_effects(y, labels, covariance="cluster_robust", **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="too_few_clusters"):
        ct.categorical_effects(y, labels, covariance="cluster_robust", cluster=[1] * n, **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="lag_unread"):
        ct.categorical_effects(y, labels, covariance="hc0", serial_lag=2, **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="lag_required"):
        ct.categorical_effects(y, labels, covariance="newey_west", **base)
    for lag in (n, n + 4):
        with pytest.raises(ct.CategoricalTreatmentRefusal, match="invalid_lag"):
            ct.categorical_effects(y, labels, covariance="newey_west", serial_lag=lag, **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="weights_need_robust_covariance"):
        ct.categorical_effects(y, labels, covariance="model_based", sampling_weights=[1.0] * n, **base)
    with pytest.raises(CausalValueError, match="sampling weights were declared"):
        ct.categorical_effects(y, labels, covariance="hc0", sampling_weights=[1.0] * (n - 1), **base)
    with pytest.raises(ct.CategoricalTreatmentRefusal, match="posterior_declared_rows"):
        ct.categorical_effects(
            y, labels, covariance="hc0", sampling_weights=[1.0] * n,
            posterior=ct.CategoricalPosteriorPrior(
                level_precision=1.0, nuisance_precision=0.01, variance_shape=2.0,
                variance_scale=2.0, credible_level=0.95,
            ),
            **base,
        )


def test_categorical_plain_rows_keep_their_snapshot_and_report_no_declared_rows():
    y = [1.0, 3.0, 4.0, 6.0, 5.0, 9.0, 7.0, 11.0, 9.0]
    labels = ["a", "a", "b", "b", "b", "c", "c", "c", "c"]
    plain = ct.categorical_effects(y, labels, covariance="hc0")
    assert plain.declared_rows is None and plain.to_dict()["declared_rows"] is None
    # Equal weights at any scale are the unweighted sandwich.
    weighted = ct.categorical_effects(y, labels, covariance="hc0", sampling_weights=[4.0] * 9)
    assert weighted.row_snapshot != plain.row_snapshot
    for a, b in zip(plain.effects, weighted.effects):
        close(b.estimate, a.estimate, 1e-12)
        close(b.standard_error, a.standard_error, 1e-12)


# ---------------------------------------------------------------------------------------------
# Smoothed dose transport
# ---------------------------------------------------------------------------------------------


def density(a, z):
    return 0.25 * (1.0 + 0.6 * np.tanh(z) * (a - 2.0) / 2.0)


def smoothed_fixture(n_trial=500, n_target=400, seed=3, **declared):
    rng = np.random.default_rng(seed)
    total = n_trial + n_target
    source = rng.random(total) < n_trial / total
    z = np.where(source, rng.normal(size=total), 0.6 + rng.normal(size=total))
    a = np.empty(total)
    pending = np.arange(total)
    while pending.size:
        candidate = 4.0 * rng.random(pending.size)
        accept = rng.random(pending.size) * 0.4 < density(candidate, z[pending])
        a[pending[accept]] = candidate[accept]
        pending = pending[~accept]
    y = a * a + z * (1.0 + a / 2.0) + rng.normal(size=total)
    data = advanced.SmoothedDoseData(
        {"z": z},
        np.where(source, y, 0.0),
        np.where(source, a, 0.0),
        np.where(source, density(a, z), 0.0),
        [bool(v) for v in source],
        "nested_cohort",
        **declared,
    )
    graph = ac.Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    query = advanced.SmoothedDoseQuery(
        graph, advanced.SelectionDiagram("trial", "target", ["z"]), "a", "y", (1.0, 2.0), 0.5, (0.0, 4.0)
    )
    return query, data, total


def estimate(query, data, seed=7):
    study = advanced.prepare_smoothed_dose(
        query, data, options=advanced.SmoothedDoseOptions(folds=3), execution=ac.Execution(seed=seed)
    )
    return study, require_success(study.estimate())


def test_smoothed_dose_declared_rows_are_read_replayed_and_bound_by_the_artifact():
    query, plain_data, total = smoothed_fixture()
    _, plain = estimate(query, plain_data)
    weights = [0.0 if i % 89 == 3 else 0.6 + (i % 5) * 0.3 for i in range(total)]
    clusters = [i % 30 for i in range(total)]
    for declared in (
        {"sampling_weights": weights},
        {"cluster": clusters},
        {"serial_lag": 3},
        {"sampling_weights": weights, "cluster": clusters},
    ):
        query, data, _ = smoothed_fixture(**declared)
        _, result = estimate(query, data)
        assert all(np.isfinite(p.estimate) for p in result.grid)
        again = advanced.consume_smoothed_dose(bytes(result.export()))
        assert [p.estimate for p in again.grid] == [p.estimate for p in result.grid]
        assert again.folds == result.folds
        assert again.premises_digest == result.premises_digest
    # The recorded fold scheme names the declaration's folds (whole clusters, contiguous blocks);
    # the assignments themselves are checked row by row in the Rust reference tests.
    query, data, _ = smoothed_fixture(cluster=clusters)
    _, clustered = estimate(query, data)
    assert clustered.folds["scheme"] == "clusters_whole_stratified_seeded_by_role"
    query, data, _ = smoothed_fixture(serial_lag=3)
    _, serial = estimate(query, data)
    assert serial.folds["scheme"] == "contiguous_row_blocks"
    assert plain.folds["scheme"] == "stratified_seeded_source_and_target"
    # Equal weights at another scale are the unweighted point; real weights move it.
    query, data, _ = smoothed_fixture(sampling_weights=[5.0] * total)
    _, equal = estimate(query, data)
    for p, q in zip(plain.grid, equal.grid):
        assert abs(p.estimate - q.estimate) <= 1e-9 * max(1.0, abs(p.estimate))
    query, data, _ = smoothed_fixture(sampling_weights=weights)
    _, moved = estimate(query, data)
    assert any(p.estimate != q.estimate for p, q in zip(plain.grid, moved.grid))


def test_smoothed_dose_refuses_unreadable_declarations_and_the_gaussian_posterior():
    query, data, total = smoothed_fixture(cluster=[0] * 3 + [i % 20 for i in range(3, 900)], serial_lag=2)
    with pytest.raises((CausalUnsupportedError, CausalValueError, ValueError), match="not offered together"):
        advanced.prepare_smoothed_dose(
            query, data, options=advanced.SmoothedDoseOptions(folds=3), execution=ac.Execution(seed=1)
        )
    query, data, _ = smoothed_fixture(serial_lag=0)
    with pytest.raises((CausalUnsupportedError, CausalValueError, ValueError), match="invalid_declared_rows"):
        advanced.prepare_smoothed_dose(
            query, data, options=advanced.SmoothedDoseOptions(folds=3), execution=ac.Execution(seed=1)
        )
    query, data, _ = smoothed_fixture(sampling_weights=[-1.0] + [1.0] * 899)
    with pytest.raises((CausalUnsupportedError, CausalValueError, ValueError), match="sampling_weights"):
        advanced.prepare_smoothed_dose(
            query, data, options=advanced.SmoothedDoseOptions(folds=3), execution=ac.Execution(seed=1)
        )
    query, data, _ = smoothed_fixture(serial_lag=3)
    study = advanced.prepare_smoothed_dose(
        query, data, options=advanced.SmoothedDoseOptions(folds=3), execution=ac.Execution(seed=1)
    )
    model = advanced.SmoothedDosePosteriorModel(
        intercept_scale=10.0, coefficient_scale=10.0, residual_variance=1.0, min_local_rows=50, draws=2000
    )
    with pytest.raises((CausalUnsupportedError, CausalValueError, ValueError), match="posterior_declared_rows"):
        require_success(study.posterior(model))
