"""The direct sensitivity ranges warn, and the compiled range of the study's own result reproduces them.

``validation.sensitivity_range`` runs on a prepared study's compiled result and calls the same
kernel functions as the direct paths on the columns the study identified, so every number of
the direct call on those columns is reproduced exactly. The range of the compiled call is bound
to the study's own target; the direct call's is bound to a target of its labels.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import regimes
from antecedent import validation as v
from antecedent._deprecation import AntecedentDeprecationWarning


def _confounded():
    rng = np.random.default_rng(11)
    n = 400
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.6 * z))).astype(float)
    y = 0.7 * t + 0.8 * z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _mediated():
    rng = np.random.default_rng(13)
    n = 500
    t = rng.integers(0, 2, size=n).astype(float)
    m = 0.7 * t + rng.normal(size=n)
    y = 0.2 * t + 0.9 * m + rng.normal(size=n)
    return {"t": t, "m": m, "y": y}


def _effect_study(data):
    return ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )


def _mediation_study(data):
    return ant.prepare(
        data,
        graph=[("t", "m"), ("t", "y"), ("m", "y")],
        query=ant.MediationEffect("t", "y", mediators=["m"]),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )


def _same_numbers(compiled, direct):
    assert compiled.range is not None and direct.range is not None
    assert compiled.range.assumption == direct.range.assumption
    assert compiled.range.tested == direct.range.tested
    assert compiled.range.break_even == direct.range.break_even
    assert compiled.range.break_even_significance == direct.range.break_even_significance


def test_omitted_variable_bias_warns_and_the_compiled_range_reproduces_it():
    data = _confounded()
    compiled = v.sensitivity_range(_effect_study(data), "omitted_variable_bias")
    with pytest.warns(AntecedentDeprecationWarning, match="omitted_variable_bias_from_data"):
        direct = v.omitted_variable_bias_from_data(
            data, outcome="y", treatment="t", covariates=["z"]
        )
    for key in ("estimate", "standard_error", "dof", "robustness_value", "robustness_value_alpha"):
        assert compiled.summary[key] == direct.summary[key], key
    assert compiled.summary["benchmarks"] == direct.summary["benchmarks"]
    _same_numbers(compiled, direct)
    with pytest.warns(AntecedentDeprecationWarning, match="omitted_variable_bias"):
        summary = v.omitted_variable_bias(
            compiled.summary["estimate"],
            compiled.summary["standard_error"],
            compiled.summary["dof"],
        )
    _same_numbers(compiled, summary)
    assert compiled.range.target.family == "scalar_average_effect"


def test_evalue_of_a_standardized_difference_warns_and_the_compiled_range_reproduces_it():
    compiled = v.sensitivity_range(_effect_study(_confounded()), "evalue")
    with pytest.warns(AntecedentDeprecationWarning, match="evalue"):
        direct = v.evalue(
            compiled.summary["standardized_effect"],
            standard_error=compiled.summary["standardized_standard_error"],
            scale="standardized_mean_difference",
        )
    for key in ("risk_ratio", "risk_ratio_limit", "evalue", "evalue_limit"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)


def test_evalue_of_a_ratio_warns_and_the_compiled_supplied_estimate_reproduces_it():
    with pytest.warns(AntecedentDeprecationWarning, match="supplied_evalue"):
        direct = v.evalue(3.9, lower=1.8, upper=8.7, scale="risk_ratio")
    compiled = v.supplied_evalue(3.9, lower=1.8, upper=8.7, scale="risk_ratio")
    for key in ("risk_ratio", "risk_ratio_limit", "evalue", "evalue_limit"):
        assert compiled.summary[key] == direct.summary[key], key
    assert compiled.range == direct.range


def test_mediation_component_and_path_warn_for_one_outcome_and_the_compiled_ranges_reproduce_them():
    data = _mediated()
    study = _mediation_study(data)
    for component in ("indirect", "direct"):
        compiled = v.sensitivity_range(study, "mediation_component", component=component)
        with pytest.warns(AntecedentDeprecationWarning, match="mediation_sensitivity_vector"):
            direct = v.mediation_sensitivity_vector(
                data["t"], data["m"], [data["y"]], [], [component], component=component
            )
        assert compiled.summary["coordinates"] == direct.summary["coordinates"]
        _same_numbers(compiled, direct)
    for path in ("treatment_mediator", "mediator_outcome", "treatment_outcome", "indirect"):
        compiled = v.sensitivity_range(study, "mediation_path", path=path)
        with pytest.warns(AntecedentDeprecationWarning, match="mediation_path_sensitivity"):
            direct = v.mediation_path_sensitivity(
                data["t"], data["m"], [data["y"]], [], [path], path=path
            )
        assert compiled.summary["coordinates"] == direct.summary["coordinates"]
        _same_numbers(compiled, direct)


def test_the_compiled_mediation_curve_reproduces_the_deprecated_direct_curve():
    data = _mediated()
    compiled = v.sensitivity_range(_mediation_study(data), "mediation")
    with pytest.warns(AntecedentDeprecationWarning, match="mediation_sensitivity"):
        direct = v.mediation_sensitivity(data, treatment="t", mediator="m", outcome="y")
    for key in ("beta_mediator", "sigma_total", "sigma_mediator", "rho_tilde"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)
    assert compiled.range.target.family == "mediation"
    assert compiled.range.target.labels == ("total", "direct", "indirect")


def _two_mediated():
    rng = np.random.default_rng(42)
    n = 500
    t = rng.integers(0, 2, size=n).astype(float)
    m = 0.7 * t + rng.normal(size=n)
    y1 = 0.2 * t + 0.9 * m + rng.normal(size=n)
    y2 = 0.5 * t + 0.3 * m + rng.normal(size=n)
    return {"t": t, "m": m, "y1": y1, "y2": y2}


def _two_outcomes():
    rng = np.random.default_rng(41)
    n = 400
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y1 = 0.8 * t + 0.6 * z + rng.normal(size=n)
    y2 = 0.4 * t - 0.5 * z + rng.normal(size=n)
    return {"t": t, "y1": y1, "y2": y2, "z": z}


def _outcome_study(data, outcome, inference=None):
    return ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y1"), ("z", "y2"), ("t", "y1"), ("t", "y2")],
        query=ant.AverageEffect("t", outcome),
        inference=ant.Frequentist(bootstrap=0) if inference is None else inference,
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )


def _mediation_outcome_study(data, outcome):
    return ant.prepare(
        data,
        graph=[("t", "m"), ("t", "y1"), ("m", "y1"), ("t", "y2"), ("m", "y2")],
        query=ant.MediationEffect("t", outcome, mediators=["m"]),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=5, threads=1),
    )


def test_the_vector_mediation_warns_and_the_compiled_vector_over_studies_reproduces_it():
    data = _two_mediated()
    studies = [_mediation_outcome_study(data, "y1"), _mediation_outcome_study(data, "y2")]
    for component in ("indirect", "direct"):
        compiled = v.sensitivity_vector_range(studies, "mediation_component", component=component)
        with pytest.warns(AntecedentDeprecationWarning, match="mediation_sensitivity_vector"):
            direct = v.mediation_sensitivity_vector(
                data["t"],
                data["m"],
                [data["y1"], data["y2"]],
                [],
                ["y1", "y2"],
                component=component,
            )
        assert compiled.summary["coordinates"] == direct.summary["coordinates"]
        _same_numbers(compiled, direct)
        assert compiled.range.target.labels == ("y1", "y2")
    compiled = v.sensitivity_vector_range(studies, "mediation_path", path="indirect")
    with pytest.warns(AntecedentDeprecationWarning, match="mediation_path_sensitivity"):
        direct = v.mediation_path_sensitivity(
            data["t"], data["m"], [data["y1"], data["y2"]], [], ["y1", "y2"], path="indirect"
        )
    assert compiled.summary["coordinates"] == direct.summary["coordinates"]
    _same_numbers(compiled, direct)


def test_the_vector_evalue_and_partial_r2_warn_and_the_compiled_vector_reproduces_them():
    data = _two_outcomes()
    studies = [_outcome_study(data, "y1"), _outcome_study(data, "y2")]
    compiled = v.sensitivity_vector_range(studies, "evalue")
    with pytest.warns(AntecedentDeprecationWarning, match="evalue_vector"):
        direct = v.evalue_vector(
            compiled.summary["standardized_effect"],
            ["y1", "y2"],
            standard_errors=compiled.summary["standardized_standard_error"],
            scale="standardized_mean_difference",
        )
    for key in ("risk_ratio", "risk_ratio_limit", "evalue", "evalue_limit"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)
    compiled = v.sensitivity_vector_range(studies, "omitted_variable_bias")
    with pytest.warns(AntecedentDeprecationWarning, match="omitted_variable_bias_vector"):
        direct = v.omitted_variable_bias_vector(
            compiled.summary["estimate"],
            compiled.summary["standard_error"],
            compiled.summary["dof"],
            ["y1", "y2"],
        )
    assert compiled.summary["robustness_value"] == direct.summary["robustness_value"]
    _same_numbers(compiled, direct)


def test_the_posterior_entries_warn_and_the_compiled_posteriors_reproduce_them():
    data = _two_outcomes()
    study = _outcome_study(data, "y1", inference=ant.Bayesian(n_draws=400))
    compiled = v.sensitivity_range(study, "evalue_posterior")
    with pytest.warns(AntecedentDeprecationWarning, match="evalue_posterior"):
        direct = v.evalue_posterior(
            [compiled.summary["draws"]], ["ate"], scale="standardized_mean_difference"
        )
    for key in ("median_risk_ratio", "credible_interval", "evalue", "evalue_limit"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)
    compiled = v.sensitivity_range(study, "omitted_variable_bias_posterior")
    with pytest.warns(AntecedentDeprecationWarning, match="omitted_variable_bias_posterior"):
        direct = v.omitted_variable_bias_posterior(
            [compiled.summary["coefficients"]], [compiled.summary["bias_scale"]], ["ate"]
        )
    _same_numbers(compiled, direct)

    mediated = _two_mediated()
    compiled = v.sensitivity_range(
        _mediation_outcome_study(mediated, "y1"), "mediation_posterior", draws=400
    )
    with pytest.warns(AntecedentDeprecationWarning, match="mediation_sensitivity_posterior"):
        direct = v.mediation_sensitivity_posterior(
            [compiled.summary["beta_mediator"]],
            [compiled.summary["sigma_total"]],
            [compiled.summary["sigma_mediator"]],
            [compiled.summary["rho_tilde"]],
            [compiled.summary["total_effect"]],
            ["indirect"],
        )
    _same_numbers(compiled, direct)


def _regime(n=400, seed=45):
    rng = np.random.default_rng(seed)
    l1 = rng.normal(size=n)
    p1 = np.where(l1 > 0.0, 0.7, 0.3)
    a1 = rng.random(n) < p1
    l2 = 0.5 * a1 + rng.normal(size=n)
    p2 = np.where(l2 > 0.0, 0.65, 0.35)
    a2 = rng.random(n) < p2
    y = 1.0 + 2.0 * a1 + 1.5 * a2 + 0.8 * l1 + rng.normal(size=n)
    query = regimes.LongitudinalRegime(
        outcome="y",
        treatment_history=np.column_stack([a1, a2]).tolist(),
        actions=[False, False],
        treatment_probabilities=np.column_stack([p1, p2]).tolist(),
        censoring_probabilities=np.ones((n, 2)).tolist(),
        subject_ids=[f"s{i}" for i in range(n)],
        fold_ids=[i % 5 for i in range(n)],
        method="marginal_structural_model",
        stabilizing_numerator_probabilities=[0.5, 0.5],
        covariates=["l1", "l2"],
    )
    data = {"y": y, "l1": l1, "l2": l2}
    study = ant.prepare(data, query=query, refute="none", execution=ant.Execution(seed=7))
    exposures = [a1.astype(float), a2.astype(float)]
    weights = (0.5 / np.where(a1, p1, 1.0 - p1)) * (0.5 / np.where(a2, p2, 1.0 - p2))
    return study, data, exposures, weights


def test_the_msm_parameter_entries_warn_and_the_compiled_regime_reading_reproduces_them():
    study, data, exposures, weights = _regime()
    compiled = v.sensitivity_range(study, "msm_parameters")
    assert compiled.summary["weights"] == pytest.approx(weights.tolist(), rel=1e-12)
    with pytest.warns(AntecedentDeprecationWarning, match="msm_parameter_sensitivity"):
        direct = v.msm_parameter_sensitivity(
            data["y"], exposures, compiled.summary["weights"], ["period[0]", "period[1]"]
        )
    for key in ("estimate", "standard_error", "dof", "robustness_value"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)
    compiled = v.sensitivity_range(
        study,
        "msm_parameters_refit",
        numerators=[[], []],
        denominators=[["l1"], ["l2"]],
        replicates=99,
    )
    with pytest.warns(AntecedentDeprecationWarning, match="msm_parameter_sensitivity_refit"):
        direct = v.msm_parameter_sensitivity_refit(
            data["y"],
            exposures,
            exposures,
            [[], []],
            [[data["l1"]], [data["l2"]]],
            ["period[0]", "period[1]"],
            replicates=99,
            execution=ant.Execution(seed=7),
        )
    for key in ("estimate", "standard_error", "fixed_weight_standard_error"):
        assert compiled.summary[key] == direct.summary[key], key
    _same_numbers(compiled, direct)


def test_the_marginal_sensitivity_model_reads_the_empirical_strata_of_the_adjustment_set():
    rng = np.random.default_rng(15)
    n = 600
    z = rng.integers(0, 3, size=n).astype(float)
    t = (rng.random(n) < 0.3 + 0.2 * z).astype(float)
    y = np.minimum(np.floor(0.8 * t + 0.6 * z + 1.2 * rng.random(n)), 3.0)
    data = {"t": t, "y": y, "z": z}
    compiled = v.sensitivity_range(_effect_study(data), "marginal_sensitivity", lambda_max=2.0)
    first = compiled.summary["grid"][0]
    contrast = 0.0
    for stratum in (0.0, 1.0, 2.0):
        cell = z == stratum
        contrast += cell.mean() * (y[cell & (t == 1.0)].mean() - y[cell & (t == 0.0)].mean())
    assert compiled.summary["identified"] == pytest.approx(contrast, abs=1e-12)
    assert first["lower"] == pytest.approx(contrast, abs=1e-12)
    assert compiled.range is None
    assert compiled.summary["calibration"] == "not_measured"


def test_prior_sensitivity_target_warns_and_returns_the_same_check():
    draws = np.random.default_rng(17).normal(size=(200, 1)).tolist()
    with pytest.warns(AntecedentDeprecationWarning, match="prior_sensitivity_target"):
        check = v.prior_sensitivity_target(
            ["effect"], draws, {"wide": draws}, family="scalar_average_effect"
        )
    assert check.passed
