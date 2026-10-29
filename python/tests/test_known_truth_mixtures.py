"""Numeric evidence for the licensed graph-posterior mixture cells.

Rust owns the known-truth DGP. These tests consume the same fixture so a
Python run sees E[τ|identified] and unidentified mass, not prepared-vs-fresh
smoke on ExactDagPosterior / DbnPosterior.
"""

from __future__ import annotations

import math
from dataclasses import replace

import antecedent
import pytest
from antecedent.outcomes import require_success

from known_truth import (  # noqa: E402
    BAYES,
    FREQ,
    STATIC,
    TEMPORAL,
    TEMPORAL_MEDIATION,
    TEMPORAL_MULTI,
    static_data,
    static_posterior,
    temporal_mediation_posterior,
    temporal_mediation_series,
    temporal_posterior,
    white_noise_pulse_series,
)


def _prepare(data, *, seed, profile="interactive", inference, **options):
    return antecedent.prepare(
        data,
        inference=replace(inference, profile=profile),
        execution=antecedent.Execution(seed=seed),
        **options,
    )


_VALIDATIONS = [
    pytest.param(False, None, id="none"),
    pytest.param("cheap", "overlap+evalue", id="cheap"),
    pytest.param("full", "validation.full", id="full"),
]


def _assert_mixture_contract(
    fresh,
    click,
    prepared,
    *,
    validation_suite: str | None,
    expected_ate: float,
    tolerance: float,
    unidentified_mass: float,
    expect_ppc: bool,
    withhold_scalar: bool = False,
) -> None:
    assert prepared.evidence_status == "licensed"
    assert fresh.evidence_status == click.evidence_status == "licensed"
    assert fresh.plan.validation_suite == click.plan.validation_suite == validation_suite
    if withhold_scalar:
        assert fresh.ate is None and click.ate is None
        assert fresh.posterior is not None and click.posterior is not None
        assert fresh.posterior.effect_mean == pytest.approx(expected_ate, abs=tolerance)
        assert click.posterior.effect_mean == pytest.approx(expected_ate, abs=tolerance)
    else:
        assert fresh.ate == pytest.approx(expected_ate, abs=tolerance)
        assert click.ate == pytest.approx(expected_ate, abs=tolerance)
        assert fresh.posterior is not None and click.posterior is not None
    assert fresh.posterior.unidentified_mass == pytest.approx(unidentified_mass, abs=1e-12)
    assert click.posterior.unidentified_mass == pytest.approx(unidentified_mass, abs=1e-12)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    if validation_suite is None:
        assert not fresh.validation.reports
        assert not click.validation.reports
        assert fresh.validation.prior_predictive is None
        assert click.validation.prior_predictive is None
    else:
        assert fresh.validation.ran and click.validation.ran
        assert fresh.validation.reports and click.validation.reports
        if expect_ppc:
            assert fresh.validation.prior_predictive is not None
            assert fresh.validation.posterior_predictive is not None
            assert click.validation.prior_predictive is not None
            assert click.validation.posterior_predictive is not None
            if validation_suite == "validation.full" and not withhold_scalar:
                assert fresh.validation.prior_sensitivity is not None
                assert click.validation.prior_sensitivity is not None


@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
def test_static_known_truth_mixture(validation, validation_suite: str | None) -> None:
    data = static_data(int(STATIC["n"]))
    posterior = static_posterior()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=BAYES,
            refute=validation,
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        seed=1,
    )
    click = require_success(antecedent.estimate(prepared))
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=validation_suite,
        expected_ate=float(STATIC["expected_effect_given_identified"]),
        tolerance=float(STATIC["effect_abs_tolerance"]),
        unidentified_mass=float(STATIC["expected_unidentified_mass"]),
        expect_ppc=True,
        withhold_scalar=True,
    )


def _assert_frequentist_mixture_contract(
    fresh,
    click,
    prepared,
    *,
    validation_suite: str | None,
    expected_ate: float,
    unidentified_mass: float,
) -> None:
    assert prepared.evidence_status == "licensed"
    assert fresh.evidence_status == click.evidence_status == "licensed"
    assert fresh.plan.validation_suite == click.plan.validation_suite == validation_suite
    assert fresh.ate is None and click.ate is None
    assert fresh.posterior is None and click.posterior is None
    assert fresh.identification.status == click.identification.status == "GraphDependent"
    assert any(
        f"unidentified_mass={unidentified_mass}" in diagnostic for diagnostic in fresh.diagnostics
    )
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    if validation_suite is None:
        assert not fresh.validation.reports
        assert not click.validation.reports
    else:
        assert fresh.validation.ran and click.validation.ran
        assert fresh.validation.reports and click.validation.reports


@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
def test_static_known_truth_mixture_frequentist(validation, validation_suite: str | None) -> None:
    data = static_data(int(STATIC["n"]))
    posterior = static_posterior()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=replace(FREQ, bootstrap=0),
            refute=validation,
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=FREQ,
        refute=validation,
        seed=1,
    )
    click = require_success(antecedent.estimate(prepared))
    _assert_frequentist_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=validation_suite,
        expected_ate=float(STATIC["expected_effect_given_identified"]),
        unidentified_mass=float(STATIC["expected_unidentified_mass"]),
    )


def test_static_known_truth_mixture_refresh_reuses_identification() -> None:
    data = static_data(int(STATIC["n"]))
    posterior = static_posterior()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=1,
    )
    click = require_success(antecedent.estimate(prepared))
    refreshed = require_success(antecedent.estimate(antecedent.rebind(prepared, data=data)))
    expected = float(STATIC["expected_effect_given_identified"])
    tolerance = float(STATIC["effect_abs_tolerance"])
    assert click.ate is None and refreshed.ate is None
    assert click.posterior is not None and refreshed.posterior is not None
    assert click.posterior.effect_mean == pytest.approx(expected, abs=tolerance)
    assert refreshed.posterior.effect_mean == pytest.approx(expected, abs=tolerance)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    assert any(
        diagnostic.startswith("exec.identify.cached") for diagnostic in refreshed.diagnostics
    )


@pytest.mark.parametrize(
    "query",
    [
        antecedent.PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        antecedent.SustainedEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
    ],
    ids=["pulse", "sustained"],
)
def test_temporal_known_truth_mixture(query) -> None:
    data = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    posterior = temporal_posterior()
    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=BAYES,
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=11,
    )
    click = require_success(antecedent.estimate(prepared))
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=None,
        expected_ate=float(TEMPORAL["expected_effect_given_identified"]),
        tolerance=float(TEMPORAL["effect_abs_tolerance"]),
        unidentified_mass=float(TEMPORAL["expected_unidentified_mass"]),
        expect_ppc=False,
    )


def test_temporal_multistep_sustained_known_truth_mixture() -> None:
    data = white_noise_pulse_series(int(TEMPORAL_MULTI["n"]), int(TEMPORAL_MULTI["seed"]))
    posterior = temporal_posterior()
    query = antecedent.SustainedEffect(
        treatment="pressure",
        outcome="defect",
        window=(-2, -1),
        horizon_steps=1,
        active_level=1.0,
    )
    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=BAYES,
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=11,
    )
    click = require_success(antecedent.estimate(prepared))
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=None,
        expected_ate=float(TEMPORAL_MULTI["expected_effect_given_identified"]),
        tolerance=float(TEMPORAL_MULTI["effect_abs_tolerance"]),
        unidentified_mass=float(TEMPORAL_MULTI["expected_unidentified_mass"]),
        expect_ppc=False,
    )


@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS[1:])
def test_temporal_multistep_sustained_validation(validation, validation_suite) -> None:
    data = white_noise_pulse_series(int(TEMPORAL_MULTI["n"]), int(TEMPORAL_MULTI["seed"]))
    options = dict(
        discovery=temporal_posterior(),
        query=antecedent.SustainedEffect(
            treatment="pressure",
            outcome="defect",
            window=(-2, -1),
        ),
        refute=validation,
    )
    fresh = require_success(
        antecedent.analyze(
            data, inference=BAYES, execution=antecedent.Execution(seed=11), **options
        )
    )
    prepared = _prepare(data, inference=BAYES, seed=11, **options)
    click = require_success(antecedent.estimate(prepared))
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=validation_suite,
        expected_ate=float(TEMPORAL_MULTI["expected_effect_given_identified"]),
        tolerance=float(TEMPORAL_MULTI["effect_abs_tolerance"]),
        unidentified_mass=float(TEMPORAL_MULTI["expected_unidentified_mass"]),
        expect_ppc=False,
    )
    assert all(math.isfinite(report.refuted_ate) for report in fresh.validation.reports)

    # Temporal results expose each mechanism separately as validation reports;
    # the scalar static-DTO predictive convenience fields do not apply here.
    for result in [fresh, click]:
        names = {report.refuter for report in result.validation.reports}
        assert any(name.startswith("prior_predictive.mechanism.") for name in names)
        assert any(name.startswith("posterior_predictive.mechanism.") for name in names)
        if validation == "full":
            assert "prior_sensitivity" in names
            assert "placebo.treatment" in names


_MEDIATION_VALIDATIONS = [
    pytest.param(False, None, id="none"),
    pytest.param("cheap", "temporal.mediation.cheap", id="cheap"),
    pytest.param("full", "temporal.mediation.full", id="full"),
]


@pytest.mark.parametrize("validation, validation_suite", _MEDIATION_VALIDATIONS)
def test_temporal_mediation_known_truth_mixture(validation, validation_suite: str | None) -> None:
    data = temporal_mediation_series(int(TEMPORAL_MEDIATION["n"]))
    posterior = temporal_mediation_posterior()
    query = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated")
    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=BAYES,
            refute=validation,
            execution=antecedent.Execution(seed=int(TEMPORAL_MEDIATION["seed"])),
        )
    )
    prepared = _prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        seed=int(TEMPORAL_MEDIATION["seed"]),
    )
    click = require_success(antecedent.estimate(prepared))
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=validation_suite,
        expected_ate=float(TEMPORAL_MEDIATION["expected_effect_given_identified"]),
        tolerance=float(TEMPORAL_MEDIATION["effect_abs_tolerance"]),
        unidentified_mass=float(TEMPORAL_MEDIATION["expected_unidentified_mass"]),
        expect_ppc=False,
    )
    assert any("identify.dbn_posterior.per_atom_horizon" in d for d in fresh.diagnostics)
    assert fresh.mediation is not None
    assert fresh.mediation.mediated == pytest.approx(fresh.ate, abs=1e-12)
    if validation_suite is not None:
        assert fresh.validation.reports and click.validation.reports
        assert any("refute.envelope.effect_mixture" in d for d in click.diagnostics)


def test_dbn_posterior_response_curve_runs() -> None:
    data = white_noise_pulse_series(64, 1)
    result = require_success(
        antecedent.analyze(
            data,
            discovery=temporal_posterior(),
            query=antecedent.ResponseCurve(
                treatment="pressure",
                outcome="defect",
                grid=[0.0, 1.0],
                horizons=[1],
                policy="pulse",
            ),
            inference=BAYES,
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.evidence_status == "licensed"
    assert result.response is not None
    # defect_t = 0.9 * pressure_{t-1}: the identified-mixture pulse response at level a is 0.9 * a
    # (a = 0 is the baseline), so the curve over the grid [0, 1] is [0, 0.9] (observed deviation
    # 1e-4).
    assert result.response.points == ((0.0, 1.0), (1.0, 1.0))
    values = [row[0] for row in result.response.values]
    assert values == pytest.approx([0.0, 0.9], abs=0.005)


def test_frequentist_dbn_mediation_mixture_follows_the_latency_tier() -> None:
    """Single-horizon Frequentist DBN mediation is reachable from Python.

    Atoms share one circular-block bootstrap for the aggregate SE, so the
    replicate count follows the latency tier: interactive runs none and
    withholds the SE, standard and an omitted tier run the omitted budget.
    """
    data = temporal_mediation_series(int(TEMPORAL_MEDIATION["n"]))
    posterior = temporal_mediation_posterior()
    query = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated")
    seed = int(TEMPORAL_MEDIATION["seed"])
    expected = float(TEMPORAL_MEDIATION["expected_effect_given_identified"])
    tolerance = float(TEMPORAL_MEDIATION["effect_abs_tolerance"])
    shared_block = "estimate.dbn_posterior.mediation.shared_block"

    interactive = require_success(
        antecedent.estimate(
            _prepare(data, discovery=posterior, query=query, inference=FREQ, seed=seed)
        )
    )
    assert interactive.ate == pytest.approx(expected, abs=tolerance)
    assert interactive.estimate.se_bootstrap is None
    assert not any(d.startswith(shared_block) for d in interactive.diagnostics)

    standard = require_success(
        antecedent.estimate(
            _prepare(
                data,
                discovery=posterior,
                query=query,
                inference=FREQ,
                seed=seed,
                profile="standard",
            )
        )
    )
    assert standard.ate == pytest.approx(interactive.ate, abs=1e-12)
    assert standard.estimate.se_bootstrap is not None
    assert math.isfinite(standard.estimate.se_bootstrap) and standard.estimate.se_bootstrap > 0
    assert any(d.startswith(shared_block) for d in standard.diagnostics)

    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=FREQ,
            refute=False,
            execution=antecedent.Execution(seed=seed),
        )
    )
    assert fresh.ate == pytest.approx(interactive.ate, abs=1e-12)
    assert fresh.estimate.se_bootstrap is not None


@pytest.mark.parametrize("kind", ["pulse", "sustained"])
def test_frequentist_dbn_pulse_sustained_mixture_follows_the_latency_tier(kind) -> None:
    """Frequentist DBN-posterior Pulse / Sustained is reachable from Python.

    Atoms share one circular-block bootstrap for the aggregate SE, so the
    replicate count follows the latency tier (interactive runs none and
    withholds the SE; an omitted tier keeps the omitted replicate budget), and
    the structural masses ride the result.
    """
    data = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    posterior = temporal_posterior()
    effect = antecedent.PulseEffect if kind == "pulse" else antecedent.SustainedEffect
    query = effect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
    )
    expected = float(TEMPORAL["expected_effect_given_identified"])
    tolerance = float(TEMPORAL["effect_abs_tolerance"])
    unidentified = float(TEMPORAL["expected_unidentified_mass"])

    def assert_structural(result) -> None:
        assert result.structural_weight_basis == "posterior_probability"
        assert result.structural_unidentified_mass == pytest.approx(unidentified, abs=1e-12)
        assert result.structural_identified_mass == pytest.approx(1.0 - unidentified, abs=1e-12)
        assert result.structural_unevaluable_mass == 0.0
        assert result.identification.status == "GraphDependent"

    interactive = require_success(
        antecedent.estimate(
            _prepare(
                data,
                discovery=posterior,
                query=query,
                inference=FREQ,
                refute=False,
                seed=11,
            )
        )
    )
    assert interactive.ate == pytest.approx(expected, abs=tolerance)
    assert interactive.estimate.se_bootstrap is None
    assert_structural(interactive)

    standard = require_success(
        antecedent.estimate(
            _prepare(
                data,
                discovery=posterior,
                query=query,
                inference=FREQ,
                refute=False,
                seed=11,
                profile="standard",
            )
        )
    )
    assert standard.ate == pytest.approx(interactive.ate, abs=1e-12)
    assert standard.estimate.se_bootstrap is not None
    assert math.isfinite(standard.estimate.se_bootstrap) and standard.estimate.se_bootstrap > 0
    assert any(
        d.startswith("estimate.dbn_posterior.frequentist") and "shared circular-block" in d
        for d in standard.diagnostics
    )
    assert_structural(standard)

    fresh = require_success(
        antecedent.analyze(
            data,
            discovery=posterior,
            query=query,
            inference=FREQ,
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    assert fresh.ate == pytest.approx(interactive.ate, abs=1e-12)
    assert fresh.estimate.se_bootstrap is not None
    assert_structural(fresh)


def test_frequentist_discovered_dbn_posterior_pulse_is_routed() -> None:
    """``discovery=DbnPosterior()`` with Frequentist inference no longer refuses."""
    data = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    query = antecedent.PulseEffect(
        treatment="pressure", outcome="defect", treatment_lag=1, horizon_steps=1
    )
    result = require_success(
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.DbnPosterior(max_lag=1).run(
                data, execution=antecedent.Execution(seed=11)
            ),
            query=query,
            inference=replace(FREQ, bootstrap=24),
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    # The DGP is defect_t = 0.9 * pressure_{t-1} with no noise, and every identified atom
    # recovers that law, so the mixture over identified atoms is 0.9 (observed deviation 2e-16).
    assert result.ate == pytest.approx(float(TEMPORAL["expected_effect_given_identified"]), abs=1e-6)
    assert result.estimate.se_bootstrap is not None and result.estimate.se_bootstrap > 0
    assert result.structural_weight_basis == "posterior_probability"
    masses = (
        result.structural_identified_mass
        + result.structural_unidentified_mass
        + result.structural_unevaluable_mass
    )
    assert masses == pytest.approx(1.0, abs=1e-9)
