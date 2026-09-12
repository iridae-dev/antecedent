"""1.1 numeric evidence for the licensed graph-posterior mixture cells.

Rust owns the known-truth DGP. These tests consume the same fixture so a
Python run sees E[τ|identified] and unidentified mass, not prepared-vs-fresh
smoke on ExactDagPosterior / DbnPosterior.
"""

from __future__ import annotations

import math

import pytest

antecedent = pytest.importorskip("antecedent")

from antecedent.errors import CausalUnsupportedError  # noqa: E402

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
) -> None:
    assert prepared.evidence_status == "licensed"
    assert fresh.evidence_status == click.evidence_status == "licensed"
    assert fresh.plan.validation_suite == click.plan.validation_suite == validation_suite
    assert fresh.ate == pytest.approx(expected_ate, abs=tolerance)
    assert click.ate == pytest.approx(expected_ate, abs=tolerance)
    assert fresh.posterior is not None and click.posterior is not None
    assert fresh.posterior.unidentified_mass == pytest.approx(unidentified_mass, abs=1e-12)
    assert click.posterior.unidentified_mass == pytest.approx(unidentified_mass, abs=1e-12)
    assert all(
        not diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics
    )
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
            if validation_suite == "validation.full":
                assert fresh.validation.prior_sensitivity is not None
                assert click.validation.prior_sensitivity is not None


@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
def test_static_known_truth_mixture(validation, validation_suite: str | None) -> None:
    data = static_data(int(STATIC["n"]))
    posterior = static_posterior()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    fresh = antecedent.analyze(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        bootstrap=0,
        seed=1,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    _assert_mixture_contract(
        fresh,
        click,
        prepared,
        validation_suite=validation_suite,
        expected_ate=float(STATIC["expected_effect_given_identified"]),
        tolerance=float(STATIC["effect_abs_tolerance"]),
        unidentified_mass=float(STATIC["expected_unidentified_mass"]),
        expect_ppc=True,
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
    assert fresh.ate == pytest.approx(expected_ate, abs=1e-8)
    assert click.ate == pytest.approx(expected_ate, abs=1e-8)
    assert fresh.posterior is None and click.posterior is None
    assert fresh.identification.status == click.identification.status == "GraphDependent"
    assert any(
        f"unidentified_mass={unidentified_mass}" in diagnostic for diagnostic in fresh.diagnostics
    )
    assert all(
        not diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics
    )
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
    fresh = antecedent.analyze(
        data,
        discovery=posterior,
        query=query,
        inference=FREQ,
        refute=validation,
        bootstrap=0,
        seed=1,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=FREQ,
        refute=validation,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
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
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    refreshed = prepared.refresh(data, seed=1)
    expected = float(STATIC["expected_effect_given_identified"])
    tolerance = float(STATIC["effect_abs_tolerance"])
    assert click.ate == pytest.approx(expected, abs=tolerance)
    assert refreshed.ate == pytest.approx(expected, abs=tolerance)
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
    fresh = antecedent.analyze(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        bootstrap=0,
        seed=11,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=11,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=11)
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
    fresh = antecedent.analyze(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        bootstrap=0,
        seed=11,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=False,
        seed=11,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=11)
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
        inference=BAYES,
        refute=validation,
        bootstrap=0,
        seed=11,
    )
    fresh = antecedent.analyze(data, **options)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(data, **options)
    click = prepared.estimate(data, seed=11)
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
    fresh = antecedent.analyze(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        bootstrap=0,
        seed=int(TEMPORAL_MEDIATION["seed"]),
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=posterior,
        query=query,
        inference=BAYES,
        refute=validation,
        seed=int(TEMPORAL_MEDIATION["seed"]),
        latency="interactive",
    )
    click = prepared.estimate(data, seed=int(TEMPORAL_MEDIATION["seed"]))
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


def test_dbn_posterior_response_curve_stays_refused() -> None:
    data = white_noise_pulse_series(64, 1)
    with pytest.raises(CausalUnsupportedError, match="response mixture"):
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
            bootstrap=0,
            seed=1,
        )
