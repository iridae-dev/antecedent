"""Manufacturing-style Bayesian temporal pulse dual (P0)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def test_manufacturing_bayesian_pulse_recovers_effect():
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = require_success(
        antecedent.analyze(
            {"pressure": pressure, "defect": defect},
            graph=[("pressure", 1, "defect", 0)],
            query=antecedent.PulseEffect(
                treatment="pressure",
                outcome="defect",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Bayesian(n_draws=256),
            refute=False,
            execution=antecedent.Execution(seed=42),
        )
    )
    assert result.posterior is not None
    # The series is noiseless, so the posterior is concentrated on 0.9 (sd 4e-4, mean error 5e-5).
    assert abs(result.posterior.effect_mean - 0.9) < 0.005
    assert abs(result.ate - result.posterior.effect_mean) < 1e-12
    assert np.isfinite(result.posterior.p_below_zero)
    assert result.estimate.estimator_id == "bayesian.temporal.gcomp"
    assert result.identification.method  # non-empty
    # Full draw artifacts are opt-in on static analyze; temporal defaults to summaries.
    assert result.posterior.n_draws is not None and result.posterior.n_draws > 0
    assert result.posterior.artifact is None


def test_manufacturing_bayesian_sustained_recovers_effect():
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = require_success(
        antecedent.analyze(
            {"pressure": pressure, "defect": defect},
            graph=[("pressure", 1, "defect", 0)],
            query=antecedent.SustainedEffect(
                treatment="pressure",
                outcome="defect",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Bayesian(n_draws=256),
            refute=False,
            execution=antecedent.Execution(seed=42),
        )
    )
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - 0.9) < 0.005
    assert result.evidence_status == "licensed"


def _pulse_query_and_data(n: int = 200):
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]
    data = {"pressure": pressure, "defect": defect}
    graph = [("pressure", 1, "defect", 0)]
    query = antecedent.PulseEffect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
    )
    return data, graph, query


def test_temporal_pulse_bayesian_composed_prior_is_hydrated():
    """A composed external prior reaches the temporal Bayesian executor.

    `Bayesian(prior_from=ComposedPrior(...))` used to be refused on temporal
    queries because the native temporal entry points took no prior transfer.
    They now do: a tight external prior centred away from the data pulls the
    posterior mean toward it and changes the bound inference identity, so the
    declared prior is part of the claim rather than silently dropped.
    """
    data, graph, query = _pulse_query_and_data()
    tight = antecedent.priors.compose_external_priors(
        [
            antecedent.priors.ExternalPriorSourceSpec(
                id="s1", mean=(5.0, 5.0), variance=(0.001, 0.001)
            )
        ],
        weights=[1.0],
    )

    def run(inference):
        return require_success(
            antecedent.analyze(
                data,
                graph=graph,
                query=query,
                inference=inference,
                refute=False,
                execution=antecedent.Execution(seed=42),
            )
        )

    flat = run(antecedent.Bayesian(n_draws=64))
    informed = run(antecedent.Bayesian(n_draws=64, prior_from=tight))
    assert flat.posterior is not None and informed.posterior is not None
    assert abs(flat.posterior.effect_mean - 0.9) < 0.005
    # The prior N(5, 0.001) on the coefficients dominates the 0.9 data slope: the posterior
    # mean is 4.54, between the data and the prior centre, nowhere near 0.9.
    assert 4.0 < informed.posterior.effect_mean < 5.0, informed.posterior.effect_mean
    contract = antecedent.artifacts.loads(bytes(informed.export())).contract
    flat_contract = antecedent.artifacts.loads(bytes(flat.export())).contract
    assert (
        contract["identities"]["inference_binding"]
        != flat_contract["identities"]["inference_binding"]
    )


def test_temporal_pulse_bayesian_prior_artifact_and_mapping_are_hydrated():
    """A posterior artifact plus its mapping transfers into a temporal pulse.

    A mapping with no artifact to read is a permanent refusal: it names which
    estimand of an artifact to use and has nothing to map on its own.
    """
    data, graph, query = _pulse_query_and_data()
    source = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=antecedent.Bayesian(n_draws=64),
            refute=False,
            execution=antecedent.Execution(seed=42),
        )
    )
    assert source.posterior is not None
    artifact = bytes(source.export(quantity="posterior"))

    transferred = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=antecedent.Bayesian(
                n_draws=64,
                prior_from=artifact,
                mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
            ),
            refute=False,
            execution=antecedent.Execution(seed=42),
        )
    )
    assert transferred.posterior is not None
    # The source posterior is centred on the data slope 0.9 with sd 4e-4, so hydrating it
    # keeps the effect there (observed error 1e-5).
    assert transferred.posterior.effect_mean == pytest.approx(0.9, abs=0.005)
    flat = antecedent.artifacts.loads(bytes(source.export())).contract
    bound = antecedent.artifacts.loads(bytes(transferred.export())).contract
    assert bound["identities"]["inference_binding"] != flat["identities"]["inference_binding"]

    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="prior_mapping"):
        require_success(
            antecedent.analyze(
                data,
                graph=graph,
                query=query,
                inference=antecedent.Bayesian(
                    n_draws=64,
                    mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
                ),
                refute=False,
                execution=antecedent.Execution(seed=42),
            )
        )


def test_temporal_pulse_bayesian_without_composed_prior_still_works():
    """Control case: plain `Bayesian(...)` (no `prior_from`/`mapping`) is unaffected by the
    new guard — it is the explicitly documented supported combination.
    """
    data, graph, query = _pulse_query_and_data()

    result = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=antecedent.Bayesian(n_draws=64),
            refute=False,
            execution=antecedent.Execution(seed=42),
        )
    )
    assert result.posterior is not None


def _pulse_with_residual_ar(coefficient: float, n: int = 300, seed: int = 5):
    rng = np.random.default_rng(seed)
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = coefficient * noise[t - 1] + rng.normal(scale=0.3)
    defect = np.zeros(n)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1] + noise[t]
    return pressure, defect


def _posterior_predictive_failure(pressure, defect, refute: str) -> str | None:
    result = require_success(
        antecedent.analyze(
            {"pressure": pressure, "defect": defect},
            graph=[("pressure", 1, "defect", 0)],
            query=antecedent.PulseEffect(
                treatment="pressure",
                outcome="defect",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Bayesian(n_draws=256),
            refute=refute,
            execution=antecedent.Execution(seed=42),
        )
    )
    (report,) = [r for r in result.validation.reports if r.refuter == "posterior_predictive"]
    return report.failure_condition


def test_the_serial_discrepancy_of_the_full_suite_reports_lag_one_residual_autocorrelation():
    pressure, defect = _pulse_with_residual_ar(0.85)
    # Independent oracle: lag-1 residual autocorrelation sum(r_t r_{t-1}) / sum(r_t^2) of the
    # regression of the outcome on the lagged treatment.
    y = defect[1:]
    design = np.column_stack([np.ones(len(y)), pressure[:-1]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    r = y - design @ beta
    oracle = float(r[1:] @ r[:-1] / (r @ r))
    assert oracle > 0.8

    text = _posterior_predictive_failure(pressure, defect, "full")
    assert text is not None
    assert (
        "posterior predictive serial-dependence check failed: lag-1 residual autocorrelation"
        in text
    )
    observed = float(text.split("lag-1 residual autocorrelation ")[1].split(" vs replicated")[0])
    assert observed == pytest.approx(oracle, abs=0.02)  # measured gap 0.0115
    # Replicated rows are iid, so their lag-1 autocorrelation is about -1/n.
    replicated = float(text.split("vs replicated ")[1].split(" (p=")[0])
    assert abs(replicated) < 0.05
    # The temporal Bayesian interval is already tempered for this dependence: the report says so.
    assert "already tempered for this dependence" in text
    kappa = float(text.split("kappa=")[1].split(",")[0])
    # The long-run-variance inflation of an AR(1) residual is (1 + rho) / (1 - rho) = 13.8 at the
    # oracle rho 0.865; the kernel estimate (16.3) sits within 30% of it, far above 1.
    assert kappa == pytest.approx((1.0 + oracle) / (1.0 - oracle), rel=0.3)

    # The cheap suite has no serial axis, so the same data records no serial finding.
    cheap = _posterior_predictive_failure(pressure, defect, "cheap")
    assert cheap is None or "serial-dependence" not in cheap


def test_the_serial_discrepancy_is_silent_on_exchangeable_residuals():
    pressure, defect = _pulse_with_residual_ar(0.0, seed=11)
    text = _posterior_predictive_failure(pressure, defect, "full")
    assert text is None or "serial-dependence" not in text
