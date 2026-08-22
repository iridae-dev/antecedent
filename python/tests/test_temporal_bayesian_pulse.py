"""Manufacturing-style Bayesian temporal pulse dual (P0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_manufacturing_bayesian_pulse_recovers_effect():
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = antecedent.analyze(
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
        bootstrap=0,
        seed=42,
    )
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - 0.9) < 0.05
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

    result = antecedent.analyze(
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
        bootstrap=0,
        seed=42,
    )
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - 0.9) < 0.05
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


def test_temporal_pulse_bayesian_composed_prior_raises_unsupported():
    """Regression: `Bayesian(prior_from=ComposedPrior(...))` on a temporal query used to
    reach the native temporal entry points unguarded and blow up as a raw
    ``TypeError: analyze_temporal_discover() got an unexpected keyword argument
    'composed_prior'`` — the native temporal signatures (`python/src/temporal_api.rs`,
    `apply_temporal_inference`) never accepted `composed_prior`; only the static ATE path
    (`analyze_ate` / `analyze_ate_discover`) does. `_reject_unsupported_temporal` now catches
    this in Python before any native call is made, raising `CausalUnsupportedError` instead.
    """
    data, graph, query = _pulse_query_and_data()
    composed = antecedent.priors.compose_external_priors(
        [antecedent.priors.ExternalPriorSourceSpec(id="s1", mean=(0.0, 0.0), variance=(1.0, 1.0))],
        weights=[1.0],
    )

    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="composed_prior"):
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=antecedent.Bayesian(n_draws=64, prior_from=composed),
            refute=False,
            bootstrap=0,
            seed=42,
        )


def test_temporal_pulse_bayesian_prior_mapping_raises_unsupported():
    """Same regression as above, for `Bayesian(mapping=PriorMapping(...))` — also only
    accepted on the static ATE path, also used to reach a raw `TypeError` from the native
    temporal call instead of a clear, typed rejection.
    """
    data, graph, query = _pulse_query_and_data()

    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="prior_mapping"):
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=antecedent.Bayesian(
                n_draws=64,
                mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
            ),
            refute=False,
            bootstrap=0,
            seed=42,
        )


def test_temporal_pulse_bayesian_without_composed_prior_still_works():
    """Control case: plain `Bayesian(...)` (no `prior_from`/`mapping`) is unaffected by the
    new guard — it is the explicitly documented supported combination.
    """
    data, graph, query = _pulse_query_and_data()

    result = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        inference=antecedent.Bayesian(n_draws=64),
        refute=False,
        bootstrap=0,
        seed=42,
    )
    assert result.posterior is not None
