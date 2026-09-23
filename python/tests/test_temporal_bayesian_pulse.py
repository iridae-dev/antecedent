"""Manufacturing-style Bayesian temporal pulse dual (P0)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest


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
        return antecedent.analyze(
            data,
            graph=graph,
            query=query,
            inference=inference,
            refute=False,
            bootstrap=0,
            seed=42,
        )

    flat = run(antecedent.Bayesian(n_draws=64))
    informed = run(antecedent.Bayesian(n_draws=64, prior_from=tight))
    assert flat.posterior is not None and informed.posterior is not None
    assert abs(flat.posterior.effect_mean - 0.9) < 0.05
    assert informed.posterior.effect_mean > 1.5, informed.posterior.effect_mean
    contract = antecedent.artifacts.loads(informed.export()).contract
    flat_contract = antecedent.artifacts.loads(flat.export()).contract
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
    source = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        inference=antecedent.Bayesian(n_draws=64),
        refute=False,
        bootstrap=0,
        seed=42,
        return_posterior_artifact=True,
    )
    assert source.posterior is not None and source.posterior.artifact is not None
    artifact = bytes(source.posterior.artifact)

    transferred = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        inference=antecedent.Bayesian(
            n_draws=64,
            prior_from=artifact,
            mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
        ),
        refute=False,
        bootstrap=0,
        seed=42,
    )
    assert transferred.posterior is not None
    assert math.isfinite(transferred.posterior.effect_mean)
    flat = antecedent.artifacts.loads(source.export()).contract
    bound = antecedent.artifacts.loads(transferred.export()).contract
    assert bound["identities"]["inference_binding"] != flat["identities"]["inference_binding"]

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
