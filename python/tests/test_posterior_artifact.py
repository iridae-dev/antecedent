"""posterior artifact encode/decode + no-default-artifact (BACKLOG E)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
from antecedent.outcomes import require_success

from _scm import confounded_scm


def test_bayesian_default_omits_posterior_artifact():
    data, edges = confounded_scm(n=400, seed=11)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=128),
            execution=antecedent.Execution(seed=3),
            refute=False,
        )
    )
    assert result.posterior is not None
    assert result.posterior.artifact is None
    assert result.posterior.n_draws == 128
    assert result.posterior.effect_mean is not None
    assert math.isfinite(result.posterior.effect_mean)


def test_posterior_artifact_round_trip_opt_in():
    data, edges = confounded_scm(n=400, seed=11)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=128),
            execution=antecedent.Execution(seed=3),
            refute=False,
        )
    )
    assert result.posterior is not None
    assert result.posterior.n_draws == 128
    assert result.posterior.effect_mean is not None

    exported = result.export(quantity="posterior")
    art = require_success(antecedent.load(exported, expected=exported.expectation))
    assert art.n_draws == 128
    effect_idx = art.quantity_names.index("ate") if "ate" in art.quantity_names else -1
    assert abs(art.mean[effect_idx] - result.posterior.effect_mean) < 1e-12
    assert art.backend_id
    draws = np.column_stack([art.quantity(name).draws for name in art.quantity_names])
    assert draws.size == art.n_draws * len(art.quantity_names)

    again = art.export()
    art2 = require_success(antecedent.load(again, expected=again.expectation))
    assert art2.n_draws == art.n_draws
    for name in art.quantity_names:
        np.testing.assert_array_equal(art2.quantity(name).draws, art.quantity(name).draws)
    assert bytes(again) == bytes(exported)
    assert art2.mean == art.mean
    assert abs(art2.mean[effect_idx] - 2.0) < 0.5


def test_posterior_artifact_payload_size_vs_summaries():
    data, edges = confounded_scm(n=200, seed=5)
    summary = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=2000),
            execution=antecedent.Execution(seed=5),
            refute=False,
        )
    )
    full = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=2000),
            execution=antecedent.Execution(seed=5),
            refute=False,
        )
    )
    assert summary.posterior.artifact is None
    full_artifact = full.export(quantity="posterior")
    art = require_success(antecedent.load(full_artifact))
    draws = np.column_stack([art.quantity(name).draws for name in art.quantity_names])
    assert draws.size == art.n_draws * len(art.quantity_names)
    assert draws.size > 0
    assert len(bytes(full_artifact)) > len(art.mean) * 8
    assert full.posterior.effect_mean == summary.posterior.effect_mean
    assert full.posterior.n_draws == summary.posterior.n_draws


def test_posterior_artifact_draws_are_a_float64_array_matching_the_buffer_view():
    """`draws` is a NumPy array, not a boxed-float list rebuilt on every access."""
    art = antecedent.inference.PosteriorArtifact(
        3,
        [0.5],
        [0.25],
        [0.2625],
        [0.7375],
        [0.25, 0.5, 0.75],
        "test",
        "identified",
        ["ate"],
    )
    draws = art.draws
    assert isinstance(draws, np.ndarray)
    assert draws.dtype == np.float64
    np.testing.assert_array_equal(draws, [0.25, 0.5, 0.75])
    # The zero-copy buffer view and the attribute agree element for element.
    np.testing.assert_array_equal(np.asarray(art), draws)
