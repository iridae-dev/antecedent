"""Prior sensitivity on the Bayesian analyze path (Shared UX)."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.outcomes import require_success


def _confounded(n: int = 140, seed: int = 11):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_prior_sensitivity_on_refute_full():
    data, edges = _confounded()
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=48, backend="conjugate"),
            refute="full",
            execution=antecedent.Execution(seed=1),
        )
    )
    sens = result.validation.prior_sensitivity
    assert sens is not None
    assert len(sens.scales) >= 3
    assert sens.alphas is None
    assert len(sens.effect_means) == len(sens.scales)
    assert len(sens.effect_sds) == len(sens.scales)
    assert all(np.isfinite(m) for m in sens.effect_means)
    # The truth of y = 2 t + z + noise adjusted for z is 2. A tighter zero-centred prior shrinks
    # the effect toward 0 and loosens toward the least-squares value, so the means rise with the
    # scale to within a few hundredths of the truth, and the posterior spread falls.
    assert list(sens.scales) == sorted(sens.scales)
    assert list(sens.effect_means) == sorted(sens.effect_means)
    assert 1.6 < sens.effect_means[0] < 1.9
    assert abs(sens.effect_means[-1] - 2.0) < 0.06
    assert list(sens.effect_sds) == sorted(sens.effect_sds, reverse=True)


def test_prior_sensitivity_skipped_on_default_refute():
    data, edges = _confounded()
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=32),
            # "default refute" is spelled by leaving it unset; an explicit
            # refute=True is a TypeError because it never said which suite.
            execution=antecedent.Execution(seed=2),
        )
    )
    assert result.validation.prior_predictive is not None
    assert result.validation.prior_sensitivity is None
