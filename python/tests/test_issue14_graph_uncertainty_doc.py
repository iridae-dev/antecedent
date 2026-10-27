"""Live check that a compact marketing-style SCM yields ~50% unidentified mass.

The practitioner page and index link are documentation, not something this
test pins by phrase.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.outcomes import require_success


def test_vignette_unidentified_mass_matches_live_analyze():
    """Earn the ~50% unidentified-mass claim against a compact marketing-style SCM."""
    import antecedent

    seed = 7
    n = 400
    rng = np.random.default_rng(seed)
    market_demand_index = rng.normal(size=n)
    paid_search_spend_kgbp = 20.0 + 4.0 * market_demand_index + 5.0 * rng.normal(size=n)
    qualified_pipeline_kgbp = (
        80.0 + 1.5 * paid_search_spend_kgbp + 12.0 * market_demand_index + 8.0 * rng.normal(size=n)
    )
    data = {
        "market_demand_index": market_demand_index,
        "paid_search_spend_kgbp": paid_search_spend_kgbp,
        "qualified_pipeline_kgbp": qualified_pipeline_kgbp,
    }

    result = require_success(
        antecedent.analyze(
            data=data,
            discovery=antecedent.discovery.ExactDagPosterior().run(data, seed=seed),
            query=antecedent.AverageEffect(
                treatment="paid_search_spend_kgbp",
                outcome="qualified_pipeline_kgbp",
            ),
            inference=antecedent.Bayesian(n_draws=80, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=seed),
        )
    )
    assert result.posterior is not None
    mass = float(result.posterior.unidentified_mass)
    # Notebook pins 0.5 on the full vignette; compact replay should stay near it.
    assert mass == pytest.approx(0.5, abs=0.15), (
        f"expected ~0.5 unidentified mass for marketing-style MEC, got {mass}"
    )
