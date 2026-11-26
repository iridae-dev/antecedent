"""The direct fitted-model contrasts warn and agree with the compiled counterfactual route."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success

EDGES = [("x", "m"), ("m", "y"), ("x", "y")]


def _data(n: int = 2000, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=3.0, scale=1.0, size=n)
    m = 1.0 + 0.8 * x + rng.normal(scale=0.3, size=n)
    y = 0.5 * m + 0.2 * x + rng.normal(scale=0.1, size=n)
    return {"x": x, "m": m, "y": y}


def _fit(data: dict[str, np.ndarray]):
    return ant.model.fit_gcm(list(data), list(data.values()), EDGES)


def test_counterfactual_contrasts_warn_and_match_the_compiled_route() -> None:
    data = _data()
    gcm = _fit(data)
    with pytest.warns(
        AntecedentDeprecationWarning,
        match="antecedent.counterfactual.FittedGcm.counterfactual_contrasts",
    ):
        direct = gcm.counterfactual_contrasts(
            ["m", "y"], {"x": 1.0}, {"x": 0.0}, execution=ant.Execution(seed=5)
        )
    compiled = require_success(
        ant.analyze(
            data,
            graph=EDGES,
            query=ant.Counterfactual("x", "m", also_outcomes=("y",)),
            refute="none",
            execution=ant.Execution(seed=5),
        )
    ).counterfactual_contrasts
    assert [c.outcome for c in compiled] == [c.outcome for c in direct.contrasts] == ["m", "y"]
    # The route selects mechanisms from the heterogeneity-capable candidate set, the direct fit
    # from the standard one; on additive linear mechanisms both select the same families.
    for route, kernel in zip(compiled, direct.contrasts, strict=True):
        np.testing.assert_allclose(route.unit_effects, kernel.unit_effects, rtol=1e-9)
        assert route.mean_effect == pytest.approx(kernel.mean_ite, rel=1e-9)


def test_joint_baseline_matches_the_compiled_route() -> None:
    data = _data()
    gcm = _fit(data)
    with pytest.warns(AntecedentDeprecationWarning):
        direct = gcm.counterfactual_contrasts(
            ["y"], {"x": 1.0, "m": 2.0}, {"x": 0.0, "m": 0.0}, execution=ant.Execution(seed=5)
        )
    compiled = require_success(
        ant.analyze(
            data,
            graph=EDGES,
            query=ant.Counterfactual(
                "x",
                "y",
                active_interventions={"m": 2.0},
                control_interventions={"m": 0.0},
            ),
            refute="none",
            execution=ant.Execution(seed=5),
        )
    )
    np.testing.assert_allclose(compiled.unit_effects, direct.contrast("y").unit_effects, rtol=1e-9)
