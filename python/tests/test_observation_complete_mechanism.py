"""``observation.Complete()`` must behave exactly like no observation mechanism.

``Complete()`` is the documented "outcome is observed directly" spelling
(``observation.py``'s ``_stage_inputs`` already treats it as equivalent to
``None``). ``_analyze.py`` used to read ``query.observation`` directly and
branch on ``mechanism is not None``, so a query carrying ``Complete()`` was
routed onto the observation-aware path -- which does not support ``Complete``
-- and died with ``unsupported observation mechanism Complete``. It also
spuriously rejected ``refute=`` and ``estimator_config=`` for such queries.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import observation
from antecedent.errors import CausalValueError


def _curve_data(seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    confounder = rng.normal(size=400)
    treatment = 0.7 * confounder + rng.normal(size=400)
    outcome = 2.0 * treatment + confounder + rng.normal(scale=0.2, size=400)
    return {"x": confounder, "a": treatment, "y": outcome}


_GRAPH = [("x", "a"), ("x", "y"), ("a", "y")]


def test_complete_observation_runs_through_analyze_like_no_mechanism():
    data = _curve_data()

    bare = antecedent.analyze(
        data,
        query=antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]),
        graph=_GRAPH,
    )
    explicit_complete = antecedent.analyze(
        data,
        query=antecedent.ResponseCurve(
            "a", "y", grid=[-0.5, 0.0, 0.5], observation=observation.Complete()
        ),
        graph=_GRAPH,
    )

    assert explicit_complete.provenance["operation_id"] == bare.provenance["operation_id"]
    assert explicit_complete.identification.status == bare.identification.status
    assert list(explicit_complete.response.points) == list(bare.response.points)
    assert np.allclose(
        np.asarray(explicit_complete.response.values),
        np.asarray(bare.response.values),
    )


def test_complete_observation_accepts_refute_like_no_mechanism():
    data = _curve_data(seed=172)
    query = antecedent.ResponseCurve(
        "a", "y", grid=[-0.4, 0.0, 0.4], observation=observation.Complete()
    )

    result = antecedent.analyze(
        {"a": data["a"], "y": data["y"]},
        query=query,
        graph=[("a", "y")],
        refute="cheap",
        seed=9,
    )

    assert result.validation is not None
    assert [check.id for check in result.validation.checks] == [
        "overlap.support",
        "data.subset",
        "scalar_ate_refuters",
    ]


def test_complete_observation_accepts_estimator_config_like_no_mechanism():
    data = _curve_data(seed=18)
    query = antecedent.ResponseCurve(
        "a", "y", grid=[-0.5, 0.0, 0.5], observation=observation.Complete()
    )

    result = antecedent.analyze(
        {"a": data["a"], "y": data["y"]},
        query=query,
        graph=[("a", "y")],
        estimator_config={"bandwidth": 0.3},
    )

    assert result.response is not None


def test_complete_observation_with_assumptions_raises_same_error_as_no_mechanism():
    data = _curve_data(seed=1)

    with pytest.raises(ValueError, match="observation_assumptions require an explicit observation mechanism"):
        antecedent.analyze(
            {"a": data["a"], "y": data["y"]},
            query=antecedent.ResponseCurve(
                "a",
                "y",
                grid=[-0.5, 0.0, 0.5],
                observation=observation.Complete(),
                observation_assumptions=[observation.IndependentGiven(())],
            ),
            graph=[("a", "y")],
        )

    with pytest.raises(ValueError, match="observation_assumptions require an explicit observation mechanism"):
        antecedent.analyze(
            {"a": data["a"], "y": data["y"]},
            query=antecedent.ResponseCurve(
                "a",
                "y",
                grid=[-0.5, 0.0, 0.5],
                observation_assumptions=[observation.IndependentGiven(())],
            ),
            graph=[("a", "y")],
        )


def test_complete_observation_still_rejected_by_observation_stage_helpers():
    """``Complete()`` is only equivalent to no-mechanism in `analyze()`.

    The dedicated observation stage helpers (`adjusted_outcome`,
    `gaussian_log_likelihood`) still fail closed on `Complete()` -- they
    require a real non-complete mechanism, per `_stage_inputs`.
    """

    query = antecedent.ResponseCurve(
        "a", "y", grid=[0.0, 1.0], observation=observation.Complete()
    )
    with pytest.raises(CausalValueError, match="non-complete"):
        observation.adjusted_outcome({}, query)
