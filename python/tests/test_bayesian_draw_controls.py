"""``Bayesian(draw_controls="measured")`` bounds the draws behind a Bayesian interval.

The draw count must reach the backend's supported minimum (refused at build, naming
it, never rewritten), the endpoint Monte Carlo error is reported beside the interval,
and the draw count and controls are part of the inference identity. A route that
does not enforce the controls refuses them rather than ignore them.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError

STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y")]
EFFECT = 0.4


def _linear_scm(n: int = 1000, seed: int = 3272793) -> dict[str, np.ndarray]:
    """Confounded binary treatment with a linear outcome: the adjusted effect is 0.4."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(0.8 - z))).astype(float)
    y = EFFECT * t + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _analyze(data: dict[str, np.ndarray], **kwargs: Any) -> Any:
    return ant.analyze(
        data,
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Bayesian(**kwargs),
        refute="none",
    )


@pytest.mark.parametrize(
    ("backend", "draws", "minimum"),
    [
        ("laplace", 200, "2000"),
        ("conjugate", 1999, "2000"),
        ("hmc", 2000, "3000"),
    ],
)
def test_a_draw_count_below_the_supported_minimum_is_refused_naming_it(
    backend: str, draws: int, minimum: str
) -> None:
    with pytest.raises(CausalUnsupportedError) as caught:
        _analyze(_linear_scm(), backend=backend, n_draws=draws, draw_controls="measured")
    assert caught.value.reason_code == "posterior_draws_below_supported"
    assert f"supported minimum is {minimum}" in str(caught.value)


def test_the_controls_are_asked_for_not_assumed() -> None:
    # Without them a small draw count runs as released.
    small = _analyze(_linear_scm(), backend="conjugate", n_draws=200)
    assert small.answer.value == pytest.approx(EFFECT, abs=0.35)
    # With them the default draw count is below the minimum too.
    with pytest.raises(CausalUnsupportedError) as caught:
        _analyze(_linear_scm(), draw_controls="measured")
    assert caught.value.reason_code == "posterior_draws_below_supported"


def test_supplied_draws_report_endpoint_error_and_enter_the_identity() -> None:
    data = _linear_scm()
    plain = _analyze(data, n_draws=3000)
    controlled = _analyze(data, n_draws=3000, draw_controls="measured")
    other_draws = _analyze(data, n_draws=4000, draw_controls="measured")
    assert controlled.answer.value == plain.answer.value
    notes = [line.split(":", 1)[0] for line in controlled.inspect().diagnostics]
    assert "estimate.bayesian.draw_controls" in notes
    assert "estimate.bayesian.draw_controls" not in [
        line.split(":", 1)[0] for line in plain.inspect().diagnostics
    ]
    identities = {
        result.inspect().to_dict()["inference_binding_id"]
        for result in (plain, controlled, other_draws)
    }
    assert len(identities) == 3
    repeat = _analyze(data, n_draws=3000, draw_controls="measured")
    assert (
        repeat.inspect().to_dict()["inference_binding_id"]
        == controlled.inspect().to_dict()["inference_binding_id"]
    )
    assert repeat.answer == controlled.answer


def test_a_route_that_does_not_enforce_the_controls_refuses_them() -> None:
    cpdag = ant.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    with pytest.raises(CausalUnsupportedError) as caught:
        ant.analyze(
            _linear_scm(),
            graph=cpdag,
            query=ant.AverageEffect("t", "y"),
            inference=ant.Bayesian(n_draws=3000, draw_controls="measured"),
        )
    assert caught.value.reason_code == "invalid_argument"


def test_unknown_draw_controls_are_invalid() -> None:
    with pytest.raises(CausalValueError, match="unknown Bayesian draw_controls 'strict'"):
        _analyze(_linear_scm(200), draw_controls="strict")  # type: ignore[arg-type]
