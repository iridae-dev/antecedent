"""``Bayesian(likelihood=...)`` chooses the g-computation outcome model.

A Bernoulli or Poisson likelihood is fitted where Rust implements one (a
tabular ``AverageEffect`` on a ``Dag``), is part of the inference binding and
survives export and load; every other route refuses it by reason code rather
than fitting a Gaussian model. A Gaussian fit to a 0/1 or count outcome is
disclosed.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError

STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y")]
DISCLOSURE = "estimate.bayesian.gaussian_likelihood_discrete_outcome"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary(n: int = 3000, seed: int = 3) -> tuple[dict[str, np.ndarray], float]:
    """Binary outcome from a logit model; returns data and the in-sample risk difference."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < _sigmoid(z)).astype(float)
    y = (rng.uniform(size=n) < _sigmoid(-0.5 + 1.2 * t + 0.8 * z)).astype(float)
    truth = float(np.mean(_sigmoid(0.7 + 0.8 * z) - _sigmoid(-0.5 + 0.8 * z)))
    return {"t": t, "y": y, "z": z}, truth


def _count(n: int = 1500, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < _sigmoid(z)).astype(float)
    y = rng.poisson(np.exp(0.2 + 0.5 * t + 0.3 * z)).astype(float)
    return {"t": t, "y": y, "z": z}


def _continuous(n: int = 800, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < _sigmoid(z)).astype(float)
    y = 2.0 * t + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _codes(result: Any) -> list[str]:
    return [line.split(":", 1)[0] for line in result.inspect().diagnostics]


def _bayes(data: dict[str, np.ndarray], **kwargs: Any) -> Any:
    return ant.analyze(
        data,
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Bayesian(**kwargs),
        refute="none",
    )


# ----------------------------------------------------------------- likelihood


def test_logit_likelihood_recovers_the_risk_difference_of_a_binary_outcome() -> None:
    data, truth = _binary()
    logit = _bayes(data, likelihood="logit")
    assert logit.answer.value == pytest.approx(truth, abs=0.04)
    assert DISCLOSURE not in _codes(logit)
    gaussian = _bayes(data)
    assert gaussian.answer.value != logit.answer.value


@pytest.mark.parametrize("likelihood", ["logit", "probit", "poisson"])
def test_likelihood_reaches_rust_and_is_bound_into_the_inference_identity(likelihood: str) -> None:
    data = _count() if likelihood == "poisson" else _binary(600)[0]
    default = _bayes(data).inspect().to_dict()
    chosen_result = _bayes(data, likelihood=likelihood)
    chosen = chosen_result.inspect().to_dict()
    assert chosen["target_id"] == default["target_id"]
    assert chosen["inference_binding_id"] != default["inference_binding_id"]
    rust_name = {"logit": "bernoulli_logit", "probit": "bernoulli_probit", "poisson": "poisson_log"}
    # The fitted outcome model is the declared one, not a Gaussian regression.
    family = {"logit": "Bernoulli logit", "probit": "Bernoulli probit", "poisson": "Poisson log"}
    assert family[likelihood] in str(chosen["assumptions"])
    assert rust_name[likelihood] in str(chosen_result.inspect().to_dict()["contract"])

    loaded = ant.load(chosen_result.export())
    loaded_report = loaded.inspect().to_dict()
    assert loaded.acceptance.verified
    assert loaded_report["inference_binding_id"] == chosen["inference_binding_id"]
    assert loaded.answer == chosen_result.answer
    assert rust_name[likelihood] in str(loaded_report["contract"])


def test_gaussian_fit_to_a_discrete_outcome_is_disclosed() -> None:
    binary = _bayes(_binary(600)[0])
    assert DISCLOSURE in _codes(binary)
    count = _bayes(_count())
    assert DISCLOSURE in _codes(count)
    assert DISCLOSURE not in _codes(_bayes(_count(), likelihood="poisson"))
    assert DISCLOSURE not in _codes(_bayes(_continuous()))
    loaded = ant.load(binary.export())
    assert any(DISCLOSURE in line for line in loaded.inspect().to_dict()["diagnostics"])


@pytest.mark.parametrize(
    ("name", "call"),
    [
        (
            "conjugate",
            lambda d: ant.analyze(
                d,
                graph=STATIC_DAG,
                query=ant.AverageEffect("t", "y"),
                inference=ant.Bayesian(likelihood="logit", backend="conjugate"),
            ),
        ),
        (
            "cpdag",
            lambda d: ant.analyze(
                d,
                graph=ant.Cpdag.from_directed_undirected(
                    ["t", "y", "z"], [("z", "y"), ("t", "y")], [("z", "t")]
                ),
                query=ant.AverageEffect("t", "y"),
                inference=ant.Bayesian(likelihood="logit"),
            ),
        ),
        (
            "pulse",
            lambda d: ant.analyze(
                d,
                graph=[("t", 1, "y", 0)],
                query=ant.PulseEffect("t", "y", treatment_lag=1),
                inference=ant.Bayesian(likelihood="logit"),
            ),
        ),
    ],
)
def test_a_route_that_cannot_fit_the_likelihood_refuses_by_code(name: str, call: Any) -> None:
    data = _binary(400)[0]
    with pytest.raises(CausalUnsupportedError) as caught:
        call(data)
    assert caught.value.reason_code == "likelihood_not_supported", name


def test_unknown_likelihood_is_invalid() -> None:
    with pytest.raises(ant.errors.CausalValueError):
        _bayes(_binary(200)[0], likelihood="student_t")  # type: ignore[arg-type]
