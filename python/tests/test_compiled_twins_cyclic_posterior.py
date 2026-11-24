"""The direct cyclic parameter posterior warns and equals the compiled posterior family.

``PreparedCyclicSystem.posterior`` samples the parameter posterior without the compiler. The
compiled ``parameter_posterior`` cyclic family (``posterior_responses``) samples it through the
same kernel under the same seed, so its executed answer carries the same draws and diagnostics.
"""

from __future__ import annotations

import warnings

import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.cyclic import Equation, EquilibriumModel, SamplingPolicy, posterior_responses
from antecedent.execution import Execution
from antecedent.outcomes import require_success

PARAMETERS = [{"kind": "weight", "equation": 1, "parent": 0}]
OPTIONS = {"prior_sd": 0.5, "chains": 2, "warmup": 40, "draws": 40, "proposal_scale": 0.05}


def _model() -> EquilibriumModel:
    return EquilibriumModel(
        (
            Equation("a", 0.0, (0.0, 0.0), "identity"),
            Equation("b", 0.0, (0.0, 0.0), "identity"),
        ),
        (1.0, 0.0, 0.0, 1.0),
    )


def _rows():
    # b = 0.3 a + u_b on clamped a: the planted weight is 0.3.
    noise = [0.05 * ((i * 7) % 5 - 2) for i in range(24)]
    return [
        {
            "values": [float(i % 6 - 3), 0.3 * float(i % 6 - 3) + noise[i]],
            "clamps": [{"variable": 0, "value": float(i % 6 - 3)}],
        }
        for i in range(24)
    ]


def _direct(seed: int):
    prepared = _model().prepare([[]], [[0.0, 0.0]], parameters=PARAMETERS, rows=_rows())
    return require_success(prepared.posterior(**OPTIONS, execution=Execution(seed=seed)))


def _compiled(seed: int):
    family = posterior_responses(
        _model(),
        PARAMETERS,
        _rows(),
        [[]],
        [[0.0, 0.0]],
        prior_sd=OPTIONS["prior_sd"],
        sampling=SamplingPolicy(
            chains=OPTIONS["chains"],
            warmup=OPTIONS["warmup"],
            draws=OPTIONS["draws"],
            proposal_scale=OPTIONS["proposal_scale"],
        ),
        execution=Execution(seed=seed),
    )
    answer = require_success(family.execute()).to_dict()["answer"]
    assert answer["family"] == "parameter_posterior"
    return answer["result"]["posterior"]


def test_direct_posterior_warns_and_equals_the_compiled_family():
    with pytest.warns(AntecedentDeprecationWarning, match="PreparedCyclicSystem.posterior"):
        direct = _direct(7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        compiled = _compiled(7)
    assert direct == compiled


def test_direct_posterior_still_returns_its_draws_after_the_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntecedentDeprecationWarning)
        first = _direct(9)
        second = _direct(9)
    assert first == second
    assert len(first["samples"]) == OPTIONS["chains"] * OPTIONS["draws"]


def test_compiled_posterior_recovers_the_planted_weight():
    posterior = _compiled(11)
    draws = [sample[0] for sample in posterior["samples"]]
    assert sum(draws) / len(draws) == pytest.approx(0.3, abs=0.1)
