"""A requested bootstrap interval runs the complete resampling procedure.

The request travels in ``estimator_config["bootstrap_interval"]``; the replicate cap is the
estimator's ``bootstrap_replicates``. The result reports the rule, the interval, every wave, the
Monte Carlo errors and the stop reason under ``result.performance.resampling``. Bit-for-bit
equality with the fixed-count bootstrap, and the early stop on a wave boundary, are also checked
per estimator in Rust (``util::interval_test_support``).
"""

from __future__ import annotations

import math

import antecedent as ant
import pytest

from _scm import confounded_scm

CAP = 120
WAVE = 30
NEVER = {
    "level": 0.9,
    "wave_size": WAVE,
    "min_replicates": 0,
    "se_rel_tol": 1e-12,
    "half_width_rel_tol": 1e-12,
}
EARLY = {
    "level": 0.9,
    "wave_size": WAVE,
    "min_replicates": 60,
    "se_rel_tol": 0.5,
    "half_width_rel_tol": 0.95,
}


def _run(estimator: str, interval: dict | None, **config: object):
    data, edges = confounded_scm(n=400, seed=5)
    settings: dict[str, object] = {"bootstrap_replicates": CAP, **config}
    if interval is not None:
        settings["bootstrap_interval"] = interval
    return ant.analyze(
        data,
        graph=edges,
        query=ant.AverageEffect(treatment="t", outcome="y"),
        estimator=estimator,
        estimator_config=settings,
        refute="none",
        seed=3,
    )


@pytest.mark.parametrize(
    "estimator", ["linear.adjustment.ate", "propensity.weighting", "glm.adjustment", "aipw"]
)
def test_never_met_tolerances_reproduce_the_fixed_count_and_early_stop_on_a_wave(estimator):
    config = {"family": "gaussian_identity"} if estimator == "glm.adjustment" else {}
    fixed = _run(estimator, None, **config)
    assert fixed.performance.resampling is None

    full = _run(estimator, NEVER, **config)
    report = full.performance.resampling
    assert report is not None
    assert report["stop"] == "max_replicates"
    assert report["attempted"] == CAP
    assert len(report["waves"]) == math.ceil(CAP / WAVE)
    assert report["max_replicates"] == CAP and report["level"] == 0.9
    lower, upper = report["interval"]
    assert lower < upper
    assert report["se_rel_mc_error"] is not None
    assert full.estimate.se_bootstrap == fixed.estimate.se_bootstrap
    assert full.estimate.ate == fixed.estimate.ate

    early = _run(estimator, EARLY, **config)
    stopped = early.performance.resampling
    assert stopped["stop"] == "converged"
    assert stopped["attempted"] < CAP and stopped["attempted"] % WAVE == 0
    assert sum(wave["attempted"] for wave in stopped["waves"]) == stopped["attempted"]


def test_a_bad_request_is_refused_with_the_offending_key():
    with pytest.raises(ValueError, match="bootstrap_interval"):
        _run("linear.adjustment.ate", {"level": 1.5})
    with pytest.raises(ValueError, match="unknown estimator_config\\['bootstrap_interval'\\]"):
        _run("linear.adjustment.ate", {"width": 3})


def test_an_estimator_whose_bootstrap_is_not_the_wave_procedure_refuses_the_key():
    with pytest.raises(ValueError, match="bootstrap_interval"):
        _run("propensity.matching", NEVER)
