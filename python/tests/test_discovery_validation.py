"""The direct discovery validators warn, and the compiled ``DiscoveryValidation`` check reproduces
them number for number on the same data, parameters and execution seed."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution, validation
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.discovery_validation import validate_discovery


def _series(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


DATA = _series(160, 5)


def _execution() -> Execution:
    return Execution(seed=7)


def _same(direct: dict, compiled: dict, check: str) -> None:
    for key, value in direct.items():
        assert compiled[key] == value, key
    assert compiled["check"] == check
    assert compiled["route"] == "DiscoveryValidation"
    assert len(compiled["program"]) == 64


def test_block_bootstrap_warns_and_the_compiled_check_matches():
    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_block_bootstrap"):
        direct = validation.validate_pcmci_block_bootstrap(
            DATA, alpha=0.1, replicates=6, block_size=15, execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "block_bootstrap", alpha=0.1, replicates=6, block_size=15, execution=_execution()
    )
    _same(direct, compiled, "block_bootstrap")


@pytest.mark.parametrize("transform", ["permute", "phase"])
def test_false_positive_warns_and_the_compiled_check_matches(transform):
    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_false_positive"):
        direct = validation.validate_pcmci_false_positive(
            DATA, transform=transform, replicates=5, execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "false_positive", transform=transform, replicates=5, execution=_execution()
    )
    _same(direct, compiled, "false_positive")


def test_parameter_grids_warn_and_the_compiled_checks_match():
    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_alpha_sensitivity"):
        direct = validation.validate_pcmci_alpha_sensitivity(
            DATA, [0.01, 0.1], execution=_execution()
        )
    compiled = validate_discovery(DATA, "alpha", alphas=[0.01, 0.1], execution=_execution())
    _same(direct, compiled, "alpha")

    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_lag_sensitivity"):
        direct = validation.validate_pcmci_lag_sensitivity(
            DATA, [1, 2], alpha=0.1, execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "lag_window", max_lags=[1, 2], alpha=0.1, execution=_execution()
    )
    _same(direct, compiled, "lag_window")

    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_ci_sensitivity"):
        direct = validation.validate_pcmci_ci_sensitivity(
            DATA, ["parcorr", "robust_parcorr"], execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "ci_test", ci_names=["parcorr", "robust_parcorr"], execution=_execution()
    )
    _same(direct, compiled, "ci_test")


def test_orientation_and_synthetic_null_warn_and_the_compiled_checks_match():
    with pytest.warns(AntecedentDeprecationWarning, match="validate_pcmci_plus_orientation"):
        direct = validation.validate_pcmci_plus_orientation(
            DATA, replicates=5, block_size=15, execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "orientation", replicates=5, block_size=15, execution=_execution()
    )
    _same(direct, compiled, "orientation")

    with pytest.warns(AntecedentDeprecationWarning, match="validate_synthetic_null_calibration"):
        direct = validation.validate_synthetic_null_calibration(
            alpha=0.1, n_sim=4, n_obs=60, n_vars=2, execution=_execution()
        )
    compiled = validate_discovery(
        DATA, "synthetic_null", alpha=0.1, n_sim=4, n_obs=60, n_vars=2, execution=_execution()
    )
    _same(direct, compiled, "synthetic_null")


def test_environment_holdout_and_regime_stability_warn_and_the_compiled_checks_match():
    environments = [_series(120, seed) for seed in (11, 12, 13)]
    with pytest.warns(AntecedentDeprecationWarning, match="validate_environment_holdout"):
        direct = validation.validate_environment_holdout(
            environments, n_discovery=2, execution=_execution()
        )
    compiled = validate_discovery(
        environments, "environment_holdout", n_discovery=2, execution=_execution()
    )
    _same(direct, compiled, "environment_holdout")

    long = _series(240, 17)
    regimes = [0] * 120 + [1] * 120
    with pytest.warns(AntecedentDeprecationWarning, match="validate_regime_stability"):
        direct = validation.validate_regime_stability(
            long, regimes, replicates=4, block_size=20, execution=_execution()
        )
    compiled = validate_discovery(
        long,
        "regime_stability",
        regimes=regimes,
        replicates=4,
        block_size=20,
        execution=_execution(),
    )
    _same(direct, compiled, "regime_stability")


def test_a_grid_check_without_its_grid_is_refused():
    with pytest.raises(ValueError, match="needs alphas"):
        validate_discovery(DATA, "alpha", execution=_execution())
