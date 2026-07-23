"""Discovery stability validators.

Thin wrappers over ``causal-validate`` stability checks (PCMCI block bootstrap,
false-positive surrogates, parameter grids, orientation, null calibration,
environment holdout, regime stability).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._data import as_columns, as_multi_env_columns
from ._native import (
    validate_environment_holdout as _validate_environment_holdout,
    validate_pcmci_alpha_sensitivity as _validate_pcmci_alpha_sensitivity,
    validate_pcmci_block_bootstrap as _validate_pcmci_block_bootstrap,
    validate_pcmci_ci_sensitivity as _validate_pcmci_ci_sensitivity,
    validate_pcmci_false_positive as _validate_pcmci_false_positive,
    validate_pcmci_lag_sensitivity as _validate_pcmci_lag_sensitivity,
    validate_pcmci_plus_orientation as _validate_pcmci_plus_orientation,
    validate_regime_stability as _validate_regime_stability,
    validate_synthetic_null_calibration as _validate_synthetic_null_calibration,
)


def validate_pcmci_block_bootstrap(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_block_bootstrap(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_false_positive(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    transform: str = "permute",
    replicates: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_false_positive(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        transform=transform,
        replicates=replicates,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_alpha_sensitivity(
    data: Mapping[str, Any] | Any,
    alphas: Sequence[float],
    *,
    max_lag: int = 1,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_alpha_sensitivity(
        names,
        columns,
        list(alphas),
        max_lag=max_lag,
        fdr=fdr,
        ci=ci,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_lag_sensitivity(
    data: Mapping[str, Any] | Any,
    max_lags: Sequence[int],
    *,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_lag_sensitivity(
        names,
        columns,
        [int(m) for m in max_lags],
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_ci_sensitivity(
    data: Mapping[str, Any] | Any,
    ci_names: Sequence[str],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_ci_sensitivity(
        names,
        columns,
        list(ci_names),
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_plus_orientation(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_pcmci_plus_orientation(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        seed=seed,
        threads=threads,
    )


def validate_synthetic_null_calibration(
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_sim: int = 20,
    n_obs: int = 100,
    n_vars: int = 3,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    return _validate_synthetic_null_calibration(
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_sim=n_sim,
        n_obs=n_obs,
        n_vars=n_vars,
        seed=seed,
        threads=threads,
    )


def validate_environment_holdout(
    data: Sequence[Mapping[str, Any] | Any],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_discovery: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, env_columns = as_multi_env_columns(data)
    return _validate_environment_holdout(
        names,
        env_columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_discovery=n_discovery,
        seed=seed,
        threads=threads,
    )


def validate_regime_stability(
    data: Mapping[str, Any] | Any,
    regimes: Sequence[int],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 10,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _validate_regime_stability(
        names,
        columns,
        [int(r) for r in regimes],
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        seed=seed,
        threads=threads,
    )


__all__ = [
    "validate_environment_holdout",
    "validate_pcmci_alpha_sensitivity",
    "validate_pcmci_block_bootstrap",
    "validate_pcmci_ci_sensitivity",
    "validate_pcmci_false_positive",
    "validate_pcmci_lag_sensitivity",
    "validate_pcmci_plus_orientation",
    "validate_regime_stability",
    "validate_synthetic_null_calibration",
]
