"""Validation checks of a temporal structure search, through the compiled ``DiscoveryValidation``
route.

:func:`validate_discovery` builds a study whose temporal search is the one the check validates
(PCMCI for the link-frequency, surrogate, grid and synthetic-null checks, PCMCI+ for
``orientation``, regime PCMCI over ``regimes`` for ``regime_stability``, J-PCMCI+ over
environments for ``environment_holdout``), compiles it, and runs the check with that search's
engine on the study's own data. The returned dictionary has the keys the direct validators in
:mod:`antecedent.validation` return, with the same numbers, plus ``check``, ``route`` (always
``"DiscoveryValidation"``) and ``program`` (the identity of the lowered program that produced
it).

A grid check's study searches at the grid's first point unless ``alpha``, ``max_lag`` or ``ci``
is given, as the direct validators do. Every number is computed in Rust; the calibration of these
diagnostics is not measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ._data import as_columns, as_multi_env_columns
from ._native import discovery_validation_run as _discovery_validation_run
from ._native import (
    discovery_validation_run_environments as _discovery_validation_run_environments,
)
from .errors import CausalValueError
from .execution import Execution, native_controls

__all__ = ["DiscoveryCheckName", "validate_discovery"]

DiscoveryCheckName = Literal[
    "block_bootstrap",
    "false_positive",
    "alpha",
    "lag_window",
    "ci_test",
    "orientation",
    "synthetic_null",
    "environment_holdout",
    "regime_stability",
]


def validate_discovery(
    data: Mapping[str, Any] | Any | Sequence[Mapping[str, Any] | Any],
    check: DiscoveryCheckName,
    *,
    max_lag: int | None = None,
    alpha: float | None = None,
    fdr: bool = False,
    ci: str | None = None,
    replicates: int = 20,
    block_size: int = 20,
    transform: str = "permute",
    alphas: Sequence[float] | None = None,
    max_lags: Sequence[int] | None = None,
    ci_names: Sequence[str] | None = None,
    n_sim: int = 20,
    n_obs: int = 100,
    n_vars: int = 3,
    regimes: Sequence[int] | None = None,
    n_discovery: int = 1,
    execution: Execution | None = None,
) -> dict[str, Any]:
    """Run ``check`` on the temporal search of a study over ``data``, through the compiler.

    ``data`` is one series (a mapping or data frame of columns), or a sequence of environment
    series for ``environment_holdout``. ``max_lag``, ``alpha``, ``fdr`` and ``ci`` configure the
    study's search; a grid check (``alpha``, ``lag_window``, ``ci_test``) takes its grid from
    ``alphas``, ``max_lags`` or ``ci_names`` and, unless the search parameter is given, searches
    at the grid's first point.
    """
    execution = Execution() if execution is None else execution
    grid_alphas = None if alphas is None else [float(a) for a in alphas]
    grid_lags = None if max_lags is None else [int(m) for m in max_lags]
    grid_tests = None if ci_names is None else [str(c) for c in ci_names]
    if alpha is None:
        alpha = grid_alphas[0] if check == "alpha" and grid_alphas else 0.05
    if max_lag is None:
        max_lag = grid_lags[0] if check == "lag_window" and grid_lags else 1
    if ci is None:
        ci = grid_tests[0] if check == "ci_test" and grid_tests else "parcorr"
    controls = native_controls(execution, seed="seed", threads="threads")
    if check == "environment_holdout":
        if isinstance(data, Mapping):
            raise CausalValueError(
                "the environment_holdout check reads a sequence of environment series",
                reason_code="invalid_argument",
            )
        names, env_columns = as_multi_env_columns(data)
        return execution._call(
            _discovery_validation_run_environments,
            names,
            env_columns,
            max_lag=max_lag,
            alpha=alpha,
            fdr=fdr,
            ci=ci,
            n_discovery=n_discovery,
            **controls,
        )
    names, columns = as_columns(data)
    return execution._call(
        _discovery_validation_run,
        names,
        columns,
        check,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        transform=transform,
        alphas=grid_alphas,
        max_lags=grid_lags,
        ci_names=grid_tests,
        n_sim=n_sim,
        n_obs=n_obs,
        n_vars=n_vars,
        regimes=None if regimes is None else [int(r) for r in regimes],
        **controls,
    )
