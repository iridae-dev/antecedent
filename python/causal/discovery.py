"""Discovery algorithm configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._native import (
    DiscoveredLink,
    GraphEdge,
    GraphPosterior,
    PcmciDiscoveryResult,
    RpcmciDiscoverySummary,
    discover_ci_screened_posterior as _discover_ci_screened_posterior,
    discover_dbn_posterior as _discover_dbn_posterior,
    discover_exact_dag_posterior as _discover_exact_dag_posterior,
    discover_jpcmci_plus as _discover_jpcmci_plus,
    discover_lpcmci as _discover_lpcmci,
    discover_order_mcmc as _discover_order_mcmc,
    discover_pc as _discover_pc,
    discover_ges as _discover_ges,
    discover_lingam as _discover_lingam,
    discover_notears as _discover_notears,
    discover_fci as _discover_fci,
    discover_pcmci as _discover_pcmci,
    discover_pcmci_plus as _discover_pcmci_plus,
    discover_rfci as _discover_rfci,
    discover_rpcmci as _discover_rpcmci,
    discover_structure_mcmc as _discover_structure_mcmc,
    two_regime_half_split,
)

CiSpec = Union[str, Callable[..., Sequence[tuple[float, float]]]]


@dataclass(frozen=True)
class PC:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["pc"] = "pc"


@dataclass(frozen=True)
class PCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    kind: Literal["pcmci"] = "pcmci"


@dataclass(frozen=True)
class PCMCIPlus:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    kind: Literal["pcmci_plus"] = "pcmci_plus"


@dataclass(frozen=True)
class LPCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    kind: Literal["lpcmci"] = "lpcmci"


@dataclass(frozen=True)
class JPCMCIPlus:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    context_names: tuple[str, ...] = ()
    include_space_dummy: bool = True
    include_time_dummy: bool = False
    space_dummy_ci: Literal["scalar", "multivariate"] = "scalar"
    time_dummy_encoding: Literal["integer", "one_hot"] = "integer"
    time_dummy_ci: Literal["scalar", "multivariate"] = "scalar"
    kind: Literal["jpcmci_plus"] = "jpcmci_plus"


@dataclass(frozen=True)
class RPCMCI:
    """Regime-PCMCI. Pass ``regimes=`` to ``analyze`` / ``discover_rpcmci`` (required).

    Use ``two_regime_half_split(n)`` when a simple half-split label vector is enough.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    kind: Literal["rpcmci"] = "rpcmci"


@dataclass(frozen=True)
class GES:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    screen_pc: bool = False
    max_subset: int | None = None
    kind: Literal["ges"] = "ges"


@dataclass(frozen=True)
class LiNGAM:
    prune_threshold: float = 0.05
    max_cond_size: int = 8
    kind: Literal["lingam"] = "lingam"


@dataclass(frozen=True)
class NOTEARS:
    l1: float = 0.1
    threshold: float = 0.3
    standardize: bool = True
    max_cond_size: int = 8
    kind: Literal["notears"] = "notears"


@dataclass(frozen=True)
class FCI:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["fci"] = "fci"


@dataclass(frozen=True)
class RFCI:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["rfci"] = "rfci"


@dataclass(frozen=True)
class ExactDagPosterior:
    """Exact DAG posterior enumeration (hard limit: n ≤ 6, Gaussian BIC).

    For more variables use ``OrderMcmc``, ``StructureMcmc``, or ``CiScreenedPosterior``.
    """

    kind: Literal["exact_dag_posterior"] = "exact_dag_posterior"


@dataclass(frozen=True)
class OrderMcmc:
    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    require_diagnostics_gate: bool = True
    kind: Literal["order_mcmc"] = "order_mcmc"


@dataclass(frozen=True)
class StructureMcmc:
    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    kind: Literal["structure_mcmc"] = "structure_mcmc"


@dataclass(frozen=True)
class CiScreenedPosterior:
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    max_cond_size: int = 2
    soft_weight: Literal["none", "bayes_factor", "posterior_dependence"] = "none"
    n_chains: int = 2
    n_warmup: int = 300
    n_draws: int = 600
    thin: int = 1
    kind: Literal["ci_screened_posterior"] = "ci_screened_posterior"


@dataclass(frozen=True)
class DbnPosterior:
    """Bounded-lag DBN posterior (Gaussian BIC).

    Exact enumeration only when ``p ≤ 4`` and ``max_lag ≤ 2``; larger templates
    automatically use MCMC (or set ``force_mcmc=True``).
    """

    max_lag: int = 1
    force_mcmc: bool = False
    n_chains: int = 2
    n_warmup: int = 200
    n_draws: int = 400
    kind: Literal["dbn_posterior"] = "dbn_posterior"


# Alias: DiscoveryResult is the preferred name; PcmciDiscoveryResult kept for compat.
DiscoveryResult = PcmciDiscoveryResult


def discovery_to_dag(result: DiscoveryResult) -> "Dag":
    """Build a ``Dag`` from a discovery result's directed ``graph_edges``.

    Raises ``ValueError`` if any undirected/circle marks remain.
    """
    from .graph import Dag

    names: list[str] = []
    seen: set[str] = set()
    directed: list[tuple[str, str]] = []
    for e in result.graph_edges:
        for n in (e.source, e.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
        if e.at_source == "tail" and e.at_target == "arrow":
            directed.append((e.source, e.target))
        elif e.at_source == "arrow" and e.at_target == "tail":
            directed.append((e.target, e.source))
        else:
            raise ValueError(
                f"cannot coerce edge {e.source}->{e.target} "
                f"({e.at_source}/{e.at_target}) into a DAG; "
                "use graph_edges or a CPDAG/PAG constructor"
            )
    return Dag.from_edges(names, directed)


def _coerce_tabular(
    names_or_data: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    names: list[str] | None = None,
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Accept ``discover_*(data)``, ``discover_*(names, columns)``, or kwargs."""
    from ._data import as_columns, coerce_data_args, to_f64

    if data is not None:
        return as_columns(data)
    if columns is not None:
        if names_or_data is None:
            raise TypeError("columns= requires names as the first argument")
        return [str(n) for n in names_or_data], [to_f64(c) for c in columns]
    if names is not None:
        # names= kw-only without columns — need columns via coerce
        return coerce_data_args(None, names=names, columns=None)
    if names_or_data is not None:
        return as_columns(names_or_data)
    raise TypeError("provide data=… or names + columns")


def discover_pc(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_pc(
        n,
        cols,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_ges(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
    screen_pc: bool = False,
    max_subset: int | None = None,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_ges(
        n,
        cols,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
        screen_pc=screen_pc,
        max_subset=max_subset,
    )


def discover_lingam(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    prune_threshold: float = 0.05,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_lingam(
        n,
        cols,
        prune_threshold=prune_threshold,
        seed=seed,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_notears(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    l1: float = 0.1,
    threshold: float = 0.3,
    standardize: bool = True,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_notears(
        n,
        cols,
        l1=l1,
        threshold=threshold,
        standardize=standardize,
        seed=seed,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_fci(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_fci(
        n,
        cols,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_rfci(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_rfci(
        n,
        cols,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_pcmci(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_pcmci(
        n,
        cols,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_pcmci_plus(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_pcmci_plus(
        n,
        cols,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_lpcmci(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> DiscoveryResult:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_lpcmci(
        n,
        cols,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_jpcmci_plus(
    names: list[str],
    env_columns: Sequence[Sequence[NDArray[np.float64]]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
    context_names: Sequence[str] | None = None,
    include_space_dummy: bool = True,
    include_time_dummy: bool = False,
    space_dummy_ci: str = "scalar",
    time_dummy_encoding: str = "integer",
    time_dummy_ci: str = "scalar",
) -> PcmciDiscoveryResult:
    return _discover_jpcmci_plus(
        names,
        [list(cols) for cols in env_columns],
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
        context_names=list(context_names) if context_names is not None else None,
        include_space_dummy=include_space_dummy,
        include_time_dummy=include_time_dummy,
        space_dummy_ci=space_dummy_ci,
        time_dummy_encoding=time_dummy_encoding,
        time_dummy_ci=time_dummy_ci,
    )


def discover_rpcmci(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    regimes: Sequence[int],
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> RpcmciDiscoverySummary:
    """Run RPCMCI. ``regimes`` is required (length = series length); no silent half-split.

    Call ``two_regime_half_split(len(series))`` for an explicit two-regime mid-point split.
    """
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_rpcmci(
        n,
        cols,
        regimes=list(regimes),
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_exact_dag_posterior(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior:
    """Exact DAG posterior (hard limit n ≤ 6). Prefer MCMC helpers for larger graphs."""
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_exact_dag_posterior(n, cols, seed=seed, threads=threads)


def discover_order_mcmc(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    require_diagnostics_gate: bool = True,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_order_mcmc(
        n,
        cols,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_draws=n_draws,
        thin=thin,
        require_diagnostics_gate=require_diagnostics_gate,
        seed=seed,
        threads=threads,
    )


def discover_structure_mcmc(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_structure_mcmc(
        n,
        cols,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_draws=n_draws,
        thin=thin,
        seed=seed,
        threads=threads,
    )


def discover_ci_screened_posterior(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    soft_weight: str = "none",
    n_chains: int = 2,
    n_warmup: int = 300,
    n_draws: int = 600,
    thin: int = 1,
    threads: int = 1,
) -> GraphPosterior:
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_ci_screened_posterior(
        n,
        cols,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        max_cond_size=max_cond_size,
        soft_weight=soft_weight,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_draws=n_draws,
        thin=thin,
        seed=seed,
        threads=threads,
    )


def discover_dbn_posterior(
    names: Any | None = None,
    columns: Sequence[NDArray[np.float64]] | None = None,
    *,
    data: Any | None = None,
    max_lag: int = 1,
    force_mcmc: bool = False,
    n_chains: int = 2,
    n_warmup: int = 200,
    n_draws: int = 400,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior:
    """DBN template posterior; exact only for p ≤ 4 and max_lag ≤ 2, else MCMC."""
    n, cols = _coerce_tabular(names, columns, data=data)
    return _discover_dbn_posterior(
        n,
        cols,
        max_lag=max_lag,
        force_mcmc=force_mcmc,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_draws=n_draws,
        seed=seed,
        threads=threads,
    )


__all__ = [
    "CiScreenedPosterior",
    "DbnPosterior",
    "DiscoveredLink",
    "DiscoveryResult",
    "ExactDagPosterior",
    "FCI",
    "GES",
    "GraphEdge",
    "GraphPosterior",
    "JPCMCIPlus",
    "LPCMCI",
    "LiNGAM",
    "NOTEARS",
    "OrderMcmc",
    "PC",
    "PCMCI",
    "PCMCIPlus",
    "PcmciDiscoveryResult",
    "RFCI",
    "RPCMCI",
    "RpcmciDiscoverySummary",
    "StructureMcmc",
    "discover_ci_screened_posterior",
    "discover_dbn_posterior",
    "discover_exact_dag_posterior",
    "discover_jpcmci_plus",
    "discover_lpcmci",
    "discover_order_mcmc",
    "discover_pc",
    "discover_ges",
    "discover_lingam",
    "discover_notears",
    "discover_fci",
    "discover_rfci",
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
    "discover_structure_mcmc",
    "discovery_to_dag",
    "two_regime_half_split",
]
