"""Discovery algorithm configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._native import (
    DiscoveredLink,
    GraphEdge,
    PcmciDiscoveryResult,
    RpcmciDiscoverySummary,
    discover_jpcmci_plus as _discover_jpcmci_plus,
    discover_lpcmci as _discover_lpcmci,
    discover_pc as _discover_pc,
    discover_ges as _discover_ges,
    discover_lingam as _discover_lingam,
    discover_fci as _discover_fci,
    discover_pcmci as _discover_pcmci,
    discover_pcmci_plus as _discover_pcmci_plus,
    discover_rfci as _discover_rfci,
    discover_rpcmci as _discover_rpcmci,
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
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    kind: Literal["rpcmci"] = "rpcmci"


def discover_pc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_pc(
        names,
        columns,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_ges(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_ges(
        names,
        columns,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_lingam(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    prune_threshold: float = 0.05,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_lingam(
        names,
        columns,
        prune_threshold=prune_threshold,
        seed=seed,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_fci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_fci(
        names,
        columns,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_rfci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_rfci(
        names,
        columns,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        max_cond_size=max_cond_size,
        threads=threads,
    )


def discover_pcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_pcmci(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_pcmci_plus(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_pcmci_plus(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
    )


def discover_lpcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult:
    return _discover_lpcmci(
        names,
        columns,
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
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
    regimes: Sequence[int] | None = None,
) -> RpcmciDiscoverySummary:
    return _discover_rpcmci(
        names,
        columns,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        ci=ci,
        weights=weights,
        threads=threads,
        regimes=list(regimes) if regimes is not None else None,
    )


__all__ = [
    "DiscoveredLink",
    "GraphEdge",
    "JPCMCIPlus",
    "LPCMCI",
    "PC",
    "PCMCI",
    "PCMCIPlus",
    "PcmciDiscoveryResult",
    "RPCMCI",
    "RpcmciDiscoverySummary",
    "discover_jpcmci_plus",
    "discover_lpcmci",
    "discover_pc",
    "discover_ges",
    "discover_lingam",
    "discover_fci",
    "discover_rfci",
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
]
