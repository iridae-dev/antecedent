"""Discovery algorithm configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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
    discover_pcmci as _discover_pcmci,
    discover_pcmci_plus as _discover_pcmci_plus,
    discover_rpcmci as _discover_rpcmci,
)


@dataclass(frozen=True)
class PC:
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    max_cond_size: int = 2
    kind: Literal["pc"] = "pc"


@dataclass(frozen=True)
class PCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    kind: Literal["pcmci"] = "pcmci"


@dataclass(frozen=True)
class PCMCIPlus:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    kind: Literal["pcmci_plus"] = "pcmci_plus"


@dataclass(frozen=True)
class LPCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    kind: Literal["lpcmci"] = "lpcmci"


@dataclass(frozen=True)
class JPCMCIPlus:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    kind: Literal["jpcmci_plus"] = "jpcmci_plus"


@dataclass(frozen=True)
class RPCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
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
    return _discover_jpcmci_plus(
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
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
]
