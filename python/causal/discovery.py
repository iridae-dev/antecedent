"""Discovery stage wrappers (DESIGN.md §25.1)."""

from __future__ import annotations

from ._native import (
    DiscoveredLink,
    PcmciDiscoveryResult,
    RpcmciDiscoverySummary,
    discover_jpcmci_plus,
    discover_lpcmci,
    discover_pcmci,
    discover_pcmci_plus,
    discover_rpcmci,
)

__all__ = [
    "DiscoveredLink",
    "PcmciDiscoveryResult",
    "RpcmciDiscoverySummary",
    "discover_jpcmci_plus",
    "discover_lpcmci",
    "discover_pcmci",
    "discover_pcmci_plus",
    "discover_rpcmci",
]
