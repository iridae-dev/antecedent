"""Discovery stage wrappers (DESIGN.md §25.1)."""

from ._native import (
    DiscoveredLink as DiscoveredLink,
    GraphEdge as GraphEdge,
    PcmciDiscoveryResult as PcmciDiscoveryResult,
    RpcmciDiscoverySummary as RpcmciDiscoverySummary,
    discover_jpcmci_plus as discover_jpcmci_plus,
    discover_lpcmci as discover_lpcmci,
    discover_pcmci as discover_pcmci,
    discover_pcmci_plus as discover_pcmci_plus,
    discover_rpcmci as discover_rpcmci,
)

__all__: list[str]
