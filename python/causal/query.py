"""Query helpers (DESIGN.md §25.1 / §8).

Typed query objects are constructed inside Rust bindings from keyword args.
These helpers expose the interventional-distribution and path-specific surfaces
shipped with ``CausalQuery::Distribution`` / ``CausalQuery::PathSpecific``.
Identify/estimate algorithms for those queries remain deferred (IDC / path-
restricted ID); GCM sampling and path contribution are the interim paths.
"""

from __future__ import annotations

from ._native import gcm_attribute_path_specific, gcm_sample_interventional_distribution

__all__ = [
    "gcm_attribute_path_specific",
    "gcm_sample_interventional_distribution",
]
