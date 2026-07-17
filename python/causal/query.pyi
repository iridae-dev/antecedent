"""Query helpers (DESIGN.md §25.1 / §8)."""

from __future__ import annotations

from ._native import gcm_attribute_path_specific, gcm_sample_interventional_distribution

__all__: list[str]

def gcm_sample_interventional_distribution(
    names: list[str],
    columns: object,
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    outcome: str | None = None,
    seed: int = 0,
    threads: int = 1,
) -> object: ...

def gcm_attribute_path_specific(
    names: list[str],
    columns: object,
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    path_nodes: list[str] | None = None,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 0,
    threads: int = 1,
) -> tuple[float, list[tuple[list[str], float]]]: ...
