"""One warning for a direct entry that a compiled study replaces.

A Python function that reaches a scientific computation without the causal compiler calls
``direct_path_deprecated`` first, so a caller learns which compiled entry to use before the direct
entry is removed.
"""

from __future__ import annotations

import warnings


class AntecedentDeprecationWarning(FutureWarning):
    """A direct entry that bypasses the causal compiler and will be removed.

    A ``FutureWarning`` so it reaches end users (a ``DeprecationWarning`` is hidden outside
    ``__main__`` by default).
    """


def direct_path_deprecated(name: str, replacement: str) -> None:
    """Warn that ``name`` bypasses the compiler and names the compiled ``replacement``."""
    warnings.warn(
        f"{name} runs without the causal compiler and is deprecated; use {replacement} instead",
        AntecedentDeprecationWarning,
        stacklevel=3,
    )
