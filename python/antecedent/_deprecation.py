"""One warning for a direct entry that a compiled study replaces.

A Python function that reaches a scientific computation without the causal compiler calls
``direct_path_deprecated`` first, so a caller learns which compiled entry to use before the direct
entry is removed.
"""

from __future__ import annotations

import os
import sys
import types
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


def unbound_artifact_deprecated(name: str, replacement: str) -> None:
    """Warn that ``name`` admitted an artifact without an identity its consumer retained.

    Called through ``antecedent._retained.admit``, often under a decorator, so the warning is
    attributed to the first frame outside the ``antecedent`` package: the caller of the loader.
    """
    warnings.warn(
        f"{name} admits an artifact without the identity its consumer retained and is "
        f"deprecated; use {replacement} instead",
        AntecedentDeprecationWarning,
        stacklevel=_caller_level(),
    )


def _caller_level() -> int:
    """``stacklevel`` of the first frame outside this package, as seen from ``warnings.warn``."""
    package = os.path.dirname(os.path.abspath(__file__))
    level = 1
    frame: types.FrameType | None = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame = frame.f_back
        level += 1
    return level
