"""Run a native joint-panel entry under a caller's ``Execution``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .execution import Execution, native_controls, resolve_execution

T = TypeVar("T")


def run_panel(
    execution: Execution | None, operation: Callable[..., T], /, *args: Any, **kwargs: Any
) -> T:
    """Call ``operation`` with the execution's seed, threads, memory limit and cancellation.

    The panel natives take ``(seed, threads, max_bytes, cancel)``; the execution is their
    only owner, and its progress and stage observers run through its ambient policy.
    ``threads=None`` means the engine's single-thread default.
    """
    execution = resolve_execution(execution)
    controls = native_controls(
        execution, seed="seed", threads="threads", memory="max_bytes", cancel="cancel"
    )
    if controls["threads"] is None:
        controls["threads"] = 1
    return execution._call(operation, *args, **kwargs, **controls)
