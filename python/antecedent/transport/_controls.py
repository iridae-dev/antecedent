"""Run a native transport entry under one ``Execution``.

Public transport entries take ``execution=`` only; the physical controls (seed, threads,
cancellation, memory limit) are read from it here and handed to the native entry that
accepts them under its own keyword spelling.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ..errors import CausalTypeError
from ..execution import Execution, native_controls

T = TypeVar("T")


def resolve(execution: Execution | None) -> Execution:
    """The declared execution, or the default one."""
    execution = Execution() if execution is None else execution
    if not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an antecedent.Execution")
    return execution


def call_native(
    execution: Execution | None,
    native: Any,
    /,
    *args: Any,
    cancel: str | None = "cancel",
    memory: str | None = "memory_bytes",
    seed: str | None = None,
    threads: str | None = None,
    **keywords: Any,
) -> Any:
    """Call ``native`` with the execution's controls under its ambient policy.

    ``cancel``, ``memory``, ``seed`` and ``threads`` name the native keyword that receives
    the matching control (``None`` withholds it).
    """
    execution = resolve(execution)
    controls = native_controls(execution, seed=seed, threads=threads, cancel=cancel, memory=memory)
    return execution._call(native, *args, **controls, **keywords)
