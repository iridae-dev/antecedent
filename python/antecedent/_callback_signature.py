"""Check the calling convention of a declared Python callback without invoking it."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .errors import CausalTypeError


def check_signature(fn: Callable[..., Any], role: str, /, *args: object, **kwargs: object) -> None:
    if not callable(fn) or inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise CausalTypeError(f"{role} must be a synchronous callable")
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{role} has an incompatible signature: {error}") from error
