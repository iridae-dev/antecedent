"""Shared public-operation bookkeeping; scientific decisions remain in Rust."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from .errors import CausalError

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class RefusalReport:
    """Descriptive refusal without inferring scientific facts from error prose."""

    operation: str
    code: str
    message: str
    query: Any = None
    hint: str | None = None
    pending_edges: tuple[Any, ...] = ()
    identification: str = "unavailable"
    support: str = "unavailable"
    uncertainty: str = "unavailable"
    assumptions: str = "unavailable"
    calibration: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        from .results._slots import json_value

        return json_value(self)


def _binds(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether ``args`` / ``kwargs`` bind to ``fn``'s signature."""
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def describe_refusal(fn: Callable[P, R]) -> Callable[P, R]:
    """Keep original exceptions and attach machine-readable operation context."""

    @wraps(fn)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            result = fn(*args, **kwargs)
        except (CausalError, ValueError, TypeError) as error:
            message = str(error)
            if getattr(error, "reason_code", None) is None and message.startswith("reason="):
                code, _, _ = message.partition(":")
                cast(Any, error).reason_code = code.removeprefix("reason=").strip()
            if (
                getattr(error, "reason_code", None) is None
                and isinstance(error, TypeError)
                and not _binds(fn, args, kwargs)
            ):
                # The call itself did not match the signature (an unknown or
                # missing keyword): the argument, not the analysis, is refused.
                cast(Any, error).reason_code = "invalid_argument"
            if not hasattr(error, "report"):
                report = RefusalReport(
                    operation=fn.__name__,
                    # An attribute that exists but is ``None`` falls through to
                    # the next source rather than becoming the code.
                    code=getattr(error, "reason_code", None)
                    or getattr(error, "reason", None)
                    or type(error).__name__,
                    message=message,
                    query=kwargs.get("query", getattr(args[0], "_query", None) if args else None),
                    hint=getattr(error, "hint", None),
                    pending_edges=tuple(getattr(error, "pending_edges", ())),
                )
                cast(Any, error).report = report
            if args and hasattr(args[0], "_native"):
                cast(Any, error).study = args[0]
            raise
        return result

    return call
