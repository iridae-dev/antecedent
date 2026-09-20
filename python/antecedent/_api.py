"""Shared public-operation bookkeeping; scientific decisions remain in Rust."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from .errors import CausalError, named_pending_edges, next_action, pending_edges

P = ParamSpec("P")
R = TypeVar("R")


class RefusalReport(BaseModel):
    """Descriptive refusal without inferring scientific facts from error prose."""

    model_config = ConfigDict(frozen=True, extra="allow")

    operation: str
    code: str
    message: str
    query: Any = None
    hint: str | None = None
    pending_edges: tuple[Any, ...] = ()
    next: str | None = None
    identification: str = "unavailable"
    support: str = "unavailable"
    uncertainty: str = "unavailable"
    assumptions: str = "unavailable"
    calibration: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
            names = _caller_names(args, kwargs)
            raw_edges = pending_edges(error)
            existing = getattr(error, "report", None)
            # Inner @describe_refusal wraps attach first (often on `self`).
            # Rebuild when this frame has names so V3 resolves at analyze().
            if existing is None or names:
                edges = named_pending_edges(raw_edges, names) if names else raw_edges
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
                    pending_edges=edges,
                    next=next_action(error, edges),
                )
                cast(Any, error).report = report
            if args and hasattr(args[0], "_native"):
                cast(Any, error).study = args[0]
            raise
        return result

    return call


def _caller_names(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Column / node names from the refused public call, when they are in hand."""
    data = kwargs.get("data")
    graph = kwargs.get("graph")
    if data is None and args:
        first = args[0]
        if not hasattr(first, "_native") and not hasattr(first, "status"):
            data = first
    names: list[str] = []
    if graph is not None and hasattr(graph, "nodes"):
        try:
            names = [str(node) for node in graph.nodes()]
        except Exception:  # noqa: BLE001 — names are best-effort
            names = []
    if not names and data is not None:
        try:
            from ._data import ingest_columns

            names, _ = ingest_columns(data)
        except Exception:  # noqa: BLE001 — names are best-effort
            names = []
    return tuple(names)
