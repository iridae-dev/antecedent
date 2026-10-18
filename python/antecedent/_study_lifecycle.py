"""Retain declarations and check named designs without changing scientific semantics."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

from ._columns import (
    _data_column_map,
    collect_column_names,
    guard_identifiers,
    resolve_columns,
)
from ._freeze import freeze
from .errors import CausalValueError


def retain(value: Any) -> Any:
    """Detach request collections, including fields of frozen query dataclasses."""
    if is_dataclass(value) and not isinstance(value, type):
        import copy

        result = copy.copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, retain(getattr(value, field.name)))
        return result
    if isinstance(value, Mapping):
        return freeze({key: retain(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(retain(item) for item in value)
    return freeze(value)


def replay_declaration(value: Any) -> Any:
    """Copy retained declarations into constructor mappings without flattening types."""
    if is_dataclass(value) and not isinstance(value, type):
        import copy

        result = copy.copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, replay_declaration(getattr(value, field.name)))
        return result
    if isinstance(value, Mapping):
        return {key: replay_declaration(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(replay_declaration(item) for item in value)
    if isinstance(value, list):
        return [replay_declaration(item) for item in value]
    return value


P = ParamSpec("P")
R = TypeVar("R")


def retain_preparation(fn: Callable[P, R]) -> Callable[P, R]:
    """Capture original declarations for every preparation route, including early returns."""

    @wraps(fn)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        request = {
            key: retain(value)
            for key, value in bound.arguments.items()
            if key not in ("cls", "data")
        }
        data = bound.arguments["data"]
        query = request["query"]
        names = collect_column_names(query)
        columns = _data_column_map(data, names) if names else {}
        resolved = resolve_columns(query, columns)
        result = fn(*args, **kwargs)
        # Native preparation owns the checked declaration. Detach its Python views too,
        # so a caller editing the original query cannot relabel that retained product.
        for attribute in ("_query", "_display_query", "_transport_stage"):
            if hasattr(result, attribute):
                setattr(result, attribute, retain(getattr(result, attribute)))
        cast(Any, result)._preparation_request = request
        cast(Any, result)._named_design_bindings = retain(columns)
        cast(Any, result)._declared_query = query
        cast(Any, result)._resolved_design_query = resolved
        return result

    return call


def guard_design(study: Any, data: Any) -> None:
    """Validate supplied design and identifier columns before dropping them from ingest."""
    guard_identifiers(study, data)
    bindings = getattr(study, "_named_design_bindings", {})
    if not bindings or data is None:
        return
    names = (
        set(data)
        if isinstance(data, Mapping)
        else set(getattr(data, "column_names", getattr(data, "columns", ())))
    )
    present = names.intersection(bindings)
    supplied = _data_column_map(data, present)
    for name in present:
        merged = dict(bindings)
        merged[name] = supplied[name]
        try:
            normalized = resolve_columns(study._declared_query, merged)
            # Compare normalized query fields, not the dtype of the supplied table.
            equal = _same(normalized, study._resolved_design_query)
        except (ValueError, TypeError) as error:
            refusal = CausalValueError(
                f"frozen design column {name!r} is invalid: {error}; use study.reprepare(data)",
                reason_code="invalid_argument",
            )
            setattr(refusal, "offending", name)  # noqa: B010 — structured optional error field
            refusal.remedy = "use study.reprepare(data) to resolve and check the new design"
            raise refusal from error
        if not equal:
            refusal = CausalValueError(
                f"frozen design column {name!r} changed values or row order; "
                "use study.reprepare(data) to declare a new design",
                reason_code="invalid_argument",
            )
            setattr(refusal, "offending", name)  # noqa: B010 — structured optional error field
            refusal.remedy = "use study.reprepare(data) to resolve and check the new design"
            raise refusal


def _same(left: Any, right: Any) -> bool:
    if is_dataclass(left) and is_dataclass(right):
        return type(left) is type(right) and all(
            _same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple, np.ndarray)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)
