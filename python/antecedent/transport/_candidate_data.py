"""Immutable report storage and detached copies for bounded native candidate payloads."""

from collections.abc import Mapping
from typing import Any

from .._freeze import freeze

__all__ = ["detached", "freeze"]


def detached(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [detached(item) for item in value]
    return value
