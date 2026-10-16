"""Private issuance snapshots for source-bound mean adaptation.

Only factories reading a checked native product stamp a reading. Editable exports and
public dataclass reconstruction never issue a new scientific identity binding.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from ._freeze import freeze
from .errors import CausalValueError

P = ParamSpec("P")
R = TypeVar("R")


def bind_mean_reading(factory: Callable[P, R]) -> Callable[P, R]:
    @wraps(factory)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        reading: Any = factory(*args, **kwargs)
        object.__setattr__(reading, "_mean_binding", freeze(reading.to_dict()))
        return reading

    return call


def check_mean_reading(reading: Any) -> None:
    binding = getattr(reading, "_mean_binding", None)
    if binding is None or freeze(reading.to_dict()) != binding:
        raise CausalValueError(
            "mean adaptation requires the unchanged retained producer binding; use the original result or consume its artifact"
        )
