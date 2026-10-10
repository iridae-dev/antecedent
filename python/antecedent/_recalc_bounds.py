"""Bounded column ingest shared by selective recalculation families."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .errors import CausalValueError
from .recalc import _column

MAX_ROWS = 100_000
MAX_COLUMNS = 256
MAX_VALUES = 1_000_000


def _columns(data: Mapping[str, ArrayLike]) -> tuple[list[str], list[Any]]:
    names = list(data)
    if len(names) > MAX_COLUMNS:
        raise CausalValueError(
            "recalc.limits_exceeded: data exceed column limit",
            reason_code="invalid_argument",
        )
    views = [np.asarray(data[name]) for name in names]
    if any(view.size > MAX_ROWS for view in views) or sum(view.size for view in views) > MAX_VALUES:
        raise CausalValueError(
            "recalc.limits_exceeded: data exceed row/value limit",
            reason_code="invalid_argument",
        )
    return names, [_column(name, values) for name, values in zip(names, views, strict=True)]
