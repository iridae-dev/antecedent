"""Portable fitted CATE prediction, bound to an independently verified parent claim."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EffectPrediction:
    """CATE point predictions; no marginal or pointwise interval is implied."""

    values: tuple[float, ...]
    parent_claim: str

    def to_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "parent_claim": self.parent_claim,
            "uncertainty": {"status": "unavailable", "reason": "prediction_only"},
        }


def _feature_column(value: Any, name: str) -> Any:
    """One feature column through the strict numeric ingest every analysis uses.

    A column the strict reader refuses (digit strings, integers beyond 2**53) but a
    plain float64 cast accepts was predicted from before; it still is, with a
    ``FutureWarning`` naming the column, rather than failing in a patch release.
    """
    from ._data import to_f64
    from .errors import CausalTypeError

    if np.ndim(value) != 1:
        return np.asarray(value, dtype=np.float64)  # shape is refused by predict()
    try:
        return to_f64(value, name=name)
    except CausalTypeError as refusal:
        try:
            column = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise refusal from None
        warnings.warn(
            f"{refusal}; parsed as float64 for this prediction. A future release "
            "refuses it: pass numeric columns.",
            FutureWarning,
            stacklevel=3,
        )
        return column


class FittedEffectModel:
    """Immutable model loaded from a verified result, without refitting."""

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    @classmethod
    def load(cls, artifact: bytes) -> FittedEffectModel:
        from ._native import FittedEffectModel as NativeModel

        return cls(NativeModel.load(artifact))

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self._native.features)

    @property
    def parent_claim(self) -> str:
        return str(self._native.parent_claim)

    def export(self) -> bytes:
        """Export the complete parent claim, including the bound model payload."""
        return bytes(self._native.export())

    def predict(self, data: Mapping[str, Any] | Any) -> EffectPrediction:
        """Predict from named columns; unrelated columns are ignored.

        Missing, duplicate, nonnumeric, nonfinite, or misaligned features fail.
        Column order is resolved from the retained feature schema.
        """
        names = tuple(data.columns) if hasattr(data, "columns") else tuple(data)
        if len(set(names)) != len(names):
            raise ValueError("prediction data has duplicate column names")
        missing = set(self.features) - set(names)
        if missing:
            raise ValueError(f"missing prediction features: {sorted(missing)}")
        columns = [_feature_column(data[name], name) for name in self.features]
        if any(c.ndim != 1 or not np.isfinite(c).all() for c in columns):
            raise ValueError("prediction features must be finite one-dimensional columns")
        if columns:
            n = len(columns[0])
        elif hasattr(data, "index"):
            n = len(data.index)
        elif names:
            n = len(data[names[0]])
        else:
            raise ValueError("intercept-only prediction needs data with an explicit row count")
        if any(len(c) != n for c in columns):
            raise ValueError("prediction columns have different lengths")
        values = self._native.predict([np.ascontiguousarray(c) for c in columns], n)
        return EffectPrediction(tuple(values.tolist()), self.parent_claim)


__all__ = ["FittedEffectModel", "EffectPrediction"]
