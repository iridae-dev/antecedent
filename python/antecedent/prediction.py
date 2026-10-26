"""Portable fitted CATE prediction, bound to an independently verified parent claim."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from .errors import CausalValueError


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
    def load(
        cls, artifact: bytes, *, expected_parent_claim: str | None = None
    ) -> FittedEffectModel:
        """Load the portable fitted effect of a verified result.

        ``expected_parent_claim`` is the claim identity the consumer retained; a different
        claim refuses ``cohort_decision.wrong_parent_identity``.
        """
        artifact = container_bytes(artifact)
        from ._native import FittedEffectModel as NativeModel

        model = cls(NativeModel.load(artifact))
        if expected_parent_claim is not None and model.parent_claim != expected_parent_claim:
            from .cohort_decision import CohortDecisionRefusal

            raise CohortDecisionRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "bind",
                    "detail": "cohort_decision.wrong_parent_identity",
                    "offending": model.parent_claim,
                    "message": "the model comes from a different parent claim than the one retained",
                    "remedy": "load the model from the claim the consumer retained",
                }
            )
        return model

    def decision_source(
        self,
        *,
        cohort: Any,
        outcome_units: str,
        shift: Any = None,
        expected_contrast: str | None = None,
    ) -> Any:
        """Bind a declared cohort and issue one native cohort-mean contrast source.

        ``cohort`` is a :class:`antecedent.cohort_decision.CohortBinding`; ``outcome_units``
        label the original outcome scale (never inferred or converted). Every row is checked
        against the model's own support contract. A cohort from another population needs a
        :class:`antecedent.cohort_decision.CovariateShift`. The result feeds
        :meth:`antecedent.decision.Contract.evaluate` and answers the expectation of an affine
        utility only. Raises :class:`antecedent.cohort_decision.CohortDecisionRefusal`.
        """
        from .cohort_decision import decision_source_from_model

        return decision_source_from_model(
            self,
            cohort=cohort,
            outcome_units=outcome_units,
            shift=shift,
            expected_contrast=expected_contrast,
        )

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self._native.features)

    @property
    def parent_claim(self) -> str:
        return str(self._native.parent_claim)

    @artifact_export
    def export(self) -> ExportedArtifact:
        """Export the complete parent claim, including the bound model payload."""
        return artifact_from_receipt(self, bytes(self._native.export()))

    def predict(self, data: Mapping[str, Any] | Any) -> EffectPrediction:
        """Predict from named columns; unrelated columns are ignored.

        Missing, duplicate, nonnumeric, nonfinite, or misaligned features fail.
        Column order is resolved from the retained feature schema.
        """
        names = tuple(data.columns) if hasattr(data, "columns") else tuple(data)
        if len(set(names)) != len(names):
            raise CausalValueError(
                "prediction data has duplicate column names",
                reason_code="invalid_argument",
            )
        missing = set(self.features) - set(names)
        if missing:
            raise CausalValueError(
                f"missing prediction features: {sorted(missing)}",
                reason_code="invalid_argument",
            )
        columns = [_feature_column(data[name], name) for name in self.features]
        if any(c.ndim != 1 or not np.isfinite(c).all() for c in columns):
            raise CausalValueError(
                "prediction features must be finite one-dimensional columns",
                reason_code="invalid_argument",
            )
        if columns:
            n = len(columns[0])
        elif hasattr(data, "index"):
            n = len(data.index)
        elif names:
            n = len(data[names[0]])
        else:
            raise CausalValueError(
                "intercept-only prediction needs data with an explicit row count",
                reason_code="invalid_argument",
            )
        if any(len(c) != n for c in columns):
            raise CausalValueError(
                "prediction columns have different lengths",
                reason_code="invalid_argument",
            )
        values = self._native.predict([np.ascontiguousarray(c) for c in columns], n)
        return EffectPrediction(tuple(values.tolist()), self.parent_claim)


__all__ = ["FittedEffectModel", "EffectPrediction"]
