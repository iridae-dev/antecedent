"""Typed unmeasured nested Fisher candidate lifecycle for internal acceptance builds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..errors import CausalTypeError, CausalUnsupportedError

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Intervals3 = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True, init=False)
class NestedFisherCandidate:
    """Original checked-ID Fisher/delta approximation, available in internal builds.

    Coordinates are the two intervention means and their difference. The full
    covariance retains the shared fitted mechanisms. Nominal 90%/95% intervals
    remain unmeasured candidates; neither this object nor artifact replay licenses
    repeated-sampling coverage. The graph and IID sampling model are declarations.
    """

    values: tuple[float, float, float]
    covariance: Matrix3
    intervals: Intervals3
    nominal_level: float
    calibration: Literal["unmeasured"]
    inference: Literal["interval_withheld_calibration_unmeasured"]
    identification: Literal["nonparametrically_identified"]
    method: str
    identity: str
    _native: Any

    def __init__(self) -> None:
        raise CausalTypeError(
            "use binary_nested_markov_fisher_interval or load with retained identity"
        )

    @classmethod
    def _from_native(cls, native: Any) -> NestedFisherCandidate:
        from .. import _native

        native_type = getattr(_native, "NativeNestedFisherCandidate", None)
        if native_type is None or not isinstance(native, native_type):
            raise CausalTypeError("candidate requires an original native Fisher result")
        payload = json.loads(native.payload())
        contrast = payload["point"]["receipt"]["contrast"]
        means = contrast["model_means"]
        values = dict(
            values=(means[0], means[1], contrast["model_contrast"]),
            covariance=cast(Matrix3, tuple(tuple(row) for row in payload["effect_covariance"])),
            intervals=cast(Intervals3, tuple(tuple(row) for row in payload["interval_candidates"])),
            nominal_level=payload["nominal_level"],
            calibration=payload["calibration"],
            inference=payload["point"]["receipt"]["status"]["inference"],
            identification=payload["point"]["receipt"]["status"]["identification"],
            method=payload["method"],
            identity=native.identity,
            _native=native,
        )
        result = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result

    def export(self) -> bytes:
        """Export original graph, counts, fit settings, covariance and standing."""
        return bytes(self._native.export())

    def to_dict(self) -> dict[str, Any]:
        """Inspect complete premises, eleven-parameter covariance and fit receipt."""
        return cast(dict[str, Any], json.loads(self._native.payload()))

    @classmethod
    def load(cls, artifact: bytes, *, expected_identity: str) -> NestedFisherCandidate:
        """Fresh consumer rechecks licensed ID and refits every original output."""
        from .. import _native

        if not isinstance(artifact, bytes) or not isinstance(expected_identity, str):
            raise CausalTypeError("artifact must be bytes and expected_identity must be str")
        consumer = getattr(_native, "consume_nested_markov_fisher_candidate", None)
        if consumer is None:
            raise CausalUnsupportedError(
                "nested_markov.route_frozen: the interval consumer remains closed pending calibration",
                reason_code="cell_not_licensed",
            )
        return cls._from_native(consumer(artifact, expected_identity))
