"""Independent native-backed consumption of durable posterior records.

Historical posterior bytes retain summaries and optionally draws. They do not
retain the fitted model or its observations and cannot resume an analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from ._artifact_protocol import ArtifactAcceptance, ExportedArtifact, container_bytes
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError
from .outcomes import Answer
from .results._report import InspectionReport, SlotModel


@dataclass(frozen=True, slots=True, repr=False)
class PosteriorQuantityRecord(ScientificResult):
    """One explicitly selected column of a retained native posterior record."""

    _native: Any = field(repr=False, compare=False)
    _index: int

    @property
    def name(self) -> str:
        return str(self._native.quantity_names[self._index])

    @property
    def n_draws(self) -> int:
        return int(self._native.n_draws)

    @property
    def mean(self) -> float:
        return float(self._native.mean[self._index])

    @property
    def sd(self) -> float:
        return float(self._native.sd[self._index])

    @property
    def q025(self) -> float:
        return float(self._native.q025[self._index])

    @property
    def q975(self) -> float:
        return float(self._native.q975[self._index])

    @property
    def draws(self) -> NDArray[np.float64]:
        """Read-only native column, empty when only summaries were retained."""
        start = self._index * self.n_draws
        return np.asarray(self._native)[start : start + self.n_draws]

    @property
    def answer(self) -> Answer:
        return Answer("structured", detail="posterior_quantity_record")

    def _summary_fields(self) -> dict[str, Any]:
        return {"quantity": self.name, "n_draws": self.n_draws, "mean": self.mean}

    def inspect(self) -> InspectionReport:
        return _inspection(
            self.answer,
            {
                "quantity": self.name,
                "n_draws": self.n_draws,
                "mean": self.mean,
                "sd": self.sd,
                "q025": self.q025,
                "q975": self.q975,
            },
            operations=("inspect", "to_dict"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.inspect().details,
            "draws": self.draws.tolist(),
        }


@dataclass(frozen=True, slots=True, repr=False)
class ConsumedPosterior(ScientificResult):
    """A sealed posterior record, with no live study, model or training data."""

    _native: Any = field(repr=False, compare=False)
    _artifact_bytes: bytes = field(repr=False)
    acceptance: ArtifactAcceptance

    @property
    def quantity_names(self) -> tuple[str, ...]:
        return tuple(self._native.quantity_names)

    @property
    def n_draws(self) -> int:
        return int(self._native.n_draws)

    @property
    def mean(self) -> tuple[float, ...]:
        return tuple(self._native.mean)

    @property
    def sd(self) -> tuple[float, ...]:
        return tuple(self._native.sd)

    @property
    def q025(self) -> tuple[float, ...]:
        return tuple(self._native.q025)

    @property
    def q975(self) -> tuple[float, ...]:
        return tuple(self._native.q975)

    @property
    def backend_id(self) -> str:
        return str(self._native.backend_id)

    @property
    def identification(self) -> str:
        """The stored status label; it is not independently reidentified on load."""
        return str(self._native.identification)

    @property
    def unidentified_mass(self) -> float:
        return float(self._native.unidentified_mass)

    @property
    def subsampled_out_mass(self) -> float:
        return float(self._native.subsampled_out_mass)

    @property
    def converged(self) -> bool:
        return bool(self._native.converged)

    @property
    def hessian_condition(self) -> float:
        return float(self._native.hessian_condition)

    @property
    def treatment_contrast(self) -> float | None:
        return self._native.treatment_contrast

    @property
    def study(self) -> None:
        return None

    @property
    def answer(self) -> Answer:
        return Answer("structured", detail="sealed_posterior_record")

    def quantity(self, name: str | None = None) -> PosteriorQuantityRecord:
        """Select one recorded quantity; never infer a causal law from its mean."""
        names = self.quantity_names
        if name is None:
            if len(names) != 1:
                raise CausalValueError("select quantity(name) explicitly for multiple quantities")
            name = names[0]
        if not isinstance(name, str):
            raise CausalTypeError("quantity name must be a string")
        matches = [index for index, candidate in enumerate(names) if candidate == name]
        if len(matches) != 1:
            raise CausalValueError(f"quantity {name!r} is missing or ambiguous")
        return PosteriorQuantityRecord(self._native, matches[0])

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "kind": "causal_posterior",
            "n_draws": self.n_draws,
            "quantities": len(self._native.quantity_names),
            "backend": self.backend_id,
            "status": self.acceptance.status,
        }

    def inspect(self) -> InspectionReport:
        return _inspection(
            self.answer,
            self._summary_fields(),
            operations=("inspect", "quantity", "export", "to_dict"),
        )

    def export(self) -> ExportedArtifact[ConsumedPosterior]:
        from ._artifact_protocol import ArtifactExpectation

        return ExportedArtifact(
            self._artifact_bytes,
            "causal_posterior",
            ArtifactExpectation(
                "causal_posterior",
                {"expected_encoding_sha256": sha256(self._artifact_bytes).hexdigest()},
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Explicit summary serialization; draws are read from a selected quantity."""
        return {
            "quantity_names": list(self.quantity_names),
            "n_draws": self.n_draws,
            "mean": list(self.mean),
            "sd": list(self.sd),
            "q025": list(self.q025),
            "q975": list(self.q975),
            "backend_id": self.backend_id,
            "identification": self.identification,
            "unidentified_mass": self.unidentified_mass,
            "subsampled_out_mass": self.subsampled_out_mass,
            "converged": self.converged,
            "hessian_condition": self.hessian_condition,
            "treatment_contrast": self.treatment_contrast,
            "acceptance": self.acceptance.to_dict(),
        }


def _inspection(
    answer: Answer, details: dict[str, Any], *, operations: tuple[str, ...]
) -> InspectionReport:
    unavailable = SlotModel(
        available=False,
        reason="model_and_observations_not_retained",
        summary="Stored posterior record; scientific claim is not replayed",
    )
    return InspectionReport(
        identification=unavailable,
        support=unavailable,
        uncertainty=SlotModel(
            available=True,
            summary="Recorded posterior summaries; no sampling law is reconstructed",
        ),
        assumptions=unavailable,
        answer=answer,
        stage="consumed",
        family="causal_posterior",
        details=details,
        operations=operations,
    )


def consume_posterior(
    data: Any, *, expected_encoding_sha256: str | None = None
) -> ConsumedPosterior:
    """Parse historical posterior bytes natively and retain sealed acceptance.

    An optional independently retained SHA-256 binds the encoding bytes only.
    It does not establish the posterior's identification or scientific validity.
    """
    raw = container_bytes(data)
    if expected_encoding_sha256 is not None:
        if not isinstance(expected_encoding_sha256, str):
            raise CausalTypeError("expected_encoding_sha256 must be a hexadecimal string")
        if (
            len(expected_encoding_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_encoding_sha256)
        ):
            raise CausalValueError("expected_encoding_sha256 must contain 64 hexadecimal digits")
        if sha256(raw).hexdigest() != expected_encoding_sha256.lower():
            raise CausalValueError("posterior encoding SHA-256 differs from expected encoding")
    native = _native.decode_posterior_artifact(raw)
    return ConsumedPosterior(
        native,
        raw,
        ArtifactAcceptance(
            "sealed",
            expected_encoding_sha256 is not None,
            False,
            False,
            proof="native posterior layout and retained encoding; no model or data replay",
            unresolved=("model_not_retained", "training_data_not_retained"),
        ),
    )


__all__ = ["ConsumedPosterior", "PosteriorQuantityRecord", "consume_posterior"]
