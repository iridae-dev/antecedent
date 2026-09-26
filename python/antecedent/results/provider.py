"""Result view for explicitly dispatched Python providers."""

from __future__ import annotations

from typing import Any

from ._execution import Answer, CalibrationInfo, ResultAPI
from ._report import InspectionReport, ResultModel, SlotModel


class ProviderAnalysisResult(ResultModel, ResultAPI):
    """Provider output carried by the shared result API without native claims.

    The output remains family-shaped and externally attested. Antecedent does
    not reinterpret it as a point estimate, interval, or licensed causal claim.
    """

    provider_name: str
    query_family: str
    provider_result: Any
    query: Any = None
    diagnostics: tuple[str, ...] = ()

    @property
    def answer(self) -> Answer:
        return Answer("structured", detail="provider_output_requires_family_interpretation")

    @property
    def calibration(self) -> CalibrationInfo:
        return CalibrationInfo(status="unavailable", reason="attested_not_reverifiable")

    @property
    def estimate(self) -> Any:
        return self.provider_result.estimate

    @property
    def uncertainty(self) -> Any:
        return self.provider_result.uncertainty

    @property
    def assumptions(self) -> tuple[str, ...]:
        return self.provider_result.assumptions

    @property
    def support(self) -> tuple[str, ...]:
        return (self.provider_result.support_status,)

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.provider_result.provenance)

    @property
    def artifact(self) -> bytes | None:
        return self.provider_result.artifact

    def claim(self) -> str:
        return (
            f"Provider {self.provider_name!r} returned output for family "
            f"{self.query_family!r}. Antecedent preserves the provider's declared "
            "assumptions, uncertainty semantics, provenance, and artifact; it does "
            "not translate this output into a native licensed causal claim."
        )

    def inspect(self) -> InspectionReport:
        provider = self.provider_result
        uncertainty = provider.uncertainty
        estimate = provider.estimate
        if hasattr(uncertainty, "tolist"):
            uncertainty = uncertainty.tolist()
        if hasattr(estimate, "tolist"):
            estimate = estimate.tolist()
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="provider_identification_not_independently_verified",
                summary="Provider-declared requirements only",
            ),
            support=SlotModel(
                available=True,
                summary=provider.support_status,
                payload={"status": provider.support_status},
            ),
            uncertainty=SlotModel(
                available=provider.uncertainty is not None,
                reason=None if provider.uncertainty is not None else "not_supplied",
                summary=provider.uncertainty_semantics,
                payload={"value": uncertainty, "estimate": estimate},
            ),
            assumptions=SlotModel(
                available=bool(provider.assumptions),
                reason=None if provider.assumptions else "not_supplied",
                summary="Provider-declared assumptions",
                payload={"assumptions": provider.assumptions},
            ),
            answer=self.answer,
            calibration=self.calibration,
            diagnostics=self.diagnostics,
            contract={
                "provider": self.provider_name,
                "query_family": self.query_family,
                "trust": provider.trust.value,
                "provenance": dict(provider.provenance),
                "artifact_present": provider.artifact is not None,
            },
        )

    def export(self) -> bytes:
        if self.artifact is None:
            raise ValueError("provider result has no portable artifact")
        return self.artifact

    def to_dict(self) -> dict[str, Any]:
        from ._slots import json_value

        result = self.provider_result
        result_dict = {
            "estimate": _portable_value(result.estimate),
            "uncertainty": _portable_value(result.uncertainty),
            "assumptions": list(result.assumptions),
            "support_status": result.support_status,
            "provenance": dict(result.provenance),
            "trust": result.trust.value,
            "uncertainty_semantics": result.uncertainty_semantics,
            "artifact_present": result.artifact is not None,
        }
        dumped = super().to_dict()
        dumped["provider_result"] = json_value(result_dict)
        return dumped


__all__ = ["ProviderAnalysisResult"]


def _portable_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
