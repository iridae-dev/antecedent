"""Nuisance scores of a prepared study, retained as artifacts that are refit on load.

A prepared average-effect study knows its treatment and its certified adjustment set.
:meth:`antecedent.estimation.PreparedAnalysis.treatment_score_posterior` returns the Laplace
posterior of the logistic treatment model on exactly those columns under a stated
``N(0, prior_sd**2)`` prior on every coefficient, the intercept included. It is a posterior of a
**nuisance**, never of an effect: the scores, their spread and the probability of falling outside
a positivity band are about the treatment model alone.

A posterior of the *weighting effect* cannot be built from these draws. Under a likelihood that
factors into a treatment model and an outcome model with independent priors, the propensity
coefficients are ancillary for the average effect, so propensity draws pushed through the
weighting identity carry propensity uncertainty alone. Asking for a Bayesian
``propensity.weighting`` estimator raises ``CausalUnsupportedError`` with
``reason_code="route_not_supported"`` naming the missing outcome law and the posteriors that do
answer that effect (the Bayesian-bootstrap posterior of the cross-fitted ridge-orthogonal score,
or the Bayesian g-computation posterior of an outcome model).

The product is unlicensed and its calibration is unmeasured. Its artifact embeds the design it was
fitted on; :func:`load_artifact` refits from it and refuses any edited, swapped or resealed one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._freeze import freeze, thaw
from ._native import consume_array_fit_artifact as _consume
from .outcomes import Answer
from .results._report import InspectionReport, SlotModel

__all__ = ["NuisanceScoreProduct", "load_artifact"]


@dataclass(frozen=True, slots=True, repr=False)
class NuisanceScoreProduct:
    """A retained nuisance-score posterior with its standing and its artifact."""

    estimator_id: str
    #: What the numbers are about: ``propensity_score_nuisance``.
    target: str
    #: ``nuisance_posterior_laplace_not_an_effect_posterior``.
    uncertainty: str
    #: Always ``"unlicensed"``.
    licensing: str
    #: Always ``"unmeasured"``.
    calibration: str
    #: Named reals: ``prior_sd``, ``mode.<j>`` and per summary row ``row<i>.mean``, ``.at_mode``,
    #: ``.sd`` and ``.outside_band``.
    values: Mapping[str, float]
    assumptions: tuple[str, ...]
    request_digest: str
    #: The binary artifact; refit on load by :func:`load_artifact`.
    artifact: bytes
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze(self.values))
        object.__setattr__(self, "assumptions", freeze(self.assumptions))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "artifact", bytes(self.artifact))

    @property
    def answer(self) -> Answer:
        """Named nuisance summaries, without publishing an effect or target draws."""
        return Answer(
            "structured",
            detail=self.target,
            structured={"target": self.target, "value_count": len(self.values)},
        )

    @property
    def identities(self) -> Mapping[str, str]:
        """The native request digest; no causal claim or dataset identity is implied."""
        return freeze({"request_digest": self.request_digest})

    @property
    def provenance(self) -> Mapping[str, str]:
        return freeze({"estimator_id": self.estimator_id, "request_digest": self.request_digest})

    def inspect(self) -> InspectionReport:
        """Read retained native standing without fitting or decoding the artifact."""
        posterior = self.uncertainty == "nuisance_posterior_laplace_not_an_effect_posterior"
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="nuisance_not_causal_effect",
                summary="treatment-model nuisance; no causal-effect identification claim",
            ),
            support=SlotModel(
                available=False,
                reason=self.licensing,
                summary=self.licensing,
                payload={"licensing": self.licensing, "calibration": self.calibration},
            ),
            uncertainty=SlotModel(
                available=posterior,
                reason=None if posterior else self.uncertainty,
                summary=self.uncertainty,
                payload={"target": self.target, "kind": self.uncertainty},
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared nuisance-model assumptions",
                payload={"declarations": self.assumptions},
            ),
            answer=self.answer,
            diagnostics=self.diagnostics,
            contract={"provenance": self.provenance},
            target={"kind": self.target},
            calibration=self.calibration,
            family="nuisance_scores",
            stage="executed",
            details={"value_count": len(self.values), "request_digest": self.request_digest},
            operations=("inspect", "export", "to_dict"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Materialize named summaries; obtain the encoded design through export()."""
        return thaw(
            {
                "estimator_id": self.estimator_id,
                "target": self.target,
                "uncertainty": self.uncertainty,
                "licensing": self.licensing,
                "calibration": self.calibration,
                "values": self.values,
                "assumptions": self.assumptions,
                "diagnostics": self.diagnostics,
                "request_digest": self.request_digest,
            }
        )

    def __repr__(self) -> str:
        return (
            f"NuisanceScoreProduct(estimator_id={self.estimator_id!r}, "
            f"target={self.target!r}, uncertainty={self.uncertainty!r}, "
            f"value_count={len(self.values)})"
        )

    def _repr_html_(self) -> str:
        return self.inspect()._repr_html_()

    @property
    def licensed(self) -> bool:
        """Always ``False``: no support license covers this product."""
        return False

    @property
    def calibration_measured(self) -> bool:
        """Always ``False``: repeated-sampling calibration is unmeasured."""
        return False

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    @artifact_export
    def export(self) -> ExportedArtifact:
        """The retained artifact."""
        return artifact_from_receipt(self, self.artifact)

    def explain(self) -> str:
        """What the product is, and what it is not."""
        return (
            f"{self.estimator_id}: {self.target}, {self.uncertainty}; {self.licensing}, "
            f"calibration {self.calibration}. A posterior of a nuisance, not of an effect."
        )


def _product(report_json: str, artifact: bytes) -> NuisanceScoreProduct:
    artifact = container_bytes(artifact)
    report: Mapping[str, Any] = json.loads(report_json)
    result = report["result"]
    return NuisanceScoreProduct(
        estimator_id=result["estimator_id"],
        target=result["target"],
        uncertainty=result["uncertainty"],
        licensing=result["licensing"],
        calibration=result["calibration"],
        values={entry["name"]: float(entry["value"]) for entry in result["values"]},
        assumptions=tuple(result["assumptions"]),
        request_digest=report["request_digest"],
        artifact=bytes(artifact),
        diagnostics=tuple(result["diagnostics"]),
    )


def load_artifact(artifact: bytes) -> NuisanceScoreProduct:
    """Load a retained artifact, refit it from its own design and return the product.

    Raises:
        CausalUnsupportedError: an edited, swapped, license-claiming or resealed artifact, or a
            stored result that does not replay from the retained arrays.
    """
    artifact = container_bytes(artifact)
    report_json, _ = _consume(bytes(artifact))
    return _product(report_json, bytes(artifact))
