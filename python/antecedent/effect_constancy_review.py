"""Downstream advisory consumers of independently verified effect-constancy evidence.

Transport still needs identification. Prior preferences are data dependent and do not
license pooling. Policy agreement is a point comparison with no generalization guarantee.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import _native
from .decision import Contract, Decision, MeanSource
from .errors import CausalValueError
from .external import ScientificQuantity
from .priors import CompatibilityReport, EstimandFingerprint, PriorCatalog
from .temporal import Contrast, EffectConstancyIdentity, EffectConstancyResult

__all__ = [
    "TransportDiagnostic",
    "PriorSourceRanking",
    "PolicyPartition",
    "PolicyReview",
    "transport_diagnostic",
    "rank_prior_sources",
    "policy_review",
]


def _review(source: EffectConstancyResult, request: Mapping[str, Any]) -> dict[str, Any]:
    if len(source.artifact) > 64 * 1024 * 1024:
        raise CausalValueError(
            "effect_constancy_consumer.invalid_request: artifact exceeds bounds",
            reason_code="invalid_argument",
        )
    text = json.dumps(request)
    if len(text.encode()) > 64 * 1024 * 1024:
        raise CausalValueError(
            "effect_constancy_consumer.invalid_request: request exceeds bounds",
            reason_code="invalid_argument",
        )
    return dict(
        json.loads(
            _native.review_effect_constancy(
                source.artifact, json.dumps(source.identity._wire()), text
            )
        )
    )


@dataclass(frozen=True, slots=True)
class TransportDiagnostic:
    identity: EffectConstancyIdentity
    contrast: Contrast
    calibration: str
    inference_claim: str
    caveats: tuple[str, ...]
    separate_transport_identification_required: bool


@dataclass(frozen=True, slots=True)
class PriorSourceRanking:
    identity: EffectConstancyIdentity
    compatibility: tuple[CompatibilityReport, ...]
    scores: tuple[tuple[str, float], ...]
    ranked: tuple[CompatibilityReport, ...]
    calibration: str
    inference_claim: str
    caveats: tuple[str, ...]
    data_dependent_selection: bool
    posterior_transfer_licensed: bool


@dataclass(frozen=True, slots=True)
class PolicyPartition:
    label: str
    coordinate: str
    source: MeanSource
    decision: Decision


@dataclass(frozen=True, slots=True)
class PolicyReview:
    identity: EffectConstancyIdentity
    partitions: tuple[PolicyPartition, ...]
    common_leaders: tuple[str, ...]
    calibration: str
    inference_claim: str
    caveats: tuple[str, ...]
    generalization_guarantee: bool


def transport_diagnostic(
    source: EffectConstancyResult, *, left: str, right: str
) -> TransportDiagnostic:
    """Original oriented covariance-aware Holm contrast, with unchanged family and caveats."""
    out = _review(source, {"mode": "transport", "left": left, "right": right})
    ev = out["evidence"]
    return TransportDiagnostic(
        EffectConstancyIdentity(**ev["identity"]),
        Contrast(**{("se" if k == "standard_error" else k): v for k, v in out["contrast"].items()}),
        ev["calibration"],
        ev["inference_claim"],
        tuple(ev["caveats"]),
        out["separate_transport_identification_required"],
    )


def rank_prior_sources(
    source: EffectConstancyResult,
    *,
    catalog: PriorCatalog,
    target: EstimandFingerprint,
    target_partition: str,
    partitions: Mapping[str, str],
    variables: Sequence[str] = (),
    tags: Mapping[str, str] | None = None,
    allow_unidentified: bool = False,
) -> PriorSourceRanking:
    """Original catalog filtering followed by explicit data-dependent effect proximity ranking."""
    if (
        sum(len(s.artifact or b"") for s in catalog.sources) > 16 * 1024 * 1024
        or len(catalog.sources) > 1024
        or len(partitions) > 1024
        or any(
            s.artifact is not None and len(s.artifact) > 16 * 1024 * 1024 for s in catalog.sources
        )
    ):
        raise CausalValueError(
            "effect_constancy_consumer.invalid_request: catalog exceeds bounds",
            reason_code="invalid_argument",
        )
    out = _review(
        source,
        {
            "mode": "prior",
            "sources": [
                {
                    "meta": s.meta.to_dict(),
                    "artifact": None if s.artifact is None else list(s.artifact),
                }
                for s in catalog.sources
            ],
            "target": {
                "estimand": asdict(target),
                "variables": list(variables),
                "tags": dict(tags or {}),
                "allow_unidentified": allow_unidentified,
            },
            "target_partition": target_partition,
            "bindings": [
                {"artifact_id": key, "partition": value} for key, value in partitions.items()
            ],
        },
    )
    ev = out["evidence"]
    return PriorSourceRanking(
        EffectConstancyIdentity(**ev["identity"]),
        tuple(CompatibilityReport.from_dict(r) for r in out["compatibility"]),
        tuple((key, float(value)) for key, value in out["scores"]),
        tuple(CompatibilityReport.from_dict(r) for r in out["ranked"]),
        ev["calibration"],
        ev["inference_claim"],
        tuple(ev["caveats"]),
        out["data_dependent_selection"],
        out["posterior_transfer_licensed"],
    )


def policy_review(
    source: EffectConstancyResult, *, contract: Contract, effect: ScientificQuantity
) -> PolicyReview:
    """Original affine mean-policy evaluation over every fully supported original partition."""
    out = _review(
        source, {"mode": "policy", "contract": contract._wire(), "effect": effect._wire()}
    )
    ev = out["evidence"]
    results = []
    for partition in out["partitions"]:
        body = partition["result"]
        means = MeanSource(
            (effect,),
            (float(partition["mean"]),),
            body["provider_id"],
            body["snapshot_id"],
            body["causal_contract_id"],
            body["rng_id"],
        )
        results.append(
            PolicyPartition(
                partition["label"], partition["coordinate"], means, Decision(contract, means, body)
            )
        )
    return PolicyReview(
        EffectConstancyIdentity(**ev["identity"]),
        tuple(results),
        tuple(out["common_leaders"]),
        ev["calibration"],
        ev["inference_claim"],
        tuple(ev["caveats"]),
        out["generalization_guarantee"],
    )
