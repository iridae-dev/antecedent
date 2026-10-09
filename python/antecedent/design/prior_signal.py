"""Checked original posterior sources paired with a future candidate likelihood."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .. import _native
from ..errors import CausalTypeError, CausalValueError
from ..joint_distribution import ScientificQuantity
from ..priors import PriorCatalog, _query_estimand
from ._declarations import BinomialSignal, GaussianMeanSignal, StatePrior, _raise


@dataclass(frozen=True, slots=True)
class PriorSignalSource:
    """One original catalog posterior quantity and its explicit evidence declarations.

    ``quantity`` names an effect/scalar column in the original artifact. Observation
    and population identities are declarations: old posterior artifacts do not retain
    those identities themselves. They must describe the observations actually fit.
    ``intercept + slope * source`` maps into the target scalar decision coordinate.
    """

    artifact_id: str
    quantity: str
    source_population: str
    lineage: tuple[str, ...]
    observation_ids: tuple[str, ...]
    weight: float = 1.0
    prior_strength: float = 1.0
    conflict_shrinkage: float = 0.0
    intercept: float = 0.0
    slope: float = 1.0

    def __post_init__(self) -> None:
        for name in ("artifact_id", "quantity", "source_population"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name} must be a nonempty string")
        for name in ("lineage", "observation_ids"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise CausalTypeError(f"{name} must be a sequence of strings")
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise CausalValueError(f"{name} must contain nonempty strings")
            object.__setattr__(self, name, tuple(values))


@dataclass(frozen=True, slots=True, init=False)
class CheckedPriorSignal:
    """Native checked prior/provider pair for the canonical study ranker.

    Use ``prior`` in :class:`DesignDecision` and ``signal`` as each candidate's
    provider. The native handoff preserves original observations and source digests;
    changing the signal family or prior identity refuses. Posterior summaries and
    coefficient-only priors cannot stand in for retained state draws.
    """

    prior: StatePrior
    signal: GaussianMeanSignal | BinomialSignal
    prior_id: str

    def __init__(self) -> None:
        raise CausalTypeError("use design.adapt_prior_to_signal to create a checked pair")

    @classmethod
    def _from_native(
        cls,
        native: _native.CheckedPriorSignal,
        signal: GaussianMeanSignal | BinomialSignal,
        prior_id: str,
    ) -> CheckedPriorSignal:
        result = object.__new__(cls)
        object.__setattr__(result, "prior", StatePrior("draws", _checked=native))
        object.__setattr__(result, "signal", signal)
        object.__setattr__(result, "prior_id", prior_id)
        return result

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Copy of the actual Rust adapter's source/transport/overlap checks."""
        native = self.prior._checked
        if native is None:
            raise CausalTypeError("checked prior native binding is unavailable")
        return dict(json.loads(native.diagnostics_json))


def adapt_prior_to_signal(
    catalog: PriorCatalog,
    *,
    query: Any,
    sources: Sequence[PriorSignalSource],
    signal: GaussianMeanSignal | BinomialSignal,
    state: ScientificQuantity,
    target_population: str,
    prior_id: str,
    variables: Sequence[str] = (),
    transport_policy_id: str | None = None,
    candidate_observation_ids: Sequence[str] = (),
    resolution: int = 1024,
) -> CheckedPriorSignal:
    """Adapt compatible original posterior rows through Rust's checked adapter.

    Supports Gaussian-mean and binomial future studies of one scalar state. Source
    draws are decoded natively from catalog bytes, never reconstructed from moments.
    Sources must be complete, converged, compatible named effect/scalar posteriors.
    Transport policy is explicit when populations differ. Shared observations refuse
    both now and later at ranking, including reuse omitted from this declaration.
    Resolution is bounded by the native adapter (1,000,000); total source artifacts
    and request declarations are bounded to 16 MiB and 64 sources.
    """
    if not isinstance(state, ScientificQuantity):
        raise CausalTypeError("state must be a ScientificQuantity")
    if not isinstance(catalog, PriorCatalog):
        raise CausalTypeError("catalog must be a priors.PriorCatalog")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise CausalTypeError("sources must be a sequence of PriorSignalSource")
    # Bound counts before type iteration, metadata conversion, or catalog tuple copies.
    if len(sources) > 64 or len(catalog._sources) > 64:
        raise CausalValueError("prior signal adaptation supports at most 64 sources")
    if not all(isinstance(s, PriorSignalSource) for s in sources):
        raise CausalTypeError("sources must contain PriorSignalSource declarations")
    if not isinstance(signal, (GaussianMeanSignal, BinomialSignal)):
        raise CausalTypeError("signal must be GaussianMeanSignal or BinomialSignal")
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise CausalTypeError("resolution must be an integer")
    for name, values in (
        ("variables", variables),
        ("candidate_observation_ids", candidate_observation_ids),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise CausalTypeError(f"{name} must be a sequence of strings")
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise CausalValueError(f"{name} must contain nonempty strings")
    payload = [{"meta": s.meta.to_dict(), "artifact": s.artifact} for s in catalog.sources]
    estimand = _query_estimand(query)
    request = {
        "query_kind": estimand.query_kind,
        "treatment": estimand.treatment,
        "outcome": estimand.outcome,
        "variables": list(variables),
        "sources": [asdict(s) for s in sources],
        "signal": signal._wire(),
        "target_population": target_population,
        "prior_id": prior_id,
        "state": state._wire(),
        "transport_policy_id": transport_policy_id,
        "resolution": resolution,
        "candidate_observation_ids": list(candidate_observation_ids),
    }
    try:
        text = json.dumps(request, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CausalValueError("invalid prior signal declaration") from error
    native, refusal = _native.adapt_prior_signal(payload, text)
    _raise(refusal)
    assert native is not None
    return CheckedPriorSignal._from_native(native, signal, prior_id)


__all__ = ["CheckedPriorSignal", "PriorSignalSource", "adapt_prior_to_signal"]
