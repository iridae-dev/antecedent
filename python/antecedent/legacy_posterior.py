"""Convert a legacy posterior artifact into an aligned joint distribution.

The conversion is the Rust facade, which calls the io-core function and returns
that artifact. This module does not transpose draws, recompute moments, or
replace the identity's meaning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._artifact_protocol import container_bytes
from ._native import convert_legacy_posterior as _convert_legacy_posterior
from .errors import CausalTypeError
from .joint_distribution import DistributionIdentity, JointDistributionArtifact, ScientificQuantity


@dataclass(frozen=True, slots=True)
class LegacyPosteriorBinding:
    """One source quantity of a legacy posterior bound to one target coordinate.

    ``source`` is the posterior wire quantity, passed through unchanged:
    ``{"coefficient": {"index": 0, "name": "beta"}}``, ``"residual_variance"``,
    ``{"effect": {"name": "ate"}}`` or ``{"scalar": {"name": "x"}}``.
    ``target`` is the scientific coordinate that column becomes.
    """

    source: str | Mapping[str, Any]
    target: ScientificQuantity

    def _wire(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target._wire()}


def convert_legacy_posterior(
    legacy: bytes,
    identity: DistributionIdentity,
    bindings: Sequence[LegacyPosteriorBinding],
) -> JointDistributionArtifact:
    """Return the joint artifact produced by the io-core legacy-posterior conversion.

    ``legacy`` is the encoded posterior container (bytes or an
    :class:`~antecedent.artifacts.ExportedArtifact`). ``identity`` is the
    consumer's distribution identity, including its draw meaning.
    ``identity.semantic`` is stored as given: ``parameter_posterior``,
    ``posterior_predictive``, ``interventional_predictive``, ``bootstrap``,
    ``estimator_sampling``, ``empirical_outcome`` and
    ``causal_functional_posterior`` stay distinct. A posterior that does not
    name its outcome quantities is refused (``distribution_meaning_mismatch``).

    The returned artifact is the native conversion, including its legacy
    receipt. Calibration is ``not_measured`` and trust is ``unverified``.

    Raises:
        CausalTypeError: ``identity`` or ``bindings`` has the wrong type.
        CausalSerializationError: the container or the conversion is refused.
            Unnamed outcome quantities and a meaning the source quantities do
            not support carry ``distribution_meaning_mismatch``. A source id
            that is not the artifact id carries ``quantity_semantics_mismatch``.
    """
    if not isinstance(identity, DistributionIdentity):
        raise CausalTypeError("identity must be a DistributionIdentity")
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise CausalTypeError("bindings must be LegacyPosteriorBinding values")
    bound = tuple(bindings)
    if any(not isinstance(item, LegacyPosteriorBinding) for item in bound):
        raise CausalTypeError("bindings must be LegacyPosteriorBinding values")
    native = _convert_legacy_posterior(
        container_bytes(legacy),
        json.dumps(identity._wire()),
        json.dumps([item._wire() for item in bound]),
    )
    return JointDistributionArtifact._from_native(native)


__all__ = ["LegacyPosteriorBinding", "convert_legacy_posterior"]
