"""Typed, bounded 2.3 aligned distribution artifacts.

``numpy.asarray(artifact)`` makes one bounded copy into NumPy-owned row-major
storage. The array never aliases the validated Rust artifact. Loading requires
an expected identity supplied by the consumer, separate from the input bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._native import JointDistributionArtifact as _NativeJointDistributionArtifact

DistributionMeaning = Literal[
    "parameter_posterior",
    "causal_functional_posterior",
    "posterior_predictive",
    "interventional_predictive",
    "estimator_sampling",
    "bootstrap",
    "empirical_outcome",
]
DrawAlignment = Literal["joint", "independent_marginals"]


@dataclass(frozen=True, slots=True)
class QuantityCondition:
    variable_id: str
    value_id: str


@dataclass(frozen=True, slots=True)
class ScientificQuantity:
    variable_id: str
    variable_name: str
    role: Literal["treatment", "outcome", "covariate", "mediator", "selection", "utility"]
    units: str
    population_id: str
    regime_id: str
    horizon: int
    functional_id: str
    conditioning: tuple[QuantityCondition, ...] = ()
    transform_id: str = "identity"

    def _wire(self) -> dict[str, Any]:
        return {"version": 1, **asdict(self)}

    @classmethod
    def _from_wire(cls, wire: dict[str, Any]) -> ScientificQuantity:
        return cls(
            variable_id=wire["variable_id"],
            variable_name=wire["variable_name"],
            role=wire["role"],
            units=wire["units"],
            population_id=wire["population_id"],
            regime_id=wire["regime_id"],
            horizon=wire["horizon"],
            functional_id=wire["functional_id"],
            conditioning=tuple(QuantityCondition(**item) for item in wire["conditioning"]),
            transform_id=wire["transform_id"],
        )


@dataclass(frozen=True, slots=True)
class DistributionIdentity:
    semantic: DistributionMeaning
    quantities: tuple[ScientificQuantity, ...]
    alignment: DrawAlignment
    source_id: str
    provider_id: str
    rng_id: str
    snapshot_id: str
    causal_contract_id: str

    def _wire(self) -> dict[str, Any]:
        return {**asdict(self), "quantities": [quantity._wire() for quantity in self.quantities]}

    @classmethod
    def _from_wire(cls, wire: dict[str, Any]) -> DistributionIdentity:
        return cls(
            semantic=wire["semantic"],
            quantities=tuple(ScientificQuantity._from_wire(q) for q in wire["quantities"]),
            alignment=wire["alignment"],
            source_id=wire["source_id"],
            provider_id=wire["provider_id"],
            rng_id=wire["rng_id"],
            snapshot_id=wire["snapshot_id"],
            causal_contract_id=wire["causal_contract_id"],
        )


class JointDistributionArtifact:
    """Finite draws with typed scientific coordinates and portable identity."""

    def __init__(
        self,
        identity: DistributionIdentity,
        draws: NDArray[np.float64],
        *,
        weights: tuple[float, ...] | None = None,
        supported: tuple[bool, ...] | None = None,
        calibration: Literal["exact", "point_only", "measured", "unmeasured"] = "unmeasured",
        trust: Literal[
            "native_licensed", "external_attested", "verified_extension", "unverified"
        ] = "unverified",
    ) -> None:
        if not isinstance(draws, np.ndarray) or draws.dtype != np.float64 or draws.ndim != 2:
            raise ValueError("draws must be a two-dimensional float64 NumPy array")
        if draws.shape[0] > 100_000 or draws.shape[1] > 1_024 or draws.size * 8 > 16 * 1024 * 1024:
            raise ValueError("distribution draw array exceeds its bounds")
        metadata = {
            "version": 1,
            "identity": identity._wire(),
            "axes": ["draw", "quantity"],
            "shape": list(draws.shape),
            "weights": weights,
            "supported": supported,
            "calibration": calibration,
            "trust": trust,
        }
        self._native = _NativeJointDistributionArtifact(json.dumps(metadata), draws)
        self._identity = identity

    @classmethod
    def load(cls, data: bytes, *, expected_identity: DistributionIdentity) -> JointDistributionArtifact:
        """Load only under the consumer's independently retained identity."""
        native = _NativeJointDistributionArtifact.load(
            data, json.dumps(expected_identity._wire())
        )
        obj = cls.__new__(cls)
        obj._native = native
        obj._identity = expected_identity
        return obj

    @property
    def semantic(self) -> DistributionMeaning:
        return self._identity.semantic

    @property
    def shape(self) -> tuple[int, int]:
        return self._native.shape

    @property
    def n_draws(self) -> int:
        return self._native.n_draws

    @property
    def axes(self) -> tuple[str, str]:
        axes = self._native.axes
        return (axes[0], axes[1])

    @property
    def quantities(self) -> tuple[ScientificQuantity, ...]:
        return self._identity.quantities

    @property
    def identity(self) -> DistributionIdentity:
        return self._identity

    @property
    def weights(self) -> tuple[float, ...] | None:
        values = json.loads(self._native.metadata_json)["weights"]
        return None if values is None else tuple(values)

    @property
    def supported(self) -> tuple[bool, ...] | None:
        values = json.loads(self._native.metadata_json)["supported"]
        return None if values is None else tuple(values)

    @property
    def calibration(self) -> str:
        return json.loads(self._native.metadata_json)["calibration"]

    @property
    def trust(self) -> str:
        return json.loads(self._native.metadata_json)["trust"]

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> NDArray[Any]:
        """Return a NumPy-owned copy; explicit ``copy=False`` cannot avoid it."""
        if copy is False:
            raise ValueError("JointDistributionArtifact requires one bounded copy into NumPy")
        array = self._native.draws_copy()
        return np.asarray(array, dtype=dtype)

    def export(self, artifact_id: str) -> bytes:
        return self._native.export(artifact_id)

    def mean(self, coordinate: int) -> float:
        return self._native.mean(coordinate)

    def covariance(self, left: int, right: int) -> float:
        return self._native.covariance(left, right)

    def joint_product_expectation(self, left: int, right: int) -> float:
        return self._native.joint_product_expectation(left, right)


__all__ = [
    "DistributionIdentity",
    "DistributionMeaning",
    "DrawAlignment",
    "JointDistributionArtifact",
    "QuantityCondition",
    "ScientificQuantity",
]
