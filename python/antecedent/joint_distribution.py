"""Typed, bounded 2.3 aligned distribution artifacts.

``numpy.asarray(artifact)`` makes one bounded copy into NumPy-owned row-major
storage. The array never aliases the validated Rust artifact. Loading requires
an expected identity supplied by the consumer, separate from the input bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from ._native import JointDistributionArtifact as _NativeJointDistributionArtifact
from .errors import CausalTypeError, CausalValueError

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
QuantityRole = Literal["treatment", "outcome", "covariate", "mediator", "selection", "utility"]


@dataclass(frozen=True, slots=True)
class QuantityCondition:
    variable_id: str
    value_id: str


@dataclass(frozen=True, slots=True)
class ScientificQuantity:
    variable_id: str
    variable_name: str
    role: QuantityRole
    units: str
    population_id: str
    regime_id: str
    horizon: int
    functional_id: str
    conditioning: tuple[QuantityCondition, ...] = ()
    transform_id: str = "identity"

    @classmethod
    def of(
        cls,
        role: QuantityRole,
        variable: str,
        *,
        units: str,
        population: str,
        regime: str,
        horizon: int = 0,
        functional: str = "mean",
        conditioning: Iterable[QuantityCondition] = (),
        transform: str = "identity",
        variable_id: str | None = None,
    ) -> ScientificQuantity:
        """Declare a quantity without spelling every wire field.

        Only the genuinely conventional fields have defaults: ``horizon=0`` (a static,
        single-period quantity), ``functional="mean"`` (the expectation), ``transform=
        "identity"`` (no transform) and ``variable_id`` (the variable's own name; pass the
        schema identity when the two differ). The scientific meaning is never defaulted:
        ``units``, ``population`` (the population the quantity is about, for example
        ``"target"``) and ``regime`` (the intervention or observation regime, for example
        ``"do(a=1)"`` or ``"observational"``) are required, as is the ``role``.

        Raises:
            CausalValueError: a required text field is blank, or ``horizon`` is not a
                non-negative integer.
        """
        if role not in get_args(QuantityRole):
            raise CausalValueError(f"role must be one of {get_args(QuantityRole)}, got {role!r}")
        for name, value in (
            ("variable", variable),
            ("units", units),
            ("population", population),
            ("regime", regime),
            ("functional", functional),
            ("transform", transform),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name}= must be a non-empty string")
        if variable_id is not None and (not isinstance(variable_id, str) or not variable_id):
            raise CausalValueError("variable_id= must be a non-empty string when given")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise CausalValueError("horizon= must be a non-negative integer")
        if isinstance(conditioning, (str, bytes)) or not isinstance(conditioning, Iterable):
            raise CausalTypeError("conditioning= must contain QuantityCondition values")
        conditions = tuple(conditioning)
        if any(not isinstance(condition, QuantityCondition) for condition in conditions):
            raise CausalTypeError("conditioning= must contain QuantityCondition values")
        for condition in conditions:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in (condition.variable_id, condition.value_id)
            ):
                raise CausalValueError("conditioning identities must be non-empty strings")
        return cls(
            variable_id=variable_id or variable,
            variable_name=variable,
            role=role,
            units=units,
            population_id=population,
            regime_id=regime,
            horizon=horizon,
            functional_id=functional,
            conditioning=conditions,
            transform_id=transform,
        )

    @classmethod
    def outcome(cls, variable: str, **kwargs: Any) -> ScientificQuantity:
        """An outcome quantity; keywords as :meth:`of` (``units``, ``population``, ``regime``)."""
        return cls.of("outcome", variable, **kwargs)

    @classmethod
    def treatment(cls, variable: str, **kwargs: Any) -> ScientificQuantity:
        """A treatment quantity; keywords as :meth:`of`."""
        return cls.of("treatment", variable, **kwargs)

    @classmethod
    def covariate(cls, variable: str, **kwargs: Any) -> ScientificQuantity:
        """A covariate quantity; keywords as :meth:`of`."""
        return cls.of("covariate", variable, **kwargs)

    @classmethod
    def mediator(cls, variable: str, **kwargs: Any) -> ScientificQuantity:
        """A mediator quantity; keywords as :meth:`of`."""
        return cls.of("mediator", variable, **kwargs)

    @classmethod
    def utility(cls, variable: str, **kwargs: Any) -> ScientificQuantity:
        """A utility quantity; keywords as :meth:`of`."""
        return cls.of("utility", variable, **kwargs)

    @classmethod
    def from_effect(
        cls, result: Any, *, outcome_units: str, population: str = "target"
    ) -> ScientificQuantity:
        """The original checked ATE contrast, not an absolute outcome mean.

        Units label the original outcome scale; this performs no conversion.
        Only an unchanged, point-identified static mean ATE is supported. The
        native producer supplies the actual variable, intervention levels and
        ``mean_difference`` semantics. Aggregate results supply no outcome law.
        """
        from .errors import CausalUnsupportedError
        from .external import ExternalRefusal
        from .results import AnalysisResult

        if not isinstance(result, AnalysisResult):
            raise CausalTypeError("from_effect needs an AnalysisResult")
        native = getattr(getattr(result, "_raw", None), "effect_source_json", None)
        if native is None:
            raise CausalUnsupportedError(
                "the result has no original checked static ATE source",
                reason_code="route_not_supported",
            )
        if not isinstance(outcome_units, str) or not outcome_units.strip():
            raise CausalValueError("outcome_units must be a non-empty string")
        wire, refusal = native(outcome_units, population, result.as_point())
        if refusal is not None:
            raise ExternalRefusal(json.loads(refusal))
        if wire is None:
            raise CausalValueError("the native effect source returned no result")
        return cls._from_wire(json.loads(wire)["coordinates"][0])

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        outcome_units: str,
        population: str = "target",
        transform: str = "identity",
    ) -> tuple[ScientificQuantity, ...]:
        """The coordinates of a response-curve result's values, one per grid point.

        ``response`` is the result of ``antecedent.analyze(..., query=ResponseCurve(...))``
        (a :class:`~antecedent.results.CausalResponseView`). The coordinates are those the
        result itself carries (``response.quantities``) or derives natively
        (``response.response_coordinates``); a result with no native payload falls back to
        the derivation from its query, which is the same one. ``outcome_units`` is required
        because a response cannot know it; nothing is converted. The tuple is in grid order;
        :meth:`from_response_dose` picks one grid point.

        Raises:
            CausalTypeError: ``response`` is not a response-curve result.
            CausalValueError: units are blank.
        """
        if not hasattr(response, "response_coordinates") or not hasattr(response, "response"):
            raise CausalTypeError(
                "ScientificQuantity.from_response needs a response-curve analysis result, "
                f"not {type(response).__name__}"
            )
        if not isinstance(outcome_units, str) or not outcome_units.strip():
            raise CausalValueError("outcome_units= is required: units are never inferred")
        if response.quantities is not None:
            coordinates = tuple(response.quantities)
            if any(
                coordinate.units != outcome_units
                or coordinate.population_id != population
                or coordinate.transform_id != transform
                for coordinate in coordinates
            ):
                raise CausalValueError(
                    "requested units, population or transform conflict with retained coordinates"
                )
        elif getattr(response, "_raw", None) is not None:
            coordinates = tuple(
                response.response_coordinates(
                    outcome_units=outcome_units, population=population, transform=transform
                )
            )
        else:
            from .results.coordinates import response_coordinates

            coordinates = response_coordinates(
                response.estimand,
                outcome_units=outcome_units,
                population=population,
                transform=transform,
            )
        return coordinates

    @classmethod
    def from_response_dose(
        cls,
        response: Any,
        dose: float,
        *,
        outcome_units: str,
        population: str = "target",
        transform: str = "identity",
    ) -> ScientificQuantity:
        """The coordinate of a response-curve result at one grid ``dose``.

        Raises:
            CausalValueError: ``dose`` is not a grid point of the response.
        """
        coordinates = cls.from_response(
            response, outcome_units=outcome_units, population=population, transform=transform
        )
        view = response.response
        points = [] if view is None else [point[0] for point in view.points]
        if isinstance(dose, bool) or not isinstance(dose, (int, float)):
            raise CausalTypeError("dose must be a number")
        if not math.isfinite(dose):
            raise CausalValueError("dose must be finite")
        if len(points) != len(coordinates):
            raise CausalValueError("response coordinates do not match its grid")
        exact = [
            coordinate
            for point, coordinate in zip(points, coordinates, strict=True)
            if point == dose
        ]
        if len(exact) == 1:
            return exact[0]
        nearby = [
            coordinate
            for point, coordinate in zip(points, coordinates, strict=True)
            if math.isclose(point, dose, rel_tol=1e-12, abs_tol=1e-12)
        ]
        if len(nearby) == 1:
            return nearby[0]
        if nearby:
            raise CausalValueError(f"dose {dose!r} ambiguously matches multiple grid points")
        raise CausalValueError(f"dose {dose!r} is not a grid point of this response: {points}")

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
        if trust == "native_licensed":
            raise CausalValueError(
                "native_distribution.authority_required: native trust requires retained producer-issued execution state",
                reason_code="invalid_argument",
            )
        if not isinstance(draws, np.ndarray) or draws.dtype != np.float64 or draws.ndim != 2:
            raise CausalValueError("draws must be a two-dimensional float64 NumPy array")
        if draws.shape[0] > 100_000 or draws.shape[1] > 1_024 or draws.size * 8 > 16 * 1024 * 1024:
            raise CausalValueError("distribution draw array exceeds its bounds")
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

    @property
    def source_evidence(self):
        """Original issuer diagnostics; historical or caller-created laws have no issued source."""
        from .source_evidence import SourceEvidence

        handle = self._native.source_evidence
        return None if handle is None else SourceEvidence(handle)

    @classmethod
    def _from_native(cls, native: _NativeJointDistributionArtifact) -> JointDistributionArtifact:
        obj = cls.__new__(cls)
        obj._native = native
        obj._identity = DistributionIdentity._from_wire(
            json.loads(native.metadata_json)["identity"]
        )
        return obj

    @classmethod
    def load(
        cls, data: bytes, *, expected_identity: DistributionIdentity
    ) -> JointDistributionArtifact:
        """Load only under the consumer's independently retained identity."""
        native = _NativeJointDistributionArtifact.load(data, json.dumps(expected_identity._wire()))
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

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        """Return a NumPy-owned copy; explicit ``copy=False`` cannot avoid it."""
        if copy is False:
            raise CausalValueError("JointDistributionArtifact requires one bounded copy into NumPy")
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
