"""Typed, bounded aligned distribution artifacts.

``numpy.asarray(artifact)`` makes one bounded copy into NumPy-owned row-major
storage. The array never aliases the validated Rust artifact. Loading requires
an expected identity supplied by the consumer, separate from the input bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._native import JointDistributionArtifact as _NativeJointDistributionArtifact
from ._scientific import ScientificResult
from ._validate import count as _count

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact
from .errors import CausalTypeError, CausalValueError
from .extensibility import ProviderTrust

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
    """One conditioning event of a :class:`ScientificQuantity`, bound by stable identities.

    A quantity conditioned on ``QuantityCondition("sex", "female")`` is about the subgroup in
    which variable ``variable_id`` takes value ``value_id`` (for example a conditional mean).

    Fields:

    * ``variable_id`` -- stable identity of the conditioning variable.
    * ``value_id`` -- stable identity of the value or event it takes.

    The constructor does not check its fields; :meth:`ScientificQuantity.of` raises
    ``CausalValueError`` when either is blank.
    """

    variable_id: str
    value_id: str


@dataclass(frozen=True, slots=True)
class ScientificQuantity:
    """The scientific meaning of one scalar column of a joint distribution.

    A quantity says which variable a number describes, in what role and units, for which
    population, under which intervention or observation regime, at which horizon, by which
    functional (for example the mean), conditional on which events and on which scale. Two
    columns with the same coordinate are the same quantity; a
    :class:`JointDistributionArtifact` refuses duplicates. The coordinate carries semantics
    only: declaring it does not make a number a supported causal estimate, and units are never
    converted.

    Build one with :meth:`of` (or the role shortcuts :meth:`outcome`, :meth:`treatment`,
    :meth:`covariate`, :meth:`mediator`, :meth:`utility`), or read it from a result with
    :meth:`from_effect`, :meth:`from_response` or :meth:`from_response_dose`.

    Fields:

    * ``variable_id`` -- stable, schema-qualified variable identity.
    * ``variable_name`` -- display name; never the sole identity.
    * ``role`` -- ``"treatment"``, ``"outcome"``, ``"covariate"``, ``"mediator"``,
      ``"selection"`` or ``"utility"``.
    * ``units`` -- declared units; no implicit conversion.
    * ``population_id`` -- the population the quantity is about.
    * ``regime_id`` -- the intervention or observation regime.
    * ``horizon`` -- ordered causal horizon, an integer in ``[0, 2**32 - 1]``.
    * ``functional_id`` -- the functional, for example ``"mean"``.
    * ``conditioning`` -- :class:`QuantityCondition` events, default ``()``.
    * ``transform_id`` -- scale or transform identity, default ``"identity"``.

    Raises:
        CausalValueError: ``horizon`` is not an integer in range (reason code
            ``invalid_argument``). The direct constructor checks nothing else; :meth:`of` checks
            every field, and the native artifact refuses an invalid coordinate when it is used.
    """

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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon",
            _count(
                self.horizon,
                "horizon",
                minimum=0,
                maximum=2**32 - 1,
                type_error=CausalValueError,
                reason_code="invalid_argument",
            ),
        )
        object.__setattr__(self, "conditioning", tuple(self.conditioning))

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> ScientificQuantity:
        """Restore a quantity from its native descriptor mapping without changing semantics.

        ``wire`` holds the field names of this class; ``conditioning`` defaults to ``()`` and
        ``transform_id`` to ``"identity"`` when absent.

        Raises:
            KeyError: a required field is missing.
            CausalValueError: ``horizon`` is out of range.
        """
        return cls(
            wire["variable_id"],
            wire["variable_name"],
            wire["role"],
            wire["units"],
            wire["population_id"],
            wire["regime_id"],
            wire["horizon"],
            wire["functional_id"],
            tuple(QuantityCondition(**item) for item in wire.get("conditioning", ())),
            wire.get("transform_id", "identity"),
        )

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

        ``conditioning`` is an iterable of :class:`QuantityCondition`; ``variable`` becomes
        ``variable_name``.

        Raises:
            CausalValueError: ``role`` is not one of the six roles, a required text field or a
                condition identity is blank, ``variable_id`` is given but empty, or ``horizon``
                is not an integer in ``[0, 2**32 - 1]``.
            CausalTypeError: ``conditioning`` is a string or holds anything other than
                :class:`QuantityCondition`.
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
        horizon = _count(
            horizon,
            "horizon",
            minimum=0,
            maximum=2**32 - 1,
            type_error=CausalValueError,
            reason_code="invalid_argument",
        )
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
        """The coordinate of a checked average treatment effect: a contrast, not an outcome mean.

        ``result`` is an :class:`~antecedent.results.AnalysisResult` (or a successful
        :class:`~antecedent.outcomes.Outcome` holding one) of an unchanged, point-identified
        static mean ATE. The native producer supplies the variable, the intervention levels and
        the ``mean_difference`` functional; only ``outcome_units`` (a label for the original
        outcome scale, never a conversion) and ``population`` (default ``"target"``) are yours.
        An aggregate result supplies no outcome law.

        Returns:
            The single :class:`ScientificQuantity` of the effect.

        Raises:
            CausalTypeError: ``result`` is not an ``AnalysisResult``.
            CausalUnsupportedError: the result carries no original checked static ATE source
                (reason code ``route_not_supported``).
            CausalValueError: ``outcome_units`` is blank.
            ExternalRefusal: the native producer refuses the result as a source.
            CausalError: ``result`` is a refused outcome; its original error is re-raised.
        """
        from .outcomes import require_success

        result = require_success(result)
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

        Parameters ``population`` (default ``"target"``) and ``transform`` (default
        ``"identity"``) label the coordinates.

        Returns:
            One :class:`ScientificQuantity` per grid point, in grid order.

        Raises:
            CausalTypeError: ``response`` is not a response-curve result.
            CausalValueError: units are blank, or the requested units, population or transform
                conflict with the coordinates the result retains.
        """
        from .outcomes import require_success

        response = require_success(response)
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

        Keywords as :meth:`from_response`. ``dose`` must equal a grid point; a value within an
        absolute and relative tolerance of ``1e-12`` of exactly one grid point also matches.
        No interpolation is done.

        Raises:
            CausalTypeError: ``dose`` is not a number.
            CausalValueError: ``dose`` is not finite, matches no grid point or several, or the
                result's coordinates do not match its grid; and as :meth:`from_response`.
        """
        from .outcomes import require_success

        response = require_success(response)
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
    """The scientific and provenance identity that a producer and a consumer of draws agree on.

    A consumer keeps its own copy of the identity, derived from its causal contract, and
    passes it to :meth:`JointDistributionArtifact.load`; loading refuses an artifact whose
    stored identity differs in any field, including quantity order.

    Fields:

    * ``semantic`` -- what each draw is (:data:`DistributionMeaning`): ``"parameter_posterior"``,
      ``"causal_functional_posterior"``, ``"posterior_predictive"`` (observational),
      ``"interventional_predictive"``, ``"estimator_sampling"`` (repeated-sampling estimator
      distribution), ``"bootstrap"`` (conditional on one data snapshot) or
      ``"empirical_outcome"`` (the observed outcome law).
    * ``quantities`` -- the :class:`ScientificQuantity` of each column, in column order. At
      least one, at most 1024, no two with the same coordinate.
    * ``alignment`` -- ``"joint"`` when each row is one joint realization across every column,
      ``"independent_marginals"`` when columns were sampled independently and rows must not be
      paired.
    * ``source_id`` -- stable source or study identity.
    * ``provider_id`` -- stable identity of the provider object that produced the draws.
    * ``rng_id`` -- random-number algorithm, seed and stream identity, or
      ``"deterministic_exact"``.
    * ``snapshot_id`` -- data snapshot or exact-law identity.
    * ``causal_contract_id`` -- identity of the independently checked causal contract.

    Every provenance identity must be nonblank. The dataclass does not check its fields; the
    native artifact does, when the identity is used to build or load one.
    """

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


class JointDistributionArtifact(ScientificResult):
    """A finite table of draws whose columns carry typed scientific coordinates and an identity.

    Rows are draws and columns are the :class:`ScientificQuantity` values of
    ``identity.quantities``, in order. With ``identity.alignment == "joint"`` each row is one
    joint realization, so covariances and nonlinear expectations across columns are defined;
    with ``"independent_marginals"`` only per-column summaries are. The artifact is evidence
    about a law, not a calibration result: ``calibration`` and ``trust`` are declarations
    that loading never upgrades.

    Constructor parameters:

    * ``identity`` -- the :class:`DistributionIdentity`.
    * ``draws`` -- a two-dimensional ``float64`` NumPy array of shape
      ``(n_draws, len(identity.quantities))``, every value finite. At most 100 000 draws,
      1024 columns and 16 MiB. It is copied.
    * ``weights`` -- optional nonnegative finite per-draw weights with a positive total,
      normalized when evaluated. Default ``None`` weighs draws equally.
    * ``supported`` -- optional per-column support mask; summaries of an unsupported column
      are refused. Default ``None`` marks every column supported.
    * ``calibration`` -- the declared inferential status: ``"exact"`` (an exact finite law
      from a checked provider), ``"point_only"`` (descriptive point summaries only),
      ``"measured"`` (a separately cited, coordinate-specific calibration record exists) or
      ``"not_measured"`` (default).
    * ``claimed_trust`` -- a caller label, not a grant: ``"unverified"`` (default) or
      ``"externally_attested"``. ``"native_supported"`` needs retained producer-issued
      execution state and ``"verified_extension"`` an exact-request verification receipt, so
      both are refused here. ``trust`` is the same argument under its older name.

    Read :attr:`identity`, :attr:`calibration` and :attr:`trust` first, then summarize with
    :meth:`mean`, :meth:`covariance` and :meth:`joint_product_expectation`. ``numpy.asarray``
    returns a bounded NumPy-owned copy of the draws.

    Raises:
        CausalValueError: both ``claimed_trust`` and ``trust`` are given; the trust is unknown,
            ``"native_supported"`` or ``"verified_extension"`` (reason code
            ``invalid_argument``); ``draws`` is not a two-dimensional ``float64`` array, has no
            rows or exceeds its bounds.
        CausalSerializationError: the native artifact rejects the identity, draws, weights or
            mask (for example a non-finite draw, a column count that differs from the
            quantities, a duplicate coordinate or a blank provenance identity).
    """

    def __init__(
        self,
        identity: DistributionIdentity,
        draws: NDArray[np.float64],
        *,
        weights: tuple[float, ...] | None = None,
        supported: tuple[bool, ...] | None = None,
        calibration: Literal["exact", "point_only", "measured", "not_measured"] = "not_measured",
        claimed_trust: str | None = None,
        trust: str | None = None,
    ) -> None:
        if claimed_trust is not None and trust is not None:
            raise CausalValueError("pass claimed_trust once")
        raw = (
            "unverified"
            if claimed_trust is None and trust is None
            else (claimed_trust or trust or "")
        )
        try:
            parsed = ProviderTrust(raw)
        except ValueError as error:
            raise CausalValueError(f"unknown distribution trust {raw!r}") from error
        if parsed is ProviderTrust.NATIVE_SUPPORTED:
            raise CausalValueError(
                "native_distribution.authority_required: native trust requires retained producer-issued execution state",
                reason_code="invalid_argument",
            )
        if parsed is ProviderTrust.VERIFIED_EXTENSION:
            raise CausalValueError(
                "distribution.verified_extension_requires_receipt: verified_extension is not a caller claim; it requires an exact-request verification receipt",
                reason_code="invalid_argument",
            )
        # The artifact codec still stores `external_attested`; the property returns the canonical wire.
        trust = "external_attested" if parsed is ProviderTrust.EXTERNALLY_ATTESTED else parsed.value
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
        """The issuing producer's diagnostics, or ``None`` when the law has no issued source.

        A :class:`~antecedent.source_evidence.SourceEvidence` for a law issued by a native
        producer; ``None`` for a law built by the caller or loaded from bytes.
        """
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
    @artifact_consumer(
        proof="joint_distribution.JointDistributionArtifact.load: independent semantic consumer checks declarations, identities and stored result",
        replayable=False,
        input="data",
    )
    def load(
        cls, data: bytes, *, expected_identity: DistributionIdentity
    ) -> JointDistributionArtifact:
        """Load exported bytes, checked against the consumer's independently retained identity.

        ``data`` is the bytes or :class:`~antecedent.artifacts.ExportedArtifact` from
        :meth:`export`. ``expected_identity`` must come from the consumer's own causal
        contract, not from the artifact being loaded; a resealed change in population, regime,
        quantity order, provider or snapshot then fails. Every check runs before any numerical
        operation. The loaded artifact has no :attr:`source_evidence`.

        Raises:
            CausalTypeError: ``data`` is neither bytes nor an ``ExportedArtifact``.
            CausalSerializationError: the bytes are oversized, truncated, corrupt or of an
                unsupported layout, or the stored identity differs from ``expected_identity``
                (reason code ``quantity_semantics_mismatch``).
        """
        data = container_bytes(data)
        native = _NativeJointDistributionArtifact.load(data, json.dumps(expected_identity._wire()))
        obj = cls.__new__(cls)
        obj._native = native
        obj._identity = expected_identity
        return obj

    @property
    def answer(self) -> Any:
        """``Answer(kind="distribution")`` with ``detail`` set to :attr:`semantic`."""
        from .outcomes import Answer

        return Answer("distribution", detail=str(self.semantic))

    @property
    def semantic(self) -> DistributionMeaning:
        """What each draw is; ``identity.semantic``."""
        return self._identity.semantic

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_draws, n_quantities)``."""
        return self._native.shape

    @property
    def n_draws(self) -> int:
        """Number of draws (rows)."""
        return self._native.n_draws

    @property
    def axes(self) -> tuple[str, str]:
        """Axis names in physical order, always ``("draw", "quantity")``."""
        axes = self._native.axes
        return (axes[0], axes[1])

    @property
    def quantities(self) -> tuple[ScientificQuantity, ...]:
        """The coordinate of each column, in column order; ``identity.quantities``."""
        return self._identity.quantities

    @property
    def identity(self) -> DistributionIdentity:
        """The :class:`DistributionIdentity` the artifact was built or loaded under."""
        return self._identity

    @property
    def weights(self) -> tuple[float, ...] | None:
        """Per-draw weights as stored (not normalized), or ``None`` for equal weights."""
        values = json.loads(self._native.metadata_json)["weights"]
        return None if values is None else tuple(values)

    @property
    def supported(self) -> tuple[bool, ...] | None:
        """Per-column support mask, or ``None`` when every column is supported."""
        values = json.loads(self._native.metadata_json)["supported"]
        return None if values is None else tuple(values)

    @property
    def calibration(self) -> str:
        """The declared calibration status of the draws.

        One of ``"exact"``, ``"point_only"``, ``"measured"`` or ``"not_measured"``; a
        declaration carried with the draws, not itself a calibration result.
        """
        return json.loads(self._native.metadata_json)["calibration"]

    @property
    def trust(self) -> str:
        """The declared provider trust, in the canonical wire spelling.

        One of ``"native_supported"``, ``"verified_extension"``, ``"externally_attested"`` or
        ``"unverified"``; the stored spelling ``external_attested`` reads as
        ``externally_attested``. Loading never upgrades it.
        """
        stored = json.loads(self._native.metadata_json)["trust"]
        return ProviderTrust(stored).value

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        """Return a NumPy-owned copy; explicit ``copy=False`` cannot avoid it."""
        if copy is False:
            raise CausalValueError("JointDistributionArtifact requires one bounded copy into NumPy")
        array = self._native.draws_copy()
        return np.asarray(array, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        """The stored metadata as a dictionary, with ``shape`` and ``n_draws``; no draws.

        Keys include ``identity``, ``axes``, ``weights``, ``supported``, ``calibration`` and
        ``trust`` (in the stored spelling).
        """
        return {
            **json.loads(self._native.metadata_json),
            "shape": list(self.shape),
            "n_draws": self.n_draws,
        }

    @artifact_export
    def export(self, artifact_id: str) -> ExportedArtifact[JointDistributionArtifact]:
        """Serialize the identity, metadata and draws as an artifact named ``artifact_id``.

        Returns an :class:`~antecedent.artifacts.ExportedArtifact`; ``bytes(exported)`` gives
        the container. Read it back with :meth:`load` and an independently retained identity.

        Raises:
            CausalSerializationError: the artifact cannot be encoded.
        """
        return artifact_from_receipt(self, self._native.export(artifact_id))

    def mean(self, coordinate: int) -> float:
        """Weighted mean of column ``coordinate`` (zero-based) over the draws.

        Raises:
            CausalSerializationError: the column is out of range or unsupported, or the mean
                overflows.
        """
        return self._native.mean(coordinate)

    def covariance(self, left: int, right: int) -> float:
        """Weighted population covariance of columns ``left`` and ``right`` over paired draws.

        Divides by the total weight, not by ``n - 1``.

        Raises:
            CausalSerializationError: either column is out of range or unsupported, the
                artifact holds independent marginals (reason code ``joint_law_required``), or
                the value overflows.
        """
        return self._native.covariance(left, right)

    def joint_product_expectation(self, left: int, right: int) -> float:
        """Weighted mean of the per-draw product of columns ``left`` and ``right``, E[XY].

        Raises:
            CausalSerializationError: as :meth:`covariance`.
        """
        return self._native.joint_product_expectation(left, right)


__all__ = [
    "DistributionIdentity",
    "DistributionMeaning",
    "DrawAlignment",
    "JointDistributionArtifact",
    "QuantityCondition",
    "ScientificQuantity",
]
