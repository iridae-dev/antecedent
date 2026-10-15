"""A retained fitted effect model decides for a declared cohort.

A fitted CATE map predicts conditional treatment contrasts. Those are differences: they give
neither two absolute potential-outcome means nor a law of the contrast. The one decision a
cohort of them supports is the expectation of an affine utility of the cohort's weighted mean
contrast. This module binds that quantity natively::

    model = antecedent.prediction.FittedEffectModel.load(result.export())
    cohort = cohort_decision.CohortBinding(
        snapshot_id="clinic-2026-q1",
        population="target",
        columns={"z": [-1.0, 0.5, 1.0]},
        weights=[0.2, 0.3, 0.5],
    )
    source = model.decision_source(cohort=cohort, outcome_units="mmHg")
    decision = contract.evaluate(source)          # affine utilities only

Rust owns every number, the support check, the identity, the artifact and every refusal.
Python declares the cohort and the retained model's bytes; it never supplies a prediction.
Every row is checked against the retained model's own support contract; a model with no
contract is refused, never assumed supported. A cohort from another population is accepted only
under a declared :class:`CovariateShift`, is labelled ``extrapolative`` and is point-only.

The value is point-only: no utility standard error, regret or interval is inherited. A
probability, quantile, nonlinear utility or joint law refuses (``decision_contract_unsatisfied``
or ``joint_law_required``). For a linear compact model that retains its coefficient covariance
the variance of the weighted cohort mean is available as
:attr:`CohortDecisionSource.uncalibrated_standard_error`, labelled model-based with calibration
unmeasured; the interval route is closed (``cell_not_licensed``).

A source built from a verified analysis result is native. A source built from a caller-built
compact export is a declared model and is never native.

A cohort of another population can also be carried under a declared selection-diagram premise
(:func:`transport_cohort_effect`): the selection node points only into the declared effect
modifiers, which are exactly the model's features, and every row lies inside the fitted support.
The result is a :class:`TransportedCohortEffect` with a typed :class:`CohortTransportReceipt`,
its identity, a point-only value and the existing :class:`CohortDecisionSource` for the same
cohort. An interval or a law is refused with the evidence that would license it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._columns import coerce_fields
from ._freeze import thaw
from ._measured_inference import MeasuredInference
from ._native import build_cohort_decision_source as _build
from ._native import cohort_decision_forward_claim as _forward_claim
from ._native import cohort_decision_interval as _interval
from ._native import cohort_decision_law_refusal as _law_refusal
from ._native import fit_native_cohort_effect as _fit_native
from ._native import consume_cohort_decision_source as _consume
from ._native import normalize_cohort_weights as _normalize
from ._native import transport_cohort_effect as _transport
from ._native import transport_cohort_effect_verify as _transport_verify
from ._native import transported_cohort_refusal as _transported_refusal
from ._native import validate_cohort_binding as _validate
from ._refusal import StructuredRefusal
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import ScientificQuantity

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact
    from .compact_export import CompactExport
    from .decision import MeanSource
    from .inverse_query import MeanClaim

__all__ = [
    "CohortBinding",
    "CohortDecisionRefusal",
    "CohortDecisionSource",
    "CohortTransportReceipt",
    "CovariateShift",
    "NativeEffectFit",
    "SelectionShift",
    "TransportedCohortEffect",
    "UncalibratedStandardError",
    "consume_cohort_decision_source",
    "decision_source_from_compact_export",
    "decision_source_from_model",
    "decision_source_from_native_fit",
    "fit_native_cohort_effect",
    "transport_cohort_effect",
]

_DEFAULT_ARTIFACT_ID = "cohort_decision_source"


class CohortDecisionRefusal(StructuredRefusal):
    """A typed refusal of the cohort decision source.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``cohort_decision.*`` ``detail`` (for example ``cohort_decision.outside_support``,
    ``cohort_decision.support_contract_absent``, ``cohort_decision.wrong_parent_identity`` or
    ``cohort_decision.interval_not_licensed``); ``offending`` names the row, feature, region or
    identity and ``remedy`` the declaration or evidence that would license the request.
    """


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _refusal(code: str, detail: str, offending: str, message: str) -> CohortDecisionRefusal:
    return CohortDecisionRefusal(
        {
            "code": code,
            "stage": "bind",
            "detail": detail,
            "offending": offending,
            "message": message,
            "remedy": None,
        }
    )


@dataclass(frozen=True, slots=True)
class CovariateShift:
    """A declared covariate shift between the model's population and the cohort's.

    The conditional contrast is declared invariant across the two populations given exactly
    ``invariant_modifiers``, which must be the model's features. The declaration cannot be
    verified from data; the result is point-only evidence under it.
    """

    source_population: str
    target_population: str
    invariant_modifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_population, "source_population")
        _text(self.target_population, "target_population")
        if isinstance(self.invariant_modifiers, str):
            raise CausalTypeError("invariant_modifiers must be an iterable of names")
        coerce_fields(
            self,
            invariant_modifiers=tuple(_text(m, "modifier") for m in self.invariant_modifiers),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "source_population": self.source_population,
            "target_population": self.target_population,
            "invariant_modifiers": list(self.invariant_modifiers),
        }


@dataclass(frozen=True, slots=True)
class UncalibratedStandardError:
    """The model-based standard error of the weighted cohort mean of a linear compact model.

    ``sqrt(phibar' V phibar)``. Not an interval, not a coverage claim, never passed to a decision;
    ``calibration`` is ``"calibration unmeasured"``.
    """

    value: float
    basis: str
    scope: str
    calibration: str


class CohortBinding:
    """A declared cohort: named feature rows, row identities and normalized weights.

    ``columns`` maps each feature name to one finite number or one level string per row.
    ``weights`` must be finite, nonnegative and sum to one; pass ``normalize=True`` to ask for
    an explicit normalization of raw weights, or omit them for equal weights. ``row_ids``
    default to ``row-0``, ``row-1``, ... The digest binds every field natively.
    """

    __slots__ = ("_columns", "_digest", "_population", "_row_ids", "_snapshot", "_weights")

    def __init__(
        self,
        *,
        snapshot_id: str,
        population: str,
        columns: Mapping[str, Any],
        weights: Sequence[float] | None = None,
        row_ids: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> None:
        _text(snapshot_id, "snapshot_id")
        _text(population, "population")
        if not isinstance(columns, Mapping) or not columns:
            raise CausalTypeError("columns must be a non-empty mapping of feature -> values")
        wired: list[tuple[str, dict[str, list[Any]]]] = []
        rows: int | None = None
        for name, values in columns.items():
            _text(name, "feature name")
            # Preserve element types while deciding whether this is a categorical or numeric
            # feature. A normal ``np.asarray`` coerces a mixed list such as ``[1, "a"]`` to
            # strings, which would silently change the caller's numeric value into a level.
            array = np.asarray(values, dtype=object)
            if array.ndim != 1:
                raise CausalValueError(f"feature {name!r} must be one-dimensional")
            rows = len(array) if rows is None else rows
            items = array.tolist()
            if any(isinstance(v, str) for v in items):
                if not all(isinstance(v, str) for v in items):
                    raise CausalTypeError(f"feature {name!r} mixes levels and numbers")
                wired.append((name, {"categorical": items}))
            else:
                numbers = np.asarray(array, dtype=np.float64)
                if not np.isfinite(numbers).all():
                    raise _refusal(
                        "invalid_argument",
                        "cohort_decision.cohort_invalid",
                        name,
                        "cohort features must be finite",
                    )
                wired.append((name, {"numeric": numbers.tolist()}))
        assert rows is not None
        if weights is None:
            if rows == 0:
                raise CausalValueError("a cohort has at least one row")
            raw = [1.0 / rows] * rows
        else:
            raw = [float(w) for w in weights]
            if any(not math.isfinite(w) for w in raw):
                raise _refusal(
                    "invalid_argument",
                    "cohort_decision.weights_invalid",
                    "weights",
                    "weights must be finite",
                )
            if normalize:
                text, refusal = _normalize(raw)
                if refusal is not None:
                    raise CohortDecisionRefusal(json.loads(refusal))
                raw = [float(w) for w in json.loads(text)]  # type: ignore[arg-type]
        if row_ids is None:
            ids = [f"row-{i}" for i in range(rows)]
        else:
            if isinstance(row_ids, (str, bytes)):
                raise CausalTypeError("row_ids must be a sequence of non-empty strings")
            ids = [_text(row_id, "row_id") for row_id in row_ids]
        self._snapshot = snapshot_id
        self._population = population
        self._row_ids = tuple(ids)
        self._columns = tuple(wired)
        self._weights = tuple(raw)
        digest, refusal = _validate(json.dumps(self._wire(), allow_nan=False))
        if refusal is not None:
            raise CohortDecisionRefusal(json.loads(refusal))
        assert digest is not None
        self._digest = digest

    def _wire(self) -> dict[str, Any]:
        return {
            "snapshot_id": self._snapshot,
            "population_id": self._population,
            "row_ids": list(self._row_ids),
            "features": [name for name, _ in self._columns],
            "columns": [column for _, column in self._columns],
            "weights": list(self._weights),
        }

    @property
    def snapshot_id(self) -> str:
        return self._snapshot

    @property
    def population(self) -> str:
        return self._population

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._columns)

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def digest(self) -> str:
        """The cohort snapshot digest every field is bound into (lowercase hex)."""
        return self._digest

    def __len__(self) -> int:
        return len(self._row_ids)

    def __repr__(self) -> str:
        return (
            f"CohortBinding(snapshot_id={self._snapshot!r}, population={self._population!r}, "
            f"rows={len(self)}, digest={self._digest[:12]}...)"
        )


def _options(shift: CovariateShift | None, expected_contrast: str | None) -> str:
    if shift is not None and not isinstance(shift, CovariateShift):
        raise CausalTypeError("shift= must be a CovariateShift")
    if expected_contrast is not None:
        _text(expected_contrast, "expected_contrast")
    return json.dumps(
        {
            "shift": None if shift is None else shift._wire(),
            "expected_contrast_regime": expected_contrast,
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class CohortDecisionSource(ScientificResult):
    """One native cohort-mean treatment-contrast source.

    ``identity`` is what a consumer retains out of band. ``quantity`` is the scientific
    coordinate (outcome, the model's contrast regime, functional ``mean_difference``, the
    cohort population, the cohort snapshot digest as conditioning); a decision contract names
    exactly that coordinate. Pass the source to :meth:`antecedent.decision.Contract.evaluate`:
    only the expectation of an affine utility is answered.
    """

    identity: str
    summary: Mapping[str, Any] = field(repr=False)
    _artifact_bytes: bytes = field(repr=False)
    extrapolative_accepted: bool = field(default=False, repr=False)
    _native_fit: str | None = field(default=None, repr=False)

    # -- values ------------------------------------------------------------------------------
    @property
    def mean_contrast(self) -> float:
        """The weighted mean of the conditional contrast over the cohort."""
        return float(self.summary["mean_contrast"])

    @property
    def row_contrasts(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.summary["row_contrasts"])

    @property
    def quantity(self) -> ScientificQuantity:
        return ScientificQuantity._from_wire(self.summary["quantity"])

    @property
    def is_native(self) -> bool:
        """True for a verified native analysis result; False for a declared compact model."""
        return bool(self.summary["is_native"])

    @property
    def parent_claim(self) -> str:
        return str(self.summary["parent_claim"])

    @property
    def cohort_digest(self) -> str:
        return str(self.summary["cohort_digest"])

    @property
    def population(self) -> str:
        return str(self.summary["population_id"])

    @property
    def outcome_units(self) -> str:
        return str(self.summary["outcome_units"])

    @property
    def support(self) -> Mapping[str, Any]:
        """The support receipt: basis, rows checked, rows outside (always 0), fitted rows."""
        return thaw(self.summary["support"])

    @property
    def support_status(self) -> str:
        """``supported`` inside the model's population, ``extrapolative`` under a shift."""
        return str(self.summary["support_status"])

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(self.summary["assumptions"])

    @property
    def uncalibrated_standard_error(self) -> UncalibratedStandardError | None:
        """Model-based diagnostic of a linear compact model; ``None`` otherwise. Never an interval."""
        raw = self.summary["uncalibrated_standard_error"]
        if raw is None:
            return None
        return UncalibratedStandardError(
            value=float(raw["value"]),
            basis=str(raw["basis"]),
            scope=str(raw["scope"]),
            calibration=str(raw["calibration"]),
        )

    # -- closed routes ---------------------------------------------------------------------
    def interval(self, *, level: float = 0.95) -> MeasuredInference:
        """The measured interval of the cohort mean of a natively fitted linear effect.

        Available for a source built with :func:`decision_source_from_native_fit`: the fit is
        replayed from its retained rows and the Student-t interval (HC3 covariance, ``n - p``
        degrees of freedom) is returned only when a currently attesting coverage record governs
        exactly this construction. A source from a declared compact export, a portable model or
        a shifted cohort raises a typed :class:`CohortDecisionRefusal`
        (``cohort_decision.interval_requires_native_fit`` or
        ``cohort_decision.interval_not_licensed``).
        """
        if isinstance(level, bool) or not isinstance(level, int | float) or not (
            0.0 < float(level) < 1.0
        ):
            raise CausalValueError("level= must be strictly between 0 and 1")
        measured, refusal = _interval(
            self._artifact_bytes, self._expectation(), self._native_fit, float(level)
        )
        if refusal is not None:
            raise CohortDecisionRefusal(json.loads(refusal))
        if measured is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native interval returned neither a result nor a refusal")
        return MeasuredInference._from_native(measured)

    def refuse(self, kind: str) -> None:
        """Raise the typed refusal of a request this source cannot answer.

        ``kind`` is ``probability``, ``quantile``, ``nonlinear_utility`` or ``joint_law``.
        """
        if kind == "interval":
            self.interval()
        raise CohortDecisionRefusal(json.loads(_law_refusal(kind)))

    # -- decision and replay -----------------------------------------------------------------
    def _expectation(self) -> str:
        return json.dumps({"identity": self.identity})

    def _verified_mean_source(self) -> Mapping[str, Any]:
        """The mean source as Rust replays it from the artifact under this identity."""
        summary, refusal = _consume(self._artifact_bytes, self._expectation())
        if refusal is not None:
            raise CohortDecisionRefusal(json.loads(refusal))
        assert summary is not None
        return dict(json.loads(summary)["mean_source"])

    @property
    def mean_source(self) -> MeanSource:
        """The single-coordinate :class:`~antecedent.decision.MeanSource` for a contract."""
        from .decision import MeanSource

        if self.support_status != "supported" and not self.extrapolative_accepted:
            raise CohortDecisionRefusal(
                {
                    "code": "cell_not_licensed",
                    "stage": "evaluate",
                    "detail": "cohort_decision.extrapolative_not_accepted",
                    "offending": self.population,
                    "message": "a cohort mean transported under a declared covariate shift is "
                    "extrapolative, point-only evidence under an assumption",
                    "remedy": "call accept_extrapolative() to decide on it explicitly",
                }
            )
        wire = self.summary["mean_source"]
        return MeanSource(
            coordinates=tuple(ScientificQuantity._from_wire(q) for q in wire["coordinates"]),
            means=tuple(wire["means"]),
            provider_id=wire["provider_id"],
            snapshot_id=wire["snapshot_id"],
            causal_contract_id=wire["causal_contract_id"],
            rng_id=wire["rng_id"],
            _original_cohort=self,
            trust="native_licensed" if self.is_native else "externally_attested",
            evidence=(self.parent_claim,),
        )

    def accept_extrapolative(self) -> CohortDecisionSource:
        """Acknowledge that the source rests on a declared covariate shift.

        The default decision path refuses an ``extrapolative`` source; this returns the same
        source with the acknowledgement recorded. It changes no number and no identity.
        """
        return replace(self, extrapolative_accepted=True)

    def _cohort_mean_source(self) -> MeanSource:
        return self.mean_source

    def forward_claim(self) -> MeanClaim:
        """The cohort mean as a forward claim of :mod:`antecedent.inverse_query`.

        A source built under a declared covariate shift is a transported claim and is refused
        (``cell_not_licensed``, ``inverse_forward_source.transported_not_accepted``); read it
        through :func:`antecedent.inverse_sources.cohort_forward_source` with
        ``accepts_transported=True`` so the result carries the transport identity.
        """
        from .inverse_query import MeanClaim

        text, refusal = _forward_claim(self._artifact_bytes, self._expectation())
        if refusal is not None:
            raise CohortDecisionRefusal(json.loads(refusal))
        assert text is not None
        wire = json.loads(text)
        return MeanClaim(
            coordinates=tuple(ScientificQuantity._from_wire(q) for q in wire["coordinates"]),
            means=tuple(wire["means"]),
            provider_id=wire["provider_id"],
            snapshot_id=wire["snapshot_id"],
            causal_contract_id=wire["causal_contract_id"],
            rng_id=wire["rng_id"],
        )

    def artifact(self) -> ExportedArtifact[CohortDecisionSource]:
        """Typed bytes envelope for shared scientific consumption."""
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self)

    def export(self) -> bytes:
        """The checksummed ``cohort_decision_source_v1`` artifact."""
        return self._artifact_bytes

    @classmethod
    def consume(
        cls,
        artifact: bytes,
        *,
        expected_identity: str,
        parent_claim: str | None = None,
        population: str | None = None,
        outcome_units: str | None = None,
        contrast: str | None = None,
        cohort_digest: str | None = None,
    ) -> CohortDecisionSource:
        """Independently replay ``artifact`` under what the consumer retained.

        Rebuilds the model and cohort from the stored fields, re-checks support and recomputes
        every number. Raises :class:`CohortDecisionRefusal` for a changed row, weight,
        coefficient, unit, population or parent (even when the container was resealed) and for
        any retained expectation that differs.
        """
        return consume_cohort_decision_source(
            artifact,
            expected_identity=expected_identity,
            parent_claim=parent_claim,
            population=population,
            outcome_units=outcome_units,
            contrast=contrast,
            cohort_digest=cohort_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return thaw(self.summary)


def _source_of(summary: str, artifact: bytes) -> CohortDecisionSource:
    parsed = json.loads(summary)
    return CohortDecisionSource(
        identity=str(parsed["identity"]), summary=parsed, _artifact_bytes=bytes(artifact)
    )


def _build_source(
    kind: str,
    model_bytes: bytes,
    expected: str | None,
    cohort: CohortBinding,
    outcome_units: str,
    shift: CovariateShift | None,
    expected_contrast: str | None,
    artifact_id: str,
) -> CohortDecisionSource:
    if not isinstance(cohort, CohortBinding):
        raise CausalTypeError("cohort= must be a CohortBinding")
    if not isinstance(outcome_units, str) or not outcome_units.strip():
        raise CausalValueError("outcome_units= is required: units are never inferred")
    summary, artifact, refusal = _build(
        kind,
        model_bytes,
        expected,
        json.dumps(cohort._wire(), allow_nan=False),
        outcome_units,
        _options(shift, expected_contrast),
        _text(artifact_id, "artifact_id"),
    )
    if refusal is not None:
        raise CohortDecisionRefusal(json.loads(refusal))
    if summary is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native binder returned neither a result nor a refusal")
    return _source_of(summary, artifact)


def decision_source_from_model(
    model: Any,
    *,
    cohort: CohortBinding,
    outcome_units: str,
    shift: CovariateShift | None = None,
    expected_contrast: str | None = None,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> CohortDecisionSource:
    """Bind ``cohort`` to a loaded :class:`antecedent.prediction.FittedEffectModel`.

    The retained model's exported parent result is re-verified natively and must carry the
    parent claim the model reports. Prefer ``model.decision_source(cohort=..., outcome_units=...)``.
    """
    return _build_source(
        "analysis_result",
        model.export(),
        model.parent_claim,
        cohort,
        outcome_units,
        shift,
        expected_contrast,
        artifact_id,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NativeEffectFit:
    """A linear conditional effect fitted natively, with its retained rows.

    Produced only by :func:`fit_native_cohort_effect`. Outcome is regressed on the declared
    basis ``(1, features)`` and its treatment interaction by ordinary least squares with an
    HC3 sandwich covariance; ``rows`` and ``width`` are the retained ``n`` and ``p``. The
    retained rows are replayed by the interval route, so a hand-built export cannot claim a
    native fit.
    """

    identity: str
    rows: int
    width: int
    coefficients: tuple[float, ...]
    covariance: tuple[float, ...]
    feature_names: tuple[str, ...]
    _export: bytes = field(repr=False)
    _fit_json: str = field(repr=False)

    def export(self) -> bytes:
        """The compact export of the fit."""
        return self._export


def fit_native_cohort_effect(
    *,
    outcome: Sequence[float],
    treatment: Sequence[float],
    features: Mapping[str, Sequence[float]],
    outcome_name: str,
    treatment_name: str,
    active: float = 1.0,
    control: float = 0.0,
    outcome_units: str,
    population: str,
    feature_units: Mapping[str, str] | None = None,
) -> NativeEffectFit:
    """Fit a linear conditional treatment effect natively and retain its rows.

    ``features`` maps each declared modifier name to its column. Treatment must be randomized
    or ignorable given exactly these modifiers; the outcome mean is assumed linear in them.
    """
    names = tuple(_text(name, "feature name") for name in features)
    units = feature_units or {}
    payload = {
        "outcome_name": _text(outcome_name, "outcome_name"),
        "treatment_name": _text(treatment_name, "treatment_name"),
        "active": float(active),
        "control": float(control),
        "outcome_units": _text(outcome_units, "outcome_units"),
        "population_id": _text(population, "population"),
        "outcome": [float(v) for v in outcome],
        "treatment": [float(v) for v in treatment],
        "feature_names": list(names),
        "feature_units": [str(units.get(name, "1")) for name in names],
        "features": [[float(v) for v in features[name]] for name in names],
    }
    text = json.dumps(payload, allow_nan=False)
    summary, export, refusal = _fit_native(text)
    if refusal is not None:
        raise CohortDecisionRefusal(json.loads(refusal))
    if summary is None or export is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native fit returned neither a result nor a refusal")
    parsed = json.loads(summary)
    return NativeEffectFit(
        identity=str(parsed["identity"]),
        rows=int(parsed["rows"]),
        width=int(parsed["width"]),
        coefficients=tuple(float(v) for v in parsed["coefficients"]),
        covariance=tuple(float(v) for v in parsed["covariance"]),
        feature_names=tuple(str(v) for v in parsed["feature_names"]),
        _export=bytes(export),
        _fit_json=text,
    )


def decision_source_from_native_fit(
    fit: NativeEffectFit,
    *,
    cohort: CohortBinding,
    outcome_units: str,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> CohortDecisionSource:
    """Bind ``cohort`` to a natively fitted effect; the source can issue a measured interval."""
    if not isinstance(fit, NativeEffectFit):
        raise CausalTypeError("fit= must be a NativeEffectFit")
    source = _build_source(
        "compact_export",
        fit._export,
        fit.identity,
        cohort,
        outcome_units,
        None,
        None,
        artifact_id,
    )
    return replace(source, _native_fit=fit._fit_json)


def decision_source_from_compact_export(
    export: CompactExport,
    *,
    cohort: CohortBinding,
    outcome_units: str,
    shift: CovariateShift | None = None,
    expected_contrast: str | None = None,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> CohortDecisionSource:
    """Bind ``cohort`` to a caller-built compact linear model that declares a contrast.

    The export's response must be an outcome whose functional is ``conditional_mean_difference``
    and whose regime starts ``contrast:``. The coefficients are the caller's declaration, so the
    source is never native; it carries the model-based variance diagnostic.
    """
    return _build_source(
        "compact_export",
        export.export(),
        export.identity,
        cohort,
        outcome_units,
        shift,
        expected_contrast,
        artifact_id,
    )


def consume_cohort_decision_source(
    artifact: bytes,
    *,
    expected_identity: str,
    parent_claim: str | None = None,
    population: str | None = None,
    outcome_units: str | None = None,
    contrast: str | None = None,
    cohort_digest: str | None = None,
) -> CohortDecisionSource:
    """Replay an exported cohort decision source under the retained identity (see
    :meth:`CohortDecisionSource.consume`)."""
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    data = bytes(artifact)
    expectation = {
        "identity": _text(expected_identity, "expected_identity"),
        "parent_claim": parent_claim,
        "population_id": population,
        "outcome_units": outcome_units,
        "contrast_regime": contrast,
        "cohort_digest": cohort_digest,
    }
    summary, refusal = _consume(data, json.dumps(expectation))
    if refusal is not None:
        raise CohortDecisionRefusal(json.loads(refusal))
    if summary is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native verifier returned neither a result nor a refusal")
    return _source_of(summary, data)


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True, slots=True)
class SelectionShift:
    """The selection-diagram premise of a shift between the model's population and a cohort's.

    The selection node ``selection_node`` marks where the populations differ. It points into
    ``selection_children`` (variables) and ``effect_modifiers`` are declared sufficient: given
    exactly those, the conditional contrast does not depend on the selection node. The
    declaration is valid only if the node points into nothing outside the modifiers and the
    modifiers are exactly the model's features. It is a premise and is never tested.
    """

    source_population: str
    target_population: str
    selection_node: str
    selection_children: tuple[str, ...]
    effect_modifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_population, "source_population")
        _text(self.target_population, "target_population")
        _text(self.selection_node, "selection_node")
        for what in ("selection_children", "effect_modifiers"):
            names = getattr(self, what)
            if isinstance(names, str):
                raise CausalTypeError(f"{what} must be an iterable of names")
            coerce_fields(self, **{what: tuple(_text(n, what) for n in names)})

    def _wire(self) -> dict[str, Any]:
        return {
            "source_population": self.source_population,
            "target_population": self.target_population,
            "selection_node": self.selection_node,
            "selection_children": list(self.selection_children),
            "effect_modifiers": list(self.effect_modifiers),
        }


@dataclass(frozen=True, slots=True)
class CohortTransportReceipt:
    """How a cohort mean reached its target population.

    ``support`` is always ``extrapolative`` and ``inference`` always ``point_only``.
    ``overlap`` is the fitted-box check that passed (per feature: fitted and target range, rows and
    weight outside, disjointness); it is a necessary condition, not a positivity guarantee.
    """

    route: str
    identity: str
    model_identity: str
    parent_claim: str
    source_population: str
    target_population: str
    selection_node: str
    selection_children: tuple[str, ...]
    effect_modifiers: tuple[str, ...]
    cohort_digest: str
    support: str
    inference: str
    assumptions: tuple[str, ...]
    overlap: Mapping[str, Any]
    scope: str

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> CohortTransportReceipt:
        return cls(
            route=str(wire["route"]),
            identity=str(wire["identity"]),
            model_identity=str(wire["model_identity"]),
            parent_claim=str(wire["parent_claim"]),
            source_population=str(wire["source_population"]),
            target_population=str(wire["target_population"]),
            selection_node=str(wire["selection_node"]),
            selection_children=tuple(wire["selection_children"]),
            effect_modifiers=tuple(wire["effect_modifiers"]),
            cohort_digest=str(wire["cohort_digest"]),
            support=str(wire["support"]),
            inference=str(wire["inference"]),
            assumptions=tuple(wire["assumptions"]),
            overlap=dict(wire["overlap"]),
            scope=str(wire["scope"]),
        )

    def as_transport_receipt(self) -> Any:
        """The shared :class:`antecedent.transported_response.TransportReceipt` of this route."""
        from .transported_response import TransportReceipt

        return TransportReceipt(
            route="cohort_effect",
            source_identity=self.model_identity,
            source_populations=(self.source_population,),
            target_population=self.target_population,
            transport_identities=(self.identity,),
            assumptions=self.assumptions,
            support=self.support,
            representation="point_only",
        )


@dataclass(frozen=True, slots=True, eq=False)
class TransportedCohortEffect(ScientificResult):
    """A fitted effect model's weighted cohort-mean contrast, carried to another population.

    ``mean_contrast`` is the expectation of one cohort-mean conditional contrast under the declared
    selection shift: point-only, never target-population data, never two absolute potential-outcome
    means. ``receipt`` and ``identity`` bind the model, the declaration, the cohort, the overlap
    check and the value. :attr:`decision_source` is the existing :class:`CohortDecisionSource` for
    the same cohort (labelled ``extrapolative``); decide on it through ``accept_extrapolative()``.
    There is no interval and no law: :meth:`interval` and :meth:`refuse` raise the typed refusal.
    """

    identity: str
    mean_contrast: float
    row_contrasts: tuple[float, ...]
    receipt: CohortTransportReceipt
    native_target_data: bool = False
    _decision: CohortDecisionSource | None = field(default=None, repr=False, compare=False)
    _retained: Any = field(default=None, repr=False, compare=False)

    @property
    def decision_source(self) -> CohortDecisionSource:
        """The existing cohort-mean decision source for the same cohort and shift."""
        assert self._decision is not None
        return self._decision

    @property
    def mean_source(self) -> MeanSource:
        """The transported mean as a decision-contract mean source (extrapolative, accepted)."""
        return self.decision_source.accept_extrapolative().mean_source

    def interval(self, *, level: float = 0.95) -> None:
        """Always refuses: no interval, band or posterior is licensed for a transported mean."""
        del level
        raise CohortDecisionRefusal(json.loads(_transported_refusal("interval")))

    def refuse(self, kind: str = "law") -> None:
        """Raise the typed refusal of a request this value cannot answer (``interval``, ``law``)."""
        if kind not in ("interval", "law"):
            self.decision_source.refuse(kind)
        raise CohortDecisionRefusal(json.loads(_transported_refusal(kind)))

    def verify(self) -> TransportedCohortEffect:
        """Re-derive the whole transport from the retained inputs and require this identity.

        Raises :class:`CohortDecisionRefusal` (``transported_effect.identity_mismatch``) when the
        recomputed identity differs. Returns the recomputed effect.
        """
        model_bytes, claim, source_population, cohort, shift = self._retained
        text, refusal = _transport_verify(
            model_bytes,
            claim,
            source_population,
            json.dumps(cohort._wire(), allow_nan=False),
            None if shift is None else json.dumps(shift._wire()),
            self.identity,
        )
        if refusal is not None:
            raise CohortDecisionRefusal(json.loads(refusal))
        assert text is not None
        wire = json.loads(text)
        return replace(self, identity=str(wire["identity"]))


def transport_cohort_effect(
    model: Any,
    *,
    cohort: CohortBinding,
    source_population: str,
    shift: SelectionShift | None,
    outcome_units: str,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> TransportedCohortEffect:
    """Carry a retained fitted effect model to a cohort of another population.

    ``model`` is a loaded :class:`antecedent.prediction.FittedEffectModel`; ``source_population``
    names the population it was fitted for (the analysis result carries no population label of its
    own, so it is declared and bound into the identity); ``cohort`` declares the target rows;
    ``shift`` is the :class:`SelectionShift` premise.

    The value is issued only when the declaration names the model's population as source and the
    cohort's as target, the selection node points only into the declared effect modifiers, those
    are exactly the model's features, and every cohort row lies inside the model's fitted support.

    Raises:
        CohortDecisionRefusal: the unmet premise, with a registered ``reason_code``, a
            ``transported_effect.*`` ``detail`` (``shift_declaration_required``,
            ``selection_enters_mechanism``, ``modifier_set_mismatch``, ``no_overlap``,
            ``out_of_support``, ...) and the ``remedy``.
    """
    if not isinstance(cohort, CohortBinding):
        raise CausalTypeError("cohort= must be a CohortBinding")
    if shift is not None and not isinstance(shift, SelectionShift):
        raise CausalTypeError("shift= must be a SelectionShift")
    _text(source_population, "source_population")
    if not isinstance(outcome_units, str) or not outcome_units.strip():
        raise CausalValueError("outcome_units= is required: units are never inferred")
    model_bytes = bytes(model.export())
    claim = model.parent_claim
    summary, decision, artifact, refusal = _transport(
        model_bytes,
        claim,
        source_population,
        json.dumps(cohort._wire(), allow_nan=False),
        outcome_units,
        None if shift is None else json.dumps(shift._wire()),
        _text(artifact_id, "artifact_id"),
    )
    if refusal is not None:
        raise CohortDecisionRefusal(json.loads(refusal))
    if summary is None or decision is None or artifact is None:  # pragma: no cover
        raise CausalValueError("the native transport returned neither a result nor a refusal")
    wire = json.loads(summary)
    return TransportedCohortEffect(
        identity=str(wire["identity"]),
        mean_contrast=float(wire["mean_contrast"]),
        row_contrasts=tuple(float(v) for v in wire["row_contrasts"]),
        receipt=CohortTransportReceipt._from_wire(wire),
        _decision=_source_of(decision, artifact),
        _retained=(model_bytes, claim, source_population, cohort, shift),
    )
