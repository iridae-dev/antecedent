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
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._columns import coerce_fields
from ._native import build_cohort_decision_source as _build
from ._native import cohort_decision_forward_claim as _forward_claim
from ._native import cohort_decision_law_refusal as _law_refusal
from ._native import consume_cohort_decision_source as _consume
from ._native import normalize_cohort_weights as _normalize
from ._native import validate_cohort_binding as _validate
from ._refusal import StructuredRefusal
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import ScientificQuantity

if TYPE_CHECKING:
    from .compact_export import CompactExport
    from .decision import MeanSource
    from .inverse_query import MeanClaim

__all__ = [
    "CohortBinding",
    "CohortDecisionRefusal",
    "CohortDecisionSource",
    "CovariateShift",
    "UncalibratedStandardError",
    "consume_cohort_decision_source",
    "decision_source_from_compact_export",
    "decision_source_from_model",
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
            array = np.asarray(values)
            if array.ndim != 1:
                raise CausalValueError(f"feature {name!r} must be one-dimensional")
            rows = len(array) if rows is None else rows
            if array.dtype.kind in "US" or array.dtype == object:
                items = [str(v) if isinstance(v, str) else v for v in array.tolist()]
                if any(not isinstance(v, str) for v in items):
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
        ids = [f"row-{i}" for i in range(rows)] if row_ids is None else [str(r) for r in row_ids]
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
class CohortDecisionSource:
    """One native cohort-mean treatment-contrast source.

    ``identity`` is what a consumer retains out of band. ``quantity`` is the scientific
    coordinate (outcome, the model's contrast regime, functional ``mean_difference``, the
    cohort population, the cohort snapshot digest as conditioning); a decision contract names
    exactly that coordinate. Pass the source to :meth:`antecedent.decision.Contract.evaluate`:
    only the expectation of an affine utility is answered.
    """

    identity: str
    summary: Mapping[str, Any] = field(repr=False)
    artifact: bytes = field(repr=False)
    extrapolative_accepted: bool = field(default=False, repr=False)

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
        return dict(self.summary["support"])

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
    def interval(self, *, level: float = 0.95) -> None:
        """Always refuses: no interval, band or posterior is licensed for a cohort mean."""
        del level
        raise CohortDecisionRefusal(json.loads(_law_refusal("interval")))

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
        summary, refusal = _consume(self.artifact, self._expectation())
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

        A source built under a declared covariate shift is refused
        (``cohort_decision.inverse_shift_not_licensed``).
        """
        from .inverse_query import MeanClaim

        text, refusal = _forward_claim(self.artifact, self._expectation())
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

    def export(self) -> bytes:
        """The checksummed ``cohort_decision_source_v1`` artifact."""
        return self.artifact

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
        return dict(self.summary)


def _source_of(summary: str, artifact: bytes) -> CohortDecisionSource:
    parsed = json.loads(summary)
    return CohortDecisionSource(
        identity=str(parsed["identity"]), summary=parsed, artifact=bytes(artifact)
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
