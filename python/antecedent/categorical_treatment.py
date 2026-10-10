"""Categorical treatments: ordered and unordered regimes with a declared reference (B4).

One question: what is the adjusted effect of each declared level of a categorical treatment
against a declared reference level, and, when the levels are ordered, do the adjusted level
effects move monotonically? The levels are dummy-coded (the reference omitted) and fitted
jointly, so every contrast uses the full covariance, off-diagonals included::

    from antecedent import categorical_treatment

    fit = categorical_treatment.categorical_effects(
        outcome,
        dose_level,                                  # one label per row
        categories=["none", "low", "high"],
        ordered=True,
        reference="none",
        monotonicity="non_decreasing",
        pairs=[("low", "high")],
        adjust={"age": age},
    )
    fit.effect("high").estimate                      # effect(high) - effect(none)
    fit.pair("low", "high").p_holm                   # Holm over the declared family
    fit.monotonicity.p_value
    again = categorical_treatment.consume_categorical_effects(fit.export(), expected_identity=fit.identity)

What the answer is **not**:

* Every p-value (per-level and pairwise Wald values, their Holm adjustment over the family of
  :attr:`CategoricalEffects.family_size` contrasts, the omnibus chi-square and the monotonicity
  bound) is asymptotic and its calibration is **unmeasured** (``calibration == "unmeasured"``):
  no coverage, Type I error or power claim is made.
* The monotonicity null is ``every adjacent step is >= 0`` (non-decreasing) or ``<= 0``
  (non-increasing); the p-value is a conservative one-sided union-intersection (Bonferroni)
  bound. Rejection is evidence *against* monotonicity in that direction; failing to reject does
  not prove it.
* Unordered levels are canonicalised (sorted) before any arithmetic, so permuting their declared
  order changes nothing. For ordered levels the declared order is the scale and part of the
  estimand.
* A declared level with no rows (``categorical_treatment.absent_level``) or fewer than
  ``min_level_rows`` (``.sparse_level``) refuses with the level named, as does a row with an
  undeclared label; no level is silently dropped or merged
  (:class:`CategoricalTreatmentRefusal`, ``arm_not_populated`` / ``invalid_argument``).

Every coefficient, covariance, contrast, Holm value, test, identity digest, refusal rule and the
artifact is computed in Rust; this module builds the declaration and presents the result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ._b4_refusal import StructuredRefusal
from ._native import consume_categorical_treatment_artifact as _consume
from ._native import evaluate_categorical_treatment as _evaluate
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "CategoricalEffects",
    "CategoricalTreatmentIdentity",
    "CategoricalTreatmentRefusal",
    "Direction",
    "LevelEffect",
    "MonotonicityStep",
    "MonotonicityTest",
    "OmnibusTest",
    "PairEffect",
    "consume_categorical_effects",
    "categorical_effects",
]

_COVARIANCES = ("model_based", "hc0", "hc1", "hc2", "hc3")
_DEFAULT_ARTIFACT_ID = "categorical_treatment"


class CategoricalTreatmentRefusal(StructuredRefusal):
    """A typed refusal of the categorical-treatment route.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``categorical_treatment.*`` (or underlying ``vector_treatment.*``) ``detail``.
    """


class Direction(StrEnum):
    """Direction of the declared monotonicity null (ordered levels only)."""

    #: Null: every adjacent step of the adjusted level effects is non-negative.
    NON_DECREASING = "non_decreasing"
    #: Null: every adjacent step of the adjusted level effects is non-positive.
    NON_INCREASING = "non_increasing"


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _column(values: object, what: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    return array


@dataclass(frozen=True, slots=True)
class LevelEffect:
    """The effect of one level against the reference (``effect(level) - effect(reference)``)."""

    level: str
    reference: str
    estimate: float
    standard_error: float
    z: float
    p_value: float
    #: Holm-adjusted p-value across the declared family.
    p_holm: float

    @property
    def se(self) -> float:
        """Alias of :attr:`standard_error`."""
        return self.standard_error


@dataclass(frozen=True, slots=True)
class PairEffect:
    """A requested pairwise contrast ``effect(to) - effect(from)``."""

    from_level: str
    to_level: str
    estimate: float
    #: Uses the off-diagonals of the dummy-regression covariance.
    standard_error: float
    z: float
    p_value: float
    #: Holm-adjusted p-value across the declared family.
    p_holm: float

    @property
    def se(self) -> float:
        """Alias of :attr:`standard_error`."""
        return self.standard_error


@dataclass(frozen=True, slots=True)
class OmnibusTest:
    """Wald chi-square of ``H0: every level has the reference level's effect``."""

    null: str
    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass(frozen=True, slots=True)
class MonotonicityStep:
    """One adjacent step ``effect(to_level) - effect(from_level)`` along the declared order."""

    from_level: str
    to_level: str
    difference: float
    standard_error: float
    z: float


@dataclass(frozen=True, slots=True)
class MonotonicityTest:
    """The declared monotonicity test and its null.

    ``statistic`` is the oriented minimum step z-score and ``p_value`` the conservative
    one-sided union-intersection (Bonferroni) bound ``min(1, m * Phi(statistic))``. Rejection is
    evidence against monotonicity in :attr:`direction`; failing to reject does not prove it.
    ``calibration`` is ``"unmeasured"``.
    """

    direction: Direction
    null: str
    steps: tuple[MonotonicityStep, ...]
    statistic: float
    p_value: float
    conservative: bool
    calibration: str


@dataclass(frozen=True, slots=True)
class CategoricalTreatmentIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_categorical_effects` as ``expected_identity=``.
    """

    snapshot_id: str
    adjustment_set_id: str
    level_scale_id: str
    design_id: str
    covariance_id: str
    family_id: str
    omnibus_null: str
    monotonicity_null: str
    result_id: str
    digest: str

    def _wire(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True, eq=False)
class CategoricalEffects:
    """The categorical fit: level and pairwise contrasts, multiplicity, tests and the artifact.

    ``level_order`` is the canonical order used (the declared order when ``ordered``, sorted
    otherwise); ``counts`` the rows per level. ``coefficients`` are the dummy coefficients
    (``level:<name>``, non-reference levels) with the read-only ``covariance`` between them.
    ``family_size`` is the size of the Holm family (levels against the reference plus the
    requested pairs). ``calibration`` is ``"unmeasured"``.
    """

    reference: str
    ordered: bool
    level_order: tuple[str, ...]
    counts: Mapping[str, int]
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray
    effects: tuple[LevelEffect, ...]
    pairs: tuple[PairEffect, ...]
    family_size: int
    omnibus: OmnibusTest
    monotonicity: MonotonicityTest | None
    covariance_kind: str
    min_level_rows: int
    replay: str
    row_snapshot: str
    adjustment: tuple[str, ...]
    caveats: tuple[str, ...]
    calibration: str
    inference_claim: str
    identity: CategoricalTreatmentIdentity
    artifact: bytes = field(repr=False)

    def export(self) -> bytes:
        """The checksummed ``categorical_treatment_v1`` artifact."""
        return self.artifact

    def effect(self, level: str) -> LevelEffect:
        """The contrast of ``level`` against the reference (the reference itself is zero)."""
        for item in self.effects:
            if item.level == level:
                return item
        if level == self.reference:
            raise KeyError(f"{level!r} is the reference level; its effect is zero by definition")
        raise KeyError(level)

    def pair(self, from_level: str, to_level: str) -> PairEffect:
        """The requested pairwise contrast ``effect(to_level) - effect(from_level)``."""
        for item in self.pairs:
            if (item.from_level, item.to_level) == (from_level, to_level):
                return item
        raise KeyError((from_level, to_level))

    def table(self) -> list[dict[str, Any]]:
        """The per-level table: level, rows, estimate, standard error, z, p and Holm p."""
        rows = [
            {
                "level": self.reference,
                "rows": self.counts[self.reference],
                "estimate": 0.0,
                "standard_error": None,
                "z": None,
                "p_value": None,
                "p_holm": None,
            }
        ]
        rows.extend(
            {
                "level": e.level,
                "rows": self.counts[e.level],
                "estimate": e.estimate,
                "standard_error": e.standard_error,
                "z": e.z,
                "p_value": e.p_value,
                "p_holm": e.p_holm,
            }
            for e in self.effects
        )
        order = {level: i for i, level in enumerate(self.level_order)}
        return sorted(rows, key=lambda row: order[row["level"]])

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes)."""
        monotonicity = None
        if self.monotonicity is not None:
            m = self.monotonicity
            monotonicity = {
                "direction": m.direction.value,
                "null": m.null,
                "statistic": m.statistic,
                "p_value": m.p_value,
                "conservative": m.conservative,
                "calibration": m.calibration,
                "steps": [
                    {
                        "from": s.from_level,
                        "to": s.to_level,
                        "difference": s.difference,
                        "standard_error": s.standard_error,
                        "z": s.z,
                    }
                    for s in m.steps
                ],
            }
        return {
            "reference": self.reference,
            "ordered": self.ordered,
            "level_order": list(self.level_order),
            "counts": dict(self.counts),
            "table": self.table(),
            "pairs": [
                {
                    "from": p.from_level,
                    "to": p.to_level,
                    "estimate": p.estimate,
                    "standard_error": p.standard_error,
                    "z": p.z,
                    "p_value": p.p_value,
                    "p_holm": p.p_holm,
                }
                for p in self.pairs
            ],
            "family_size": self.family_size,
            "omnibus": {
                "null": self.omnibus.null,
                "statistic": self.omnibus.statistic,
                "degrees_of_freedom": self.omnibus.degrees_of_freedom,
                "p_value": self.omnibus.p_value,
            },
            "monotonicity": monotonicity,
            "covariance": self.covariance.tolist(),
            "covariance_kind": self.covariance_kind,
            "replay": self.replay,
            "calibration": self.calibration,
            "inference_claim": self.inference_claim,
            "caveats": list(self.caveats),
            "identity": self.identity._wire(),
        }


def _result(report_json: str, artifact: bytes) -> CategoricalEffects:
    report = json.loads(report_json)
    result = report["result"]
    spec = report["spec"]
    coefficient_names = tuple(c["name"] for c in result["coefficients"])
    k = len(coefficient_names)
    covariance = np.array(result["covariance"], dtype=np.float64).reshape(k, k)
    covariance.setflags(write=False)
    omnibus = result["omnibus"]
    mono = result["monotonicity"]
    return CategoricalEffects(
        reference=result["reference"],
        ordered=result["scale"] == "ordered",
        level_order=tuple(result["level_order"]),
        counts={c["level"]: int(c["rows"]) for c in result["counts"]},
        coefficient_names=coefficient_names,
        covariance=covariance,
        effects=tuple(
            LevelEffect(
                level=c["level"],
                reference=c["reference"],
                estimate=float(c["estimate"]),
                standard_error=float(c["standard_error"]),
                z=float(c["z"]),
                p_value=float(c["p_value"]),
                p_holm=float(c["p_holm"]),
            )
            for c in result["level_contrasts"]
        ),
        pairs=tuple(
            PairEffect(
                from_level=c["from"],
                to_level=c["to"],
                estimate=float(c["estimate"]),
                standard_error=float(c["standard_error"]),
                z=float(c["z"]),
                p_value=float(c["p_value"]),
                p_holm=float(c["p_holm"]),
            )
            for c in result["pairwise"]
        ),
        family_size=int(result["family_size"]),
        omnibus=OmnibusTest(
            null=report["null"],
            statistic=float(omnibus["statistic"]),
            degrees_of_freedom=int(omnibus["degrees_of_freedom"]),
            p_value=float(omnibus["p_value"]),
        ),
        monotonicity=None
        if mono is None
        else MonotonicityTest(
            direction=Direction(mono["direction"]),
            null=mono["null"],
            steps=tuple(
                MonotonicityStep(
                    from_level=s["from"],
                    to_level=s["to"],
                    difference=float(s["difference"]),
                    standard_error=float(s["standard_error"]),
                    z=float(s["z"]),
                )
                for s in mono["steps"]
            ),
            statistic=float(mono["statistic"]),
            p_value=float(mono["p_value"]),
            conservative=bool(mono["conservative"]),
            calibration=mono["calibration"],
        ),
        covariance_kind=result["covariance_kind"],
        min_level_rows=int(spec["min_level_rows"]),
        replay=report["replay"],
        row_snapshot=report["row_snapshot"],
        adjustment=tuple(report["adjustment"]),
        caveats=tuple(report["caveats"]),
        calibration=report["calibration"],
        inference_claim=report["inference_claim"],
        identity=CategoricalTreatmentIdentity(**report["identity"]),
        artifact=artifact,
    )


def _content_snapshot(
    outcome: np.ndarray, labels: Sequence[str], adjustment: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(outcome, dtype="<f8").tobytes())
    for label in labels:
        digest.update(label.encode())
        digest.update(b"\0")
    for name, values in adjustment.items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return "sha256:" + digest.hexdigest()


def categorical_effects(
    outcome: Any,
    levels: Sequence[Any],
    *,
    categories: Sequence[Any] | None = None,
    ordered: bool = False,
    reference: Any | None = None,
    pairs: Sequence[tuple[Any, Any]] | None = None,
    monotonicity: Direction | str | None = None,
    min_level_rows: int = 2,
    adjust: Mapping[str, Any] | None = None,
    covariance: str = "model_based",
    snapshot: str | None = None,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> CategoricalEffects:
    """Estimate the adjusted effect of every level of a categorical treatment.

    ``levels`` holds one label per row (labels are compared as strings). ``categories`` is the
    declared level set (at least two); omitted (unordered only), it is the sorted observed
    labels, so a level that never occurs can only be caught when it is declared. With
    ``ordered=True`` the order of ``categories`` is the scale and is required; otherwise the
    levels are unordered and canonicalised.
    ``reference`` is the level every other level is compared against (default: the first
    declared level when ordered, the first sorted level otherwise). ``pairs`` requests further
    ``(from, to)`` contrasts ``effect(to) - effect(from)``; they join the per-level contrasts in
    one Holm family. ``monotonicity`` (``"non_decreasing"`` or ``"non_increasing"``) declares a
    monotonicity test and requires ``ordered=True``. ``min_level_rows`` is the fewest rows a
    declared level may have. ``adjust`` maps shared adjustment column names to their columns.
    ``covariance`` is ``"model_based"`` or a robust sandwich ``"hc0"`` to ``"hc3"`` (the rows
    are embedded in the artifact, capped).

    Raises :class:`CategoricalTreatmentRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`) for an absent or sparse declared level
    (``arm_not_populated``), an undeclared label or reference, a degenerate design
    (``invalid_argument``, ``design_rank_deficient``) and a monotonicity test on unordered
    levels (``route_not_supported``, ``categorical_treatment.monotonicity_requires_ordered``).
    """
    if isinstance(levels, str | bytes):
        raise CausalTypeError("levels must be a sequence with one label per row")
    if ordered and categories is None:
        raise CausalValueError(
            "ordered levels need categories=: the declared order is the scale and part of the "
            "estimand"
        )
    if adjust is not None and not isinstance(adjust, Mapping):
        raise CausalTypeError("adjust must be a mapping of name -> column")
    if covariance not in _COVARIANCES:
        raise CausalValueError(f"covariance must be one of {_COVARIANCES}")
    if isinstance(min_level_rows, bool) or not isinstance(min_level_rows, int):
        raise CausalTypeError("min_level_rows must be an integer")
    if min_level_rows < 0:
        raise CausalValueError("min_level_rows must not be negative")
    y = _column(outcome, "outcome")
    labels = [str(item) for item in np.asarray(levels, dtype=object).ravel()]
    adjustment = {
        _name(n, "adjustment name"): _column(v, f"adjustment {n!r}")
        for n, v in (adjust or {}).items()
    }
    declared = sorted(set(labels)) if categories is None else [str(item) for item in categories]
    if reference is None:
        chosen = declared[0] if ordered else (min(declared) if declared else "")
    else:
        chosen = str(reference)
    try:
        direction = None if monotonicity is None else Direction(monotonicity).value
    except ValueError as error:
        raise CausalValueError(
            f"monotonicity must be one of {[d.value for d in Direction]} or None"
        ) from error
    shared = snapshot if snapshot is not None else _content_snapshot(y, labels, adjustment)
    declaration = {
        "row_snapshot": _name(shared, "snapshot"),
        "adjustment": list(adjustment),
        "spec": {
            "declared_levels": declared,
            "scale": "ordered" if ordered else "unordered",
            "reference": chosen,
            "min_level_rows": min_level_rows,
            "pairwise": [[str(a), str(b)] for a, b in (pairs or ())],
            "monotonicity": direction,
            "covariance": covariance,
        },
    }
    report, artifact, refusal = _evaluate(
        y.tolist(),
        [v.tolist() for v in adjustment.values()],
        labels,
        json.dumps(declaration, allow_nan=False),
        _name(artifact_id, "artifact_id"),
    )
    if refusal is not None:
        raise CategoricalTreatmentRefusal(json.loads(refusal))
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native categorical fit returned neither a result nor a refusal")
    return _result(report, bytes(artifact))


def consume_categorical_effects(
    artifact: bytes,
    *,
    expected_identity: CategoricalTreatmentIdentity | Mapping[str, str] | None = None,
) -> CategoricalEffects:
    """Recompute an exported categorical artifact and accept only an identical one.

    Every coefficient, covariance entry, contrast, Holm value, omnibus and monotonicity result
    is recomputed from the embedded compact design and must reproduce bit for bit. With
    ``expected_identity`` (the :attr:`CategoricalEffects.identity` retained out-of-band) a changed level
    order, reference, scale, family, covariance kind, design, null or result is refused even
    when the artifact was resealed consistently (:class:`CategoricalTreatmentRefusal`,
    ``route_not_supported``, ``categorical_treatment.wrong_contract``). Corruption and unknown
    major versions raise :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, CategoricalTreatmentIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError(
            "expected_identity must be a CategoricalTreatmentIdentity or a mapping"
        )
    data = bytes(artifact)
    report, refusal = _consume(data, expected_json)
    if refusal is not None:
        raise CategoricalTreatmentRefusal(json.loads(refusal))
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)
