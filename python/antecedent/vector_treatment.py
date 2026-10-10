"""Joint vector-treatment coefficients (B4).

Several treatments are estimated *together* by one regression that shares ONE adjustment set
and ONE row snapshot. The answer is a named coefficient vector, its **full** covariance
(off-diagonals included), declared contrasts whose standard errors use that covariance, a Holm
adjustment across the declared contrast family, and the joint Wald test that every treatment
coefficient is zero::

    from antecedent import vector_treatment

    fit = vector_treatment.joint_effects(
        outcome,
        {"dose_a": a, "dose_b": b},
        adjust={"age": age},
        contrasts=[vector_treatment.Contrast("a_minus_b", {"dose_a": 1.0, "dose_b": -1.0})],
    )
    fit.coefficient("dose_a").estimate
    fit.covariance                       # k x k, off-diagonals included
    fit.contrast("a_minus_b").standard_error
    fit.joint_wald.p_value
    again = vector_treatment.consume_joint_effects(fit.export(), expected_identity=fit.identity)

What the answer is **not**:

* Every p-value (the per-coefficient and per-contrast Wald values, their Holm adjustment and the
  joint chi-square) is asymptotic and its calibration is **unmeasured**
  (``calibration == "unmeasured"``): no coverage, Type I error or power claim is made, and no
  interval is produced.
* It is a regression-coefficient estimate, not an identification statement. The caller owns that
  the shared adjustment set is valid.
* A treatment that declares a different adjustment set, row snapshot or row count than the
  shared one refuses (:class:`VectorTreatmentRefusal`, ``route_not_supported``,
  ``vector_treatment.adjustment_set_mismatch`` / ``.row_snapshot_mismatch`` /
  ``.row_count_mismatch``); a treatment with no variation or a collinear design refuses with
  ``design_rank_deficient``. Nothing is dropped silently.

Every coefficient, covariance entry, contrast, Holm value, identity digest, refusal rule and the
artifact is computed in Rust; this module builds the declaration and presents the result. The
exported artifact embeds a compact design (``X'X``, ``X'y`` and the residual sum of squares for
the model-based covariance; the rows, capped, for a heteroskedasticity-robust covariance) from
which the consumer recomputes everything bit for bit. Pass the :attr:`JointEffects.identity` you
retained out-of-band as ``expected_identity=`` to refuse a *resealed* change.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._b4_refusal import StructuredRefusal
from ._native import consume_vector_treatment_artifact as _consume
from ._native import evaluate_vector_treatment as _evaluate
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "Coefficient",
    "Contrast",
    "ContrastEffect",
    "JointEffects",
    "JointWald",
    "Treatment",
    "VectorTreatmentIdentity",
    "VectorTreatmentRefusal",
    "consume_joint_effects",
    "joint_effects",
]

#: Frozen null of the joint Wald test; it is not configurable.
NULL = "all treatment coefficients are zero"
_COVARIANCES = ("model_based", "hc0", "hc1", "hc2", "hc3")
_DEFAULT_ARTIFACT_ID = "vector_treatment"


class VectorTreatmentRefusal(StructuredRefusal):
    """A typed refusal of the joint vector-treatment route.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    ``vector_treatment.*`` ``detail`` (for example
    ``vector_treatment.adjustment_set_mismatch``).
    """


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


@dataclass(frozen=True, slots=True, eq=False)
class Treatment:
    """One treatment column with the adjustment set and snapshot it was prepared under.

    ``adjustment_set`` and ``snapshot`` default to the shared ones given to
    :func:`joint_effects`; declaring a different one makes the joint fit refuse.
    """

    values: Any
    adjustment_set: Sequence[str] | None = None
    snapshot: str | None = None


@dataclass(frozen=True, slots=True)
class Contrast:
    """A linear contrast of the named treatment coefficients: ``sum(weight * coefficient)``."""

    name: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        _name(self.name, "contrast name")
        if not isinstance(self.weights, Mapping):
            raise CausalTypeError("contrast weights must be a mapping of coefficient -> weight")

    def _wire(self) -> dict[str, Any]:
        pairs = []
        for coefficient, weight in self.weights.items():
            number = float(weight)
            if not math.isfinite(number):
                raise VectorTreatmentRefusal(
                    {
                        "code": "invalid_argument",
                        "stage": "fit",
                        "detail": "vector_treatment.non_finite_value",
                        "offending": self.name,
                        "message": f"contrast {self.name!r} has a non-finite weight",
                    }
                )
            pairs.append([_name(coefficient, "coefficient name"), number])
        return {"name": self.name, "weights": pairs}


def _contrast(item: Contrast | Mapping[str, Any] | Sequence[Any]) -> Contrast:
    if isinstance(item, Contrast):
        return item
    if isinstance(item, Mapping):
        try:
            return Contrast(**dict(item))
        except TypeError as error:
            raise CausalTypeError(f"a contrast mapping has unexpected keys: {error}") from error
    if isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
        return Contrast(item[0], item[1])
    raise CausalTypeError(
        "a contrast is a Contrast, a {'name', 'weights'} mapping or a (name, weights) pair"
    )


@dataclass(frozen=True, slots=True)
class Coefficient:
    """One named treatment coefficient."""

    name: str
    estimate: float
    standard_error: float
    z: float
    #: Two-sided asymptotic normal p-value (calibration unmeasured).
    p_value: float

    @property
    def se(self) -> float:
        """Alias of :attr:`standard_error`."""
        return self.standard_error


@dataclass(frozen=True, slots=True)
class ContrastEffect:
    """One declared contrast of the coefficient vector."""

    name: str
    estimate: float
    #: ``sqrt(w' V w)``: uses the off-diagonals of the covariance.
    standard_error: float
    #: ``sqrt(sum w_i^2 V_ii)``: the standard error that wrongly treats the coefficients as
    #: independent, reported so the size of the covariance correction is visible.
    naive_independent_standard_error: float
    z: float
    p_value: float
    #: Holm-adjusted p-value across the declared contrast family.
    p_holm: float

    @property
    def se(self) -> float:
        """Alias of :attr:`standard_error`."""
        return self.standard_error


@dataclass(frozen=True, slots=True)
class JointWald:
    """Wald chi-square of ``H0: all treatment coefficients are zero``."""

    statistic: float
    degrees_of_freedom: int
    #: Asymptotic chi-square upper-tail p-value (calibration unmeasured).
    p_value: float


@dataclass(frozen=True, slots=True)
class VectorTreatmentIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_joint_effects` as ``expected_identity=``.
    """

    snapshot_id: str
    adjustment_set_id: str
    treatment_id: str
    design_id: str
    covariance_id: str
    contrast_id: str
    null: str
    result_id: str
    digest: str

    def _wire(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True, eq=False)
class JointEffects:
    """The joint fit: named coefficients, full covariance, contrasts and the portable artifact.

    ``covariance`` is the read-only ``k x k`` matrix in the order of :attr:`names`.
    ``calibration`` is ``"unmeasured"`` and ``inference_claim`` names the asymptotic Wald basis;
    read :attr:`caveats`. ``replay`` is ``"summary"`` (model-based covariance) or ``"rows"``
    (robust covariance) and states what the artifact embeds.
    """

    null: str
    names: tuple[str, ...]
    coefficients: tuple[Coefficient, ...]
    covariance: np.ndarray
    contrasts: tuple[ContrastEffect, ...]
    joint_wald: JointWald
    covariance_kind: str
    n_rows: int
    residual_df: int
    residual_variance: float
    row_snapshot: str
    adjustment: tuple[str, ...]
    replay: str
    caveats: tuple[str, ...]
    calibration: str
    inference_claim: str
    identity: VectorTreatmentIdentity
    artifact: bytes = field(repr=False)

    def export(self) -> bytes:
        """The checksummed ``vector_treatment_v1`` artifact."""
        return self.artifact

    @property
    def estimates(self) -> np.ndarray:
        """The coefficient vector, in :attr:`names` order."""
        return np.array([c.estimate for c in self.coefficients], dtype=np.float64)

    def coefficient(self, name: str) -> Coefficient:
        """The coefficient named ``name``."""
        for item in self.coefficients:
            if item.name == name:
                return item
        raise KeyError(name)

    def contrast(self, name: str) -> ContrastEffect:
        """The declared contrast named ``name``."""
        for item in self.contrasts:
            if item.name == name:
                return item
        raise KeyError(name)

    def covariance_of(self, first: str, second: str) -> float:
        """The covariance entry between two named coefficients."""
        i, j = self.names.index(first), self.names.index(second)
        return float(self.covariance[i, j])

    def table(self) -> list[dict[str, Any]]:
        """The coefficient table: name, estimate, standard error, z and p-value."""
        return [
            {
                "name": c.name,
                "estimate": c.estimate,
                "standard_error": c.standard_error,
                "z": c.z,
                "p_value": c.p_value,
            }
            for c in self.coefficients
        ]

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes)."""
        return {
            "null": self.null,
            "names": list(self.names),
            "coefficients": self.table(),
            "covariance": self.covariance.tolist(),
            "contrasts": [
                {
                    "name": c.name,
                    "estimate": c.estimate,
                    "standard_error": c.standard_error,
                    "naive_independent_standard_error": c.naive_independent_standard_error,
                    "z": c.z,
                    "p_value": c.p_value,
                    "p_holm": c.p_holm,
                }
                for c in self.contrasts
            ],
            "joint_wald": {
                "statistic": self.joint_wald.statistic,
                "degrees_of_freedom": self.joint_wald.degrees_of_freedom,
                "p_value": self.joint_wald.p_value,
            },
            "covariance_kind": self.covariance_kind,
            "n_rows": self.n_rows,
            "residual_df": self.residual_df,
            "residual_variance": self.residual_variance,
            "row_snapshot": self.row_snapshot,
            "adjustment": list(self.adjustment),
            "replay": self.replay,
            "calibration": self.calibration,
            "inference_claim": self.inference_claim,
            "caveats": list(self.caveats),
            "identity": self.identity._wire(),
        }


def _result(report_json: str, artifact: bytes) -> JointEffects:
    report = json.loads(report_json)
    result = report["result"]
    names = tuple(report["treatments"])
    k = len(names)
    covariance = np.array(result["covariance"], dtype=np.float64).reshape(k, k)
    covariance.setflags(write=False)
    wald = result["joint_wald"]
    return JointEffects(
        null=report["null"],
        names=names,
        coefficients=tuple(
            Coefficient(
                name=c["name"],
                estimate=float(c["estimate"]),
                standard_error=float(c["standard_error"]),
                z=float(c["z"]),
                p_value=float(c["p_value"]),
            )
            for c in result["coefficients"]
        ),
        covariance=covariance,
        contrasts=tuple(
            ContrastEffect(
                name=c["name"],
                estimate=float(c["estimate"]),
                standard_error=float(c["standard_error"]),
                naive_independent_standard_error=float(c["naive_independent_standard_error"]),
                z=float(c["z"]),
                p_value=float(c["p_value"]),
                p_holm=float(c["p_holm"]),
            )
            for c in result["contrasts"]
        ),
        joint_wald=JointWald(
            statistic=float(wald["statistic"]),
            degrees_of_freedom=int(wald["degrees_of_freedom"]),
            p_value=float(wald["p_value"]),
        ),
        covariance_kind=report["covariance_kind"],
        n_rows=int(result["n_rows"]),
        residual_df=int(result["residual_df"]),
        residual_variance=float(result["residual_variance"]),
        row_snapshot=report["row_snapshot"],
        adjustment=tuple(report["adjustment"]),
        replay=report["replay"],
        caveats=tuple(report["caveats"]),
        calibration=report["calibration"],
        inference_claim=report["inference_claim"],
        identity=VectorTreatmentIdentity(**report["identity"]),
        artifact=artifact,
    )


def _content_snapshot(columns: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, values in columns:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return "sha256:" + digest.hexdigest()


def joint_effects(
    outcome: Any,
    treatments: Mapping[str, Treatment | Any],
    *,
    adjust: Mapping[str, Any] | None = None,
    contrasts: Sequence[Contrast | Mapping[str, Any] | Sequence[Any]] | None = None,
    covariance: str = "model_based",
    snapshot: str | None = None,
    artifact_id: str = _DEFAULT_ARTIFACT_ID,
) -> JointEffects:
    """Estimate ``treatments`` jointly with one shared adjustment set and row snapshot.

    ``treatments`` maps each treatment name to its column (or a :class:`Treatment`); the
    mapping order is the coefficient order and at least two are required. ``adjust`` maps the
    shared adjustment column names to their columns (the intercept is implicit). ``contrasts``
    declares linear contrasts of the named coefficients; their standard errors use the full
    covariance and their p-values get a Holm adjustment across the declared family.
    ``covariance`` is ``"model_based"`` (``sigma^2 (X'X)^-1``) or the robust sandwich ``"hc0"``,
    ``"hc1"``, ``"hc2"`` or ``"hc3"``; a robust covariance embeds the rows in the artifact and
    refuses above a row cap (``vector_treatment.hc_replay_row_cap_exceeded``). ``snapshot``
    names the row snapshot all columns come from; omitted, it is a content digest of the
    columns.

    Raises :class:`VectorTreatmentRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`) for treatments that do not share the
    adjustment set, snapshot or row count (``route_not_supported``), for a treatment without
    variation or a collinear design (``design_rank_deficient``), and for invalid inputs
    (``invalid_argument``: ``.too_few_treatments``, ``.too_few_rows``, ``.duplicate_name``,
    ``.non_finite_value``, ``.degenerate_covariance``, contrast details, ...).
    """
    if not isinstance(treatments, Mapping):
        raise CausalTypeError("treatments must be a mapping of name -> column")
    if adjust is not None and not isinstance(adjust, Mapping):
        raise CausalTypeError("adjust must be a mapping of name -> column")
    if covariance not in _COVARIANCES:
        raise CausalValueError(f"covariance must be one of {_COVARIANCES}")
    y = _column(outcome, "outcome")
    adjustment = {
        _name(n, "adjustment name"): _column(v, f"adjustment {n!r}")
        for n, v in (adjust or {}).items()
    }
    prepared: list[tuple[str, np.ndarray, Treatment]] = []
    for name, item in treatments.items():
        spec = item if isinstance(item, Treatment) else Treatment(item)
        column = _column(spec.values, f"treatment {name!r}")
        prepared.append((_name(name, "treatment name"), column, spec))
    shared = snapshot
    if shared is None:
        shared = _content_snapshot(
            [("outcome", y), *adjustment.items(), *[(n, v) for n, v, _ in prepared]]
        )
    _name(shared, "snapshot")
    declared_contrasts = [_contrast(c)._wire() for c in (contrasts or ())]
    declaration = {
        "row_snapshot": shared,
        "adjustment": list(adjustment),
        "treatments": [
            {
                "name": name,
                "adjustment_set": list(spec.adjustment_set)
                if spec.adjustment_set is not None
                else list(adjustment),
                "row_snapshot": spec.snapshot if spec.snapshot is not None else shared,
            }
            for name, _, spec in prepared
        ],
        "covariance": covariance,
        "contrasts": declared_contrasts,
    }
    report, artifact, refusal = _evaluate(
        y.tolist(),
        [v.tolist() for v in adjustment.values()],
        [v.tolist() for _, v, _ in prepared],
        json.dumps(declaration, allow_nan=False),
        _name(artifact_id, "artifact_id"),
    )
    if refusal is not None:
        raise VectorTreatmentRefusal(json.loads(refusal))
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native joint fit returned neither a result nor a refusal")
    return _result(report, bytes(artifact))


def consume_joint_effects(
    artifact: bytes,
    *,
    expected_identity: VectorTreatmentIdentity | Mapping[str, str] | None = None,
) -> JointEffects:
    """Recompute an exported joint-fit artifact and accept only an identical one.

    The coefficients, full covariance, contrasts, Holm values and joint Wald test are recomputed
    from the embedded compact design and every stored value must reproduce bit for bit. With
    ``expected_identity`` (the :attr:`JointEffects.identity` retained out-of-band) a changed snapshot,
    adjustment set, coefficient order, design, covariance kind, contrasts, null or result is
    refused even when the artifact was resealed consistently
    (:class:`VectorTreatmentRefusal`, ``route_not_supported``,
    ``vector_treatment.wrong_contract``). Corruption and unknown major versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, VectorTreatmentIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError("expected_identity must be a VectorTreatmentIdentity or a mapping")
    data = bytes(artifact)
    report, refusal = _consume(data, expected_json)
    if refusal is not None:
        raise VectorTreatmentRefusal(json.loads(refusal))
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)
