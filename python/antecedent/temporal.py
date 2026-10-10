"""Effect constancy across a declared time or region partition (F18).

One question: does *the same defined effect* take the same value in every declared partition?
You supply one effect estimate and standard error per partition (disjoint periods or regions),
the shared effect definition, and, when the partition estimates are dependent, their full
covariance. The test is a global heterogeneity statistic (Cochran's ``Q`` for independent
partitions, the Wald chi-square of the contrasts against the first partition when a covariance
is supplied) with per-pair contrasts adjusted by Holm's step-down procedure::

    from antecedent import temporal

    effect = temporal.EffectEstimand("ate_difference", "outcome_units", "treat_vs_control", "h2")
    result = temporal.effect_constancy(
        [("p1", 1.0, 0.5), ("p2", 2.0, 0.5)], estimand=effect
    )
    result.statistic, result.p_value, result.conclusion      # 2.0, 0.157..., NOT_REJECTED
    again = temporal.consume_effect_constancy_artifact(result.export(), expected_identity=result.identity)

What the answer is **not**:

* Failing to reject (:attr:`ConstancyConclusion.NOT_REJECTED`) never proves the effect is
  constant. It is the absence of evidence against the null at the stated level.
* The test's Type I error and power are **unmeasured** (``calibration == "unmeasured"``): no
  false-positive or power claim is made, and the result grants no causal or interval license.
  The claim is a point (``inference_claim == "point_only"``).
* The null (the effect is equal across all declared partitions) is frozen; it is not a
  parameter.

Partitions that change the effect definition, units, regime or population refuse the global
test (:class:`TemporalRefusal`, ``reason_code="route_not_supported"``, detail
``effect_constancy.incompatible_partitions``). Partitions are put in label order before any
arithmetic, so permuting the input (and a covariance consistently) gives bit-identical output.

Every statistic, identity digest, refusal rule and artifact is computed in Rust; this module
builds the declaration and presents the result. The exported artifact is a checksummed
container whose consumer recomputes the statistic, p-value and Holm values from the embedded
effects, standard errors and covariance. Pass the :attr:`EffectConstancyResult.identity` you
retained out-of-band as ``expected_identity=`` to refuse a *resealed* change of partition identity,
estimand, covariance, null or multiplicity family.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ._native import consume_effect_constancy_artifact as _consume
from ._native import evaluate_effect_constancy as _evaluate
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

__all__ = [
    "ConstancyConclusion",
    "Contrast",
    "EffectConstancyIdentity",
    "EffectConstancyResult",
    "EffectEstimand",
    "Partition",
    "PartitionEffect",
    "TemporalRefusal",
    "consume_effect_constancy_artifact",
    "effect_constancy",
]

#: Frozen null of the global test; it is not configurable.
NULL = "effect equal across all declared partitions"
_SUPPORT = ("supported", "partial", "unsupported")
_MAX_PARTITIONS = 1024


class TemporalRefusal(CausalUnsupportedError):
    """A :class:`CausalUnsupportedError` carrying the structured Rust refusal fields.

    ``reason_code`` and ``remedy`` are the inherited, registered fields; ``detail`` is the
    namespaced ``family.slot`` slot (for example ``effect_constancy.incompatible_partitions``
    or ``temporal_counterfactual.unpaired_histories``). ``witness`` is the refusing witness
    the evaluator retained (unit, history, world, node, residual, bound) and
    ``missing_gates`` / ``missing_factors`` list what a closed transported route lacks.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        parts = [detail]
        message = refusal.get("message")
        if message:
            parts[0] = f"{detail}: {message}"
        if refusal.get("offending"):
            parts.append(f"at {refusal['offending']}")
        super().__init__(" ".join(parts), reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``family.slot`` detail.
        self.detail: str = detail
        #: Offending unit, node, field or factor, when there is one.
        self.offending: str | None = refusal.get("offending")
        #: The refusing witness as a mapping, when the evaluator retained one.
        self.witness: Mapping[str, Any] | None = refusal.get("witness")
        #: Prerequisite gates a closed route lacks, in a fixed order.
        self.missing_gates: tuple[str, ...] = tuple(refusal.get("missing_gates", ()))
        #: Regime factors (``source:<regime>`` / ``target:<regime>``) the evidence map lacks.
        self.missing_factors: tuple[str, ...] = tuple(refusal.get("missing_factors", ()))


def _raise_refusal(payload: str | None) -> None:
    """Raise the structured refusal in ``payload`` (a JSON string), if there is one."""
    if payload is not None:
        raise TemporalRefusal(json.loads(payload))


def _finite(value: object, what: str, detail: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise TemporalRefusal(
            {
                "code": "invalid_argument",
                "stage": "test",
                "detail": detail,
                "offending": what,
                "message": f"{what} is not finite",
            }
        )
    return number


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


class ConstancyConclusion(StrEnum):
    """Decision of the global test at the declared level."""

    #: The null was not rejected. This does **not** establish constancy.
    NOT_REJECTED = "not_rejected"
    #: The null was rejected: the effect differs across partitions.
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EffectEstimand:
    """The effect every partition must estimate, identically.

    ``estimand`` defines the effect (for example ``"ate_difference"``), ``units`` its scale,
    ``regime`` the intervention contrast and ``population`` the population and horizon.
    Partitions that disagree on any of the four refuse the global test.
    """

    estimand: str
    units: str
    regime: str
    population: str

    def __post_init__(self) -> None:
        for field_name in ("estimand", "units", "regime", "population"):
            _name(getattr(self, field_name), field_name)

    def _wire(self) -> dict[str, str]:
        return {
            "estimand": self.estimand,
            "units": self.units,
            "regime": self.regime,
            "population": self.population,
        }


@dataclass(frozen=True, slots=True)
class Partition:
    """One partition's effect estimate.

    ``coordinate`` is the partition's typed coordinate (a period or region identifier); when
    omitted it is derived as ``"<kind>:<label>"`` from the ``coordinate=`` kind given to
    :func:`effect_constancy`. ``estimand`` overrides the shared one for this partition (a
    partition that changes it refuses the test). ``support`` is ``"supported"``,
    ``"partial"`` or ``"unsupported"``; an unsupported partition refuses.
    """

    label: str
    effect: float
    se: float
    coordinate: str | None = None
    estimand: EffectEstimand | None = None
    support: str = "supported"

    def __post_init__(self) -> None:
        _name(self.label, "label")
        if self.support not in _SUPPORT:
            raise CausalValueError(f"support must be one of {_SUPPORT}")
        if self.coordinate is not None:
            _name(self.coordinate, "coordinate")
        if self.estimand is not None and not isinstance(self.estimand, EffectEstimand):
            raise CausalTypeError("estimand must be an EffectEstimand")
        object.__setattr__(self, "effect", _finite(self.effect, "effect", _NON_FINITE))
        object.__setattr__(self, "se", _finite(self.se, "se", _NON_FINITE))


_NON_FINITE = "effect_constancy.non_finite_estimate"


def _partition(item: Partition | Mapping[str, Any] | Sequence[Any]) -> Partition:
    if isinstance(item, Partition):
        return item
    if isinstance(item, Mapping):
        try:
            return Partition(**dict(item))
        except TypeError as error:
            raise CausalTypeError(f"a partition mapping has unexpected keys: {error}") from error
    if isinstance(item, Sequence) and not isinstance(item, str) and len(item) in (3, 4):
        return Partition(*item)
    raise CausalTypeError(
        "a partition is a Partition, a mapping with label/effect/se, or a "
        "(label, effect, se[, coordinate]) tuple"
    )


@dataclass(frozen=True, slots=True)
class PartitionEffect:
    """One partition as reported, in canonical (label) order."""

    label: str
    coordinate: str
    support: str
    effect: float
    se: float


@dataclass(frozen=True, slots=True)
class Contrast:
    """One multiplicity-adjusted contrast ``effect(left) - effect(right)``."""

    left: str
    right: str
    difference: float
    se: float
    z: float
    p_value: float
    #: Holm-adjusted p-value over the declared family.
    p_holm: float
    #: Whether ``p_holm`` is below the level.
    rejected: bool


@dataclass(frozen=True, slots=True)
class EffectConstancyIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Each is BLAKE3 over a canonical encoding in label order. Pass it to
    :func:`consume_effect_constancy_artifact` as ``expected_identity=``.
    """

    partition_id: str
    estimand_id: str
    covariance_id: str
    null: str
    family_id: str
    evidence_id: str
    digest: str

    def _wire(self) -> dict[str, str]:
        return {
            "partition_id": self.partition_id,
            "estimand_id": self.estimand_id,
            "covariance_id": self.covariance_id,
            "null": self.null,
            "family_id": self.family_id,
            "evidence_id": self.evidence_id,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class EffectConstancyResult:
    """The global constancy test, its per-partition table and its portable artifact.

    ``partitions`` carries each partition's effect (canonical label order) so transport
    diagnostics, prior transfer and policy generalization can consume them directly.
    ``conclusion`` is never a proof of constancy and ``calibration`` is ``"unmeasured"``;
    read :attr:`caveats`. ``pooled_effect`` is the inverse-variance pooled effect and exists
    only for independent partitions.
    """

    null: str
    estimand: EffectEstimand
    partitions: tuple[PartitionEffect, ...]
    statistic_kind: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    alpha: float
    conclusion: ConstancyConclusion
    dependence: str
    family: str
    reference: str | None
    contrasts: tuple[Contrast, ...]
    pooled_effect: float | None
    caveats: tuple[str, ...]
    calibration: str
    inference_claim: str
    identity: EffectConstancyIdentity
    artifact: bytes

    def export(self) -> bytes:
        """The checksummed ``effect_constancy_v1`` artifact."""
        return self.artifact

    def table(self) -> list[dict[str, Any]]:
        """The per-partition table: label, coordinate, support, effect and standard error."""
        return [
            {
                "label": p.label,
                "coordinate": p.coordinate,
                "support": p.support,
                "effect": p.effect,
                "se": p.se,
            }
            for p in self.partitions
        ]

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes)."""
        return {
            "null": self.null,
            "statistic_kind": self.statistic_kind,
            "statistic": self.statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "conclusion": self.conclusion.value,
            "calibration": self.calibration,
            "inference_claim": self.inference_claim,
            "partitions": self.table(),
            "contrasts": [
                {
                    "left": c.left,
                    "right": c.right,
                    "difference": c.difference,
                    "se": c.se,
                    "z": c.z,
                    "p_value": c.p_value,
                    "p_holm": c.p_holm,
                    "rejected": c.rejected,
                }
                for c in self.contrasts
            ],
            "caveats": list(self.caveats),
            "identity": self.identity._wire(),
        }


def _result(report_json: str, artifact: bytes) -> EffectConstancyResult:
    report = json.loads(report_json)
    result = report["result"]
    family = report["family"]
    return EffectConstancyResult(
        null=report["null"],
        estimand=EffectEstimand(**report["estimand"]),
        partitions=tuple(
            PartitionEffect(
                label=p["label"],
                coordinate=p["coordinate"],
                support=p["support"],
                effect=float(p["effect"]),
                se=float(p["standard_error"]),
            )
            for p in report["partitions"]
        ),
        statistic_kind=result["statistic_kind"],
        statistic=float(result["statistic"]),
        degrees_of_freedom=int(result["degrees_of_freedom"]),
        p_value=float(result["p_value"]),
        alpha=float(report["alpha"]),
        conclusion=ConstancyConclusion(result["conclusion"]),
        dependence=report["dependence"],
        family=family["kind"],
        reference=family["reference"],
        contrasts=tuple(
            Contrast(
                left=c["left"],
                right=c["right"],
                difference=float(c["difference"]),
                se=float(c["standard_error"]),
                z=float(c["z"]),
                p_value=float(c["p_value"]),
                p_holm=float(c["p_holm"]),
                rejected=bool(c["rejected"]),
            )
            for c in result["contrasts"]
        ),
        pooled_effect=None if result["pooled_effect"] is None else float(result["pooled_effect"]),
        caveats=tuple(report["caveats"]),
        calibration=report["calibration"],
        inference_claim=report["inference_claim"],
        identity=EffectConstancyIdentity(**report["identity"]),
        artifact=artifact,
    )


def _covariance(covariance: Any) -> list[float]:
    """Row-major flattening; Rust owns the shape, symmetry and definiteness rules."""
    try:
        flat = np.asarray(covariance, dtype=np.float64).ravel()
    except (TypeError, ValueError) as error:
        raise CausalTypeError("covariance must be a numeric square matrix") from error
    if not np.all(np.isfinite(flat)):
        # JSON cannot carry a non-finite number, so this one rule is applied here with the
        # same code and detail the Rust test reports.
        raise TemporalRefusal(
            {
                "code": "invalid_argument",
                "stage": "test",
                "detail": "effect_constancy.invalid_covariance",
                "message": "the covariance has a non-finite entry",
            }
        )
    return [float(x) for x in flat]


def effect_constancy(
    partitions: Sequence[Partition | Mapping[str, Any] | Sequence[Any]],
    *,
    estimand: EffectEstimand | None = None,
    covariance: Any = None,
    against: str | None = None,
    alpha: float = 0.05,
    coordinate: str = "partition",
) -> EffectConstancyResult:
    """Test whether one effect is constant across the declared ``partitions``.

    ``partitions`` are :class:`Partition` objects, ``{"label", "effect", "se"}`` mappings or
    ``(label, effect, se)`` tuples; their order does not matter. ``estimand`` is the effect
    definition shared by every partition that does not carry its own; every partition must end
    up with one, and they must all be equal. ``covariance`` is the full ``k x k`` covariance of
    the estimates *in the order supplied* (omit it only when the partitions use disjoint,
    independent evidence); its diagonal must equal the squared standard errors. ``against``
    names a reference partition for the multiplicity family (every other partition against
    it); otherwise every pair is tested. ``alpha`` is the level of the decisions and
    ``coordinate`` the kind used to derive a missing partition coordinate
    (``"<kind>:<label>"``).

    Raises :class:`TemporalRefusal` (a :class:`~antecedent.errors.CausalUnsupportedError`) for
    incompatible partitions (``route_not_supported``,
    ``effect_constancy.incompatible_partitions`` / ``.unsupported_partition``) and for invalid
    numbers (``invalid_argument``: ``.too_few_partitions``, ``.invalid_covariance``,
    ``.invalid_standard_error``, ``.unknown_reference``, ``.invalid_alpha``, ...).
    """
    if isinstance(partitions, str | bytes) or not isinstance(partitions, Sequence):
        raise CausalTypeError("partitions must be a sequence")
    if estimand is not None and not isinstance(estimand, EffectEstimand):
        raise CausalTypeError("estimand must be an EffectEstimand")
    kind = _name(coordinate, "coordinate")
    declared = [_partition(item) for item in partitions]
    if len(declared) > _MAX_PARTITIONS:
        raise CausalValueError(
            f"effect_constancy.too_many_partitions: at most {_MAX_PARTITIONS} partitions",
            reason_code="invalid_argument",
        )
    wire_partitions = []
    for p in declared:
        shared = p.estimand if p.estimand is not None else estimand
        if shared is None:
            raise CausalValueError(
                "effect_constancy.estimand_missing: declare the shared effect with estimand= or "
                f"on partition {p.label!r}; the global test never assumes partitions estimate "
                "the same effect",
                reason_code="invalid_argument",
            )
        wire_partitions.append(
            {
                "label": p.label,
                "coordinate": p.coordinate if p.coordinate is not None else f"{kind}:{p.label}",
                "support": p.support,
                "estimand": shared._wire(),
                "effect": p.effect,
                "standard_error": p.se,
            }
        )
    if covariance is None:
        dependence: dict[str, Any] = {"kind": "independent", "covariance": []}
    else:
        dependence = {"kind": "covariance", "covariance": _covariance(covariance)}
    if against is not None:
        _name(against, "against")
    family = {
        "kind": "all_pairs" if against is None else "against_reference",
        "reference": against,
    }
    request = {
        "partitions": wire_partitions,
        "dependence": dependence,
        "family": family,
        "alpha": _finite(alpha, "alpha", "effect_constancy.invalid_alpha"),
    }
    report, artifact, refusal = _evaluate(json.dumps(request, allow_nan=False), "effect_constancy")
    _raise_refusal(refusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native effect-constancy test returned neither a result nor a refusal"
        )
    return _result(report, bytes(artifact))


def consume_effect_constancy_artifact(
    artifact: bytes,
    *,
    expected_identity: EffectConstancyIdentity | Mapping[str, str] | None = None,
) -> EffectConstancyResult:
    """Recompute an exported constancy artifact and accept only an identical one.

    The statistic, p-value, per-pair contrasts and Holm values are recomputed from the
    embedded effects, standard errors and covariance, and every stored value must reproduce
    bit for bit. With ``expected_identity`` (the :attr:`EffectConstancyResult.identity` retained
    out-of-band) a changed partition identity, estimand, covariance, null or multiplicity
    family is refused even when the artifact was resealed consistently
    (:class:`TemporalRefusal`, ``route_not_supported``, ``effect_constancy.wrong_contract``).
    Corruption and unknown major versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, EffectConstancyIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError("expected_identity must be an EffectConstancyIdentity or a mapping")
    data = bytes(artifact)
    report, refusal = _consume(data, expected_json)
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)
