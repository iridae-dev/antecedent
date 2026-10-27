"""Nonlinear continuous-mediator mediation: natural direct, indirect and total effects.

A binary treatment ``A`` acts on an outcome ``Y`` directly and through a continuous mediator
``M``. This module estimates the **natural** effects, averaged over the observed covariate rows::

    NDE = E[Y(1, M(0))] - E[Y(0, M(0))]
    NIE = E[Y(1, M(1))] - E[Y(1, M(0))]
    total = NDE + NIE

    from antecedent.nonlinear_mediation import MediationPremises, mediation_effects

    result = mediation_effects(
        treatment, mediator, outcome, {"x": x},
        premises=MediationPremises.sequentially_ignorable(),
    )
    result.natural_direct, result.natural_indirect, result.total
    again = consume_mediation_artifact(result.export(), expected_identity=result.identity)

The mediator is linear-Gaussian in ``(A, X)``; the outcome is a polynomial of ``outcome_degree``
in the standardized mediator with a treatment interaction, linear in ``A`` and ``X``. The
integral over the cross-world mediator law is a Gauss-Hermite rule with ``quadrature_nodes``
nodes; the estimate is computed with ``n`` and ``2n`` nodes and the disagreement is reported as
``integration_error`` (an error above ``integration_tolerance`` is refused).

Identification is **declared**, never inferred: pass a :class:`MediationPremises`. A declared
unmeasured treatment-outcome, treatment-mediator or mediator-outcome confounder, a
treatment-induced mediator-outcome confounder, or an undeclared cross-world independence
refuses. The interventional (randomized-draw) estimand is closed and refuses with
``nonlinear_mediation.interventional_effects_closed``.

Bootstrap standard errors (fixed seed, replicate ids ``0 .. replicates``) have calibration
``"unmeasured"``: no coverage claim is made and no public interval is produced
(``interval_status == "closed_calibration_unmeasured"``). The point estimates and the
diagnostics are the claim.

:func:`consume_mediation_artifact` re-estimates from the artifact's embedded dataset under the
stored seed and refuses unless every stored value reproduces bit for bit; with ``expected_identity`` (the
:attr:`MediationEffects.identity` retained out-of-band) a resealed change of premises,
configuration, data or result is refused too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._native import consume_nonlinear_mediation_artifact as _consume
from ._native import evaluate_nonlinear_mediation as _evaluate
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import count, non_finite_refusal, real
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, resolve_execution

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact

__all__ = [
    "DEFAULT_SEED",
    "MediationBootstrap",
    "MediationEffects",
    "MediationEstimand",
    "MediationIdentity",
    "MediationOverlap",
    "MediationPremise",
    "MediationPremises",
    "MediationRefusal",
    "consume_mediation_artifact",
    "mediation_effects",
]

#: Default bootstrap seed (the Rust estimator's).
DEFAULT_SEED = 0x4234_4D45_4449_4154
_MAX_COUNT = 2**32 - 1


class MediationRefusal(StructuredRefusal):
    """A :class:`StructuredRefusal` for the structured Rust refusal fields.

    ``reason_code`` is the registered code; ``detail`` is the namespaced ``family.slot`` slot
    (for example ``nonlinear_mediation.confounding``,
    ``nonlinear_mediation.interventional_effects_closed``, ``nonlinear_mediation.overlap`` or
    ``nonlinear_mediation.integration_error``). ``stage`` is ``estimate`` or ``consume`` and
    ``offending`` names a changed identity field when a consumer refused one.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        super().__init__(refusal)


class MediationEstimand(StrEnum):
    """Which mediation estimand is requested."""

    #: Natural direct and indirect effects (the one open route).
    NATURAL_EFFECTS = "natural_effects"
    #: Interventional (randomized-draw) effects. Closed: asking for them refuses.
    INTERVENTIONAL_EFFECTS = "interventional_effects"


@dataclass(frozen=True, slots=True)
class MediationPremises:
    """The declared graph premises of the natural effects.

    Every field is a declaration; nothing here is tested against the data. A declared
    confounder, a treatment-induced mediator-outcome confounder or a missing cross-world
    independence declaration refuses the estimate.
    """

    #: An unmeasured common cause of treatment and outcome is declared.
    unmeasured_treatment_outcome_confounding: bool = False
    #: An unmeasured common cause of treatment and mediator is declared.
    unmeasured_treatment_mediator_confounding: bool = False
    #: An unmeasured common cause of mediator and outcome is declared.
    unmeasured_mediator_outcome_confounding: bool = False
    #: Names of declared mediator-outcome confounders that are descendants of the treatment.
    treatment_induced_confounders: tuple[str, ...] = ()
    #: The cross-world independence ``Y(a, m) _||_ M(a') | X`` is declared.
    cross_world_independence: bool = False

    def __post_init__(self) -> None:
        for name in (
            "unmeasured_treatment_outcome_confounding",
            "unmeasured_treatment_mediator_confounding",
            "unmeasured_mediator_outcome_confounding",
            "cross_world_independence",
        ):
            if not isinstance(getattr(self, name), bool):
                raise CausalTypeError(f"{name} must be a bool")
        names = self.treatment_induced_confounders
        if isinstance(names, str) or not isinstance(names, Sequence):
            raise CausalTypeError("treatment_induced_confounders must be a sequence of names")
        if not all(isinstance(n, str) and n for n in names):
            raise CausalTypeError("treatment_induced_confounders must be non-empty strings")
        coerce_fields(self, treatment_induced_confounders=tuple(names))

    @classmethod
    def sequentially_ignorable(cls) -> MediationPremises:
        """No unmeasured confounding of any of the three relations beyond the covariates, no
        treatment-induced confounder, and the cross-world independence declared."""
        return cls(cross_world_independence=True)

    def _wire(self) -> dict[str, Any]:
        return {
            "unmeasured_treatment_outcome_confounding": (
                self.unmeasured_treatment_outcome_confounding
            ),
            "unmeasured_treatment_mediator_confounding": (
                self.unmeasured_treatment_mediator_confounding
            ),
            "unmeasured_mediator_outcome_confounding": (
                self.unmeasured_mediator_outcome_confounding
            ),
            "treatment_induced_mediator_outcome_confounders": list(
                self.treatment_induced_confounders
            ),
            "cross_world_independence": self.cross_world_independence,
        }


@dataclass(frozen=True, slots=True)
class MediationPremise:
    """One premise of the artifact and its status."""

    name: str
    #: ``"declared"`` (asserted by the caller) or ``"checked"`` (verified from the data).
    status: str
    holds: bool


@dataclass(frozen=True, slots=True)
class MediationOverlap:
    """Overlap diagnostics of the cross-world integral."""

    treated_count: int
    control_count: int
    #: Observed mediator range of the treated arm.
    treated_mediator_range: tuple[float, float]
    #: Average mass of ``M | A = 0, X`` outside that range: the share the outcome model
    #: extrapolates.
    mediator_support_violation: float


@dataclass(frozen=True, slots=True)
class MediationBootstrap:
    """The bootstrap record: seed, replicate ids and the closed interval status."""

    seed: int
    replicates_requested: int
    replicates_succeeded: int
    replicate_ids: tuple[int, ...]
    failed_replicate_ids: tuple[int, ...]
    id_scheme: str
    #: ``"closed_calibration_unmeasured"``: no public interval is produced.
    interval_status: str


@dataclass(frozen=True, slots=True)
class MediationIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_mediation_artifact` as ``expected_identity=``.
    """

    premises_id: str
    config_id: str
    data_id: str
    result_id: str
    digest: str

    def _wire(self) -> dict[str, str]:
        return {
            "premises_id": self.premises_id,
            "config_id": self.config_id,
            "data_id": self.data_id,
            "result_id": self.result_id,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class MediationEffects(ScientificResult):
    """Natural effects, their diagnostics and the portable artifact.

    ``natural_direct``, ``natural_indirect`` and ``total`` are the claim; ``*_se`` are bootstrap
    standard errors with ``calibration == "unmeasured"`` and no interval. ``integration_error``
    is the ``coarse_nodes`` versus ``fine_nodes`` (``2n``) disagreement as a fraction of the
    outcome standard deviation; the point estimates use ``fine_nodes``.
    """

    estimand: MediationEstimand
    natural_direct: float
    natural_indirect: float
    total: float
    natural_direct_se: float | None
    natural_indirect_se: float | None
    total_se: float | None
    identity_residual: float
    integration_error: float
    coarse_nodes: int
    fine_nodes: int
    outcome_degree: int
    n_rows: int
    overlap: MediationOverlap
    #: Mediator model coefficients: intercept, treatment, then covariates.
    mediator_coefficients: tuple[float, ...]
    mediator_residual_variance: float
    mediator_residual_skewness: float
    mediator_residual_excess_kurtosis: float
    bootstrap: MediationBootstrap
    premises: tuple[MediationPremise, ...]
    model_specs: Mapping[str, Any]
    assumptions: tuple[str, ...]
    calibration: str
    interval_status: str
    inference_claim: str
    caveats: tuple[str, ...]
    identity: MediationIdentity
    #: ``None`` when the artifact would exceed its export byte budget; the estimate above stands.
    _artifact_bytes: bytes | None
    _export_refusal: str | None = field(default=None, repr=False)

    @artifact_export
    def export(self) -> ExportedArtifact[MediationEffects]:
        """The checksummed ``nonlinear_mediation_v1`` artifact.

        Raises :class:`MediationRefusal` (``route_not_supported``,
        ``nonlinear_mediation.export_budget_exceeded``) when the embedded dataset would not fit
        the export byte budget; the estimate itself is unaffected.
        """
        if self._artifact_bytes is None:
            raise_refusal(self._export_refusal, MediationRefusal)
        assert self._artifact_bytes is not None
        return artifact_from_receipt(self, self._artifact_bytes)

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes)."""
        return {
            "estimand": self.estimand.value,
            "natural_direct": self.natural_direct,
            "natural_indirect": self.natural_indirect,
            "total": self.total,
            "natural_direct_se": self.natural_direct_se,
            "natural_indirect_se": self.natural_indirect_se,
            "total_se": self.total_se,
            "identity_residual": self.identity_residual,
            "integration_error": self.integration_error,
            "coarse_nodes": self.coarse_nodes,
            "fine_nodes": self.fine_nodes,
            "outcome_degree": self.outcome_degree,
            "n_rows": self.n_rows,
            "overlap": {
                "treated_count": self.overlap.treated_count,
                "control_count": self.overlap.control_count,
                "treated_mediator_range": list(self.overlap.treated_mediator_range),
                "mediator_support_violation": self.overlap.mediator_support_violation,
            },
            "premises": [
                {"name": p.name, "status": p.status, "holds": p.holds} for p in self.premises
            ],
            "bootstrap": {
                "seed": self.bootstrap.seed,
                "replicates_requested": self.bootstrap.replicates_requested,
                "replicates_succeeded": self.bootstrap.replicates_succeeded,
                "failed_replicate_ids": list(self.bootstrap.failed_replicate_ids),
                "interval_status": self.bootstrap.interval_status,
            },
            "calibration": self.calibration,
            "interval_status": self.interval_status,
            "inference_claim": self.inference_claim,
            "caveats": list(self.caveats),
            "identity": self.identity._wire(),
        }


def _result(
    report_json: str, artifact: bytes | None, export_refusal: str | None = None
) -> MediationEffects:
    artifact = container_bytes(artifact) if artifact is not None else None
    meta = json.loads(report_json)
    r = meta["result"]
    overlap = r["overlap"]
    boot = r["bootstrap"]
    mediator = r["mediator"]

    def optional(value: float | None) -> float | None:
        return None if value is None else float(value)

    return MediationEffects(
        estimand=MediationEstimand(meta["estimand"]),
        natural_direct=float(r["natural_direct"]),
        natural_indirect=float(r["natural_indirect"]),
        total=float(r["total"]),
        natural_direct_se=optional(r["natural_direct_se"]),
        natural_indirect_se=optional(r["natural_indirect_se"]),
        total_se=optional(r["total_se"]),
        identity_residual=float(r["identity_residual"]),
        integration_error=float(r["integration_error"]),
        coarse_nodes=int(r["coarse_nodes"]),
        fine_nodes=int(r["fine_nodes"]),
        outcome_degree=int(r["outcome_degree"]),
        n_rows=int(r["n_rows"]),
        overlap=MediationOverlap(
            treated_count=int(overlap["treated_count"]),
            control_count=int(overlap["control_count"]),
            treated_mediator_range=(
                float(overlap["treated_mediator_min"]),
                float(overlap["treated_mediator_max"]),
            ),
            mediator_support_violation=float(overlap["mediator_support_violation"]),
        ),
        mediator_coefficients=tuple(float(c) for c in mediator["coefficients"]),
        mediator_residual_variance=float(mediator["residual_variance"]),
        mediator_residual_skewness=float(mediator["residual_skewness"]),
        mediator_residual_excess_kurtosis=float(mediator["residual_excess_kurtosis"]),
        bootstrap=MediationBootstrap(
            seed=int(boot["seed"]),
            replicates_requested=int(boot["replicates_requested"]),
            replicates_succeeded=int(boot["replicates_succeeded"]),
            replicate_ids=tuple(int(i) for i in boot["replicate_ids"]),
            failed_replicate_ids=tuple(int(i) for i in boot["failed_replicate_ids"]),
            id_scheme=boot["id_scheme"],
            interval_status=boot["interval_status"],
        ),
        premises=tuple(
            MediationPremise(p["name"], p["status"], bool(p["holds"]))
            for p in meta["premise_records"]
        ),
        model_specs=dict(meta["model_specs"]),
        assumptions=tuple(r["assumptions"]),
        calibration=meta["calibration"],
        interval_status=meta["interval_status"],
        inference_claim=meta["inference_claim"],
        caveats=tuple(meta["caveats"]),
        identity=MediationIdentity(**meta["identity"]),
        _artifact_bytes=artifact,
        _export_refusal=export_refusal,
    )


def _column(values: object, what: str) -> np.ndarray:
    try:
        column = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a numeric array") from error
    if column.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional", reason_code="invalid_argument")
    return np.ascontiguousarray(column)


def _covariate_columns(covariates: Any) -> tuple[list[str], list[np.ndarray]]:
    """Names and columns of the covariates: a mapping, one 1-D array or a 2-D array."""
    if covariates is None:
        return [], []
    if isinstance(covariates, Mapping):
        names = [str(name) for name in covariates]
        columns = [_column(covariates[name], f"covariate {name!r}") for name in covariates]
    else:
        try:
            array = np.asarray(covariates, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CausalTypeError("covariates must be a mapping or a numeric array") from error
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise CausalValueError(
                "covariates must be one- or two-dimensional", reason_code="invalid_argument"
            )
        columns = [np.ascontiguousarray(array[:, j]) for j in range(array.shape[1])]
        names = [f"x{j}" for j in range(len(columns))]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise CausalValueError(
            "covariate names must be unique and non-empty", reason_code="invalid_argument"
        )
    return names, columns


_count = partial(count, minimum=0, maximum=_MAX_COUNT, reason_code="invalid_argument")


_real = partial(
    real,
    finite=True,
    non_finite=non_finite_refusal(
        MediationRefusal, stage="estimate", detail="nonlinear_mediation.non_finite_option"
    ),
)


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise CausalTypeError("seed must be an integer")
    number = int(value)
    if not 0 <= number < 2**64:
        raise CausalValueError("seed must lie in 0..2**64", reason_code="invalid_argument")
    return number


def mediation_effects(
    treatment: Any,
    mediator: Any,
    outcome: Any,
    covariates: Any = None,
    *,
    premises: MediationPremises,
    estimand: MediationEstimand | str = MediationEstimand.NATURAL_EFFECTS,
    outcome_degree: int = 2,
    quadrature_nodes: int = 16,
    integration_tolerance: float = 1e-6,
    min_arm_count: int = 10,
    max_support_violation: float = 0.25,
    bootstrap_replicates: int = 200,
    seed: int = DEFAULT_SEED,
    data_binding: Literal["embedded", "digest"] = "embedded",
) -> MediationEffects:
    """Estimate the natural direct, indirect and total effects through a continuous mediator.

    ``treatment`` is binary (0/1), ``mediator`` continuous, ``outcome`` any real; ``covariates``
    is a mapping of name to column, one array or a 2-D array of pre-treatment covariates
    (adjusted for in both models and averaged over). ``premises`` is required: the identification
    premises are declared, not tested. ``outcome_degree`` (1 to 4) is the polynomial degree of
    the outcome in the standardized mediator, ``quadrature_nodes`` (1 to 64) the declared
    Gauss-Hermite node count ``n`` (``2n`` is also evaluated), ``integration_tolerance`` the
    largest accepted ``n`` versus ``2n`` disagreement as a fraction of the outcome standard
    deviation, ``min_arm_count`` the fewest rows per arm and ``max_support_violation`` the
    largest accepted share of the cross-world mediator law outside the treated arm's observed
    mediator range. ``bootstrap_replicates`` and ``seed`` fix the bootstrap (``0`` reports no
    standard errors). ``data_binding="digest"`` exports a record that holds no rows, bound to the
    dataset by its digest; :func:`consume_mediation_artifact` then needs the dataset passed back.

    The estimator has no row ceiling. The artifact embeds the dataset (at most 64 covariates,
    2000 bootstrap replicates) when it fits the 16 MiB export budget; above it the estimate is
    returned with no artifact and :meth:`MediationEffects.export` raises :class:`MediationRefusal`
    (``nonlinear_mediation.export_budget_exceeded``).

    Raises :class:`MediationRefusal` (a :class:`~antecedent.errors.CausalUnsupportedError`) for
    declared confounding (``effect_not_identified``, ``nonlinear_mediation.confounding``), a
    treatment-induced confounder or undeclared cross-world independence
    (``cross_world_not_identified``), the interventional estimand
    (``route_not_supported``, ``nonlinear_mediation.interventional_effects_closed``), weak
    overlap (``nonlinear_mediation.overlap``), an integration error above tolerance
    (``nonlinear_mediation.integration_error``) and degenerate or rank-deficient models.
    """
    if not isinstance(premises, MediationPremises):
        raise CausalTypeError("premises must be a MediationPremises declaration")
    try:
        requested = MediationEstimand(estimand)
    except ValueError as error:
        raise CausalValueError(
            f"estimand must be one of {[e.value for e in MediationEstimand]}",
            reason_code="invalid_argument",
        ) from error
    names, columns = _covariate_columns(covariates)
    a = _column(treatment, "treatment")
    m = _column(mediator, "mediator")
    y = _column(outcome, "outcome")
    if not (len(a) == len(m) == len(y) and all(len(c) == len(a) for c in columns)):
        raise CausalValueError(
            "treatment, mediator, outcome and covariates must share one length",
            reason_code="invalid_argument",
        )
    config = {
        "estimand": requested.value,
        "outcome_degree": _count(outcome_degree, "outcome_degree"),
        "quadrature_nodes": _count(quadrature_nodes, "quadrature_nodes"),
        "integration_tolerance": _real(integration_tolerance, "integration_tolerance"),
        "min_arm_count": _count(min_arm_count, "min_arm_count"),
        "max_support_violation": _real(max_support_violation, "max_support_violation"),
        "bootstrap_replicates": _count(bootstrap_replicates, "bootstrap_replicates"),
        "seed": _seed(seed),
    }
    report, artifact, refusal = _evaluate(
        a,
        m,
        y,
        columns,
        names,
        json.dumps(premises._wire(), allow_nan=False),
        json.dumps(config, allow_nan=False),
        "nonlinear_mediation",
        digest_bound=data_binding == "digest",
    )
    if report is not None and artifact is None and refusal is not None:
        return _result(report, None, refusal)  # the export was refused; the estimate stands
    raise_refusal(refusal, MediationRefusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native mediation estimator returned neither a result nor a refusal"
        )
    return _result(report, bytes(artifact))


@artifact_consumer(
    proof="nonlinear_mediation.consume_mediation_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_mediation_artifact(
    artifact: bytes,
    *,
    expected_identity: MediationIdentity | Mapping[str, str] | None = None,
    treatment: Any = None,
    mediator: Any = None,
    outcome: Any = None,
    covariates: Any = None,
    execution: Execution | None = None,
) -> MediationEffects:
    """Re-estimate an exported mediation artifact and accept only an identical one.

    The dataset embedded in the artifact is re-run through the estimator under the stored
    premises, configuration and seed, and every stored value must reproduce bit for bit. With
    ``expected_identity`` (the :attr:`MediationEffects.identity` retained out-of-band) a changed premise,
    configuration, dataset or result is refused even when the artifact was resealed consistently
    (:class:`MediationRefusal`, ``route_not_supported``, ``nonlinear_mediation.wrong_contract``).
    Corruption and unknown major versions raise
    :class:`~antecedent.errors.CausalSerializationError`.

    ``execution.limits.memory_bytes`` bounds the re-estimation's planned working set
    (``nonlinear_mediation.memory_budget_exceeded``). An artifact exported with
    ``data_binding="digest"`` holds no rows: pass ``treatment``, ``mediator``, ``outcome`` and
    ``covariates`` (the dataset it was estimated on); any other dataset is refused
    (``IdentityMismatch``, ``data``) and omitting it refuses with
    ``nonlinear_mediation.data_required``.
    """
    execution = resolve_execution(execution)
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, MediationIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError("expected_identity must be a MediationIdentity or a mapping")
    data = bytes(artifact)
    given = [treatment, mediator, outcome]
    supplied = None
    if any(v is not None for v in given):
        if any(v is None for v in given):
            raise CausalValueError(
                "treatment, mediator and outcome must be supplied together",
                reason_code="invalid_argument",
            )
        names, columns = _covariate_columns(covariates)
        supplied = (
            _column(treatment, "treatment"),
            _column(mediator, "mediator"),
            _column(outcome, "outcome"),
            names,
            columns,
        )
    elif covariates is not None:
        raise CausalValueError(
            "covariates need treatment, mediator and outcome", reason_code="invalid_argument"
        )
    report, refusal = _consume(
        data, expected_json, memory_limit_bytes=execution.limits.memory_bytes, data=supplied
    )
    raise_refusal(refusal, MediationRefusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)
