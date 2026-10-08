"""B4 latent-class (finite mixture) regime effects.

The population is a mixture of ``K`` latent regimes (``2 <= K <= 4``, declared, never
selected). Within class ``k`` the outcome follows a Gaussian linear model in the treatment ``A``
and the covariates ``X``::

    Y | A, X, class = k  ~  N(alpha_k + tau_k * A + gamma_k' X, sigma_k^2),   P(class = k) = pi_k

``tau_k`` is the class-specific effect. The class weights ``pi_k`` are constants (a
covariate-dependent class prior is not offered)::

    from antecedent.latent_class import consume_latent_class_artifact, latent_class_effects

    result = latent_class_effects(
        y, a, {"x": x}, classes=2, within_class_randomization=True, seed=11
    )
    result.mixture_effect                   # sum_k pi_k * tau_k
    [(c.weight, c.effect) for c in result.classes]
    result.responsibilities                 # (n, K) posterior class probabilities
    again = consume_latent_class_artifact(result.export(), expected_identity=result.identity)

Labels carry no meaning. Classes are reported in a **canonical order**: ascending effect, then
intercept, then weight. ``raw_index`` and :attr:`LatentClassEffects.class_order` keep the mapping
to the raw EM components, so a permuted initial labelling, another seed or a reordering of rows
cannot change the reported classes. "Class 0" is the lowest-effect class, not a substantive
name; ``min_effect_gap == 0`` means two classes are not separately identified in effect.

Identification is **declared**: treatment must be as-if randomized within class given the
covariates (pass ``within_class_randomization=True``; without it the fit refuses with
``latent_class.randomization_not_declared``), the Gaussian-linear class model and constant class
weights hold, and ``K`` is the true number of classes. Weak or degenerate classes refuse
(``latent_class.weak_class``, ``latent_class.degenerate_class``).

Standard errors are a seeded, label-aligned bootstrap with calibration ``"unmeasured"``: no
coverage claim and no interval is made. :func:`consume_latent_class_artifact` refits from the
artifact's embedded dataset under the stored seed and configuration and refuses unless the
canonical output reproduces bit for bit; with ``expected_identity`` (the
:attr:`LatentClassEffects.identity` retained out-of-band) a resealed change of configuration,
data or result is refused too.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._native import consume_latent_class_artifact as _consume
from ._native import evaluate_latent_class_effects as _evaluate
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

__all__ = [
    "LatentClass",
    "LatentClassBootstrap",
    "LatentClassEffects",
    "LatentClassIdentity",
    "LatentClassPremise",
    "LatentClassRefusal",
    "ResponsibilitiesSummary",
    "consume_latent_class_artifact",
    "latent_class_effects",
]

_MAX_COUNT = 2**32 - 1


class LatentClassRefusal(CausalUnsupportedError):
    """A :class:`CausalUnsupportedError` carrying the structured Rust refusal fields.

    ``reason_code`` is the registered code; ``detail`` is the namespaced ``family.slot`` slot:
    ``latent_class.randomization_not_declared``, ``latent_class.weak_class``,
    ``latent_class.degenerate_class``, ``latent_class.not_converged``,
    ``latent_class.likelihood_not_monotone``, ``latent_class.treatment_constant``,
    ``latent_class.design_support_too_small``, ``latent_class.constant_outcome`` and the input
    validation details. ``stage`` is ``fit`` or ``consume``; ``offending`` names a changed
    identity field when a consumer refused one.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        text = f"{detail}: {message}" if message else detail
        if refusal.get("offending"):
            text = f"{text} at {refusal['offending']}"
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``family.slot`` detail.
        self.detail: str = detail
        #: Changed identity field, when a consumer refused one.
        self.offending: str | None = refusal.get("offending")


def _raise_refusal(payload: str | None) -> None:
    if payload is not None:
        raise LatentClassRefusal(json.loads(payload))


@dataclass(frozen=True, slots=True)
class LatentClass:
    """One class, in canonical order (ascending effect)."""

    #: Canonical index.
    index: int
    #: Raw EM component index of the selected fit this class came from.
    raw_index: int
    weight: float
    weight_se: float | None
    intercept: float
    #: Class effect ``tau_k``: change in the class mean outcome per unit of treatment.
    effect: float
    effect_se: float | None
    #: Covariate coefficients ``gamma_k`` by covariate name.
    covariate_coefficients: Mapping[str, float]
    #: Maximum-likelihood residual variance ``sigma_k^2``.
    residual_variance: float
    #: Sum of the class's responsibilities.
    effective_n: float


@dataclass(frozen=True, slots=True)
class ResponsibilitiesSummary:
    """Summary of the posterior class responsibilities."""

    rows: int
    #: Mean responsibility of each canonical class.
    class_means: tuple[float, ...]
    #: Units whose most probable class is each canonical class.
    hard_counts: tuple[int, ...]
    #: Mean largest responsibility (1.0 is perfect separation).
    separation: float
    #: BLAKE3 over every responsibility's bits and the hard assignment.
    digest: str


@dataclass(frozen=True, slots=True)
class LatentClassBootstrap:
    """Bootstrap bookkeeping (replicates that did not converge are skipped)."""

    requested: int
    succeeded: int
    failed: int


@dataclass(frozen=True, slots=True)
class LatentClassPremise:
    """One identification premise and its status."""

    name: str
    #: ``"declared"`` (asserted by the caller) or ``"checked"`` (verified from the data).
    status: str


@dataclass(frozen=True, slots=True)
class LatentClassIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_latent_class_artifact` as ``expected_identity=``.
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
class LatentClassEffects:
    """A fitted mixture of class-specific effects and its portable artifact.

    ``classes`` are in canonical order; ``mixture_effect`` is ``sum_k pi_k tau_k``.
    ``responsibilities`` is the ``(n, K)`` matrix of posterior class probabilities (canonical
    columns) and ``hard_assignment`` each unit's most probable canonical class. Bootstrap
    standard errors have ``calibration == "unmeasured"``: they carry no coverage claim.
    """

    classes_declared: int
    classes: tuple[LatentClass, ...]
    #: ``class_order[c]`` is the raw EM component index of canonical class ``c``.
    class_order: tuple[int, ...]
    mixture_effect: float
    mixture_effect_se: float | None
    #: Smallest gap between adjacent canonical effects (0: not separately identified).
    min_effect_gap: float
    responsibilities: np.ndarray
    hard_assignment: np.ndarray
    responsibilities_summary: ResponsibilitiesSummary
    log_likelihood: float
    #: First entries of the (non-decreasing) EM log-likelihood trace.
    log_likelihood_trace_head: tuple[float, ...]
    iterations: int
    bic: float
    restarts_attempted: int
    restarts_converged: int
    bootstrap: LatentClassBootstrap | None
    premises: tuple[LatentClassPremise, ...]
    seed: int
    calibration: str
    inference_claim: str
    caveat: str
    identity: LatentClassIdentity
    artifact: bytes

    def export(self) -> bytes:
        """The checksummed ``latent_class_effects_v1`` artifact."""
        return self.artifact

    @property
    def separation(self) -> float:
        """Mean over units of the largest posterior responsibility."""
        return self.responsibilities_summary.separation

    def table(self) -> list[dict[str, Any]]:
        """The per-class table in canonical order."""
        return [
            {
                "index": c.index,
                "raw_index": c.raw_index,
                "weight": c.weight,
                "weight_se": c.weight_se,
                "intercept": c.intercept,
                "effect": c.effect,
                "effect_se": c.effect_se,
                "residual_variance": c.residual_variance,
                "effective_n": c.effective_n,
            }
            for c in self.classes
        ]

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes, no per-unit matrix)."""
        summary = self.responsibilities_summary
        return {
            "classes_declared": self.classes_declared,
            "class_order": list(self.class_order),
            "classes": [
                {**row, "covariate_coefficients": dict(c.covariate_coefficients)}
                for row, c in zip(self.table(), self.classes, strict=True)
            ],
            "mixture_effect": self.mixture_effect,
            "mixture_effect_se": self.mixture_effect_se,
            "min_effect_gap": self.min_effect_gap,
            "responsibilities": {
                "rows": summary.rows,
                "class_means": list(summary.class_means),
                "hard_counts": list(summary.hard_counts),
                "separation": summary.separation,
                "digest": summary.digest,
            },
            "log_likelihood": self.log_likelihood,
            "iterations": self.iterations,
            "bic": self.bic,
            "premises": [{"name": p.name, "status": p.status} for p in self.premises],
            "calibration": self.calibration,
            "inference_claim": self.inference_claim,
            "caveat": self.caveat,
            "identity": self.identity._wire(),
        }


def _optional(value: float | None) -> float | None:
    return None if value is None else float(value)


def _result(report_json: str, artifact: bytes) -> LatentClassEffects:
    report = json.loads(report_json)
    meta = report["meta"]
    names = meta["covariate_names"]
    summary = meta["responsibilities"]
    likelihood = meta["likelihood"]
    boot = meta["bootstrap"]
    k = int(meta["classes_declared"])
    matrix = np.asarray(report["responsibilities"], dtype=np.float64).reshape(-1, k)
    return LatentClassEffects(
        classes_declared=k,
        classes=tuple(
            LatentClass(
                index=int(c["index"]),
                raw_index=int(c["raw_index"]),
                weight=float(c["weight"]),
                weight_se=_optional(c["weight_se"]),
                intercept=float(c["intercept"]),
                effect=float(c["effect"]),
                effect_se=_optional(c["effect_se"]),
                covariate_coefficients=dict(
                    zip(names, (float(v) for v in c["covariate_coefficients"]), strict=True)
                ),
                residual_variance=float(c["residual_variance"]),
                effective_n=float(c["effective_n"]),
            )
            for c in meta["classes"]
        ),
        class_order=tuple(int(i) for i in meta["class_order"]),
        mixture_effect=float(meta["mixture_average_effect"]),
        mixture_effect_se=_optional(meta["mixture_average_se"]),
        min_effect_gap=float(meta["min_effect_gap"]),
        responsibilities=matrix,
        hard_assignment=np.asarray(report["hard_assignment"], dtype=np.int64),
        responsibilities_summary=ResponsibilitiesSummary(
            rows=int(summary["rows"]),
            class_means=tuple(float(v) for v in summary["class_means"]),
            hard_counts=tuple(int(v) for v in summary["hard_counts"]),
            separation=float(summary["separation"]),
            digest=summary["digest"],
        ),
        log_likelihood=float(likelihood["log_likelihood"]),
        log_likelihood_trace_head=tuple(float(v) for v in likelihood["trace_head"]),
        iterations=int(likelihood["iterations"]),
        bic=float(likelihood["bic"]),
        restarts_attempted=int(likelihood["restarts_attempted"]),
        restarts_converged=int(likelihood["restarts_converged"]),
        bootstrap=None
        if boot is None
        else LatentClassBootstrap(
            requested=int(boot["requested"]),
            succeeded=int(boot["succeeded"]),
            failed=int(boot["failed"]),
        ),
        premises=tuple(LatentClassPremise(p["name"], p["status"]) for p in meta["premises"]),
        seed=int(meta["config"]["seed"]),
        calibration=meta["calibration"],
        inference_claim=meta["inference_claim"],
        caveat=meta["caveat"],
        identity=LatentClassIdentity(**meta["identity"]),
        artifact=artifact,
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


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise CausalTypeError(f"{what} must be an integer")
    number = int(value)
    if not 0 <= number <= _MAX_COUNT:
        raise CausalValueError(
            f"{what} must lie in 0..={_MAX_COUNT}", reason_code="invalid_argument"
        )
    return number


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise LatentClassRefusal(
            {
                "code": "invalid_argument",
                "stage": "fit",
                "detail": "latent_class.invalid_config",
                "offending": what,
                "message": f"{what} is not finite",
            }
        )
    return number


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise CausalTypeError("seed must be an integer")
    number = int(value)
    if not 0 <= number < 2**64:
        raise CausalValueError("seed must lie in 0..2**64", reason_code="invalid_argument")
    return number


def _labels(labels: Sequence[int] | None) -> list[int] | None:
    if labels is None:
        return None
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence | np.ndarray):
        raise CausalTypeError("initial_labels must be a sequence of class labels")
    return [_count(label, "an initial label") for label in labels]


def latent_class_effects(
    outcome: Any,
    treatment: Any,
    covariates: Any = None,
    *,
    classes: int,
    within_class_randomization: bool = False,
    seed: int = 0,
    restarts: int = 10,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    min_class_weight: float = 0.05,
    min_separation: float = 0.7,
    variance_floor: float = 1e-8,
    bootstrap_replicates: int = 200,
    initial_labels: Sequence[int] | None = None,
) -> LatentClassEffects:
    """Fit a ``classes``-component mixture of Gaussian linear outcome models and report effects.

    ``outcome`` and ``treatment`` (binary or numeric) are columns; ``covariates`` is a mapping
    of name to column, one array or a 2-D array of covariates that enter every class's mean.
    ``classes`` (2 to 4) is declared, not selected. ``within_class_randomization=True`` declares
    treatment as-if randomized within class given the covariates; without it the fit refuses.
    ``seed`` fixes every deterministic stream (initialization restarts, bootstrap).
    ``restarts`` EM starts are tried (the highest converged log-likelihood is reported; one
    ``initial_labels`` hard labelling replaces them), each for at most ``max_iterations`` with
    relative log-likelihood ``tolerance``. A class lighter than ``min_class_weight``, or a mean
    largest posterior responsibility below ``min_separation``, refuses as weak.
    ``variance_floor`` (a fraction of the outcome variance) marks a class as degenerate.
    ``bootstrap_replicates`` (0 reports no standard errors) resamples rows with label alignment.

    The artifact embeds the dataset (at most 10000 rows, 32 covariates, 1000 bootstrap
    replicates).

    Raises :class:`LatentClassRefusal` (a :class:`~antecedent.errors.CausalUnsupportedError`):
    ``required_option_missing`` / ``latent_class.randomization_not_declared``;
    ``population_not_estimable`` / ``latent_class.weak_class``, ``.degenerate_class``,
    ``.constant_outcome``; ``mechanism_fit_not_converged`` / ``latent_class.not_converged``,
    ``.likelihood_not_monotone``; ``effect_not_identified`` / ``latent_class.treatment_constant``;
    ``design_rank_deficient`` / ``latent_class.design_support_too_small``; and
    ``invalid_argument`` input details.
    """
    if not isinstance(within_class_randomization, bool):
        raise CausalTypeError("within_class_randomization must be a bool")
    names, columns = _covariate_columns(covariates)
    y = _column(outcome, "outcome")
    a = _column(treatment, "treatment")
    if not (len(y) == len(a) and all(len(c) == len(y) for c in columns)):
        raise CausalValueError(
            "outcome, treatment and covariates must share one length",
            reason_code="invalid_argument",
        )
    config = {
        "classes": _count(classes, "classes"),
        "seed": _seed(seed),
        "restarts": _count(restarts, "restarts"),
        "max_iterations": _count(max_iterations, "max_iterations"),
        "tolerance": _real(tolerance, "tolerance"),
        "min_class_weight": _real(min_class_weight, "min_class_weight"),
        "min_separation": _real(min_separation, "min_separation"),
        "variance_floor": _real(variance_floor, "variance_floor"),
        "bootstrap_replicates": _count(bootstrap_replicates, "bootstrap_replicates"),
        "initial_labels": _labels(initial_labels),
        "assume_conditional_randomization": within_class_randomization,
    }
    report, artifact, refusal = _evaluate(
        y, a, columns, names, json.dumps(config, allow_nan=False), "latent_class_effects"
    )
    _raise_refusal(refusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native latent-class fit returned neither a result nor a refusal"
        )
    return _result(report, bytes(artifact))


def consume_latent_class_artifact(
    artifact: bytes,
    *,
    expected_identity: LatentClassIdentity | Mapping[str, str] | None = None,
) -> LatentClassEffects:
    """Refit an exported latent-class artifact and accept only an identical one.

    The dataset embedded in the artifact is refit under the stored seed and configuration, and
    the canonical output (classes, weights, effects, responsibilities digest, log-likelihood
    trace) must reproduce bit for bit. With ``expected_identity`` (the
    :attr:`LatentClassEffects.identity` retained out-of-band) a changed configuration, dataset
    or result is refused even when the artifact was resealed consistently
    (:class:`LatentClassRefusal`, ``route_not_supported``, ``latent_class.wrong_contract``).
    Corruption and unknown major versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, LatentClassIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError("expected_identity must be a LatentClassIdentity or a mapping")
    data = bytes(artifact)
    report, refusal = _consume(data, expected_json)
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _result(report, data)
