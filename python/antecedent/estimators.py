"""Typed front-end over the ``estimator_config=`` dict kwarg.

``estimator_config`` (parsed Rust-side by ``python/src/estimator_config.rs``) is a
plain ``dict``: it works, but the key spelling, accepted value vocabularies, and
which-key-belongs-to-which-estimator rules live only in that Rust table. The
dataclasses here are a typed layer on top: one frozen, ``slots=True`` dataclass
per configurable estimator, fields named and defaulted to match the Rust setter
surface exactly, with a ``_wire()`` method that renders the dict the Rust parser
expects — omitting any key the caller did not set, so an all-defaults instance
is indistinguishable from passing no ``estimator_config`` at all.

``__post_init__`` validates fail-fast, in Python, combinations that Rust either
rejects with a less specific message or — in a couple of cases (bare
``cluster_ids``/``multiway_ids`` without a matching ``se``) — silently accepts
and then ignores. Catching those here means a caller learns about a
mismatched config before it reaches the Rust boundary, rather than getting a
result that quietly didn't honor part of what they asked for.

Every class also exposes ``estimator_id``, the wire id to pass as
``estimator=``. ``analyze(..., estimator=cfg)`` also works directly with a
config instance from this module — :func:`antecedent._analyze.analyze`
detects a non-``str``/``Estimator`` ``estimator=`` value, pulls
``estimator_id`` and ``cfg._wire()`` off it, and forwards them as
``estimator=``/``estimator_config=`` — so both spellings below are
equivalent::

    from antecedent import analyze
    from antecedent.estimators import LinearAdjustment

    cfg = LinearAdjustment(bootstrap=500, se="cluster", cluster_ids=ids)

    # Single-argument spelling.
    result = analyze(data, graph=g, query=q, estimator=cfg)

    # Equivalent two-argument spelling.
    result = analyze(
        data, graph=g, query=q,
        estimator=cfg.estimator_id,
        estimator_config=cfg._wire(),
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from ._defaults import OMITTED
from .errors import CausalValueError
from .ids import Estimator
from .learners import LearnerSpec, _learner_wire

SeKind = Literal[
    "homoskedastic",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "cluster",
    "multiway",
    "newey_west",
    "panel_cluster_hac",
]
FitKind = Literal["ols", "ridge", "lasso", "huber"]
PropensityPenaltyKind = Literal["ridge_logistic", "lasso"]
NuisanceFallbackName = Literal["none", "ml", "ridge_logistic", "lasso"]
IndependenceUnitName = Literal["cluster", "dyad"]
GlmFamilyName = Literal[
    "binomial_logit",
    "binomial_probit",
    "gaussian_identity",
    "poisson_log",
    "negative_binomial",
]

_SE_KINDS_NEEDING_LAG = ("newey_west", "panel_cluster_hac")


def _omit_empty(out: dict[str, Any]) -> dict[str, Any]:
    """All-defaults configuration is a strict no-op wire, not a missing one."""
    return out


class _Unset:
    """Sentinel distinguishing "field not set" from an explicit ``None``.

    Only ``GlmOptions.ridge_on_separation`` needs this: Rust's parser treats
    the key being *absent* (keep ``GlmOptions::default()``'s
    ``Some(1e-4)``) differently from the key being present with a Python
    ``None`` value (explicitly clears it to ``None``, disabling the
    ridge-on-separation fallback). A plain ``float | None`` field cannot
    distinguish those two cases; this sentinel is the third state.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset()


# --- Validation helpers, shared across the estimator dataclasses below --------------------


def _validate_bootstrap(bootstrap: int | None) -> None:
    # Non-negative, not strictly positive: bootstrap=0 is a legitimate, common way to
    # disable bootstrap replicate computation (Rust's own `get_u32` accepts it, and the
    # top-level `analyze(..., bootstrap=0)` kwarg is used exactly this way throughout the
    # test suite) — rejecting it here would break that pattern for no reason.
    if bootstrap is not None and bootstrap < 0:
        raise ValueError(f"bootstrap must be a non-negative int, got {bootstrap!r}")


def _validate_positive(name: str, value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _validate_se(
    *,
    se: SeKind | None,
    se_lag: int | None,
    cluster_ids: Sequence[int] | None,
    multiway_ids: Sequence[Sequence[int]] | None = None,
) -> None:
    """Cross-field ``se`` rules shared by every estimator that exposes ``se_kind``.

    Mirrors Rust's own ``se_lag`` requirement (``estimator_config.rs``'s
    ``build_se_kind``) and additionally rejects the inverse for ``cluster_ids`` /
    ``multiway_ids``: Rust's ``build_configured_spec`` calls
    ``est.with_cluster_ids(ids)`` unconditionally whenever ``cluster_ids`` is present,
    regardless of ``se``, so supplying ``cluster_ids`` with a non-cluster ``se`` is
    silently accepted and then never used by the SE formula — a caller mistake Rust
    does not name. Python catches it here instead.
    """
    needs_lag = se in _SE_KINDS_NEEDING_LAG
    if needs_lag and se_lag is None:
        raise ValueError(
            f"se={se!r} requires se_lag (newey_west/panel_cluster_hac are lag-parameterized)"
        )
    if not needs_lag and se_lag is not None:
        raise ValueError(
            f"se_lag is only valid when se is 'newey_west' or 'panel_cluster_hac', got se={se!r}"
        )
    if se == "cluster" and cluster_ids is None:
        raise ValueError("se='cluster' requires cluster_ids")
    if cluster_ids is not None and se != "cluster":
        raise ValueError(f"cluster_ids requires se='cluster'; got se={se!r}")
    if se == "multiway" and multiway_ids is None:
        raise ValueError("se='multiway' requires multiway_ids")
    if multiway_ids is not None and se != "multiway":
        raise ValueError(f"multiway_ids requires se='multiway'; got se={se!r}")


def _validate_linear_fit(
    *,
    fit: FitKind | None,
    fit_lambda: float | None,
    fit_c: float | None,
    se: SeKind | None,
) -> None:
    """``fit``-dependent rules for :class:`LinearAdjustment`, including the lasso trap."""
    if fit in ("ridge", "lasso"):
        if fit_lambda is None:
            raise ValueError(f"fit={fit!r} requires fit_lambda (the ridge/lasso penalty)")
    elif fit_lambda is not None:
        raise ValueError(f"fit_lambda requires fit='ridge' or fit='lasso'; got fit={fit!r}")
    if fit == "huber":
        if fit_c is None:
            raise ValueError("fit='huber' requires fit_c (the Huber tuning constant)")
    elif fit_c is not None:
        raise ValueError(f"fit_c requires fit='huber'; got fit={fit!r}")
    if fit == "lasso" and se is not None:
        # Rust's own doc comment on `LinearFitKind::Lasso`
        # (crates/antecedent-estimate/src/adjustment.rs): "Analytic SE is permanently
        # omitted: classical / active-set sandwich SEs are invalid after selection, and
        # debiased Lasso changes the point estimator. Use bootstrap
        # (bootstrap_replicates > 0); se_analytic is NaN." Rust's own setter stays
        # infallible and does not enforce this pairing (see that file's `with_fit_kind`
        # doc: "this setter stays dumb and does not enforce that pairing") — so a
        # caller who sets both `se=` and `fit="lasso"` gets a config that silently
        # produces `se_analytic = NaN` no matter what `se=` they chose. Naming that here
        # is exactly the point of a typed, validating front-end.
        raise ValueError(
            f"LinearAdjustment(fit='lasso', se={se!r}) is invalid: Lasso's analytic SE is "
            "permanently omitted — classical / active-set sandwich SEs are invalid after "
            "selection, and debiased Lasso changes the point estimator itself. se_analytic "
            f"is NaN for fit='lasso' regardless of se=; requesting se={se!r} would be "
            "silently ignored. Drop se= and set bootstrap=... (bootstrap_replicates > 0) "
            "instead to get a usable standard error."
        )


# --- Shared nested config ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlmOptions:
    """``glm_options`` sub-dict shared by every propensity-model-backed estimator.

    ``nb_alpha`` (the NB2 dispersion policy on the Rust side's ``GlmOptions``) is
    deliberately not exposed: Rust's own parser (``estimator_config.rs``'s
    ``build_glm_options``) pins it to ``MethodOfMoments`` and does not accept it from
    Python either, so there would be nothing for a field here to wire through.
    """

    max_iter: int | None = None
    tol: float | None = None
    ridge_on_separation: float | None | _Unset = UNSET

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.max_iter is not None:
            out["max_iter"] = self.max_iter
        if self.tol is not None:
            out["tol"] = self.tol
        if not isinstance(self.ridge_on_separation, _Unset):
            out["ridge_on_separation"] = self.ridge_on_separation
        return out


@dataclass(frozen=True, slots=True)
class PropensityPenalty:
    """Explicit penalized binary propensity for :class:`Aipw` (``propensity_penalty=``).

    ``kind="ridge_logistic"`` fits a ridge-penalized logistic propensity on each cross-fit
    fold's *training rows only*; ``kind="lasso"`` fits an L1-penalized one and also selects
    its support (the covariates with a nonzero coefficient) on those training rows. The
    penalty is the member of ``lambdas`` (a fixed grid; default ``(0.01, 0.1, 1, 10, 100,
    1000)`` for ridge and ``(0.5, 1, 2, 5, 10, 20, 50, 100)`` for lasso) with the smallest
    ``inner_folds``-fold cross-validated log loss on those training rows (default 5 folds,
    seeded by the analysis seed and replayable); the evaluation rows never inform the
    penalty or the support. Penalties are on a sum-scale log likelihood with each
    non-intercept covariate standardized by the training rows. This is a declared nuisance
    choice, distinct from ``GlmOptions.ridge_on_separation`` (a rescue that estimation paths
    refuse to keep).

    The route publishes the cross-fitted point estimate, the score table (row identity,
    overlap report and retargeting preserved, retarget covariance as for the unpenalized
    route), the cross-fitted influence-function SE of the out-of-fold scores, and, with
    ``bootstrap > 0``, a bootstrap SE whose every replicate repeats the fold plan, the
    penalty (and lasso support) selection and all nuisance fits on the resample. A lasso
    records the selected support of each fold on the estimate (``penalized_support``). Both
    intervals rest on a stated remainder condition (``docs/guides/penalized-aipw.md``);
    their coverage is what the calibration suite measures. Outcome models stay arm-wise OLS.
    A lasso outside the cross-fitted untrimmed ``AllObserved`` mean ATE is refused with
    ``selection_inference_not_licensed``.
    """

    kind: PropensityPenaltyKind = "ridge_logistic"
    lambdas: Sequence[float] | None = None
    inner_folds: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("ridge_logistic", "lasso"):
            raise CausalValueError(
                f"PropensityPenalty.kind must be 'ridge_logistic' or 'lasso', got {self.kind!r}"
            )
        if self.lambdas is not None:
            values = list(self.lambdas)
            if not values:
                raise CausalValueError("PropensityPenalty.lambdas must not be empty")
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not float(value) > 0.0
                    or float(value) == float("inf")
                ):
                    raise CausalValueError(
                        f"PropensityPenalty.lambdas must be finite and positive, got {value!r}"
                    )
        if self.inner_folds is not None and (
            isinstance(self.inner_folds, bool)
            or not isinstance(self.inner_folds, int)
            or not 2 <= self.inner_folds <= 20
        ):
            raise CausalValueError(
                f"PropensityPenalty.inner_folds must be an int in [2, 20], got {self.inner_folds!r}"
            )

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.lambdas is not None:
            out["lambdas"] = [float(value) for value in self.lambdas]
        if self.inner_folds is not None:
            out["inner_folds"] = int(self.inner_folds)
        return out


@dataclass(frozen=True, slots=True)
class ClusterDml:
    """Declared independence unit of a cross-fitted :class:`Aipw` (``cluster_dml=``).

    ``cluster_ids`` is the cluster label of every *complete-case* row (aligned to the rows
    the estimator uses, as for ``se="cluster"``). Whole clusters share a cross-fit fold, so
    the nuisances scoring a cluster are fit on other clusters only; this is not an IID
    cross-fit followed by a cluster standard error. ``min_clusters`` (default 20, at least
    10) is the smallest cluster count accepted: fewer clusters refuse
    ``too_few_clusters`` rather than forming a sandwich over a handful of cluster sums.

    The route publishes the cross-fitted **point estimate and score table** and **no
    interval** (``Aipw(bootstrap=0, cluster_dml=...)`` is required; a requested interval
    refuses ``cluster_interval_not_licensed``). It is licensed for the untrimmed
    ``AllObserved`` mean ATE and is not combined with ``propensity_penalty``.

    ``unit="dyad"`` declares two-way (dyadic) dependence: ``cluster_ids`` are the first
    endpoint of every row and ``second_cluster_ids`` the second, in separate label sets
    (an entity that is a first endpoint of some rows and a second endpoint of others refuses
    ``dyadic_dependence_not_licensed``). Folds own whole connected components of the endpoint
    graph, so no endpoint crosses folds; ``min_components_per_fold`` (default 4, at least 2)
    is the fewest components each of the five folds must own. One giant component (more than
    a fold's share of the rows) also refuses ``dyadic_dependence_not_licensed``. The
    two-way variance is a Rust receipt only; no interval is published.
    """

    cluster_ids: Sequence[int]
    min_clusters: int | None = None
    unit: IndependenceUnitName = "cluster"
    second_cluster_ids: Sequence[int] | None = None
    min_components_per_fold: int | None = None

    def __post_init__(self) -> None:
        if self.unit not in ("cluster", "dyad"):
            raise CausalValueError(
                f"ClusterDml.unit must be 'cluster' or 'dyad', got {self.unit!r}"
            )
        if len(self.cluster_ids) == 0:
            raise CausalValueError("ClusterDml.cluster_ids must not be empty")
        for label in self.cluster_ids:
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise CausalValueError(
                    f"ClusterDml.cluster_ids must be non-negative ints, got {label!r}"
                )
        if self.min_clusters is not None and (
            isinstance(self.min_clusters, bool)
            or not isinstance(self.min_clusters, int)
            or self.min_clusters < 10
        ):
            raise CausalValueError(
                f"ClusterDml.min_clusters must be an int of at least 10, got {self.min_clusters!r}"
            )
        if self.unit == "cluster":
            if self.second_cluster_ids is not None or self.min_components_per_fold is not None:
                raise CausalValueError(
                    "ClusterDml.second_cluster_ids and min_components_per_fold belong to "
                    "unit='dyad'"
                )
            return
        if self.second_cluster_ids is None:
            raise CausalValueError("ClusterDml(unit='dyad') requires second_cluster_ids")
        if len(self.second_cluster_ids) != len(self.cluster_ids):
            raise CausalValueError(
                "ClusterDml.second_cluster_ids must have one label per cluster_ids row, got "
                f"{len(self.second_cluster_ids)} and {len(self.cluster_ids)}"
            )
        for label in self.second_cluster_ids:
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise CausalValueError(
                    f"ClusterDml.second_cluster_ids must be non-negative ints, got {label!r}"
                )
        if self.min_components_per_fold is not None and (
            isinstance(self.min_components_per_fold, bool)
            or not isinstance(self.min_components_per_fold, int)
            or self.min_components_per_fold < 2
        ):
            raise CausalValueError(
                "ClusterDml.min_components_per_fold must be an int of at least 2, got "
                f"{self.min_components_per_fold!r}"
            )

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cluster_ids": [int(label) for label in self.cluster_ids]}
        if self.min_clusters is not None:
            out["min_clusters"] = int(self.min_clusters)
        if self.unit != "cluster":
            out["unit"] = self.unit
        if self.second_cluster_ids is not None:
            out["second_cluster_ids"] = [int(label) for label in self.second_cluster_ids]
        if self.min_components_per_fold is not None:
            out["min_components_per_fold"] = int(self.min_components_per_fold)
        return out


@dataclass(frozen=True, slots=True)
class Overlap:
    """Propensity clipping and trimming for a propensity-score estimator.

    Every propensity-score estimator (``PropensityWeighting``,
    ``PropensityMatching``, ``PropensityStratification``, ``DistanceMatching``,
    ``Aipw``) applies the native default policy (``omitted_defaults()``:
    clip at 0.01, no trim) unless it is given an ``overlap``. ``clip`` bounds the
    propensities used in the weights to ``[clip, 1 - clip]``; ``trim`` drops units whose propensity
    lies outside ``[trim, 1 - trim]``, which also narrows the population the
    effect describes. ``None`` turns that operation off. Each bound lies in
    ``(0, 0.5)``.

    ``Overlap()`` is the default policy, so passing it changes nothing. Any other
    policy is a different interval construction: its calibration binds only to
    coverage records measured under that policy, and otherwise reports
    ``scope_not_assessed``.
    """

    clip: float | None = OMITTED["overlap_clip"]
    trim: float | None = OMITTED["overlap_trim"]

    def __post_init__(self) -> None:
        for name in ("clip", "trim"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalValueError(f"Overlap.{name} must be a float or None, got {value!r}")
            if not 0.0 < float(value) < 0.5:
                raise CausalValueError(f"Overlap.{name} must lie in (0, 0.5), got {value!r}")

    def _wire(self) -> dict[str, Any]:
        return {
            "clip": None if self.clip is None else float(self.clip),
            "trim": None if self.trim is None else float(self.trim),
        }


def _wire_overlap(overlap: Overlap | None) -> dict[str, Any]:
    if overlap is None:
        return {}
    if not isinstance(overlap, Overlap):
        raise CausalValueError(f"overlap must be an Overlap, got {overlap!r}")
    return {"overlap": overlap._wire()}


def _wire_glm_options(glm_options: GlmOptions | None) -> dict[str, Any]:
    if glm_options is None:
        return {}
    sub = glm_options._wire()
    return {"glm_options": sub} if sub else {}


def _wire_se_common(
    *,
    bootstrap: int | None,
    se: SeKind | None,
    se_lag: int | None,
    cluster_ids: Sequence[int] | None,
    multiway_ids: Sequence[Sequence[int]] | None = None,
    panel_times: Sequence[int] | None = None,
) -> dict[str, Any]:
    """``bootstrap``/``se``/``se_lag``/``cluster_ids``/``multiway_ids``/``panel_times`` wiring
    shared by every estimator config that carries these fields.

    ``multiway_ids``/``panel_times`` default to ``None`` so callers whose dataclass has no
    such field (``FrontdoorLinearTwoStage``) can simply omit them rather than inventing values.
    """
    out: dict[str, Any] = {}
    if bootstrap is not None:
        out["bootstrap_replicates"] = bootstrap
    if se is not None:
        out["se_kind"] = se
    if se_lag is not None:
        out["se_lag"] = se_lag
    if cluster_ids is not None:
        out["cluster_ids"] = list(cluster_ids)
    if multiway_ids is not None:
        out["multiway_ids"] = [list(group) for group in multiway_ids]
    if panel_times is not None:
        out["panel_times"] = list(panel_times)
    return out


# --- Estimator configs -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearAdjustment:
    """``linear.adjustment.ate`` — OLS/ridge/lasso/Huber backdoor adjustment.

    ``fit="lasso"`` combined with ``se=...`` raises: Lasso's analytic SE is
    permanently ``NaN`` (see :func:`_validate_linear_fit`); use ``bootstrap=``.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    fit: FitKind | None = None
    fit_lambda: float | None = None
    fit_c: float | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )
        _validate_linear_fit(fit=self.fit, fit_lambda=self.fit_lambda, fit_c=self.fit_c, se=self.se)

    @property
    def estimator_id(self) -> str:
        return str(Estimator.LINEAR_ADJUSTMENT_ATE)

    def _wire(self) -> dict[str, Any]:
        out = _wire_se_common(
            bootstrap=self.bootstrap,
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
            panel_times=self.panel_times,
        )
        if self.fit is not None:
            out["fit_kind"] = self.fit
        if self.fit_lambda is not None:
            out["fit_lambda"] = self.fit_lambda
        if self.fit_c is not None:
            out["fit_c"] = self.fit_c
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class PropensityWeighting:
    """``propensity.weighting`` — Hajek-normalized IPW.

    No ``se``/``cluster_ids``/... fields: the Rust struct carries no
    ``AnalyticSeKind`` at all for this estimator (the Hajek SE isn't
    parameterized that way) — only ``bootstrap_replicates`` and ``glm_options``
    are configurable, plus the propensity ``overlap`` policy.
    """

    bootstrap: int | None = None
    glm_options: GlmOptions | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)

    @property
    def estimator_id(self) -> str:
        return str(Estimator.PROPENSITY_WEIGHTING)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        out.update(_wire_glm_options(self.glm_options))
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class PropensityMatching:
    """``propensity.matching`` — nearest-neighbor propensity matching.

    ``caliper`` is a maximum matching distance, interpreted on ``caliper_scale``.
    The default scale is ``"logit"``, matching the field convention: the familiar
    0.2 rule of thumb (Rosenbaum & Rubin 1985; Austin 2011) is 0.2 standard
    deviations *of the logit* propensity, not of the raw probability. The raw
    probability scale compresses near 0 and 1, exactly where match quality matters
    most, so a caliper given on it behaves quite differently. Pass
    ``caliper_scale="raw"`` to match on the clipped propensity directly.

    Only the homoskedastic Abadie–Imbens standard error is published.
    ``se="cluster"``, ``"multiway"``, ``"newey_west"`` and ``"panel_cluster_hac"``
    are accepted here but refused at fit (``estimator_inference_mismatch``): no
    clustered influence function exists for fixed-match nearest-neighbour matching.
    Use :class:`Aipw` for clustered or serially dependent data.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    caliper: float | None = None
    caliper_scale: Literal["logit", "raw"] | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )
        _validate_positive("caliper", self.caliper)
        if self.caliper_scale is not None and self.caliper_scale not in ("logit", "raw"):
            raise CausalValueError(
                f'caliper_scale must be "logit" or "raw", got {self.caliper_scale!r}'
            )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.PROPENSITY_MATCHING)

    def _wire(self) -> dict[str, Any]:
        out = _wire_se_common(
            bootstrap=self.bootstrap,
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
            panel_times=self.panel_times,
        )
        out.update(_wire_glm_options(self.glm_options))
        if self.caliper is not None:
            out["caliper"] = self.caliper
        if self.caliper_scale is not None:
            out["caliper_scale"] = self.caliper_scale
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class PropensityStratification:
    """``propensity.stratification`` — propensity-score strata (default 5 strata)."""

    bootstrap: int | None = None
    glm_options: GlmOptions | None = None
    n_strata: int | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_positive("n_strata", self.n_strata)

    @property
    def estimator_id(self) -> str:
        return str(Estimator.PROPENSITY_STRATIFICATION)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        out.update(_wire_glm_options(self.glm_options))
        if self.n_strata is not None:
            out["n_strata"] = self.n_strata
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class DistanceMatching:
    """``distance.matching`` — Mahalanobis/caliper covariate-distance matching.

    Only the homoskedastic Abadie–Imbens standard error is published; clustered,
    multiway and HAC ``se`` kinds are refused at fit (``estimator_inference_mismatch``),
    as for :class:`PropensityMatching`.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    caliper: float | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )
        _validate_positive("caliper", self.caliper)

    @property
    def estimator_id(self) -> str:
        return str(Estimator.DISTANCE_MATCHING)

    def _wire(self) -> dict[str, Any]:
        out = _wire_se_common(
            bootstrap=self.bootstrap,
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
            panel_times=self.panel_times,
        )
        out.update(_wire_glm_options(self.glm_options))
        if self.caliper is not None:
            out["caliper"] = self.caliper
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class Aipw:
    """``aipw`` — augmented inverse propensity weighting.

    The point estimate is doubly robust: consistent when either the propensity or
    the outcome model is. The analytic standard error (``se=...``) is not: it
    corrects for parametric nuisances only, is not valid for flexible or
    nonparametric nuisances, and for ATT/ATC is not robust to a misspecified
    propensity or outcome model. The default inference is the bootstrap, which
    refits the nuisance models on every resample.

    ``propensity_penalty`` declares a ridge- or lasso-logistic propensity chosen on each
    fold's training rows (:class:`PropensityPenalty`); the cross-fitted influence-function
    SE is published, and ``bootstrap > 0`` publishes a bootstrap SE that repeats penalty
    selection and nuisance fitting on every resample.
    ``nuisance_fallback`` declares the destination a failed GLM propensity fit falls back to:
    ``"ridge_logistic"`` / ``"lasso"`` (default tuning) or a :class:`PropensityPenalty`
    (its kind and tuning) re-run the whole cross-fitted route with that penalized propensity
    and record the failed fit (``penalized_fallback``) beside the result; the claim is the
    destination's and nothing is silently substituted. ``"ml"`` (a flexible learner) is
    closed (``nuisance_fallback_not_licensed``): a failed fit is refused with the failure
    recorded. A fallback is not combined with ``propensity_penalty``.
    ``cluster_dml`` declares whole-cluster cross-fitting (:class:`ClusterDml`); it requires
    ``bootstrap=0`` and publishes the point estimate and score table with no interval.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    overlap: Overlap | None = None
    propensity_penalty: PropensityPenalty | None = None
    nuisance_fallback: NuisanceFallbackName | PropensityPenalty | None = None
    cluster_dml: ClusterDml | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )
        if self.cluster_dml is not None:
            if not isinstance(self.cluster_dml, ClusterDml):
                raise CausalValueError(
                    f"cluster_dml must be a ClusterDml, got {self.cluster_dml!r}"
                )
            if self.bootstrap != 0 or self.se not in (None, "homoskedastic"):
                raise CausalValueError(
                    "Aipw(cluster_dml=...) publishes no interval: pass bootstrap=0 and leave "
                    "se unset (reason=cluster_interval_not_licensed)"
                )
            if self.cluster_ids is not None:
                raise CausalValueError(
                    "Aipw(cluster_dml=...) carries its own cluster_ids; do not also pass "
                    "cluster_ids"
                )
            if self.propensity_penalty is not None:
                raise CausalValueError(
                    "Aipw(cluster_dml=...) is not combined with propensity_penalty "
                    "(reason=route_not_supported)"
                )
        if self.propensity_penalty is not None:
            if not isinstance(self.propensity_penalty, PropensityPenalty):
                raise CausalValueError(
                    "propensity_penalty must be a PropensityPenalty, "
                    f"got {self.propensity_penalty!r}"
                )
            if self.nuisance_fallback not in (None, "none"):
                raise CausalValueError(
                    "Aipw(nuisance_fallback=...) replaces a failed GLM propensity fit and is not "
                    "combined with propensity_penalty (reason=invalid_argument)"
                )
        if isinstance(self.nuisance_fallback, PropensityPenalty):
            pass
        elif self.nuisance_fallback not in (None, "none", "ml", "ridge_logistic", "lasso"):
            raise CausalValueError(
                "nuisance_fallback must be 'none', 'ml', 'ridge_logistic', 'lasso' or a "
                f"PropensityPenalty, got {self.nuisance_fallback!r}"
            )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.AIPW)

    def _wire(self) -> dict[str, Any]:
        out = _wire_se_common(
            bootstrap=self.bootstrap,
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
            panel_times=self.panel_times,
        )
        out.update(_wire_glm_options(self.glm_options))
        out.update(_wire_overlap(self.overlap))
        if self.propensity_penalty is not None:
            out["propensity_penalty"] = self.propensity_penalty._wire()
        if isinstance(self.nuisance_fallback, PropensityPenalty):
            out["nuisance_fallback"] = self.nuisance_fallback._wire()
        elif self.nuisance_fallback is not None:
            out["nuisance_fallback"] = self.nuisance_fallback
        if self.cluster_dml is not None:
            out["cluster_dml"] = self.cluster_dml._wire()
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class GlmAdjustment:
    """``glm.adjustment`` — GLM-family outcome-model adjustment (default binomial-logit)."""

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    family: GlmFamilyName | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.GLM_ADJUSTMENT)

    def _wire(self) -> dict[str, Any]:
        out = _wire_se_common(
            bootstrap=self.bootstrap,
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
            panel_times=self.panel_times,
        )
        out.update(_wire_glm_options(self.glm_options))
        if self.family is not None:
            out["family"] = self.family
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class FrontdoorLinearTwoStage:
    """``frontdoor.linear_two_stage`` — linear product-of-coefficients front-door estimator.

    Multiplies OLS coefficients (``T -> M`` times ``M -> Y`` given ``T``) rather than
    evaluating the front-door functional, so it is exact only when ``E[M|T]`` is linear
    and ``E[Y|M,T]`` has no treatment-mediator interaction. The result is therefore
    reported as identified under parametric restrictions, with
    ``frontdoor.linear_path_product`` among its identification assumptions. For a
    discrete treatment prefer
    ``estimator="frontdoor.functional"``, which estimates the functional itself.

    No ``multiway_ids``/``panel_times`` fields: the Rust struct carries only
    ``cluster_ids`` (no multiway/panel SE machinery for this estimator) — matches
    ``estimator_config.rs``'s ``ESTIMATOR_KEYS`` row for ``frontdoor.linear_two_stage``,
    which likewise omits those two keys.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(se=self.se, se_lag=self.se_lag, cluster_ids=self.cluster_ids)

    @property
    def estimator_id(self) -> str:
        return str(Estimator.FRONTDOOR_LINEAR_TWO_STAGE)

    def _wire(self) -> dict[str, Any]:
        return _omit_empty(
            _wire_se_common(
                bootstrap=self.bootstrap,
                se=self.se,
                se_lag=self.se_lag,
                cluster_ids=self.cluster_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class IvWald:
    """``iv.wald`` — single binary-instrument Wald ratio.

    The ratio ``(E[Y|Z=1] - E[Y|Z=0]) / (E[T|Z=1] - E[T|Z=0])`` is the average
    treatment effect only when the treatment's effect on the outcome is the same
    constant, linear effect for every unit. When effects differ across units it is,
    for a binary treatment and under monotonicity (no defiers), the local average
    treatment effect for compliers, not the population average. The instrument
    cannot tell which case holds; the result is reported as identified under
    parametric restrictions and carries
    ``iv.constant_linear_effect_or_monotonicity`` among its assumptions.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.IV_WALD)

    def _wire(self) -> dict[str, Any]:
        return _omit_empty(
            _wire_se_common(
                bootstrap=self.bootstrap,
                se=self.se,
                se_lag=self.se_lag,
                cluster_ids=self.cluster_ids,
                multiway_ids=self.multiway_ids,
                panel_times=self.panel_times,
            )
        )


@dataclass(frozen=True, slots=True)
class Iv2Sls:
    """``iv.2sls`` — two-stage least squares with one or more instruments.

    The coefficient on the instrumented treatment is the average treatment effect
    only under a constant linear structural effect. With heterogeneous effects it
    is an instrument-weighted average of complier effects (for one binary
    instrument and a binary treatment, the complier local average treatment
    effect under monotonicity), not the population average. The result carries
    ``iv.constant_linear_effect_or_monotonicity`` among its assumptions.
    """

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap(self.bootstrap)
        _validate_se(
            se=self.se,
            se_lag=self.se_lag,
            cluster_ids=self.cluster_ids,
            multiway_ids=self.multiway_ids,
        )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.IV_2SLS)

    def _wire(self) -> dict[str, Any]:
        return _omit_empty(
            _wire_se_common(
                bootstrap=self.bootstrap,
                se=self.se,
                se_lag=self.se_lag,
                cluster_ids=self.cluster_ids,
                multiway_ids=self.multiway_ids,
                panel_times=self.panel_times,
            )
        )


@dataclass(frozen=True, slots=True)
class DML:
    """``dml`` — cross-fitted DML / AIPW.

    These flexible-learner estimators take no cluster or dependence option: an
    ``estimator_config`` carrying ``cluster_ids``, ``cluster_dml`` or ``multiway_ids`` for
    them refuses ``route_not_supported`` (``cluster_dml.flexible_learner_closed``) instead of
    being ignored; use ``Aipw(bootstrap=0, cluster_dml=ClusterDml(...))`` for clustered data.
    """

    learner: LearnerSpec | str | None = None
    outcome: LearnerSpec | str | None = None
    treatment: LearnerSpec | str | None = None
    score: str | None = None
    folds: int | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        if self.folds is not None and self.folds < 2:
            raise ValueError(f"DML folds must be at least 2, got {self.folds!r}")
        if self.score is not None and self.score not in {"aipw", "partially_linear"}:
            raise ValueError(f"DML score must be 'aipw' or 'partially_linear', got {self.score!r}")

    @property
    def estimator_id(self) -> str:
        return str(Estimator.DML)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.learner is not None:
            out["learner"] = _learner_wire(self.learner)
        if self.outcome is not None:
            out["outcome"] = _learner_wire(self.outcome)
        if self.treatment is not None:
            out["treatment"] = _learner_wire(self.treatment)
        if self.score is not None:
            out["score"] = self.score
        if self.folds is not None:
            out["folds"] = self.folds
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class DRLearner:
    """``dr.learner`` — cross-fitted CATE learner on the doubly robust score.

    The doubly robust property is that of the score: the effect estimate is
    consistent when either the propensity or the outcome nuisance is. It is a
    property of the point estimate, not of any reported standard error.

    These flexible-learner estimators take no cluster or dependence option: an
    ``estimator_config`` carrying ``cluster_ids``, ``cluster_dml`` or ``multiway_ids`` for
    them refuses ``route_not_supported`` (``cluster_dml.flexible_learner_closed``) instead of
    being ignored; use ``Aipw(bootstrap=0, cluster_dml=ClusterDml(...))`` for clustered data.
    """

    learner: LearnerSpec | str | None = None
    outcome: LearnerSpec | str | None = None
    treatment: LearnerSpec | str | None = None
    final_learner: LearnerSpec | str | None = None
    folds: int | None = None
    overlap: Overlap | None = None

    def __post_init__(self) -> None:
        if self.folds is not None and self.folds < 2:
            raise ValueError(f"DRLearner folds must be at least 2, got {self.folds!r}")

    @property
    def estimator_id(self) -> str:
        return str(Estimator.DR_LEARNER)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.learner is not None:
            out["learner"] = _learner_wire(self.learner)
        if self.outcome is not None:
            out["outcome"] = _learner_wire(self.outcome)
        if self.treatment is not None:
            out["treatment"] = _learner_wire(self.treatment)
        if self.final_learner is not None:
            out["final_learner"] = _learner_wire(self.final_learner)
        if self.folds is not None:
            out["folds"] = self.folds
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class CausalForest:
    """``causal.forest`` — honest causal-forest CATE.

    These flexible-learner estimators take no cluster or dependence option: an
    ``estimator_config`` carrying ``cluster_ids``, ``cluster_dml`` or ``multiway_ids`` for
    them refuses ``route_not_supported`` (``cluster_dml.flexible_learner_closed``) instead of
    being ignored; use ``Aipw(bootstrap=0, cluster_dml=ClusterDml(...))`` for clustered data.
    """

    n_trees: int | None = None
    min_leaf: int | None = None
    max_depth: int | None = None
    honesty: bool | None = None

    def __post_init__(self) -> None:
        if self.n_trees is not None and self.n_trees < 1:
            raise ValueError(f"CausalForest n_trees must be at least 1, got {self.n_trees!r}")
        if self.min_leaf is not None and self.min_leaf < 1:
            raise ValueError(f"CausalForest min_leaf must be at least 1, got {self.min_leaf!r}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"CausalForest max_depth must be at least 1, got {self.max_depth!r}")

    @property
    def estimator_id(self) -> str:
        return str(Estimator.CAUSAL_FOREST)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.n_trees is not None:
            out["n_trees"] = self.n_trees
        if self.min_leaf is not None:
            out["min_leaf"] = self.min_leaf
        if self.max_depth is not None:
            out["max_depth"] = self.max_depth
        if self.honesty is not None:
            out["honesty"] = self.honesty
        return _omit_empty(out)


__all__ = [
    "Aipw",
    "CausalForest",
    "ClusterDml",
    "DML",
    "DRLearner",
    "DistanceMatching",
    "FitKind",
    "FrontdoorLinearTwoStage",
    "GlmAdjustment",
    "GlmFamilyName",
    "GlmOptions",
    "Iv2Sls",
    "IvWald",
    "LinearAdjustment",
    "Overlap",
    "PropensityMatching",
    "PropensityPenalty",
    "PropensityStratification",
    "PropensityWeighting",
    "SeKind",
    "UNSET",
]
