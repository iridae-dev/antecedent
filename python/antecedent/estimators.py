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
from typing import Any, Final, Literal, get_args

from .errors import CausalValueError
from .ids import Estimator

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
RdSeKind = Literal["hc1", "homoskedastic", "hc0", "hc2", "hc3"]
"""Analytic SE kinds :class:`SharpRd` accepts (``hc1`` is the default)."""
FitKind = Literal["ols", "ridge", "lasso", "huber"]
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
class Overlap:
    """Propensity clipping and trimming for a propensity-score estimator.

    Every propensity-score estimator (``PropensityWeighting``,
    ``PropensityMatching``, ``PropensityStratification``, ``DistanceMatching``,
    ``Aipw``) clips fitted propensities into ``[0.01, 0.99]`` and trims no unit
    unless it is given an ``overlap``. ``clip`` bounds the propensities used in
    the weights to ``[clip, 1 - clip]``; ``trim`` drops units whose propensity
    lies outside ``[trim, 1 - trim]``, which also narrows the population the
    effect describes. ``None`` turns that operation off. Each bound lies in
    ``(0, 0.5)``.

    ``Overlap()`` is the default policy, so passing it changes nothing. Any other
    policy is a different interval construction: its calibration binds only to
    coverage records measured under that policy, and otherwise reports
    ``scope_not_assessed``.
    """

    clip: float | None = 0.01
    trim: float | None = None

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
    such field (``FrontdoorTwoStage``) can simply omit them rather than inventing values.
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
    """``distance.matching`` — Mahalanobis/caliper covariate-distance matching."""

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
    """``aipw`` — augmented inverse propensity weighting (doubly robust)."""

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    overlap: Overlap | None = None

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
class FrontdoorTwoStage:
    """``frontdoor.two_stage`` — two-stage front-door estimator.

    No ``multiway_ids``/``panel_times`` fields: the Rust struct carries only
    ``cluster_ids`` (no multiway/panel SE machinery for this estimator) — matches
    ``estimator_config.rs``'s ``ESTIMATOR_KEYS`` row for ``frontdoor.two_stage``,
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
        return str(Estimator.FRONTDOOR_TWO_STAGE)

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
    """``iv.wald`` — single-instrument Wald IV estimator."""

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
    """``iv.2sls`` — two-stage least squares, multi-instrument IV estimator."""

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
class SharpRd:
    """``rd.sharp`` — sharp regression discontinuity.

    Unlike every other config in this module, there is no meaningful
    all-defaults instance: ``rd.sharp`` cannot run without a running variable,
    a cutoff, and a bandwidth, so ``SharpRd()`` raises immediately rather than
    producing an empty ``_wire()``. This retires the three loose
    ``running_variable``/``cutoff``/``bandwidth`` kwargs on ``analyze()`` in
    favor of one typed, validated config.

    ``se`` selects the analytic SE of the jump coefficient: ``None`` keeps the
    ``hc1`` residual sandwich default; ``hc0``/``hc2``/``hc3`` are the other
    heteroskedasticity-robust variants and ``homoskedastic`` opts into the
    classical constant-variance formula. Cluster-, multiway-, and lag-based
    kinds do not apply to the single local-linear fit and are rejected.
    """

    running_variable: str | None = None
    cutoff: float | None = None
    bandwidth: float | None = None
    se: RdSeKind | None = None

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("running_variable", self.running_variable),
                ("cutoff", self.cutoff),
                ("bandwidth", self.bandwidth),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and "
                f"bandwidth; missing: {', '.join(missing)}"
            )
        assert self.bandwidth is not None  # narrowed by the check above
        if self.bandwidth <= 0:
            raise ValueError(f"SharpRd bandwidth must be positive, got {self.bandwidth!r}")
        if self.se is not None and self.se not in get_args(RdSeKind):
            raise ValueError(
                f"SharpRd se must be one of {', '.join(get_args(RdSeKind))}; got {self.se!r}"
            )

    @property
    def estimator_id(self) -> str:
        return str(Estimator.RD_SHARP)

    def _wire(self) -> dict[str, Any]:
        return {
            "running_variable": self.running_variable,
            "cutoff": self.cutoff,
            "bandwidth": self.bandwidth,
            **({"se_kind": self.se} if self.se is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class DML:
    """``dml`` — cross-fitted DML / AIPW."""

    learner: str | None = None
    outcome: str | None = None
    treatment: str | None = None
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
            out["learner"] = self.learner
        if self.outcome is not None:
            out["outcome"] = self.outcome
        if self.treatment is not None:
            out["treatment"] = self.treatment
        if self.score is not None:
            out["score"] = self.score
        if self.folds is not None:
            out["folds"] = self.folds
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class DRLearner:
    """``dr.learner`` — doubly robust CATE learner."""

    learner: str | None = None
    outcome: str | None = None
    treatment: str | None = None
    final_learner: str | None = None
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
            out["learner"] = self.learner
        if self.outcome is not None:
            out["outcome"] = self.outcome
        if self.treatment is not None:
            out["treatment"] = self.treatment
        if self.final_learner is not None:
            out["final_learner"] = self.final_learner
        if self.folds is not None:
            out["folds"] = self.folds
        out.update(_wire_overlap(self.overlap))
        return _omit_empty(out)


@dataclass(frozen=True, slots=True)
class CausalForest:
    """``causal.forest`` — honest causal-forest CATE."""

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
    "DML",
    "DRLearner",
    "DistanceMatching",
    "FitKind",
    "FrontdoorTwoStage",
    "GlmAdjustment",
    "GlmFamilyName",
    "GlmOptions",
    "Iv2Sls",
    "IvWald",
    "LinearAdjustment",
    "Overlap",
    "PropensityMatching",
    "PropensityStratification",
    "PropensityWeighting",
    "RdSeKind",
    "SeKind",
    "SharpRd",
    "UNSET",
]
