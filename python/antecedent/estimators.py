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
``estimator=``. There is currently no single-argument spelling — the
`estimator=` and `estimator_config=` kwargs of :func:`antecedent.analyze` are
independent, so both must be supplied::

    from antecedent import analyze
    from antecedent.estimators import LinearAdjustment

    cfg = LinearAdjustment(bootstrap=500, se="cluster", cluster_ids=ids)
    result = analyze(
        data, graph=g, query=q,
        estimator=cfg.estimator_id,
        estimator_config=cfg._wire(),
    )

See this module's docstring companion in the P6b report for the exact
``_analyze.py`` change that would let ``analyze(..., estimator=cfg)`` work
directly (not implemented here: ``_analyze.py`` is out of scope for this
change).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

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
FitKind = Literal["ols", "ridge", "lasso", "huber"]
GlmFamilyName = Literal[
    "binomial_logit",
    "binomial_probit",
    "gaussian_identity",
    "poisson_log",
    "negative_binomial",
]

_SE_KINDS_NEEDING_LAG = ("newey_west", "panel_cluster_hac")


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


def _wire_glm_options(glm_options: GlmOptions | None) -> dict[str, Any]:
    if glm_options is None:
        return {}
    sub = glm_options._wire()
    return {"glm_options": sub} if sub else {}


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        if self.fit is not None:
            out["fit_kind"] = self.fit
        if self.fit_lambda is not None:
            out["fit_lambda"] = self.fit_lambda
        if self.fit_c is not None:
            out["fit_c"] = self.fit_c
        return out


@dataclass(frozen=True, slots=True)
class PropensityWeighting:
    """``propensity.weighting`` — Hajek-normalized IPW.

    No ``se``/``cluster_ids``/... fields: the Rust struct carries no
    ``AnalyticSeKind`` at all for this estimator (the Hajek SE isn't
    parameterized that way) — only ``bootstrap_replicates`` and ``glm_options``
    are configurable.
    """

    bootstrap: int | None = None
    glm_options: GlmOptions | None = None

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
        return out


@dataclass(frozen=True, slots=True)
class PropensityMatching:
    """``propensity.matching`` — nearest-neighbor propensity matching."""

    bootstrap: int | None = None
    se: SeKind | None = None
    se_lag: int | None = None
    cluster_ids: Sequence[int] | None = None
    multiway_ids: Sequence[Sequence[int]] | None = None
    panel_times: Sequence[int] | None = None
    glm_options: GlmOptions | None = None
    caliper: float | None = None

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
        return str(Estimator.PROPENSITY_MATCHING)

    def _wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        out.update(_wire_glm_options(self.glm_options))
        if self.caliper is not None:
            out["caliper"] = self.caliper
        return out


@dataclass(frozen=True, slots=True)
class PropensityStratification:
    """``propensity.stratification`` — propensity-score strata (default 5 strata)."""

    bootstrap: int | None = None
    glm_options: GlmOptions | None = None
    n_strata: int | None = None

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
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        out.update(_wire_glm_options(self.glm_options))
        if self.caliper is not None:
            out["caliper"] = self.caliper
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        out.update(_wire_glm_options(self.glm_options))
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        out.update(_wire_glm_options(self.glm_options))
        if self.family is not None:
            out["family"] = self.family
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        return out


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
        out: dict[str, Any] = {}
        if self.bootstrap is not None:
            out["bootstrap_replicates"] = self.bootstrap
        if self.se is not None:
            out["se_kind"] = self.se
        if self.se_lag is not None:
            out["se_lag"] = self.se_lag
        if self.cluster_ids is not None:
            out["cluster_ids"] = list(self.cluster_ids)
        if self.multiway_ids is not None:
            out["multiway_ids"] = [list(group) for group in self.multiway_ids]
        if self.panel_times is not None:
            out["panel_times"] = list(self.panel_times)
        return out


@dataclass(frozen=True, slots=True)
class SharpRd:
    """``rd.sharp`` — sharp regression discontinuity.

    Unlike every other config in this module, there is no meaningful
    all-defaults instance: ``rd.sharp`` cannot run without a running variable,
    a cutoff, and a bandwidth, so ``SharpRd()`` raises immediately rather than
    producing an empty ``_wire()``. This retires the three loose
    ``running_variable``/``cutoff``/``bandwidth`` kwargs on ``analyze()`` in
    favor of one typed, validated config.
    """

    running_variable: str | None = None
    cutoff: float | None = None
    bandwidth: float | None = None

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

    @property
    def estimator_id(self) -> str:
        return str(Estimator.RD_SHARP)

    def _wire(self) -> dict[str, Any]:
        return {
            "running_variable": self.running_variable,
            "cutoff": self.cutoff,
            "bandwidth": self.bandwidth,
        }


__all__ = [
    "Aipw",
    "DistanceMatching",
    "FitKind",
    "FrontdoorTwoStage",
    "GlmAdjustment",
    "GlmFamilyName",
    "GlmOptions",
    "Iv2Sls",
    "IvWald",
    "LinearAdjustment",
    "PropensityMatching",
    "PropensityStratification",
    "PropensityWeighting",
    "SeKind",
    "SharpRd",
    "UNSET",
]
