"""Tests for `antecedent.estimators` — the typed dataclass front-end over `estimator_config=`.

Covers: every dataclass's all-defaults `_wire()` is empty and its estimator id
itself raises, since there is no meaningful all-defaults RD config); `estimator_id` matches
the Rust wire id from `antecedent.ids.Estimator`; `_wire()` round-trips through the real
`antecedent.analyze(...)` / `antecedent._native.analyze_ate` and matches a hand-written dict
exactly; and every `__post_init__` validation rule raises with a message naming the offending
field.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.estimators import (
    DML,
    UNSET,
    Aipw,
    CausalForest,
    DistanceMatching,
    DRLearner,
    FrontdoorLinearTwoStage,
    GlmAdjustment,
    GlmOptions,
    Iv2Sls,
    IvWald,
    LinearAdjustment,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
)
from antecedent.ids import Estimator

# --- fixtures -----------------------------------------------------------------------------


def _confounded_data(seed: int = 7, n: int = 400):
    """Simple confounded SCM: z -> t -> y, z -> y. Complete cases, no missing data."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) * 0.5 > 0).astype(np.float64)
    y = 2.0 * t + 1.0 * z + rng.normal(size=n) * 0.5
    names = ["z", "t", "y"]
    columns = [z, t, y]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    return names, columns, edges


def _public_scm(seed: int = 9, n: int = 400):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (0.7 * z + rng.normal(size=n) * 0.5 > 0).astype(float)
    y = 1.5 * t + z + rng.normal(size=n) * 0.4
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


# --- all-defaults _wire() is a strict no-op ------------------------------------------------

_DEFAULT_INSTANCE_CASES = [
    (LinearAdjustment(), Estimator.LINEAR_ADJUSTMENT_ATE),
    (PropensityWeighting(), Estimator.PROPENSITY_WEIGHTING),
    (PropensityMatching(), Estimator.PROPENSITY_MATCHING),
    (PropensityStratification(), Estimator.PROPENSITY_STRATIFICATION),
    (DistanceMatching(), Estimator.DISTANCE_MATCHING),
    (Aipw(), Estimator.AIPW),
    (DML(), Estimator.DML),
    (DRLearner(), Estimator.DR_LEARNER),
    (CausalForest(), Estimator.CAUSAL_FOREST),
    (GlmAdjustment(), Estimator.GLM_ADJUSTMENT),
    (FrontdoorLinearTwoStage(), Estimator.FRONTDOOR_LINEAR_TWO_STAGE),
    (IvWald(), Estimator.IV_WALD),
    (Iv2Sls(), Estimator.IV_2SLS),
]


@pytest.mark.parametrize(
    ("instance", "expected_id"),
    _DEFAULT_INSTANCE_CASES,
    ids=[type(inst).__name__ for inst, _ in _DEFAULT_INSTANCE_CASES],
)
def test_default_wire_is_empty_and_estimator_id_matches_rust(instance, expected_id):
    assert instance._wire() == {}
    assert instance.estimator_id == str(expected_id)


# --- real round-trips through analyze() / analyze_ate --------------------------------------


def test_linear_adjustment_wire_round_trips_through_public_analyze():
    data, graph = _public_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    cluster_ids = [i % 20 for i in range(len(data["z"]))]

    cfg = LinearAdjustment(bootstrap=64, se="cluster", cluster_ids=cluster_ids)
    via_dataclass = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        seed=1,
        estimator=cfg.estimator_id,
        estimator_config=cfg._wire(),
    )
    # Hand-written equivalent: `bootstrap_replicates` omitted from the dict on purpose —
    # Rust falls back to the ambient `bootstrap=` kwarg whenever it's absent, so passing
    # bootstrap=64 at the top level is exactly equivalent to cfg.bootstrap=64.
    hand_dict = {"se_kind": "cluster", "cluster_ids": cluster_ids}
    via_hand_dict = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        seed=1,
        bootstrap=64,
        estimator="linear.adjustment.ate",
        estimator_config=hand_dict,
    )

    assert via_dataclass.estimate.ate == pytest.approx(via_hand_dict.estimate.ate)
    assert via_dataclass.estimate.se_analytic == pytest.approx(via_hand_dict.estimate.se_analytic)


# --- se_kind / cluster_ids / multiway_ids / se_lag validation ------------------------------


def test_se_cluster_without_cluster_ids_raises():
    with pytest.raises(ValueError, match="cluster_ids") as exc:
        LinearAdjustment(se="cluster")
    assert "cluster_ids" in str(exc.value)


def test_cluster_ids_without_cluster_se_raises():
    # The "reject the inverse" rule: Rust's `build_configured_spec` sets cluster_ids on
    # the estimator unconditionally, regardless of se_kind, so this combination would be
    # silently accepted and then ignored by the SE formula if we didn't catch it here.
    with pytest.raises(ValueError, match="cluster_ids") as exc:
        LinearAdjustment(cluster_ids=[0, 1, 2])
    assert "se='cluster'" in str(exc.value)


def test_se_multiway_without_multiway_ids_raises():
    with pytest.raises(ValueError, match="multiway_ids") as exc:
        LinearAdjustment(se="multiway")
    assert "multiway_ids" in str(exc.value)


def test_multiway_ids_without_multiway_se_raises():
    with pytest.raises(ValueError, match="multiway_ids") as exc:
        LinearAdjustment(multiway_ids=[[0, 1], [2, 3]])
    assert "se='multiway'" in str(exc.value)


def test_se_newey_west_without_se_lag_raises():
    with pytest.raises(ValueError, match="se_lag") as exc:
        LinearAdjustment(se="newey_west")
    assert "se_lag" in str(exc.value)


def test_se_panel_cluster_hac_without_se_lag_raises():
    with pytest.raises(ValueError, match="se_lag") as exc:
        LinearAdjustment(se="panel_cluster_hac")
    assert "se_lag" in str(exc.value)


def test_se_lag_without_lag_requiring_se_raises():
    with pytest.raises(ValueError, match="se_lag") as exc:
        LinearAdjustment(se="cluster", cluster_ids=[0, 1], se_lag=2)
    assert "se_lag" in str(exc.value)


def test_valid_cluster_config_does_not_raise_and_wires_correctly():
    cfg = LinearAdjustment(se="cluster", cluster_ids=[0, 1, 2])
    assert cfg._wire() == {"se_kind": "cluster", "cluster_ids": [0, 1, 2]}


def test_valid_newey_west_config_does_not_raise_and_wires_correctly():
    cfg = LinearAdjustment(se="newey_west", se_lag=3)
    assert cfg._wire() == {"se_kind": "newey_west", "se_lag": 3}


@pytest.mark.parametrize(
    "cls",
    [PropensityMatching, DistanceMatching, Aipw, GlmAdjustment, IvWald, Iv2Sls],
    ids=lambda c: c.__name__,
)
def test_se_validation_shared_by_every_se_bearing_estimator(cls):
    with pytest.raises(ValueError, match="cluster_ids"):
        cls(se="cluster")
    with pytest.raises(ValueError, match="se='cluster'"):
        cls(cluster_ids=[0, 1])


def test_frontdoor_two_stage_has_no_multiway_or_panel_fields():
    # frontdoor.linear_two_stage's Rust struct only carries cluster_ids (no multiway/panel_times
    # SE machinery); passing those as kwargs must be a plain TypeError (unknown field),
    # not a silently-accepted-then-ignored value.
    with pytest.raises(TypeError, match="unexpected keyword argument .multiway_ids."):
        FrontdoorLinearTwoStage(multiway_ids=[[0, 1]])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument .panel_times."):
        FrontdoorLinearTwoStage(panel_times=[0, 1])  # type: ignore[call-arg]
    cfg = FrontdoorLinearTwoStage(se="cluster", cluster_ids=[0, 1, 2])
    assert cfg._wire() == {"se_kind": "cluster", "cluster_ids": [0, 1, 2]}


# --- LinearAdjustment fit_kind validation, including the lasso trap ------------------------


def test_ridge_without_fit_lambda_raises():
    with pytest.raises(ValueError, match="fit_lambda") as exc:
        LinearAdjustment(fit="ridge")
    assert "fit_lambda" in str(exc.value)


def test_lasso_without_fit_lambda_raises():
    with pytest.raises(ValueError, match="fit_lambda") as exc:
        LinearAdjustment(fit="lasso")
    assert "fit_lambda" in str(exc.value)


def test_huber_without_fit_c_raises():
    with pytest.raises(ValueError, match="fit_c") as exc:
        LinearAdjustment(fit="huber")
    assert "fit_c" in str(exc.value)


def test_fit_lambda_without_ridge_or_lasso_raises():
    with pytest.raises(ValueError, match="fit_lambda") as exc:
        LinearAdjustment(fit="ols", fit_lambda=0.5)
    assert "fit_lambda" in str(exc.value)


def test_fit_c_without_huber_raises():
    with pytest.raises(ValueError, match="fit_c") as exc:
        LinearAdjustment(fit="ridge", fit_lambda=0.5, fit_c=1.345)
    assert "fit_c" in str(exc.value)


def test_ridge_and_huber_do_not_carry_the_lasso_se_restriction():
    # Only Lasso permanently omits the analytic SE; Ridge and Huber do not.
    ridge = LinearAdjustment(fit="ridge", fit_lambda=0.5, se="cluster", cluster_ids=[0, 1])
    assert ridge._wire()["fit_kind"] == "ridge"
    huber = LinearAdjustment(fit="huber", fit_c=1.345, se="hc0")
    assert huber._wire()["fit_kind"] == "huber"


def test_lasso_with_se_raises_and_explains_why():
    with pytest.raises(ValueError, match="permanently omitted") as exc:
        LinearAdjustment(fit="lasso", fit_lambda=0.1, se="hc0")
    message = str(exc.value)
    # Names the offending field...
    assert "se" in message
    # ...and explains *why* the SE is unavailable in principle (permanently NaN after
    # selection / debiasing), not just that the combination is rejected.
    assert "permanently omitted" in message
    assert "debiased" in message
    assert "bootstrap" in message


def test_lasso_without_se_does_not_raise():
    # The recommended pattern: bootstrap-only, no se= requested.
    cfg = LinearAdjustment(fit="lasso", fit_lambda=0.1, bootstrap=500)
    assert cfg._wire() == {"bootstrap_replicates": 500, "fit_kind": "lasso", "fit_lambda": 0.1}


# --- bootstrap / n_strata / caliper positivity ----------------------------------------------


def test_bootstrap_negative_raises():
    with pytest.raises(ValueError, match="bootstrap") as exc:
        LinearAdjustment(bootstrap=-1)
    assert "bootstrap" in str(exc.value)


def test_bootstrap_zero_is_allowed():
    # Deliberately non-negative, not strictly positive: bootstrap=0 is the standard way
    # to disable bootstrap replicate computation (matches Rust's own `get_u32`, and the
    # ambient `analyze(..., bootstrap=0)` kwarg used throughout the existing test suite).
    cfg = LinearAdjustment(bootstrap=0)
    assert cfg._wire() == {"bootstrap_replicates": 0}


def test_propensity_stratification_n_strata_zero_raises():
    with pytest.raises(ValueError, match="n_strata") as exc:
        PropensityStratification(n_strata=0)
    assert "n_strata" in str(exc.value)


def test_propensity_stratification_n_strata_negative_raises():
    with pytest.raises(ValueError, match="n_strata"):
        PropensityStratification(n_strata=-2)


def test_propensity_matching_caliper_zero_raises():
    with pytest.raises(ValueError, match="caliper") as exc:
        PropensityMatching(caliper=0.0)
    assert "caliper" in str(exc.value)


def test_distance_matching_caliper_negative_raises():
    with pytest.raises(ValueError, match="caliper") as exc:
        DistanceMatching(caliper=-0.1)
    assert "caliper" in str(exc.value)


# --- GlmOptions: UNSET vs explicit None for ridge_on_separation -----------------------------


def test_glm_options_default_wire_is_empty():
    assert GlmOptions()._wire() == {}


def test_glm_options_ridge_on_separation_unset_by_default():
    assert GlmOptions().ridge_on_separation is UNSET
    assert "ridge_on_separation" not in GlmOptions()._wire()


def test_glm_options_ridge_on_separation_explicit_none_is_wired():
    # Explicit None must be distinguished from "not set": Rust treats an *absent* key as
    # "keep GlmOptions::default()'s Some(1e-4)" but a *present* key with value None as
    # "explicitly clear it to None", so the wire dict must carry the key at all in this
    # case, with value None.
    cfg = GlmOptions(ridge_on_separation=None)
    assert cfg._wire() == {"ridge_on_separation": None}


def test_glm_options_ridge_on_separation_explicit_float_is_wired():
    cfg = GlmOptions(ridge_on_separation=1e-3)
    assert cfg._wire() == {"ridge_on_separation": 1e-3}


def test_glm_options_full_wire_and_nesting_in_glm_adjustment():
    opts = GlmOptions(max_iter=25, tol=1e-6, ridge_on_separation=None)
    cfg = GlmAdjustment(family="poisson_log", glm_options=opts)
    assert cfg._wire() == {
        "family": "poisson_log",
        "glm_options": {"max_iter": 25, "tol": 1e-6, "ridge_on_separation": None},
    }


def test_propensity_weighting_only_exposes_bootstrap_and_glm_options():
    cfg = PropensityWeighting(bootstrap=10, glm_options=GlmOptions(max_iter=5))
    assert cfg._wire() == {"bootstrap_replicates": 10, "glm_options": {"max_iter": 5}}
    with pytest.raises(TypeError, match="unexpected keyword argument .se."):
        PropensityWeighting(se="cluster")  # type: ignore[call-arg]


# --- caliper scale (Rust ↔ Python parity) ----------------------------------------------------


def test_propensity_matching_caliper_scale_defaults_to_none_meaning_logit():
    """Unset leaves the wire clean; Rust's own default is `CaliperScale::Logit`.

    The 0.2 rule of thumb (Rosenbaum & Rubin 1985; Austin 2011) is a logit-scale
    quantity, so logit is the default a caller passing 0.2 expects.
    """
    est = PropensityMatching(caliper=0.2)
    assert est.caliper_scale is None
    assert "caliper_scale" not in est._wire()


def test_propensity_matching_caliper_scale_reaches_the_wire():
    for scale in ("logit", "raw"):
        wire = PropensityMatching(caliper=0.2, caliper_scale=scale)._wire()
        assert wire["caliper_scale"] == scale


def test_propensity_matching_rejects_unknown_caliper_scale():
    with pytest.raises(ValueError, match="caliper_scale"):
        PropensityMatching(caliper=0.2, caliper_scale="probit")  # type: ignore[arg-type]


# --- matching refuses clustered, multiway and HAC standard errors ----------------------------


@pytest.mark.parametrize("cls", [PropensityMatching, DistanceMatching])
@pytest.mark.parametrize("se", ["cluster", "multiway", "newey_west", "panel_cluster_hac"])
def test_matching_refuses_clustered_multiway_and_hac_se(cls, se):
    """No clustered influence function exists for fixed-M nearest-neighbour matching, so
    every dependence-robust ``se`` is refused by Rust (``estimator_inference_mismatch``);
    only the homoskedastic Abadie–Imbens SE is published. See ``docs/capabilities.md``."""
    data, graph = _public_scm()
    n = len(data["z"])
    clusters = [i % 20 for i in range(n)]
    kwargs: dict = {"se": se}
    if se == "cluster":
        kwargs["cluster_ids"] = clusters
    elif se == "multiway":
        kwargs["multiway_ids"] = [clusters, [i % 15 for i in range(n)]]
    else:
        kwargs["se_lag"] = 2
        kwargs["panel_times"] = [i // 20 for i in range(n)]
    cfg = cls(**kwargs)
    with pytest.raises(CausalUnsupportedError) as refused:
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            seed=1,
            refute=False,
            estimator=cfg.estimator_id,
            estimator_config=cfg._wire(),
        )
    assert refused.value.reason_code == "estimator_inference_mismatch"
    assert "matching has no valid cluster, multiway or HAC standard error" in str(refused.value)
