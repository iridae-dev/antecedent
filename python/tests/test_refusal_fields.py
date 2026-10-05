"""Structured ``refusal_fields`` at the existing refusal sites (2.2 preflight cell, completion).

Mirrors ``crates/antecedent-estimate/tests/refusal_fields_existing_sites.rs`` and
``crates/antecedent/tests/refusal_fields_existing_sites.rs`` through the Python surface. Each site
refuses with the same exception class, reason code and message as before; the exception now also
carries ``refusal_fields`` (stage, subject, reason, remedy and the numbers its failing step
produced). Entries the failing step did not compute are ``None`` or empty, never zero.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalEstimateError, CausalUnsupportedError
from antecedent.estimation import PreparedBatch, RetargetClaim, RetargetContrast
from antecedent.estimators import Aipw, ClusterDml

from _refusal import assert_registered_refusal

FIELD_KEYS = {
    "stage",
    "subject",
    "reason",
    "arm_ess",
    "propensity_min",
    "propensity_max",
    "propensity_quantiles",
    "cluster_count",
    "cluster_minimum",
    "numerical_rank",
    "design_columns",
    "implicated_columns",
    "remedy",
    "glm_iterations",
    "boundary_margin",
    "boundary_count",
}

GLM_REASONS = {"non_converged", "separated", "boundary_saturated"}


def _confounded(n: int = 800, seed: int = 14):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    t = ((z < 0.0) & (rng.random(n) < 0.7)).astype(float)
    y = t + 0.2 * rng.standard_normal(n)
    return {"t": t, "y": y, "z": z}


def _wide_graph(p: int):
    return [(f"x{i}", "t") for i in range(p)] + [(f"x{i}", "y") for i in range(p)] + [("t", "y")]


def _prepare_aipw(data, p: int):
    return ant.prepare(
        data,
        graph=_wide_graph(p),
        query=ant.AverageEffect(treatment="t", outcome="y"),
        estimator="aipw",
        refute=False,
        bootstrap=0,
    )


def _absent(fields: dict, *names: str) -> None:
    for name in names:
        value = fields[name]
        assert value is None or value == [], (name, value)


def test_a_retarget_onto_unsupported_rows_reports_arm_sizes_and_propensity_range():
    data = _confounded()
    prepared = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect(treatment="t", outcome="y"),
        estimator="aipw",
        refute=False,
        bootstrap=0,
    )
    weights = (np.asarray(data["z"]) > 1.2).astype(float)
    with pytest.raises(CausalUnsupportedError) as raised:
        prepared.retarget(weights, ["z"])
    error = raised.value
    # The class, code and message are those of the support refusal.
    assert str(error) == (
        "refused: retarget refused: weighted overlap failed under the declared target weights"
    )
    assert error.reason_code == "cell_not_licensed"
    assert_registered_refusal(error)
    fields = error.refusal_fields
    assert set(fields) == FIELD_KEYS
    assert fields["stage"] == "retarget"
    assert fields["subject"] == "effect(t -> y)"
    assert fields["remedy"]
    assert fields["reason"] in {
        "arm_effective_sample_size_below_minimum",
        "propensity_range_touches_zero_or_one",
        "extreme_propensity_share_above_limit",
        "propensity_range_unavailable",
    }
    labels = [label for label, _ in fields["arm_ess"]]
    assert labels == ["control", "active"]
    # The treated arm only exists where z < 0, and the target is z > 1.2: no weighted mass.
    ess = dict(fields["arm_ess"])
    assert ess["active"] < 10.0 <= ess["control"]
    assert 0.0 < fields["propensity_min"] <= fields["propensity_max"] < 1.0
    # Quantiles of the raw propensities of rows the target weights: the seven reported levels,
    # ordered and inside the range (the exact nearest-rank values are checked in the Rust test).
    quantiles = fields["propensity_quantiles"]
    assert [p for p, _ in quantiles] == [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    values = [v for _, v in quantiles]
    assert values == sorted(values)
    assert fields["propensity_min"] <= values[0] and values[-1] <= fields["propensity_max"]
    # No fitted GLM, rank or cluster step produced anything here.
    _absent(
        fields,
        "cluster_count",
        "cluster_minimum",
        "numerical_rank",
        "design_columns",
        "glm_iterations",
        "boundary_margin",
        "boundary_count",
    )


def test_a_retarget_dependence_refusal_has_stage_reason_and_remedy_but_no_numbers():
    data = _confounded()
    prepared = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect(treatment="t", outcome="y"),
        estimator="aipw",
        refute=False,
        bootstrap=0,
    )
    weights = np.exp(0.3 * np.asarray(data["z"]))
    with pytest.raises(CausalError) as raised:
        prepared.retarget(weights, ["t"])
    error = raised.value
    assert "must not include the treatment" in str(error)
    fields = error.refusal_fields
    assert fields["stage"] == "retarget"
    assert fields["subject"] == "effect(t -> y)"
    assert fields["reason"] == "weights_depend_on_treatment_or_descendant"
    assert fields["remedy"]
    # Nothing was fitted at this stage: every numeric entry is absent rather than zero.
    _absent(
        fields,
        "arm_ess",
        "propensity_min",
        "propensity_max",
        "propensity_quantiles",
        "cluster_count",
        "numerical_rank",
        "glm_iterations",
        "boundary_margin",
    )


def test_the_real_aipw_refusal_at_rank_174_of_175_carries_rank_and_columns():
    n, p = 2000, 174
    rng = np.random.default_rng(1)
    covariates = [rng.standard_normal(n) for _ in range(p - 1)]
    covariates.append(covariates[5].copy())
    t = (rng.random(n) < 0.5).astype(float)
    y = t + covariates[0] + rng.standard_normal(n)
    data = {"t": t, "y": y} | {f"x{i}": c for i, c in enumerate(covariates)}
    with pytest.raises(CausalEstimateError) as raised:
        _prepare_aipw(data, p)
    error = raised.value
    assert "rank deficient: rank=174 ncols=175" in str(error)
    fields = error.refusal_fields
    assert set(fields) == FIELD_KEYS
    assert fields["stage"] == "design_rank"
    assert fields["reason"] == "rank_deficient"
    assert (fields["numerical_rank"], fields["design_columns"]) == (174, 175)
    assert fields["remedy"]
    # The facade names the cell, and the fit-free preflight of the same design names the
    # dependent column (of the duplicated pair x5 / x173).
    assert fields["subject"] == "effect(t -> y)"
    assert len(fields["implicated_columns"]) == 1
    assert fields["implicated_columns"][0] in {"x5", "x173"}
    _absent(fields, "arm_ess", "propensity_min", "propensity_max", "cluster_count")


def test_the_real_aipw_refusal_under_joint_separation_carries_the_fit_facts():
    n, p = 240, 174
    rng = np.random.default_rng(2)
    data = {"t": (rng.random(n) < 0.5).astype(float), "y": rng.standard_normal(n)} | {
        f"x{i}": rng.standard_normal(n) for i in range(p)
    }
    with pytest.raises(CausalEstimateError) as raised:
        _prepare_aipw(data, p)
    error = raised.value
    fields = error.refusal_fields
    assert set(fields) == FIELD_KEYS
    assert fields["remedy"]
    if fields["stage"] == "glm_fit":
        assert fields["reason"] in GLM_REASONS
        assert fields["glm_iterations"] >= 1
        assert fields["boundary_margin"] is not None and fields["boundary_count"] is not None
        expected = {
            "non_converged": "GLM IRLS did not converge; refuse propensity/outcome scores",
            "separated": (
                "GLM indicates (quasi-)complete separation; refuse propensity/outcome scores"
            ),
            "boundary_saturated": (
                "GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not "
                "necessarily separation); refuse propensity/outcome scores"
            ),
        }[fields["reason"]]
        assert str(error).endswith(f"backend error: {expected}") or expected in str(error)
    else:
        # The IRLS design lost rank before any score existed: counts only, no fitted figure.
        assert fields["stage"] == "design_rank"
        assert fields["numerical_rank"] < fields["design_columns"] == p + 1
        assert str(error).count("rank deficient") == 1
        _absent(fields, "arm_ess", "propensity_min", "propensity_max", "glm_iterations")
    assert fields["subject"] == "effect(t -> y)"
    _absent(fields, "cluster_count")


def test_a_cluster_count_shortfall_reports_found_and_minimum_counts():
    rng = np.random.default_rng(7)
    groups, size = 12, 20
    cluster = np.repeat(np.arange(groups), size)
    n = groups * size
    z0, z1 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.6 * z0 - 0.4 * z1)))).astype(float)
    y = 2.0 * t + z0 + 0.5 * z1 + 0.5 * rng.normal(size=n)
    data = {"z0": z0, "z1": z1, "t": t, "y": y}
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=20))
    with pytest.raises(CausalError) as raised:
        ant.analyze(
            data,
            graph=[("z0", "t"), ("z1", "t"), ("z0", "y"), ("z1", "y"), ("t", "y")],
            query=ant.AverageEffect("t", "y"),
            estimator=cfg,
            refute=False,
            seed=1,
        )
    error = raised.value
    assert error.reason_code == "too_few_clusters"
    assert_registered_refusal(error)
    assert "12 clusters are below the declared minimum 20" in str(error)
    fields = error.refusal_fields
    assert fields["stage"] == "cluster_dml"
    assert fields["reason"] == "too_few_clusters"
    assert (fields["cluster_count"], fields["cluster_minimum"]) == (12, 20)
    assert fields["remedy"]
    _absent(fields, "arm_ess", "propensity_min", "numerical_rank", "glm_iterations")


def test_batch_retarget_members_carry_the_failing_claim_as_subject():
    rng = np.random.default_rng(11)
    n = 500
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    y1 = 2.0 * t1 + z + 0.3 * rng.normal(size=n)
    y2 = -t1 - 0.5 * z + 0.3 * rng.normal(size=n)
    data = {"t1": t1, "y1": y1, "y2": y2, "z": z}
    graph = [("z", "t1"), ("z", "y1"), ("z", "y2"), ("t1", "y1"), ("t1", "y2")]
    batch = PreparedBatch.prepare(
        data,
        graph=graph,
        queries=[ant.AverageEffect("t1", "y1"), ant.AverageEffect("t1", "y2")],
        estimator="aipw",
        refute=False,
        seed=11,
        bootstrap=0,
    )
    rows = batch.retarget_rows()
    assert rows is not None
    plus = np.exp(0.4 * np.asarray(data["z"])[list(rows)])
    spike = np.zeros_like(plus)
    spike[:3] = 1.0
    report = batch.retarget(
        [
            RetargetClaim("good", 0, plus, depends_on=["z"]),
            RetargetClaim("spiked", 1, spike, depends_on=["z"]),
            RetargetClaim("short", 1, plus[:10], depends_on=["z"]),
        ],
        [RetargetContrast("good_minus_spiked", {"good": 1.0, "spiked": -1.0})],
    )
    members = {m.name: m for m in (*report.claims, *report.contrasts)}
    assert members["good"].refusal_fields is None

    spiked = members["spiked"].refusal_fields
    assert spiked["stage"] == "batch_retarget" and spiked["subject"] == "spiked"
    assert [label for label, _ in spiked["arm_ess"]] == ["control", "active"]
    assert spiked["propensity_min"] is not None and spiked["propensity_max"] is not None
    assert spiked["remedy"]
    assert members["spiked"].refusal_code == "cell_not_licensed"

    short = members["short"].refusal_fields
    assert short["subject"] == "short" and short["stage"] == "batch_retarget"
    assert short["reason"] == "batch_retarget.incompatible_target"
    assert short["remedy"]
    # A malformed declaration has no fitted score behind it: nothing numeric is reported.
    _absent(short, "arm_ess", "propensity_min", "propensity_max", "cluster_count")

    contrast = members["good_minus_spiked"].refusal_fields
    assert contrast["subject"] == "good_minus_spiked"
    assert contrast["reason"] == "batch_retarget.contrast_member_failed"


def test_a_malformed_family_names_the_offending_claim_as_subject():
    rng = np.random.default_rng(12)
    n = 300
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    data = {"t": t, "y": t + z + rng.normal(size=n), "z": z}
    batch = PreparedBatch.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        queries=[ant.AverageEffect("t", "y")],
        estimator="aipw",
        refute=False,
        seed=12,
        bootstrap=0,
    )
    rows = batch.retarget_rows()
    plus = np.exp(0.4 * np.asarray(data["z"])[list(rows)])
    good = RetargetClaim("a", 0, plus, depends_on=["z"])
    with pytest.raises(CausalUnsupportedError) as raised:
        batch.retarget([good, good])
    fields = raised.value.refusal_fields
    assert fields["stage"] == "batch_retarget"
    assert fields["subject"] == "a"
    assert fields["reason"] == "batch_retarget.duplicate_name"
    assert raised.value.reason_code == "invalid_argument"
