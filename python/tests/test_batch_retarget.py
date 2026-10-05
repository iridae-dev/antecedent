"""2.2 E3: configured batches and batch retarget through the Python surface.

The numerical oracle for the joint covariance, the contrasts and the closed interval is the
Rust integration test ``crates/antecedent/tests/batch_retarget.rs`` (an independent calculation
from the raw score tables); this file pins the Python wiring: typed estimator configurations
accepted by the batch entry points with string compatibility, the retained scores after an
estimate, typed refusals, the failed-member report and the tidy export.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimation import (
    PreparedBatch,
    RetargetClaim,
    RetargetContrast,
    analyze_many,
)
from antecedent.estimators import Aipw, PropensityPenalty

from _refusal import assert_registered_refusal

GRAPH = [
    ("z", "t1"),
    ("z", "t2"),
    ("z", "y1"),
    ("z", "y2"),
    ("t1", "y1"),
    ("t1", "y2"),
    ("t2", "y1"),
    ("t2", "y2"),
]
Q_T1_Y1 = ant.AverageEffect("t1", "y1")
Q_T1_Y2 = ant.AverageEffect("t1", "y2")
Q_T2_Y1 = ant.AverageEffect("t2", "y1")


def frame(n: int = 500, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.1 - 0.6 * z)))).astype(float)
    y1 = 2.0 * t1 + 0.5 * t2 + z + 0.3 * rng.normal(size=n)
    y2 = -t1 + 1.5 * t2 - 0.5 * z + 0.3 * rng.normal(size=n)
    return {"t1": t1, "t2": t2, "y1": y1, "y2": y2, "z": z}


def prepare(data, estimator="aipw", **kwargs) -> PreparedBatch:
    return PreparedBatch.prepare(
        data,
        graph=GRAPH,
        queries=[Q_T1_Y1, Q_T1_Y2, Q_T2_Y1],
        estimator=estimator,
        refute=False,
        seed=11,
        **({"bootstrap": 0} if isinstance(estimator, str) else {}),
        **kwargs,
    )


def weights(batch: PreparedBatch, data, sign: float) -> np.ndarray:
    rows = batch.retarget_rows()
    assert rows is not None
    return np.exp(sign * 0.4 * np.asarray(data["z"])[list(rows)])


def test_batch_entry_points_accept_typed_configurations_and_strings_alike():
    data = frame()
    for cfg in (
        Aipw(bootstrap=0),
        Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[0.5, 5.0, 50.0])),
    ):
        solo = [
            ant.analyze(data, graph=GRAPH, query=q, estimator=cfg, refute=False, seed=11)
            for q in (Q_T1_Y1, Q_T2_Y1)
        ]
        many = analyze_many(
            data, graph=GRAPH, queries=[Q_T1_Y1, Q_T2_Y1], estimator=cfg, refute=False, seed=11
        )
        prepared = PreparedBatch.prepare(
            data,
            graph=GRAPH,
            queries=[Q_T1_Y1, Q_T2_Y1],
            estimator=cfg,
            refute=False,
            seed=11,
        ).estimate(data, seed=11)
        for alone, batch_result, again in zip(solo, many, prepared, strict=True):
            assert batch_result.effect == alone.effect
            assert again.effect == alone.effect
    # String compatibility is unchanged.
    plain = analyze_many(
        data,
        graph=GRAPH,
        queries=[Q_T1_Y1],
        estimator="aipw",
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert plain[0].effect == pytest.approx(2.0, abs=0.35)
    # The configuration dict form takes the same keys as `analyze`.
    via_dict = analyze_many(
        data,
        graph=GRAPH,
        queries=[Q_T1_Y1],
        estimator="aipw",
        estimator_config={"bootstrap_replicates": 0},
        refute=False,
        seed=11,
    )
    assert via_dict[0].effect == pytest.approx(plain[0].effect, rel=1e-9)


def test_an_unknown_configuration_key_is_refused_like_the_single_query_call():
    data = frame()
    with pytest.raises(Exception, match="(?i)unknown estimator_config key"):
        analyze_many(
            data,
            graph=GRAPH,
            queries=[Q_T1_Y1],
            estimator="aipw",
            estimator_config={"no_such_option": 1},
            refute=False,
        )


def test_a_batch_retarget_reports_points_covariance_contrasts_and_a_simultaneous_interval():
    data = frame()
    batch = prepare(data)
    plus, minus = weights(batch, data, 1.0), weights(batch, data, -1.0)
    report = batch.retarget(
        [
            RetargetClaim("a", Q_T1_Y1, plus, depends_on=["z"]),
            RetargetClaim("b", 0, minus, depends_on=["z"]),
            RetargetClaim("c", Q_T2_Y1, plus, depends_on=["z"]),
        ],
        [RetargetContrast("route_a_minus_route_b", {"a": 1.0, "b": -1.0})],
        expected_snapshot=batch.retarget_snapshot(),
    )
    assert report.complete and report.complete_family() is report
    assert report.scores_source == "prepared"
    assert report.covariance_names == ("a", "b", "c")
    assert report.covariance is not None
    cov = report.covariance
    for i, row in enumerate(cov):
        for j, value in enumerate(row):
            assert value == cov[j][i]
            assert abs(value) <= math.sqrt(cov[i][i] * cov[j][j]) * (1 + 1e-12)
    by_name = {m.name: m for m in report.claims}
    assert report.covariance_between("a", "b") == cov[0][1]
    for i, name in enumerate(("a", "b", "c")):
        member = by_name[name]
        assert member.status == "ok" and member.value is not None
        assert member.std_error == pytest.approx(math.sqrt(cov[i][i]))
        assert member.uncertainty_kind == "plug_in_score_covariance"
    assert by_name["a"].estimand.startswith("E[y1(do t1=1.0)]")
    # Contrast algebra: value a - b and variance S_aa + S_bb - 2 S_ab.
    contrast = report.contrasts[0]
    assert contrast.value == pytest.approx(by_name["a"].value - by_name["b"].value)
    variance = cov[0][0] + cov[1][1] - 2 * cov[0][1]
    assert contrast.std_error == pytest.approx(math.sqrt(variance))
    assert report.inference_claim == "point_only"

    # The family-level simultaneous interval is the max-t band of the complete family.
    band = report.simultaneous_interval()
    assert band.level == 0.95 and band.critical_value > 1.959
    for member in band.members:
        assert member.upper - member.value == pytest.approx(band.critical_value * member.std_error)

    rows = report.to_rows()
    assert [r["name"] for r in rows] == ["a", "b", "c", "route_a_minus_route_b"]
    assert {r["family_id"] for r in rows} == {report.family_id}
    assert all(r["simultaneous_interval"] == "max_t" and r["family_complete"] for r in rows)
    assert rows[3]["kind"] == "contrast" and rows[0]["scores_source"] == "prepared"


def test_claim_order_does_not_change_the_family_identity_or_values():
    data = frame()
    batch = prepare(data)
    plus, minus = weights(batch, data, 1.0), weights(batch, data, -1.0)
    a = RetargetClaim("a", 0, plus, depends_on=["z"])
    b = RetargetClaim("b", 2, minus, depends_on=["z"])
    first = batch.retarget([a, b], [RetargetContrast("d", {"a": 1.0, "b": -1.0})])
    second = batch.retarget([b, a], [RetargetContrast("d", {"b": -1.0, "a": 1.0})])
    assert first.family_id == second.family_id
    assert {m.name: m.value for m in first.claims} == {m.name: m.value for m in second.claims}
    assert first.covariance_between("a", "b") == second.covariance_between("a", "b")
    assert first.contrasts[0].value == pytest.approx(second.contrasts[0].value)


def test_a_partial_family_lists_the_failed_members_and_is_never_complete():
    data = frame()
    batch = prepare(data)
    plus = weights(batch, data, 1.0)
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
    assert not report.complete
    assert report.failed_members == ("spiked", "short", "good_minus_spiked")
    failed = {m.name: m for m in (*report.claims, *report.contrasts) if m.status == "failed"}
    assert failed["spiked"].support_status == "refused"
    assert failed["spiked"].refusal_code == "cell_not_licensed"
    assert failed["short"].refusal_code == "invalid_argument"
    assert failed["good_minus_spiked"].value is None
    with pytest.raises(CausalUnsupportedError) as partial:
        report.complete_family()
    assert partial.value.reason_code == "cell_not_licensed"
    assert "spiked" in str(partial.value) and "short" in str(partial.value)
    rows = report.to_rows()
    assert len(rows) == 4
    assert {r["name"] for r in rows if r["status"] == "failed"} == {
        "spiked",
        "short",
        "good_minus_spiked",
    }
    assert all(not r["family_complete"] and r["family_failed"] == 3 for r in rows)
    assert all(r["refusal_code"] for r in rows if r["status"] == "failed")


def test_a_malformed_family_is_refused_with_a_registered_reason():
    data = frame()
    batch = prepare(data)
    plus = weights(batch, data, 1.0)
    good = RetargetClaim("a", 0, plus, depends_on=["z"])
    cases = [
        ([], [], "invalid_argument"),
        ([good, good], [], "invalid_argument"),
        ([RetargetClaim("a", 7, plus, depends_on=["z"])], [], "invalid_argument"),
        ([good], [RetargetContrast("c", {"nope": 1.0})], "invalid_argument"),
    ]
    for claims, contrasts, code in cases:
        with pytest.raises(CausalUnsupportedError) as error:
            batch.retarget(claims, contrasts)
        assert error.value.reason_code == code
        assert_registered_refusal(error.value)
    with pytest.raises(CausalUnsupportedError) as moved:
        batch.retarget([good], expected_snapshot="not-this-snapshot")
    assert moved.value.reason_code == "row_weights_bound_to_snapshot"
    with pytest.raises(CausalValueError):
        batch.retarget([RetargetClaim("a", ant.AverageEffect("t1", "z"), plus)])


def test_a_retarget_after_estimate_reads_that_estimates_rows():
    data = frame()
    batch = prepare(data)
    prepared_rows = batch.retarget_rows()
    assert prepared_rows is not None and len(prepared_rows) == 500
    assert batch.retarget_snapshot() is not None
    fresh = frame(n=430, seed=12)
    batch.estimate(fresh, seed=11)
    rows = batch.retarget_rows()
    assert rows is not None and len(rows) == 430
    assert (
        batch.retarget_snapshot()
        != PreparedBatch.prepare(
            data,
            graph=GRAPH,
            queries=[Q_T1_Y1],
            estimator="aipw",
            refute=False,
            bootstrap=0,
            seed=11,
        ).retarget_snapshot()
    )
    report = batch.retarget(
        [RetargetClaim("a", 0, np.exp(0.4 * fresh["z"]), depends_on=["z"])],
    )
    assert report.scores_source == "estimated" and report.complete
    # Weights aligned with the old rows fail their member rather than reweighting new rows.
    stale = batch.retarget([RetargetClaim("a", 0, np.exp(0.4 * data["z"]), depends_on=["z"])])
    assert stale.claims[0].status == "failed"
    assert stale.claims[0].refusal_code == "invalid_argument"


def test_a_penalized_family_retargets_with_its_plug_in_score_covariance():
    data = frame()
    cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(lambdas=[0.5, 5.0, 50.0]))
    batch = prepare(data, estimator=cfg)
    plus = weights(batch, data, 1.0)
    report = batch.retarget(
        [
            RetargetClaim("a", 0, plus, depends_on=["z"]),
            RetargetClaim("b", 1, plus, depends_on=["z"]),
        ],
        [RetargetContrast("d", {"a": 1.0, "b": -1.0})],
    )
    # A penalized table retargets exactly like an unpenalized one: covariance, standard
    # errors, contrast standard error and a complete family.
    assert report.covariance is not None and report.covariance_names == ("a", "b")
    assert report.point_only_members == ()
    assert all(m.status == "ok" and m.std_error is not None for m in report.claims)
    assert all(math.isfinite(m.value) for m in report.claims)
    assert report.contrasts[0].std_error is not None
    assert report.complete_family() is report
    cov = report.covariance
    variance = cov[0][0] + cov[1][1] - 2 * cov[0][1]
    assert report.contrasts[0].std_error == pytest.approx(math.sqrt(variance))
    assert report.covariance_between("a", "b") == cov[0][1]


def test_a_plan_without_scores_refuses_its_members_after_an_estimate():
    data = frame()
    batch = PreparedBatch.prepare(
        data,
        graph=GRAPH,
        queries=[Q_T1_Y1],
        estimator="linear.adjustment.ate",
        refute=False,
        bootstrap=0,
        seed=11,
    )
    claim = RetargetClaim("a", 0, np.ones(500), depends_on=[])
    before = batch.retarget([claim])
    assert before.claims[0].refusal_code == "score_table_unavailable"
    batch.estimate(data, seed=11)
    after = batch.retarget([claim])
    assert after.claims[0].refusal_code == "score_table_unavailable"
    assert after.claims[0].status == "failed" and not after.complete


def test_the_simultaneous_interval_is_a_max_t_band_with_a_typed_partial_refusal():
    data = frame()
    batch = prepare(data)
    plus = weights(batch, data, 1.0)
    report = batch.retarget(
        [
            RetargetClaim("a", 0, plus, depends_on=["z"]),
            RetargetClaim("b", 2, plus, depends_on=["z"]),
        ],
        simultaneous_level=0.9,
        simultaneous_seed=5,
        simultaneous_draws=20_000,
    )
    assert report.complete
    band = report.simultaneous_interval()
    assert (band.level, band.seed, band.draws) == (0.9, 5, 20_000)
    assert 1.6448 < band.critical_value < 2.4
    by_name = {m.name: m for m in report.claims}
    for member in band.members:
        assert member.value == pytest.approx(by_name[member.name].value)
        assert member.std_error == pytest.approx(by_name[member.name].std_error)
        assert member.lower == pytest.approx(member.value - band.critical_value * member.std_error)
    # The max-t band is wider than each marginal band at the same level.
    assert all(m.upper - m.value > 1.6448 * m.std_error for m in band.members)
    # A partial family has no band: a typed refusal naming why.
    partial = batch.retarget(
        [
            RetargetClaim("a", 0, plus, depends_on=["z"]),
            RetargetClaim("short", 2, plus[:5], depends_on=["z"]),
        ]
    )
    with pytest.raises(CausalUnsupportedError) as refused:
        partial.simultaneous_interval()
    assert_registered_refusal(refused.value)
    assert refused.value.reason_code == "cell_not_licensed"
    assert "partial_family" in str(refused.value)
    with pytest.raises(CausalUnsupportedError, match="max_t_invalid_level"):
        batch.retarget([RetargetClaim("a", 0, plus, depends_on=["z"])], simultaneous_level=1.5)


def test_the_tidy_export_carries_identity_provenance_and_failed_members():
    data = frame()
    batch = prepare(data)
    plus = weights(batch, data, 1.0)
    report = batch.retarget(
        [
            RetargetClaim("ok", 0, plus, depends_on=["z"]),
            RetargetClaim("short", 0, plus[:5], depends_on=["z"]),
        ],
        [RetargetContrast("ok_only", {"ok": 1.0})],
    )
    rows = report.to_rows()
    assert [r["name"] for r in rows] == ["ok", "short", "ok_only"]
    columns = {
        "family_id",
        "family_complete",
        "family_size",
        "family_failed",
        "kind",
        "name",
        "estimand",
        "status",
        "value",
        "std_error",
        "uncertainty_kind",
        "simultaneous_interval",
        "support_status",
        "refusal_code",
        "refusal_detail",
        "refusal_message",
        "diagnostics",
        "scores_source",
        "snapshot_id",
        "estimator_fingerprint",
        "nuisance_provenance",
    }
    assert all(columns <= set(row) for row in rows)
    assert {r["family_id"] for r in rows} == {report.family_id}
    assert {r["snapshot_id"] for r in rows} == {batch.retarget_snapshot()}
    assert {r["scores_source"] for r in rows} == {"prepared"}
    assert all(r["family_size"] == 3 and r["family_failed"] == 1 for r in rows)
    assert all(not r["family_complete"] for r in rows)
    ok, short, contrast = rows
    assert ok["status"] == "ok" and ok["support_status"] == "supported"
    assert ok["uncertainty_kind"] == "plug_in_score_covariance" and ok["refusal_code"] is None
    assert short["status"] == "failed" and short["value"] is None
    assert short["refusal_code"] == "invalid_argument" and short["refusal_message"]
    assert contrast["kind"] == "contrast" and contrast["status"] == "ok"
    assert contrast["value"] == ok["value"]
    # The returned rows are copies: editing one does not edit the report.
    original = ok["value"]
    rows[0]["value"] = 0.0
    assert report.to_rows()[0]["value"] == original != 0.0
