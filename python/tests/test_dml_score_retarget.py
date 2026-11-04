"""Cross-fitted score tables of DML, DR-Learner and CausalForest, reweighted by a batch retarget.

Reference and tolerance. The retarget reweights the route's own row-identified cross-fitted
scores ``phi_i`` (the arm-1 minus arm-0 contrast score of the table the fit already built) as
``theta = sum w phi / sum w`` with the plug-in score covariance of the weighted influence values.
Under uniform weights ``theta`` is the score mean and the covariance is the iid plug-in variance
of the same scores, which is exactly how each route reports its own ATE and analytic standard
error; so the two must agree to floating-point rounding (``1e-9`` relative), independently of how
the nuisances were fitted. Two further facts have closed forms that do not depend on any score:
the contrast of two claims is ``value_a - value_b`` with variance ``S_aa + S_bb - 2 S_ab`` of the
reported covariance, and a covariate-dependent tilt of a constant effect returns that effect
within sampling error (here the structural effect ``2`` is recovered within ``4`` standard
errors). A fit whose score is partially linear keeps no per-arm table and must refuse the
retarget with ``score_table_unavailable`` rather than reweight another construction.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.estimation import RetargetClaim, RetargetContrast, prepare_many
from antecedent.estimators import DML, CausalForest, DRLearner
from antecedent.outcomes import require_success

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect("t", "y")


def frame(n: int = 400, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.1 + 0.7 * z)))).astype(float)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def estimated(data, config):
    return require_success(
        prepare_many(
            data,
            graph=GRAPH,
            queries=[QUERY],
            estimator=config,
            refute=False,
            execution=ant.Execution(seed=5),
        ).estimate()
    )


def tilt(batch, data, sign):
    rows = batch.retarget_rows()
    assert rows is not None
    return np.exp(sign * 0.4 * np.asarray(data["z"])[list(rows)])


def test_each_score_table_route_reweights_its_own_cross_fitted_scores():
    data = frame()
    for name, config in (
        ("dml", DML(folds=3)),
        ("dr_learner", DRLearner(folds=3)),
        ("causal_forest", CausalForest(n_trees=40, min_leaf=10)),
    ):
        batch = estimated(data, config)
        member = require_success(batch.member(0))
        own_effect = member.effect
        own_se = member.estimate.se_analytic
        rows = batch.retarget_rows()
        assert rows is not None, name

        # Uniform weights return the route's own point and iid plug-in standard error.
        uniform = require_success(
            batch.retarget(
                [RetargetClaim("uniform", 0, np.ones(len(rows)), depends_on=[])],
            )
        )
        point = uniform.claims[0]
        assert point.status == "ok", name
        assert point.value == pytest.approx(own_effect, rel=1e-9, abs=1e-12), name
        assert point.std_error == pytest.approx(own_se, rel=1e-9), name
        assert point.uncertainty_kind == "plug_in_score_covariance", name
        assert uniform.inference_claim == "point_only", name
        assert uniform.scores_source == "estimated", name

        # Two tilts of one constant effect: the covariance is symmetric, the contrast algebra
        # holds exactly and both claims recover the structural effect within sampling error.
        plus, minus = tilt(batch, data, 1.0), tilt(batch, data, -1.0)
        family = require_success(
            batch.retarget(
                [
                    RetargetClaim("plus", 0, plus, depends_on=["z"]),
                    RetargetClaim("minus", 0, minus, depends_on=["z"]),
                ],
                [RetargetContrast("plus_minus_minus", {"plus": 1.0, "minus": -1.0})],
            )
        )
        cov = family.covariance
        assert cov is not None and family.covariance_names == ("plus", "minus"), name
        assert cov[0][1] == cov[1][0], name
        by_name = {m.name: m for m in family.claims}
        for label in ("plus", "minus"):
            claim = by_name[label]
            assert abs(claim.value - 2.0) < 4.0 * claim.std_error, (name, label)
        contrast = family.contrasts[0]
        assert contrast.value == pytest.approx(
            by_name["plus"].value - by_name["minus"].value, abs=1e-12
        ), name
        variance = cov[0][0] + cov[1][1] - 2.0 * cov[0][1]
        assert contrast.std_error == pytest.approx(math.sqrt(variance), rel=1e-9), name
        # The tilts really reweight: the two claims are different points.
        assert by_name["plus"].value != by_name["minus"].value, name


def test_a_partially_linear_score_keeps_no_table_and_refuses_the_retarget():
    data = frame()
    batch = estimated(data, DML(score="partially_linear", folds=3))
    rows = batch.retarget_rows()
    claims = [RetargetClaim("a", 0, np.ones(len(data["t"])), depends_on=[])]
    if rows is not None:
        claims = [RetargetClaim("a", 0, np.ones(len(rows)), depends_on=[])]
    report = require_success(batch.retarget(claims))
    assert report.claims[0].status != "ok"
    assert report.claims[0].refusal_code == "score_table_unavailable"
    assert report.claims[0].value is None
