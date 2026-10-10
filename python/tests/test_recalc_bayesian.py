"""Ordinary native posterior work, independent projections and process boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.recalc import RecalcReceipt, RecalcRefusal, ResumeContext, Utility
from antecedent.recalc_bayesian import BayesianRequest, BayesianSession, PosteriorSummary
from antecedent.recalc_capabilities import (
    Family,
    Operation,
    RetainedKind,
    require_adapter,
    retained_kind,
)


def request(model="gaussian"):
    t, z, y = [], [], []
    for _ in range(20):
        for action in (0.0, 1.0):
            for covariate in (-1.0, 0.0, 1.0):
                for noise in (-0.2, 0.2):
                    t.append(action)
                    z.append(covariate)
                    y.append(
                        1
                        + 2 * action
                        + 0.5 * covariate
                        + 0.4 * action * covariate
                        + 0.3 * covariate**2
                        + noise
                    )
    return BayesianRequest(
        {"t": np.array(t), "z": np.array(z), "y": np.array(y)},
        (("t", "y"), ("z", "t"), ("z", "y")),
        "t",
        "y",
        Utility(2, 0.5),
        model=model,
        inference=ac.Bayesian(backend="conjugate", n_draws=2048),
    )


def assert_projection(result, rows, summary):
    values = np.array(rows)
    assert result.law.mean == pytest.approx(values.mean(), abs=1e-12)
    assert result.law.standard_deviation == pytest.approx(values.std(ddof=1), abs=1e-12)
    assert result.law.lower_quantile == pytest.approx(
        np.quantile(values, summary.lower_probability), abs=1e-12
    )
    assert result.law.upper_quantile == pytest.approx(
        np.quantile(values, summary.upper_probability), abs=1e-12
    )
    assert result.law.probability_below == pytest.approx(
        np.mean(values < summary.threshold), abs=1e-12
    )


@pytest.mark.parametrize("model", ["gaussian", "quadratic_basis"])
def test_bayesian_native_draw_projection_truth_and_full_rerun(model):
    req = request(model)
    session = BayesianSession()
    result = session.execute(req, seed=41)
    assert result.receipt.totals.model_fits > 0
    assert result.receipt.totals.posterior_draws == 2048
    assert result.receipt.totals.fold_fits == 0
    assert result.law.mean == pytest.approx(2, abs=0.03)
    assert_projection(result, session.effect_draws, req.summary)
    rerun = BayesianSession().execute(req, seed=41)
    assert result.law == rerun.law
    if model == "gaussian":
        full = ac.analyze(
            req.data,
            query=ac.AverageEffect("t", "y"),
            graph=req.edges,
            inference=req.inference,
            refute="none",
            bootstrap=0,
            seed=41,
        )
        assert result.law.mean == pytest.approx(full.posterior.effect_mean, abs=1e-12)
        assert result.law.standard_deviation == pytest.approx(full.posterior.effect_sd, abs=1e-12)


def test_bayesian_summary_and_utility_reuse_actual_native_rows():
    req = request()
    session = BayesianSession()
    first = session.execute(req, seed=41)
    rows = session.effect_draws
    assert session.execute(req, seed=41).receipt.totals.total == 0
    changed = replace(req, summary=PosteriorSummary(0.1, 0.9, 2.0))
    summary = session.execute(changed, seed=41)
    assert summary.receipt.totals.model_fits == summary.receipt.totals.posterior_draws == 0
    assert summary.receipt.totals.law_summaries == 1
    assert session.effect_draws == rows
    assert_projection(summary, rows, changed.summary)
    utility = session.execute(replace(changed, utility=Utility(3, 1)), seed=41)
    assert utility.receipt.totals.total == utility.receipt.totals.decisions == 1
    assert utility.law == summary.law
    assert utility.decision.net_benefit == pytest.approx(3 * first.law.mean - 1)


@pytest.mark.parametrize("mutation", ["data", "model", "inference", "graph", "seed"])
def test_bayesian_changed_inputs_refit_against_fresh_execution(mutation):
    req = request()
    session = BayesianSession()
    session.execute(req, seed=41)
    seed = 42 if mutation == "seed" else 41
    changed = {
        "data": replace(req, data={**req.data, "y": 2 * req.data["y"]}),
        "model": replace(req, model="quadratic_basis"),
        "inference": replace(
            req, inference=ac.Bayesian(backend="conjugate", n_draws=1024, prior_scale=5)
        ),
        "graph": replace(req, edges=(("t", "y"), ("z", "y"))),
        "seed": req,
    }[mutation]
    result = session.execute(changed, seed=seed)
    assert result.receipt.totals.model_fits > 0
    assert result.receipt.totals.posterior_draws > 0
    assert result.law == BayesianSession().execute(changed, seed=seed).law


def test_bayesian_exact_refusals_preserve_native_rows_and_bounds():
    req = request()
    session = BayesianSession()
    session.execute(req, seed=41)
    rows, ids = session.effect_draws, session.identities
    cases = [
        (
            replace(req, inference=ac.Bayesian(backend="hmc", n_draws=256)),
            "route_not_supported",
            "recalc.bayesian_inference_unsupported",
            "learner_folds_rng",
        ),
        (
            replace(req, summary=PosteriorSummary(0.9, 0.1)),
            "invalid_argument",
            "recalc.bayesian_invalid_summary",
            "law",
        ),
    ]
    for changed, code, detail, stage in cases:
        with pytest.raises(RecalcRefusal) as error:
            session.execute(changed, seed=41)
        assert (error.value.reason_code, error.value.detail, error.value.stage) == (
            code,
            detail,
            stage,
        )
        assert session.effect_draws == rows and session.identities == ids
    with pytest.raises(CausalValueError) as error:
        session.execute(
            replace(req, inference=ac.Bayesian(backend="conjugate", n_draws=100001)), seed=41
        )
    assert error.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(error.value)
    assert session.execute(req, seed=41).receipt.totals.total == 0


def test_bayesian_native_capabilities_require_retained_state():
    session = BayesianSession()
    assert retained_kind(session) == RetainedKind.READABLE
    session.execute(request(), seed=41)
    assert retained_kind(session) == RetainedKind.LIVE_POSTERIOR
    assert (
        require_adapter(Family.BAYESIAN, Operation.FUNCTIONAL, session).state_type
        is BayesianSession
    )
    assert session.effect_draws is not None


def test_bayesian_artifact_and_receipt_fresh_process_boundary(tmp_path):
    req = request()
    session = BayesianSession()
    result = session.execute(req, seed=41)
    artifact = tmp_path / "posterior.analysis"
    receipt = tmp_path / "posterior.receipt"
    artifact.write_bytes(session.export_result(seed=999))
    receipt.write_bytes(result.receipt.export())
    loaded = RecalcReceipt.consume(receipt.read_bytes())
    resumed = BayesianSession.resume(loaded, ResumeContext(portable_fit=True, portable_scores=True))
    with pytest.raises(RecalcRefusal) as error:
        resumed.execute(req, seed=41)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    )
    code = """
import json, pathlib, sys
import antecedent as ac
sys.path.insert(0,sys.argv[3])
from test_recalc_bayesian import request
from antecedent.recalc import RecalcReceipt, ResumeContext
from antecedent.recalc_bayesian import BayesianSession
artifact=ac.artifacts.accept(pathlib.Path(sys.argv[1]).read_bytes())
receipt=RecalcReceipt.consume(pathlib.Path(sys.argv[2]).read_bytes())
session=BayesianSession.resume(receipt,ResumeContext(supplied_data=True,portable_fit=True))
result=session.execute(request(),seed=41)
print(json.dumps({"verified":artifact["accepts_as_verified_program"],"mean":result.law.mean,"fits":result.receipt.totals.model_fits,"draws":result.receipt.totals.posterior_draws,"live":session.is_live}))
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(artifact), str(receipt), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(child.stdout)
    assert observed["verified"] and observed["live"]
    assert observed["mean"] == result.law.mean
    assert observed["fits"] > 0 and observed["draws"] == 2048


def test_bayesian_owned_prior_source_independent_transfer_and_double_use():
    req = request()
    source = BayesianSession()
    source.execute(req, seed=41)
    prior = source.export_prior_source()
    transfer = replace(
        req, inference=ac.Bayesian(backend="conjugate", n_draws=2048, prior_from=prior)
    )
    with pytest.raises(RecalcRefusal) as error:
        BayesianSession().execute(transfer, seed=43)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "recalc.bayesian_prior_likelihood_double_use",
        "prior.0",
    )
    changed = replace(transfer, data={**req.data, "y": req.data["y"] + 0.25})
    session = BayesianSession()
    result = session.execute(changed, seed=43)
    assert result.law.mean == pytest.approx(2, abs=0.03)
    assert result.receipt.totals.model_fits >= 2
    assert result.receipt.totals.posterior_draws >= 4096
    assert result.law == BayesianSession().execute(changed, seed=43).law
    assert session.execute(changed, seed=43).receipt.totals.total == 0
    invalid = replace(
        changed,
        inference=ac.Bayesian(
            backend="conjugate", n_draws=2048, prior_from=b"caller posterior metadata"
        ),
    )
    with pytest.raises(RecalcRefusal) as error:
        session.execute(invalid, seed=43)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "route_not_supported",
        "recalc.bayesian_prior_source_unverified",
        "prior.0",
    )


def test_bayesian_owned_prior_source_fresh_process_reexecutes_source_and_target(tmp_path):
    req = request()
    source = BayesianSession()
    source.execute(req, seed=41)
    path = tmp_path / "owned.prior"
    path.write_bytes(source.export_prior_source())
    target = replace(
        req,
        data={**req.data, "y": req.data["y"] + 0.25},
        inference=ac.Bayesian(backend="conjugate", n_draws=2048, prior_from=path.read_bytes()),
    )
    expected = BayesianSession().execute(target, seed=43)
    code = """
import json, pathlib, sys
from dataclasses import replace
import antecedent as ac
sys.path.insert(0,sys.argv[2])
from test_recalc_bayesian import request
from antecedent.recalc_bayesian import BayesianSession
req=request()
req=replace(req,data={**req.data,"y":req.data["y"]+0.25},inference=ac.Bayesian(backend="conjugate",n_draws=2048,prior_from=pathlib.Path(sys.argv[1]).read_bytes()))
session=BayesianSession()
result=session.execute(req,seed=43)
print(json.dumps({"mean":result.law.mean,"fits":result.receipt.totals.model_fits,"draws":result.receipt.totals.posterior_draws,"repeat":session.execute(req,seed=43).receipt.totals.total}))
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(child.stdout)
    assert observed["mean"] == expected.law.mean
    assert observed["fits"] == expected.receipt.totals.model_fits
    assert observed["fits"] >= 2 and observed["draws"] >= 4096
    assert observed["repeat"] == 0
