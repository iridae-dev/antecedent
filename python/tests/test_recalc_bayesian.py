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
from antecedent import _native
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.recalc import RecalcReceipt, RecalcRefusal, ResumeContext, Utility
from antecedent.recalc_bayesian import (
    BayesianRequest,
    BayesianSession,
    JointBayesianRequest,
    JointBayesianSession,
    JointRecalcSource,
    JointRecalcTarget,
    PosteriorSummary,
)
from antecedent.recalc_capabilities import (
    Family,
    Operation,
    RetainedKind,
    require_adapter,
    retained_kind,
)
from antecedent.transport._joint_posterior import GaussianTransportPrior, JointTransportIdentity


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
    result = require_success(session.execute(req, seed=41))
    assert result.receipt.totals.model_fits > 0
    assert result.receipt.totals.posterior_draws == 2048
    assert result.receipt.totals.fold_fits == 0
    assert result.law.mean == pytest.approx(2, abs=0.03)
    assert_projection(result, session.effect_draws, req.summary)
    rerun = require_success(BayesianSession().execute(req, seed=41))
    assert result.law == rerun.law
    if model == "gaussian":
        full = require_success(
            ac.analyze(
                req.data,
                query=ac.AverageEffect("t", "y"),
                graph=req.edges,
                inference=req.inference,
                refute="none",
                execution=ac.Execution(seed=41),
            )
        )
        assert result.law.mean == pytest.approx(full.posterior.effect_mean, abs=1e-12)
        assert result.law.standard_deviation == pytest.approx(full.posterior.effect_sd, abs=1e-12)


def test_bayesian_summary_and_utility_reuse_actual_native_rows():
    req = request()
    session = BayesianSession()
    first = require_success(session.execute(req, seed=41))
    rows = session.effect_draws
    assert require_success(session.execute(req, seed=41)).receipt.totals.total == 0
    changed = replace(req, summary=PosteriorSummary(0.1, 0.9, 2.0))
    summary = require_success(session.execute(changed, seed=41))
    assert summary.receipt.totals.model_fits == summary.receipt.totals.posterior_draws == 0
    assert summary.receipt.totals.law_summaries == 1
    assert session.effect_draws == rows
    assert_projection(summary, rows, changed.summary)
    utility = require_success(session.execute(replace(changed, utility=Utility(3, 1)), seed=41))
    assert utility.receipt.totals.total == utility.receipt.totals.decisions == 1
    assert utility.law == summary.law
    assert utility.decision.net_benefit == pytest.approx(3 * first.law.mean - 1)


@pytest.mark.parametrize("mutation", ["data", "model", "inference", "graph", "seed"])
def test_bayesian_changed_inputs_refit_against_fresh_execution(mutation):
    req = request()
    session = BayesianSession()
    require_success(session.execute(req, seed=41))
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
    result = require_success(session.execute(changed, seed=seed))
    assert result.receipt.totals.model_fits > 0
    assert result.receipt.totals.posterior_draws > 0
    assert result.law == require_success(BayesianSession().execute(changed, seed=seed)).law


def test_bayesian_exact_refusals_preserve_native_rows_and_bounds():
    req = request()
    session = BayesianSession()
    require_success(session.execute(req, seed=41))
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
            require_success(session.execute(changed, seed=41))
        assert (error.value.reason_code, error.value.detail, error.value.stage) == (
            code,
            detail,
            stage,
        )
        assert session.effect_draws == rows and session.identities == ids
    with pytest.raises(CausalValueError) as error:
        require_success(
            session.execute(
                replace(req, inference=ac.Bayesian(backend="conjugate", n_draws=100001)), seed=41
            )
        )
    assert error.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(error.value)
    assert require_success(session.execute(req, seed=41)).receipt.totals.total == 0


def test_bayesian_native_capabilities_require_retained_state():
    session = BayesianSession()
    assert retained_kind(session) == RetainedKind.READABLE
    require_success(session.execute(request(), seed=41))
    assert retained_kind(session) == RetainedKind.LIVE_POSTERIOR
    assert (
        require_adapter(Family.BAYESIAN, Operation.FUNCTIONAL, session).state_type
        is BayesianSession
    )
    assert session.effect_draws is not None


def test_bayesian_artifact_and_receipt_fresh_process_boundary(tmp_path):
    req = request()
    session = BayesianSession()
    result = require_success(session.execute(req, seed=41))
    artifact = tmp_path / "posterior.analysis"
    receipt = tmp_path / "posterior.receipt"
    artifact.write_bytes(bytes(session.export(seed=999)))
    receipt.write_bytes(bytes(result.receipt.export()))
    loaded = require_success(RecalcReceipt.consume(receipt.read_bytes()))
    resumed = require_success(
        BayesianSession.resume(loaded, ResumeContext(portable_fit=True, portable_scores=True))
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(resumed.execute(req, seed=41))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    )
    code = "from antecedent.outcomes import require_success\nimport json, pathlib, sys\nimport antecedent as ac\nsys.path.insert(0, sys.argv[3])\nfrom test_recalc_bayesian import request\nfrom antecedent.recalc import RecalcReceipt, ResumeContext\nfrom antecedent.recalc_bayesian import BayesianSession\nartifact = ac.artifacts.accept(pathlib.Path(sys.argv[1]).read_bytes())\nreceipt = require_success(RecalcReceipt.consume(pathlib.Path(sys.argv[2]).read_bytes()))\nsession = require_success(BayesianSession.resume(receipt, ResumeContext(supplied_data=True, portable_fit=True)))\nresult = require_success(session.execute(request(), seed=41))\nprint(json.dumps({'verified': artifact['accepts_as_verified_program'], 'mean': result.law.mean, 'fits': result.receipt.totals.model_fits, 'draws': result.receipt.totals.posterior_draws, 'live': session.is_live}))\n"
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
    require_success(source.execute(req, seed=41))
    prior = bytes(source.export(quantity="prior_source"))
    transfer = replace(
        req, inference=ac.Bayesian(backend="conjugate", n_draws=2048, prior_from=prior)
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(BayesianSession().execute(transfer, seed=43))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "recalc.bayesian_prior_likelihood_double_use",
        "prior.0",
    )
    changed = replace(transfer, data={**req.data, "y": req.data["y"] + 0.25})
    session = BayesianSession()
    result = require_success(session.execute(changed, seed=43))
    assert result.law.mean == pytest.approx(2, abs=0.03)
    assert result.receipt.totals.model_fits >= 2
    assert result.receipt.totals.posterior_draws >= 4096
    assert result.law == require_success(BayesianSession().execute(changed, seed=43)).law
    assert require_success(session.execute(changed, seed=43)).receipt.totals.total == 0
    invalid = replace(
        changed,
        inference=ac.Bayesian(
            backend="conjugate", n_draws=2048, prior_from=b"caller posterior metadata"
        ),
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(invalid, seed=43))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "route_not_supported",
        "recalc.bayesian_prior_source_unverified",
        "prior.0",
    )


def test_bayesian_owned_prior_source_fresh_process_reexecutes_source_and_target(tmp_path):
    req = request()
    source = BayesianSession()
    require_success(source.execute(req, seed=41))
    path = tmp_path / "owned.prior"
    path.write_bytes(bytes(source.export(quantity="prior_source")))
    target = replace(
        req,
        data={**req.data, "y": req.data["y"] + 0.25},
        inference=ac.Bayesian(backend="conjugate", n_draws=2048, prior_from=path.read_bytes()),
    )
    expected = require_success(BayesianSession().execute(target, seed=43))
    code = "from antecedent.outcomes import require_success\nimport json, pathlib, sys\nfrom dataclasses import replace\nimport antecedent as ac\nsys.path.insert(0, sys.argv[2])\nfrom test_recalc_bayesian import request\nfrom antecedent.recalc_bayesian import BayesianSession\nreq = request()\nreq = replace(req, data={**req.data, 'y': req.data['y'] + 0.25}, inference=ac.Bayesian(backend='conjugate', n_draws=2048, prior_from=pathlib.Path(sys.argv[1]).read_bytes()))\nsession = BayesianSession()\nresult = require_success(session.execute(req, seed=43))\nprint(json.dumps({'mean': result.law.mean, 'fits': result.receipt.totals.model_fits, 'draws': result.receipt.totals.posterior_draws, 'repeat': require_success(session.execute(req, seed=43)).receipt.totals.total}))\n"
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


def joint_request():
    """Two randomized sources (outcome 1 control, 3 treated, unit noise) and a 40-unit target."""

    def source(name):
        return JointRecalcSource(
            name,
            JointTransportIdentity(
                f"snapshot-{name}", tuple(f"{name}-{row}" for row in range(128))
            ),
            [row % 2 == 1 for row in range(128)],
            [3.0 if row % 2 == 1 else 1.0 for row in range(128)],
            1.0,
        )

    return JointBayesianRequest(
        names=("t", "y"),
        edges=(("t", "y"),),
        treatment="t",
        outcome="y",
        source_population="source",
        target_population="target",
        sources=(source("a"), source("b")),
        target=JointRecalcTarget(
            JointTransportIdentity("target", tuple(f"target-{row}" for row in range(40))), 40
        ),
        invariant=GaussianTransportPrior((0.0,), ((100.0,),)),
        varying_prior=GaussianTransportPrior((0.0,), ((4.0,),)),
        utility=Utility(1.0, 0.0),
        draws=1024,
        seed=19,
    )


JOINT_ENABLED = hasattr(_native, "JointBayesianSessionHandle")


def assert_joint_route_frozen():
    # An ordinary build omits the whole calibration-internal candidate family.
    with pytest.raises(CausalUnsupportedError, match="calibration-internal") as error:
        JointBayesianSession()
    assert error.value.reason_code == "route_not_supported"


def test_joint_bayesian_session_matches_gaussian_algebra_and_reuses_the_retained_fit():
    if not JOINT_ENABLED:
        assert_joint_route_frozen()
        return
    req = joint_request()
    session = JointBayesianSession()
    first = require_success(session.execute(req, seed=19))
    # Independent conjugate algebra of the shared-block model: the invariant intercept sums the
    # two sources (precision 128 + 128 + 0.01 for the prior), the shared varying block adds the
    # cross term 128 and the treated-minus-control contrast is 2 in each unit.
    a, b, c = 128.01, 128.0, 256.25
    det = a * c - b * b
    mean, variance = (c * 384.0 - b * 512.0) / det, c / det
    assert first.target_effect_mean == pytest.approx(mean, abs=1e-12)
    assert first.target_effect_variance == pytest.approx(variance, abs=1e-12)
    assert first.effect_names[-1] == "effect.target"
    assert first.effect_means[-1] == pytest.approx(mean, abs=1e-12)
    # Three counted solves: the joint model and one disagreement solve per source.
    assert first.receipt.totals.model_fits == 3
    assert first.receipt.totals.posterior_draws == 1024
    assert first.law.draws == 1024
    # The law is a summary of the native draws of that posterior, not a calibrated interval.
    assert first.law.mean == pytest.approx(mean, abs=5 * (variance / 1024) ** 0.5)
    assert first.law.standard_deviation == pytest.approx(variance**0.5, rel=0.1)
    assert first.decision.net_benefit == pytest.approx(first.law.mean, abs=1e-12)
    assert session.is_live

    changed = replace(req, summary=PosteriorSummary(0.1, 0.9, 2.0))
    summary = require_success(session.execute(changed, seed=19))
    assert summary.receipt.totals.model_fits == summary.receipt.totals.posterior_draws == 0
    assert summary.receipt.totals.law_summaries == 1
    assert summary.target_effect_mean == first.target_effect_mean
    priced = require_success(session.execute(replace(changed, utility=Utility(3.0, 1.0)), seed=19))
    assert priced.receipt.totals.total == priced.receipt.totals.decisions == 1
    assert priced.decision.net_benefit == pytest.approx(3.0 * summary.law.mean - 1.0, abs=1e-12)
    assert require_success(session.execute(changed, seed=19)).receipt.totals.total == 0

    # A different posterior stream refits the joint model and redraws.
    reseeded = require_success(session.execute(replace(changed, seed=20), seed=19))
    assert reseeded.receipt.totals.model_fits > 0
    assert reseeded.receipt.totals.posterior_draws == 1024
    assert reseeded.target_effect_mean == pytest.approx(mean, abs=1e-12)

    artifact = bytes(session.export())
    assert artifact
    assert JointBayesianSession().is_live is False


def test_joint_bayesian_refusals_keep_the_retained_fit_and_bound_the_work():
    if not JOINT_ENABLED:
        assert_joint_route_frozen()
        return
    req = joint_request()
    session = JointBayesianSession()
    require_success(session.execute(req, seed=19))
    used = replace(
        req,
        varying_prior=GaussianTransportPrior(
            (0.0,),
            ((4.0,),),
            bank_id="used-bank",
            consumed=(req.sources[0].identity,),
        ),
    )
    # A prior bank built from the very units the likelihood consumes is refused before any work.
    with pytest.raises(RecalcRefusal):
        require_success(session.execute(used, seed=19))
    assert require_success(session.execute(req, seed=19)).receipt.totals.total == 0
    oversized = replace(
        req,
        varying_prior=GaussianTransportPrior(
            (0.0,),
            ((4.0,),),
            bank_id="oversized-bank",
            consumed=(req.sources[0].identity,) * 17,
        ),
    )
    with pytest.raises(RecalcRefusal):
        require_success(session.execute(oversized, seed=19))
    # A source whose identity does not name each of its units once is an invalid declaration.
    broken = JointRecalcSource(
        "a",
        JointTransportIdentity("snapshot-a", ("a-0", "a-1")),
        [False, True, False],
        [1.0, 3.0, 1.0],
        1.0,
    )
    with pytest.raises(CausalValueError):
        require_success(session.execute(replace(req, sources=(broken, req.sources[1])), seed=19))
    with pytest.raises(CausalValueError):
        require_success(session.execute(replace(req, treatment="missing"), seed=19))
