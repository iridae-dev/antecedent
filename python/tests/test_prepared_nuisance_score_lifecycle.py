"""Nuisance products execute explicitly and keep causal and nuisance uncertainty separate."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import nuisance_scores as ns
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import Outcome, require_success


def _data():
    return {
        "z": np.r_[np.zeros(100), np.ones(100)],
        "t": np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)],
        "y": np.arange(200, dtype=float) % 7,
    }


def _study(data=None):
    return ant.prepare(
        _data() if data is None else data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=Execution(seed=13, threads=1),
        refute=False,
    )


def _sigmoid(value):
    return 1 / (1 + np.exp(-value))


def _mode(design, treatment, precision):
    beta = np.zeros(design.shape[1])
    for _ in range(40):
        p = _sigmoid(design @ beta)
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + precision
        beta += np.linalg.solve(hessian, design.T @ (treatment - p) - precision @ beta)
    return beta


def test_preparation_is_fit_free_and_explicit_nuisance_fit_matches_independent_mode():
    prepared = observe_native_attempts(_study)
    study = prepared.unwrap()
    for name in ("glm_fit", "posterior_fit", "posterior_draw", "score_construction"):
        assert prepared.report.operations[name].attempted == 0
    before = study.inspect().contract
    attempted = observe_native_attempts(
        lambda: ns.treatment_score_posterior(study, draws=500, summary_rows=[0, 150])
    )
    result = attempted.unwrap()
    assert isinstance(result, Outcome)
    product = require_success(result)
    assert not isinstance(product, Outcome)
    data = _data()
    beta = _mode(np.column_stack([np.ones(200), data["z"]]), data["t"], np.eye(2) / 4)
    assert [product["mode.0"], product["mode.1"]] == pytest.approx(beta, abs=1e-5)
    assert product["row0.at_mode"] == pytest.approx(_sigmoid(beta[0]), abs=1e-5)
    assert product["row1.at_mode"] == pytest.approx(_sigmoid(beta.sum()), abs=1e-5)
    assert "not_an_effect_posterior" in product.uncertainty
    assert product.target == "propensity_score_nuisance"
    assert attempted.report.operations["posterior_fit"].completed == 1
    assert attempted.report.operations["posterior_draw"].completed == 500
    assert attempted.report.operations["score_construction"].attempted == 0
    assert study.inspect().contract == before and study.inspect().stage == "prepared"


def test_nuisance_execution_branches_leave_original_products_and_declarations_unchanged():
    study = _study()
    original = require_success(ns.treatment_score_posterior(study, draws=300, summary_rows=[0]))
    different_seed = require_success(
        ns.treatment_score_posterior(
            study, draws=300, summary_rows=[0], execution=Execution(seed=97, threads=1)
        )
    )
    assert original["row0.at_mode"] == different_seed["row0.at_mode"]
    assert original.request_digest != different_seed.request_digest
    changed = _data()
    changed["t"] = 1 - changed["t"]
    branch = ant.rebind(study, data=changed)
    replacement = require_success(ns.treatment_score_posterior(branch, draws=300, summary_rows=[0]))
    assert replacement["row0.at_mode"] == pytest.approx(1 - original["row0.at_mode"], abs=1e-5)
    repeated = require_success(ns.treatment_score_posterior(study, draws=300, summary_rows=[0]))
    assert repeated.request_digest == original.request_digest
    assert dict(repeated.values) == dict(original.values)
    loaded = ns.load_artifact(bytes(original.export()))
    assert loaded.request_digest == original.request_digest
    assert dict(loaded.values) == dict(original.values)
    assert study.execution.seed == 13


def test_nuisance_likelihood_uses_the_completed_members_analysis_rows():
    from antecedent.estimation import CandidateScreen, analyze_many

    selected = _data()
    excluded = {**selected, "t": 1 - selected["t"]}
    data = {name: np.r_[excluded[name], selected[name]] for name in selected}
    batch = require_success(
        analyze_many(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            queries=[ant.AverageEffect("t", "y")],
            estimator="aipw",
            inference=ant.Frequentist(bootstrap=0),
            execution=Execution(seed=13, threads=1),
            candidate_screen=CandidateScreen(
                "nuisance-rows", "unrecorded", tuple(range(200)), tuple(range(200, 400))
            ),
            refute=False,
        )
    )
    member = batch.member(0)
    assert member.study._data.num_rows == 400
    actual = require_success(
        ns.treatment_score_posterior(member.study, draws=200, summary_rows=[0, 350])
    )
    reference = require_success(
        ns.treatment_score_posterior(_study(selected), draws=200, summary_rows=[0, 150])
    )
    assert actual.request_digest == reference.request_digest
    assert actual.values == reference.values
    design = np.column_stack([np.ones(200), selected["z"]])
    beta = _mode(design, selected["t"], np.eye(2) / 4)
    assert [actual["mode.0"], actual["mode.1"]] == pytest.approx(beta, abs=1e-5)
    assert actual["row0.at_mode"] == pytest.approx(_sigmoid(beta[0]), abs=1e-5)
    assert actual["row1.at_mode"] == pytest.approx(_sigmoid(beta.sum()), abs=1e-5)
    assert batch.retarget_rows() == tuple(range(200, 400))


def test_nuisance_fit_honors_actual_resources_cancellation_and_original_callback_errors():
    study = _study()
    with pytest.raises(CausalResourceError):
        ns.treatment_score_posterior(
            study, draws=100, execution=Execution(limits=ResourceLimits(memory_bytes=0))
        )
    token = ant.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ns.treatment_score_posterior(study, execution=Execution(cancel=token))
    original = CausalUnsupportedError("user observer failed", reason_code="effect_not_identified")

    def observer(_fraction, _stage):
        raise original

    with pytest.raises(CausalUnsupportedError) as raised:
        ns.treatment_score_posterior(study, execution=Execution(on_progress=observer))
    assert raised.value is original
    assert any(frame.name == "observer" for frame in raised.traceback)
    with pytest.raises(CausalUnsupportedError) as unavailable:
        ns.treatment_score_posterior(study, execution=Execution(on_stage=lambda *_args: None))
    assert unavailable.value.reason_code == "stage_stream_unavailable"
    assert require_success(ns.treatment_score_posterior(study, draws=50)).target == (
        "propensity_score_nuisance"
    )


@pytest.mark.parametrize("arguments", [{"draws": 0}, {"prior_sd": 0}, {"clip": 0.5}])
def test_invalid_nuisance_arguments_remain_typed_errors(arguments):
    with pytest.raises(CausalValueError):
        ns.treatment_score_posterior(_study(), **arguments)
    with pytest.raises(CausalTypeError):
        ns.treatment_score_posterior({}, draws=10)


def _temporal_study(*, panel=True):
    rng = np.random.default_rng(9)
    units = []
    for _ in range(160):
        c = rng.normal(size=3)
        x = (rng.random(3) < _sigmoid(0.5 + 0.9 * c)).astype(float)
        y = np.r_[0.0, _sigmoid(x[:-1] + c[:-1])]
        units.append({"x": x, "c": c, "y": y})
    data = (
        ant.data.panel(units)
        if panel
        else {k: np.concatenate([u[k] for u in units]) for k in units[0]}
    )
    study = ant.prepare(
        data,
        graph=[("c", 1, "x", 1), ("c", 1, "y", 0), ("x", 1, "y", 0)],
        query=ant.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1, active_level=1.0),
        inference=ant.Frequentist(bootstrap=0),
        execution=Execution(seed=3, threads=1),
        refute=False,
    )
    return study, units


def test_temporal_nuisance_uses_independent_subject_rows_and_declared_ridge_or_prior():
    study, units = _temporal_study()
    design = np.array([[1, u["c"][0]] for u in units])
    treatment = np.array([u["x"][0] for u in units])
    ridge = require_success(ns.temporal_treatment_score(study, [[0], [1.5]], penalty=2))
    beta = _mode(design, treatment, np.diag([0, 2]))
    assert [ridge["coefficient.0"], ridge["coefficient.1"]] == pytest.approx(beta, abs=1e-6)
    assert ridge["score1.at_mode"] == pytest.approx(_sigmoid(beta @ [1, 1.5]), abs=1e-6)
    posterior = require_success(
        ns.temporal_treatment_score(study, [[0], [1.5]], slope_sd=0.5, draws=500)
    )
    beta = _mode(design, treatment, np.diag([1, 4]))
    assert posterior["score0.at_mode"] == pytest.approx(_sigmoid(beta[0]), abs=1e-6)
    assert posterior.estimator_id == "bayesian.temporal.propensity_posterior"
    assert ridge.estimator_id == "temporal.penalized_propensity"
    assert study.inspect().stage == "prepared"


def test_dependent_series_produces_a_scientific_nuisance_refusal_without_fitting():
    study, _ = _temporal_study(panel=False)
    attempted = observe_native_attempts(
        lambda: ns.temporal_treatment_score(study, [[0]], penalty=1)
    )
    result = attempted.unwrap()
    assert result.value is None and result.refusal is not None
    assert result.refusal.code == "route_not_supported"
    assert "single dependent series" in result.refusal.message
    assert attempted.report.operations["glm_fit"].attempted == 0
    assert attempted.report.operations["posterior_fit"].attempted == 0
