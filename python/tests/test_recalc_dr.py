"""Shared selective DR execution, structural truth and portable boundaries."""

import json
import subprocess
import sys
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimators import DML, DRLearner
from antecedent.learners import Auto, Linear, Logistic, Ridge
from antecedent.recalc import (
    RecalcReceipt,
    RecalcRefusal,
    ResumeContext,
    Stage,
    TargetWeights,
    Utility,
)
from antecedent.recalc_capabilities import (
    Family,
    Operation,
    RetainedKind,
    require_adapter,
    retained_kind,
)
from antecedent.recalc_dr import DrRequest, DrSession, _spec


def request(cate=False):
    rng = np.random.default_rng(61)
    n = 840
    z = np.tile([-1.0, 0.0, 1.0], n // 3)
    w = np.tile(np.linspace(-1, 1, 20), n // 20)
    t = (rng.random(n) < 1 / (1 + np.exp(-0.4 * z + 0.2 * w))).astype(float)
    y = z - 0.5 * w + t * (2 + 0.4 * z - 0.2 * w)
    model = (DRLearner if cate else DML)(outcome=Linear(), treatment=Logistic(), folds=3)
    if cate:
        model = replace(model, final_learner=Linear())
    return DrRequest(
        {"z": z, "w": w, "t": t, "y": y},
        [("z", "t"), ("w", "t"), ("z", "y"), ("w", "y"), ("t", "y")],
        "t",
        "y",
        Utility(1),
        model,
    )


def full(request, actual, seed=3):
    expected = DrSession().execute(request, seed=seed)
    assert actual.law.ate == pytest.approx(expected.law.ate, abs=1e-11)
    assert actual.law.std_error == pytest.approx(expected.law.std_error, abs=1e-11)
    assert actual.decision == expected.decision
    return expected


def refusal(call, code, detail, stage):
    with pytest.raises(RecalcRefusal) as caught:
        call()
    error = caught.value
    assert (error.reason_code, error.detail, error.stage) == (code, detail, stage)


@pytest.mark.parametrize("cate", [False, True])
def test_dr_native_dml_cate_counts_and_full_rerun(cate):
    r = request(cate)
    session = DrSession()
    result = session.execute(r, seed=3)
    truth = 2 + 0.4 * r.data["z"] - 0.2 * r.data["w"]
    assert session.score_contrast() == pytest.approx(truth, abs=1e-10)
    assert result.law.ate == pytest.approx(np.mean(truth), abs=1e-10)
    assert result.receipt.totals.fold_fits == 9
    assert result.receipt.totals.model_fits == int(cate)
    assert result.receipt.totals.score_computations == 2
    existing = ac.analyze(
        r.data,
        graph=r.edges,
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=r.estimator,
        refute=False,
        seed=3,
    )
    assert result.law.ate == pytest.approx(existing.effect, abs=1e-10)
    assert result.law.std_error == pytest.approx(existing.estimate.se_analytic, abs=1e-10)
    full(r, result)


@pytest.mark.parametrize("cate", [False, True])
def test_dr_utility_and_same_row_target_reuse_scores(cate):
    r = request(cate)
    session = DrSession()
    session.execute(r, seed=3)
    changed = replace(r, utility=Utility(3, 0.2))
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.total == 1
    full(changed, result)
    weights = np.exp(0.3 * r.data["z"])
    changed = replace(changed, target=TargetWeights(weights, ("z",)))
    result = session.execute(changed, seed=3)
    phi = np.asarray(session.score_contrast())
    mean = np.average(phi, weights=weights)
    se = np.sqrt(len(phi) / (len(phi) - 1) * np.sum((weights / sum(weights) * (phi - mean)) ** 2))
    assert result.law.ate == pytest.approx(mean, abs=1e-11)
    assert result.law.std_error == pytest.approx(se, abs=1e-11)
    assert result.receipt.totals.fold_fits == result.receipt.totals.model_fits == 0
    full(changed, result)


@pytest.mark.parametrize("mutation", ["data", "graph", "folds", "rng", "learner"])
def test_dr_changed_snapshot_graph_folds_rng_learner_refit(mutation):
    r = request(True)
    session = DrSession()
    session.execute(r, seed=3)
    seed = 3
    if mutation == "data":
        r = replace(r, data={**r.data, "y": r.data["y"] + 0.2 * r.data["t"]})
    elif mutation == "graph":
        r = replace(r, edges=[*r.edges, ("z", "w")])
    elif mutation == "folds":
        r = replace(r, estimator=replace(r.estimator, folds=4))
    elif mutation == "rng":
        seed = 7
    else:
        r = replace(r, estimator=replace(r.estimator, outcome=Ridge(penalty=0.05)))
    result = session.execute(r, seed=seed)
    assert result.receipt.totals.fold_fits == 3 * r.estimator.folds
    assert result.receipt.totals.model_fits == 1
    full(r, result, seed)


def test_dr_complete_case_rows_and_fold_identity():
    r = request()
    data = {name: np.array(value, copy=True) for name, value in r.data.items()}
    data["y"][1] = np.nan
    data["z"][7] = np.nan
    r = replace(r, data=data)
    session = DrSession()
    first = session.execute(r, seed=3)
    assert session.row_ids == tuple(i for i in range(840) if i not in (1, 7))
    assert len(session.score_contrast()) == 838
    changed = replace(r, target=TargetWeights(np.ones(838), ("z",)))
    second = session.execute(changed, seed=3)
    assert second.receipt.totals.fold_fits == 0
    assert (
        first.receipt.requested[Stage.LEARNER_FOLDS_RNG]
        == second.receipt.requested[Stage.LEARNER_FOLDS_RNG]
    )
    full(changed, second)


def test_dr_cate_predict_named_schema_zero_fits_independent_truth():
    r = request(True)
    session = DrSession()
    session.execute(r, seed=3)
    rows = [[-1, -0.5], [0, 0], [1, 0.5]]
    assert session.prediction_columns == ("z", "w")
    result = session.predict(rows, columns=("z", "w"))
    assert result.values == pytest.approx([1.7, 2, 2.3], abs=1e-10)
    assert result.model_fits == 0
    assert result.to_dict()["uncertainty"]["status"] == "unavailable"
    fresh = DrSession()
    fresh.execute(r, seed=3)
    assert result.values == pytest.approx(fresh.predict(rows, columns=("z", "w")).values)


@pytest.mark.parametrize("cate", [False, True])
def test_dr_portable_scores_and_predictor_fresh_process(tmp_path, cate):
    r = request(cate)
    session = DrSession()
    session.execute(r, seed=3)
    scores = session.export_scores()
    model = session.export_predictor() if cate else None
    if model is not None:
        assert model.features == ("z", "w")
    else:
        refusal(
            session.export_predictor,
            "route_not_supported",
            "recalc.dr_predictor_unavailable",
            "score_artifact",
        )
    (tmp_path / "scores").write_bytes(scores.export())
    if model is not None:
        (tmp_path / "model").write_bytes(model.export())
    script = """import json,sys
from antecedent.recalc_cell import FrozenScores
from antecedent.prediction import FittedEffectModel
from antecedent.recalc import Utility
from pathlib import Path
p=Path(sys.argv[1])
f=FrozenScores.load((p/'scores').read_bytes(),expected_identity=sys.argv[2])
s=f.resume(variables=['z','w','t','y'],edges=[('z','t'),('w','t'),('z','y'),('w','y'),('t','y')],utility=Utility(1),quantity='average_effect',expected_identity=sys.argv[2])
r=s.retarget()
values=None
if (p/'model').exists():
 m=FittedEffectModel.load((p/'model').read_bytes())
 values=m.predict({'z':[-1,0,1],'w':[-.5,0,.5]}).values
print(json.dumps({'ate':r.law.ate,'fits':r.receipt.totals.fold_fits,'values':values}))"""
    ran = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), scores.identity],
        check=True,
        capture_output=True,
        text=True,
    )
    wire = json.loads(ran.stdout)
    assert wire["ate"] == pytest.approx(2)
    assert wire["fits"] == 0
    if cate:
        assert wire["values"] == pytest.approx([1.7, 2, 2.3], abs=1e-10)
    else:
        assert wire["values"] is None


def test_dr_receipt_only_resume_refuses_and_supplied_data_refits(tmp_path):
    r = request(True)
    result = DrSession().execute(r, seed=3)
    receipt = RecalcReceipt.consume(
        result.receipt.export(), expected_identity=result.receipt.identity
    )
    resumed = DrSession.resume(
        receipt,
        ResumeContext(
            portable_fit=True,
            portable_scores=True,
            supplied_provider=True,
            scores_snapshot_bound=True,
        ),
    )
    assert not resumed.is_live
    refusal(
        lambda: resumed.execute(r, seed=3),
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    )
    resumed = DrSession.resume(receipt, ResumeContext(supplied_data=True))
    rerun = resumed.execute(r, seed=3)
    assert rerun.receipt.totals.fold_fits == 9
    assert rerun.receipt.totals.model_fits == 1
    full(r, rerun)
    (tmp_path / "receipt").write_bytes(receipt.export())
    script = """import json,sys,numpy as np
from pathlib import Path
from antecedent.recalc import RecalcReceipt,ResumeContext,RecalcRefusal,Utility
from antecedent.recalc_dr import DrRequest,DrSession
from antecedent.estimators import DRLearner
from antecedent.learners import Linear,Logistic
rng=np.random.default_rng(61);n=840
z=np.tile([-1.,0.,1.],n//3);w=np.tile(np.linspace(-1,1,20),n//20)
t=(rng.random(n)<1/(1+np.exp(-.4*z+.2*w))).astype(float)
y=z-.5*w+t*(2+.4*z-.2*w)
r=DrRequest({'z':z,'w':w,'t':t,'y':y},[('z','t'),('w','t'),('z','y'),('w','y'),('t','y')],'t','y',Utility(1),DRLearner(outcome=Linear(),treatment=Logistic(),final_learner=Linear(),folds=3))
a=RecalcReceipt.consume(Path(sys.argv[1]).read_bytes())
s=DrSession.resume(a,ResumeContext(portable_fit=True,portable_scores=True,supplied_provider=True,scores_snapshot_bound=True))
try:s.execute(r,seed=3)
except RecalcRefusal as e:detail=e.detail
else:raise AssertionError('historical receipt recreated executable state')
s=DrSession.resume(a,ResumeContext(supplied_data=True));ran=s.execute(r,seed=3)
print(json.dumps({'detail':detail,'folds':ran.receipt.totals.fold_fits,'models':ran.receipt.totals.model_fits,'ate':ran.law.ate}))
"""
    ran = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "receipt")],
        check=True,
        text=True,
        capture_output=True,
    )
    wire = json.loads(ran.stdout)
    assert wire == {
        "detail": "recalc.unavailable_data",
        "folds": 9,
        "models": 1,
        "ate": pytest.approx(2),
    }


def test_dr_unsupported_score_target_and_prediction_preserve_state():
    r = request(True)
    session = DrSession()
    session.execute(r, seed=3)
    before = session.identities
    unsupported = request()
    unsupported = replace(
        unsupported, estimator=replace(unsupported.estimator, score="partially_linear")
    )
    refusal(
        lambda: session.execute(unsupported, seed=3),
        "route_not_supported",
        "recalc.unsupported_request",
        "treatment_grid",
    )
    for rows, names in [([[0, 0]], ["w", "z"]), ([[0]], ["z", "w"]), ([[np.nan, 0]], ["z", "w"])]:
        refusal(
            lambda rows=rows, names=names: session.predict(rows, columns=names),
            "invalid_argument",
            "recalc.dr_prediction_schema_mismatch",
            "score_artifact",
        )
    unsupported = replace(r, estimator=replace(r.estimator, outcome=Auto()))
    refusal(
        lambda: session.execute(unsupported, seed=3),
        "route_not_supported",
        "recalc.unsupported_request",
        "treatment_grid",
    )
    assert session.identities == before
    marginal = DrSession()
    marginal.execute(request(), seed=3)
    refusal(
        lambda: marginal.predict([[0, 0]], columns=["z", "w"]),
        "route_not_supported",
        "recalc.dr_predictor_unavailable",
        "score_artifact",
    )
    invalid = replace(r, target=TargetWeights(np.ones(1), ("z",)))
    refusal(
        lambda: session.execute(invalid, seed=3),
        "invalid_argument",
        "recalc.invalid_target_weights",
        "target_population",
    )
    refusal(
        lambda: session.predict([[5, 0]], columns=["z", "w"]),
        "route_not_supported",
        "recalc.dr_prediction_out_of_support",
        "score_artifact",
    )
    no_arm = replace(r, data={**r.data, "t": np.zeros(840)})
    refusal(
        lambda: session.execute(no_arm, seed=3),
        "arm_not_populated",
        "recalc.dr_arm_not_populated",
        "score_artifact",
    )
    malformed = replace(r, edges=[*r.edges, ("y", "t")])
    refusal(
        lambda: session.execute(malformed, seed=3),
        "invalid_argument",
        "recalc.invalid_graph",
        "graph",
    )
    invalid_query = replace(r, treatment="y")
    refusal(
        lambda: session.execute(invalid_query, seed=3),
        "invalid_argument",
        "recalc.invalid_query",
        "query",
    )
    assert session.identities == before
    prediction = session.predict([[0, 0]], columns=["z", "w"])
    assert prediction.values == pytest.approx([2])
    assert prediction.model_fits == 0
    assert session.execute(r, seed=3).receipt.totals.total == 0


def test_dr_bounded_native_input_and_learner_configs():
    r = request()
    session = DrSession()
    spec = json.dumps(_spec(r))
    with pytest.raises(CausalValueError) as caught:
        session._handle.execute(
            ["z", "w", "t", "y"], [np.zeros(1), np.zeros(100001), np.zeros(1), np.zeros(1)], spec
        )
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(caught.value)
    with pytest.raises(CausalValueError) as caught:
        session.execute(replace(r, estimator=replace(r.estimator, folds=21)))
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.dr_invalid_folds" in str(caught.value)
    cate = DrSession()
    cate.execute(request(True), seed=3)

    class NonIterable(list):
        def __iter__(self):
            raise AssertionError("native bounds must use actual list items")

    report, error = cate._handle.predict(NonIterable([NonIterable([0, 0])]), ["z", "w"])
    assert error is None
    assert json.loads(report)["model_fits"] == 0
    with pytest.raises(CausalValueError) as caught:
        session._handle.predict([[0]] * 100001, ["z"])
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(caught.value)


def test_dr_artifact_receipt_count_replay():
    result = DrSession().execute(request(True), seed=3)
    loaded = RecalcReceipt.consume(
        result.receipt.export(), expected_identity=result.receipt.identity
    )
    assert loaded.totals == result.receipt.totals
    assert loaded.totals.fold_fits == 9
    assert loaded.totals.model_fits == 1


def test_dr_capabilities_inspect_native_score_and_predictor_states():
    empty = DrSession()
    assert retained_kind(empty) == RetainedKind.READABLE
    refusal(
        lambda: empty.predict([[0, 0]], columns=["z", "w"]),
        "score_table_unavailable",
        "recalc.no_live_state",
        "score_artifact",
    )
    for cate in [False, True]:
        session = DrSession()
        session.execute(request(cate), seed=3)
        assert retained_kind(session) == RetainedKind.LIVE_SCORES
        adapter = require_adapter(Family.DOUBLY_ROBUST, Operation.DATA, session)
        assert adapter.python_path == "antecedent.recalc_dr.DrSession.execute"
        assert (session._handle.prediction_columns() is not None) == cate
        if cate:
            assert retained_kind(session.export_predictor()) == RetainedKind.PREDICTOR
