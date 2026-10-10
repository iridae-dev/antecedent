"""Checked design reuse with structural truths and source-backed artifact replay."""

import itertools
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from antecedent.recalc import RecalcReceipt, RecalcRefusal, ResumeContext, Utility
from antecedent.recalc_design import (
    DesignRequest,
    DesignSession,
    DesignWeakInstrument,
    FrontdoorModel,
    IvModel,
    RdModel,
    consume_design_result,
)


def request(kind="iv", strength=1.0):
    if kind == "iv":
        rows = np.array(
            [
                (z, strength * z + u, 2 * (strength * z + u) + 0.3 * u + 0.4 * e)
                for _ in range(20)
                for z, u, e in itertools.product([0.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])
            ]
        )
        names, edges, model = ["z", "t", "y"], [("z", "t"), ("t", "y")], IvModel("z")
    elif kind == "rd":
        rows = np.array(
            [
                (r, float(r >= 0), 1 + 2.5 * (r >= 0) + 0.3 * r + 0.8 * (r >= 0) * r + 0.4 * e)
                for _ in range(20)
                for r, e in itertools.product([-0.5, -0.25, 0.25, 0.5], [-1.0, 1.0])
            ]
        )
        names, edges, model = (
            ["r", "t", "y"],
            [("r", "t"), ("r", "y"), ("t", "y")],
            RdModel("r", 0, 1),
        )
    else:
        rows = np.array(
            [
                (t, 0.5 * t + d, 3 * (0.5 * t + d) + 0.4 * e)
                for _ in range(20)
                for t, d, e in itertools.product([0.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])
            ]
        )
        names, edges, model = ["t", "m", "y"], [("t", "m"), ("m", "y")], FrontdoorModel("m")
    return DesignRequest(dict(zip(names, rows.T, strict=True)), edges, "t", "y", model, Utility(1))


def full(r, actual):
    fresh = DesignSession().execute(r, seed=3)
    assert actual.law == fresh.law
    assert actual.decision == fresh.decision
    return fresh


def refused(call, code, detail, stage):
    with pytest.raises(RecalcRefusal) as caught:
        call()
    assert (caught.value.reason_code, caught.value.detail, caught.value.stage) == (
        code,
        detail,
        stage,
    )


@pytest.mark.parametrize(
    "kind,truth,fits", [("iv", 2.0, None), ("rd", 2.5, 1), ("frontdoor", 1.5, 2)]
)
def test_design_native_iv_rd_frontdoor_counts_and_truth(kind, truth, fits):
    r = request(kind)
    result = DesignSession().execute(r, seed=3)
    assert result.law.ate == pytest.approx(truth, abs=1e-11)
    counts = result.receipt.totals
    assert counts.model_fits >= 2 if fits is None else counts.model_fits == fits
    assert counts.fold_fits == counts.reweights == 0
    assert counts.law_summaries == 1
    if kind == "iv":
        assert result.law.std_error is None
        # Independent Wald ratio using the two instrument groups.
        z, t, y = (r.data[n] for n in ("z", "t", "y"))
        assert result.law.ate == pytest.approx(
            (y[z == 1].mean() - y[z == 0].mean()) / (t[z == 1].mean() - t[z == 0].mean())
        )
    full(r, result)


@pytest.mark.parametrize("kind", ["iv", "rd", "frontdoor"])
def test_design_utility_reuse_and_full_rerun(kind):
    r = request(kind)
    session = DesignSession()
    session.execute(r, seed=3)
    assert session.execute(r, seed=3).receipt.totals.total == 0
    changed = replace(r, utility=Utility(3, 0.2))
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.total == result.receipt.totals.decisions == 1
    full(changed, result)


@pytest.mark.parametrize("kind", ["iv", "rd", "frontdoor"])
def test_design_changed_inputs_refit(kind):
    r = request(kind)
    session = DesignSession()
    session.execute(r, seed=3)
    changed = replace(r, data={**r.data, "y": 2 * r.data["y"]})
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.model_fits > 0
    full(changed, result)


def test_design_weak_iv_retains_point_diagnostics_and_previous_state():
    session = DesignSession()
    strong = request()
    session.execute(strong, seed=3)
    before = session.identities
    with pytest.raises(DesignWeakInstrument) as caught:
        session.execute(request(strength=0.001), seed=3)
    error = caught.value
    assert (error.reason_code, error.detail, error.stage) == (
        "route_not_supported",
        "recalc.iv_decision_unavailable",
        "decision",
    )
    assert error.point == pytest.approx(2.0, abs=1e-9)
    assert error.model_fits >= 2
    assert (
        error.diagnostics["uncertainty_withheld"] or error.diagnostics["anderson_rubin"] is not None
    )
    assert session.identities == before and session.is_live
    assert session.execute(strong, seed=3).receipt.totals.total == 0


@pytest.mark.parametrize("kind", ["iv", "rd", "frontdoor"])
def test_design_source_backed_replay_and_mismatch_refusals_fresh_process(kind, tmp_path):
    r = request(kind)
    session = DesignSession()
    result = session.execute(r, seed=3)
    artifact = session.export_result(seed=3)
    replay = consume_design_result(artifact, r, seed=3)
    assert replay.session.is_live
    assert replay.result.law == result.law
    assert replay.result.receipt.totals.model_fits > 0
    assert replay.session.execute(r, seed=3).receipt.totals.total == 0
    refused(
        lambda: consume_design_result(artifact, None, seed=3),
        "score_table_unavailable",
        "recalc.design_data_unavailable",
        "data_snapshot",
    )
    refused(
        lambda: consume_design_result(
            artifact, replace(r, data={**r.data, "y": 2 * r.data["y"]}), seed=3
        ),
        "invalid_argument",
        "recalc.design_artifact_mismatch",
        "score_artifact",
    )
    refused(
        lambda: consume_design_result(artifact, r, seed=4),
        "invalid_argument",
        "recalc.design_artifact_mismatch",
        "score_artifact",
    )
    if kind == "rd":
        refused(
            lambda: consume_design_result(
                artifact, replace(r, model=RdModel("r", 0, 0.75)), seed=3
            ),
            "invalid_argument",
            "recalc.design_artifact_mismatch",
            "score_artifact",
        )
    path = tmp_path / "result.artifact"
    path.write_bytes(artifact)
    source = tmp_path / "data.json"
    source.write_text(json.dumps({n: v.tolist() for n, v in r.data.items()}))
    script = """import json,sys
from antecedent.recalc import Utility
from antecedent.recalc_design import *
k=sys.argv[3]; data=json.load(open(sys.argv[2])); model={'iv':IvModel('z'),'rd':RdModel('r',0,1),'frontdoor':FrontdoorModel('m')}[k]
edges={'iv':[('z','t'),('t','y')],'rd':[('r','t'),('r','y'),('t','y')],'frontdoor':[('t','m'),('m','y')]}[k]
r=DesignRequest(data,edges,'t','y',model,Utility(1))
p=consume_design_result(open(sys.argv[1],'rb').read(),r,seed=3)
assert p.session.is_live and p.result.receipt.totals.model_fits>0
reused=p.session.execute(r,seed=3).receipt.totals.total
print(json.dumps({'point':p.result.law.ate,'fits':p.result.receipt.totals.model_fits,'reused':reused,'digest':p.artifact_digest}))
"""
    ran = subprocess.run(
        [sys.executable, "-c", script, str(path), str(source), kind],
        check=True,
        capture_output=True,
        text=True,
    )
    child = json.loads(ran.stdout)
    assert child["point"] == pytest.approx(result.law.ate, abs=1e-11)
    assert child["fits"] == replay.result.receipt.totals.model_fits
    assert child["reused"] == 0 and child["digest"] == replay.artifact_digest


@pytest.mark.parametrize("kind", ["iv", "rd", "frontdoor"])
def test_design_receipt_only_boundary_and_supplied_raw_refit(kind):
    r = request(kind)
    first = DesignSession().execute(r, seed=3)
    receipt = RecalcReceipt.consume(first.receipt.export())
    empty = DesignSession.resume(receipt, ResumeContext(portable_fit=True))
    refused(
        lambda: empty.execute(r, seed=3),
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    )
    restored = DesignSession.resume(receipt, ResumeContext(supplied_data=True))
    result = restored.execute(r, seed=3)
    assert restored.is_live and result.receipt.totals.model_fits > 0
    full(r, result)


def test_design_resource_guards_and_native_capabilities():
    from antecedent.errors import CausalValueError
    from antecedent.recalc_capabilities import (
        Family,
        Operation,
        RetainedKind,
        require_adapter,
        retained_kind,
    )
    from antecedent.recalc_design import _spec

    r = request()
    session = DesignSession()
    assert retained_kind(session) == RetainedKind.READABLE
    with pytest.raises(CausalValueError) as caught:
        session._handle.execute(
            ["z", "t", "y"], [np.zeros(1), np.zeros(100_001), np.zeros(1)], _spec(r)
        )
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(caught.value)
    session.execute(r, seed=3)
    assert retained_kind(session) == RetainedKind.LIVE_DESIGN
    assert require_adapter(Family.DESIGN, Operation.UTILITY, session).state_type is DesignSession
    restored = DesignSession.resume(session.identities, ResumeContext(portable_fit=True))
    assert retained_kind(restored) == RetainedKind.READABLE


def test_design_typed_invalid_design_refusals():
    r = request("rd")
    session = DesignSession()
    session.execute(r, seed=3)
    identities = session.identities
    refused(
        lambda: session.execute(replace(r, model=RdModel("r", 0, -1)), seed=3),
        "invalid_argument",
        "recalc.invalid_rd_window",
        "query",
    )
    assert session.identities == identities and session.is_live


@pytest.mark.parametrize("kind", ["iv", "rd", "frontdoor"])
def test_design_original_artifact_independent_consumer_fresh_process(kind, tmp_path):
    session = DesignSession()
    session.execute(request(kind), seed=3)
    path = tmp_path / "original.artifact"
    path.write_bytes(session.export_result(seed=3))
    script = """import antecedent as a,json,sys
r=a.artifacts.accept(open(sys.argv[1],'rb').read())
if sys.argv[2]=='rd': assert 'dependencies.checked_rd_operation' in r['unresolved']
print(json.dumps({'verified':r['accepts_as_verified_program'],'unresolved':r['unresolved']}))
"""
    ran = subprocess.run(
        [sys.executable, "-c", script, str(path), kind], check=True, capture_output=True, text=True
    )
    accepted = json.loads(ran.stdout)
    if kind == "rd":
        assert "dependencies.checked_rd_operation" in accepted["unresolved"]
    else:
        assert accepted == {"verified": "true", "unresolved": ""}
