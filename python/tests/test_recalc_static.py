"""Independent finite-response and enumerated multi-source reuse acceptance."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent.recalc import RecalcReceipt, RecalcRefusal, ResumeContext, Utility
from antecedent.recalc_static import (
    MultiSourceRequest,
    MultiSourceSession,
    StaticResponseRequest,
    StaticResponseSession,
)

from test_transport_multi_source import (
    catalog,
    graph,
    identified_stage,
    laws,
    query,
    source_a,
    source_b,
    truth,
)

TABLE = [
    (0, 0, 0, 504),
    (0, 0, 1, 56),
    (0, 1, 0, 56),
    (0, 1, 1, 84),
    (1, 0, 0, 30),
    (1, 0, 1, 30),
    (1, 1, 0, 24),
    (1, 1, 1, 216),
]


def response_request():
    rows = np.array([(t, m, y) for t, m, y, n in TABLE for _ in range(n)], dtype=float)
    return StaticResponseRequest(
        dict(zip(("t", "m", "y"), rows.T, strict=True)),
        ac.Admg.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")], [("t", "y")]),
        "t",
        "y",
        [0, 1],
        Utility(1),
    )


def response_truth():
    # Front-door adjustment evaluated directly from the empirical joint counts.
    joint = np.zeros((2, 2, 2))
    for t, m, y, n in TABLE:
        joint[t, m, y] = n
    result = []
    for active in range(2):
        result.append(
            sum(
                joint[active, m].sum()
                / joint[active].sum()
                * sum(
                    joint[t, m, 1] / joint[t, m].sum() * joint[t].sum() / joint.sum()
                    for t in range(2)
                )
                for m in range(2)
            )
        )
    return result


def mz_request():
    return MultiSourceRequest(
        graph(),
        query(source_a(), source_b()),
        catalog(),
        laws(),
        [{"x": 0.0}, {"x": 1.0}],
        Utility(1),
    )


def refused(call, code, detail, stage):
    with pytest.raises(RecalcRefusal) as caught:
        call()
    assert (caught.value.reason_code, caught.value.detail, caught.value.stage) == (
        code,
        detail,
        stage,
    )


def full(session_type, request, actual):
    fresh = session_type().execute(request, seed=3)
    assert actual.means == pytest.approx(fresh.means, abs=1e-11)
    assert actual.contrast == pytest.approx(fresh.contrast, abs=1e-11)
    assert actual.decision == fresh.decision
    return fresh


def test_static_response_finite_truth_counts_and_full_rerun():
    request = response_request()
    result = StaticResponseSession().execute(request, seed=3)
    assert result.means == pytest.approx(response_truth(), abs=1e-12)
    existing = ac.prepare(
        request.data,
        graph=request.graph,
        query=ac.ResponseCurve("t", "y", grid=[0.0, 1.0]),
        identifier="general.id",
        estimator="functional.effect",
        refute="none",
        bootstrap=0,
    ).estimate(request.data, seed=3)
    assert result.means == pytest.approx(np.asarray(existing.response.values).flatten(), abs=1e-12)
    counts = result.receipt.totals
    assert counts.factor_builds > 0 and counts.factor_evaluations > 0
    assert counts.program_compilations > 0 and counts.integrations > 0
    assert counts.model_fits == counts.fold_fits == 0
    assert result.to_dict()["uncertainty"] == {"status": "unavailable", "reason": "point_only"}
    full(StaticResponseSession, request, result)


def test_static_response_utility_actions_and_contrast_reuse():
    request = response_request()
    session = StaticResponseSession()
    session.execute(request, seed=3)
    assert session.execute(request, seed=3).receipt.totals.total == 0
    changed = replace(request, utility=Utility(3, 0.2))
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.total == result.receipt.totals.decisions == 1
    full(StaticResponseSession, changed, result)
    changed = replace(changed, actions=[1, 0])
    result = session.execute(changed, seed=3)
    assert result.means == pytest.approx(response_truth()[::-1])
    assert result.receipt.totals.factor_builds == result.receipt.totals.factor_evaluations == 0
    assert result.receipt.totals.law_summaries == 1
    full(StaticResponseSession, changed, result)


def test_static_response_changed_graph_data_support_invalidate():
    request = response_request()
    session = StaticResponseSession()
    session.execute(request, seed=3)
    changed = replace(request, data={**request.data, "y": 1 - request.data["y"]})
    result = session.execute(changed, seed=3)
    assert result.means == pytest.approx(1 - np.asarray(response_truth()))
    assert result.receipt.totals.factor_builds > 0
    full(StaticResponseSession, changed, result)
    changed = replace(
        changed, graph=ac.Admg.from_edges(["t", "m", "y"], [("m", "y")], [("t", "y")])
    )
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.identifications > 0
    full(StaticResponseSession, changed, result)
    expanded = {name: np.r_[values, values[:100]] for name, values in changed.data.items()}
    expanded["t"][-100:] = 2
    changed = replace(changed, data=expanded)
    result = session.execute(changed, seed=3)
    full(StaticResponseSession, changed, result)
    changed = replace(changed, support=[0, 1, 2], actions=[0, 1, 2], active=2)
    result = session.execute(changed, seed=3)
    full(StaticResponseSession, changed, result)


def test_static_response_refusals_preserve_native_state():
    request = response_request()
    session = StaticResponseSession()
    session.execute(request, seed=3)
    identities = session.identities
    refused(
        lambda: session.execute(replace(request, actions=[0, 2]), seed=3),
        "route_not_supported",
        "recalc.static_action_out_of_support",
        "treatment_grid",
    )
    bad_graph = ac.Admg.from_edges(["t", "m", "y"], [("t", "y")], [("t", "y")])
    refused(
        lambda: session.execute(replace(request, graph=bad_graph), seed=3),
        "effect_not_identified",
        "recalc.static_not_identified",
        "identification",
    )
    refused(
        lambda: session.execute(replace(request, support=[1, 0]), seed=3),
        "invalid_argument",
        "recalc.static_invalid_query",
        "query",
    )
    refused(
        lambda: session.execute(replace(request, support=[0], actions=[0], active=0), seed=3),
        "invalid_argument",
        "recalc.static_invalid_query",
        "query",
    )
    dag = ac.Admg.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")], [])
    refused(
        lambda: session.execute(replace(request, graph=dag), seed=3),
        "route_not_supported",
        "recalc.static_graph_unsupported",
        "graph",
    )
    assert session.identities == identities and session.is_live
    assert session.execute(request, seed=3).receipt.totals.total == 0


def test_multi_source_scm_truth_counts_and_full_rerun():
    request = mz_request()
    result = MultiSourceSession().execute(request, seed=3)
    assert result.means == pytest.approx([truth(0), truth(1)], abs=1e-11)
    existing = json.loads(
        identified_stage().prepare_exact(request.laws, request.assignments).estimate()
    )
    assert result.means == pytest.approx(
        [entry["means"]["y"] for entry in existing["requests"]], abs=1e-11
    )
    counts = result.receipt.totals
    assert counts.provider_bindings > 0 and counts.provider_calls > 0
    assert counts.factor_evaluations > 0 and counts.integrations > 0
    assert counts.model_fits == counts.fold_fits == 0
    full(MultiSourceSession, request, result)


def test_multi_source_utility_and_compatible_request_reuse():
    request = mz_request()
    session = MultiSourceSession()
    session.execute(request, seed=3)
    assert session.execute(request, seed=3).receipt.totals.total == 0
    changed = replace(request, utility=Utility(3, 0.2))
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.total == result.receipt.totals.decisions == 1
    full(MultiSourceSession, changed, result)
    changed = replace(changed, assignments=[{"x": 1.0}, {"x": 0.0}])
    result = session.execute(changed, seed=3)
    assert result.means == pytest.approx([truth(1), truth(0)])
    assert result.receipt.totals.factor_evaluations == 0
    assert result.receipt.totals.provider_calls > 0
    full(MultiSourceSession, changed, result)


def test_multi_source_changed_law_catalog_regime_source_invalidate():
    request = mz_request()
    session = MultiSourceSession()
    session.execute(request, seed=3)
    changed = replace(
        request, catalog=catalog(snapshot_suffix="new"), laws=laws(snapshot_suffix="new")
    )
    result = session.execute(changed, seed=3)
    assert result.receipt.totals.provider_bindings > 0
    full(MultiSourceSession, changed, result)
    complemented = [
        replace(
            law,
            probabilities=tuple(law.probabilities[i ^ 1] for i in range(len(law.probabilities))),
        )
        for law in changed.laws
    ]
    changed = replace(changed, laws=complemented)
    result = session.execute(changed, seed=3)
    assert result.means == pytest.approx([1 - truth(0), 1 - truth(1)], abs=1e-11)
    full(MultiSourceSession, changed, result)


def test_multi_source_exact_missing_provider_support_refusals():
    request = mz_request()
    session = MultiSourceSession()
    session.execute(request, seed=3)
    identities = session.identities
    refused(
        lambda: session.execute(replace(request, laws=request.laws[:-1]), seed=3),
        "transport_missing_provider",
        "recalc.static_transport_missing_provider",
        "score_artifact",
    )
    refused(
        lambda: session.execute(replace(request, assignments=[{"x": 0}, {"x": 2}]), seed=3),
        "route_not_supported",
        "recalc.off_grid_request",
        "treatment_grid",
    )
    missing = replace(request, catalog=catalog(omit=("b_z1_0",)), laws=laws(omit=("b_z1_0",)))
    refused(
        lambda: session.execute(missing, seed=3),
        "transport_missing_evidence",
        "recalc.static_transport_missing_evidence",
        "identification",
    )
    assert session.identities == identities
    assert session.execute(request, seed=3).receipt.totals.total == 0


@pytest.mark.parametrize("kind", ["response", "mz"])
def test_static_original_artifact_fresh_process(kind, tmp_path):
    session = StaticResponseSession() if kind == "response" else MultiSourceSession()
    session.execute(response_request() if kind == "response" else mz_request(), seed=3)
    path = tmp_path / "result.artifact"
    path.write_bytes(session.export_result(seed=3))
    if kind == "response":
        script = "import antecedent as a,json,sys; r=a.artifacts.accept(open(sys.argv[1],'rb').read()); print(json.dumps({'verified':r['accepts_as_verified_program'],'unresolved':r['unresolved']}))"
    else:
        script = "import json,sys; from antecedent.transport.advanced import consume_multi_source_z_transport_artifact as c; r=json.loads(c(open(sys.argv[1],'rb').read())); print(json.dumps([entry['means']['y'] for entry in r['requests']]))"
    ran = subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, capture_output=True, text=True
    )
    consumed = json.loads(ran.stdout)
    if kind == "response":
        assert consumed == {"verified": "true", "unresolved": ""}
    else:
        assert consumed == pytest.approx([truth(0), truth(1)], abs=1e-11)


@pytest.mark.parametrize("kind", ["response", "mz"])
def test_static_receipt_only_fresh_refit_boundary(kind, tmp_path):
    cls = StaticResponseSession if kind == "response" else MultiSourceSession
    request = response_request() if kind == "response" else mz_request()
    first = cls().execute(request, seed=3)
    receipt = RecalcReceipt.consume(first.receipt.export())
    empty = cls.resume(receipt, ResumeContext(portable_fit=True, portable_scores=True))
    assert not empty.is_live
    refused(
        lambda: empty.execute(request, seed=3),
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    )
    restored = cls.resume(receipt, ResumeContext(supplied_data=True))
    result = restored.execute(request, seed=3)
    assert restored.is_live and result.receipt.totals.total > 0
    full(cls, request, result)
    path = tmp_path / "receipt.artifact"
    path.write_bytes(receipt.export())
    script = """import json,sys
sys.path.insert(0,sys.argv[3])
from test_recalc_static import response_request,mz_request
from antecedent.recalc import RecalcReceipt,RecalcRefusal,ResumeContext
from antecedent.recalc_static import StaticResponseSession,MultiSourceSession
cls=StaticResponseSession if sys.argv[2]=='response' else MultiSourceSession
r=response_request() if sys.argv[2]=='response' else mz_request()
a=RecalcReceipt.consume(open(sys.argv[1],'rb').read())
s=cls.resume(a,ResumeContext(portable_fit=True,portable_scores=True,supplied_provider=True,scores_snapshot_bound=True))
try:s.execute(r,seed=3)
except RecalcRefusal as e:refusal=[e.reason_code,e.detail,e.stage]
else:raise AssertionError('historical receipt recreated executable state')
s=cls.resume(a,ResumeContext(supplied_data=True));ran=s.execute(r,seed=3)
print(json.dumps({'refusal':refusal,'means':ran.means,'work':ran.receipt.totals.total,'live':s.is_live}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), kind, str(Path(__file__).resolve().parent)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(child.stdout)
    assert observed["refusal"] == [
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    ]
    assert observed["means"] == pytest.approx(result.means, abs=1e-11)
    assert observed["live"] and observed["work"] == result.receipt.totals.total


def test_static_binding_resource_guards():
    from antecedent.errors import CausalValueError
    from antecedent.recalc_static import _response_spec, _transport_spec

    r = response_request()
    session = StaticResponseSession()
    # A short first column must not hide a later oversized column.
    with pytest.raises(CausalValueError) as caught:
        session._handle.execute(
            ["t", "m", "y"],
            [np.zeros(1), np.zeros(100_001), np.zeros(1)],
            r.graph,
            _response_spec(r),
        )
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(caught.value)
    mz = mz_request()

    class IndexedOnly(list):
        def __iter__(self):
            raise AssertionError("native bounded sequence must bypass custom iterators")

    raw = IndexedOnly(mz.laws)
    # The actual bounded sequence, not an arbitrary iterator, supplies laws.
    result, artifact, error = MultiSourceSession()._handle.execute(
        mz.graph, mz.catalog, raw, _transport_spec(mz)
    )
    assert result is not None and artifact is not None and error is None
    oversized = replace(mz, laws=[replace(mz.laws[0], probabilities=(0.0,) * 1_000_001)])
    with pytest.raises(CausalValueError) as caught:
        MultiSourceSession()._handle.execute(
            oversized.graph, oversized.catalog, oversized.laws, _transport_spec(oversized)
        )
    assert caught.value.reason_code == "invalid_argument"
    assert "recalc.limits_exceeded" in str(caught.value)


def test_static_capabilities_require_actual_native_programs():
    from antecedent.recalc_capabilities import (
        Family,
        Operation,
        RetainedKind,
        require_adapter,
        retained_kind,
    )

    for cls, request in [
        (StaticResponseSession, response_request()),
        (MultiSourceSession, mz_request()),
    ]:
        session = cls()
        assert retained_kind(session) == RetainedKind.READABLE
        session.execute(request, seed=3)
        assert retained_kind(session) == RetainedKind.LIVE_PROGRAM
        assert require_adapter(Family.STATIC, Operation.UTILITY, session).state_type is cls
        loaded = cls.resume(session.identities, ResumeContext(portable_fit=True))
        assert retained_kind(loaded) == RetainedKind.READABLE


def test_static_factor_domain_budget_refuses_before_cartesian_allocation():
    names = ["t", "m", "z", "y"]
    request = StaticResponseRequest(
        {n: np.arange(64, dtype=float) for n in names},
        ac.Admg.from_edges(
            names, [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]], [("m", "z")]
        ),
        "t",
        "y",
        [0.0, 1.0],
        Utility(1),
    )
    session = StaticResponseSession()
    refused(
        lambda: session.execute(request, seed=3),
        "invalid_argument",
        "recalc.static_invalid_factor",
        "score_artifact",
    )
    assert not session.is_live


def test_static_retained_response_issues_native_authority_without_new_execution():
    from antecedent import external, program_claims

    session = StaticResponseSession()
    assert session.response is None
    request = response_request()
    first = session.execute(request, seed=3)
    response = session.response
    program = program_claims.ProgramBinding.from_response(
        response, outcome_units="probability", dose_units="binary"
    )
    claim = program_claims.native_claim(response, program)
    assert claim.means == pytest.approx(first.means, abs=1e-12)
    assert claim.trust.value == "native_licensed"
    spec = external.response(
        response.program_identification, outcome_units="probability", dose_units="binary"
    )
    assert program_claims.ProgramBinding.from_spec(spec) == program
    assert session.execute(request, seed=3).receipt.totals.total == 0
    utility = session.execute(replace(request, utility=Utility(3, 0.1)), seed=3)
    assert utility.receipt.totals.total == utility.receipt.totals.decisions == 1
    assert program_claims.native_claim(session.response, program).means == claim.means


def test_static_retained_response_snapshot_substitution_and_stale_projection_refuse():
    from antecedent import program_claims

    request = response_request()
    session = StaticResponseSession()
    session.execute(request, seed=3)
    previous = session.response
    program = program_claims.ProgramBinding.from_response(
        previous, outcome_units="probability", dose_units="binary"
    )
    changed = replace(request, data={**request.data, "y": 1 - request.data["y"]})
    session.execute(changed, seed=3)
    current = session.response
    assert current.data_snapshot_id != previous.data_snapshot_id
    assert (
        program_claims.ProgramBinding.from_response(
            current, outcome_units="probability", dose_units="binary"
        )
        == program
    )
    swapped = previous.model_copy()
    object.__setattr__(swapped, "_raw", current._raw)
    with pytest.raises(ac.external.ExternalRefusal) as error:
        program_claims.native_claim(swapped, program)
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "native_claims.projection_mismatch",
        "bind",
    )
    assert session.execute(changed, seed=3).receipt.totals.total == 0
