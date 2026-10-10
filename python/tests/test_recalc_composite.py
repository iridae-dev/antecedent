"""Independent finite front-door SCM and executing two-provider shared workflow."""

import os
import subprocess
import sys
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent import decision, external
from antecedent.design import StructuralCandidate, rank_structural
from antecedent.program_claims import ProgramBinding
from antecedent.recalc import RecalcReceipt, Utility
from antecedent.recalc_composite import (
    CompositeRequest,
    CompositeSession,
    ConditionalStudyPolicy,
    ConditionalStudyRanking,
    ConditionalStudyRankingRefusal,
)
from antecedent.recalc_external import (
    CallbackDescriptor,
    CallbackProvider,
    ExternalCallbackRefusal,
    ExternalCallbackRequest,
    ExternalCallbackSession,
)
from antecedent.recalc_static import StaticResponseRequest, StaticResponseSession


def fixture(*, selected_providers=False):
    rows = [
        (t, int(row < ones), 2.0 + 4.0 * int(row < ones))
        for t, ones in enumerate((20, 50, 80))
        for row in range(100)
    ]
    data = dict(zip(("t", "m", "y"), np.asarray(rows, dtype=float).T, strict=True))
    native_request = StaticResponseRequest(
        data,
        ac.Admg.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")], [("t", "y")]),
        "t",
        "y",
        [0, 1, 2],
        Utility(1),
    )
    native = StaticResponseSession()
    native.execute(native_request, seed=7)
    response = native.response
    binding = ProgramBinding.from_response(
        response, outcome_units="dimensionless", dose_units="dimensionless"
    )
    spec = external.response(
        response.program_identification, outcome_units="dimensionless", dose_units="dimensionless"
    )
    requests, providers, calls = [], {}, [[], []]
    for branch in range(2):
        obj = external.ProviderObject(
            f"provider-{branch}",
            "linear",
            "v1",
            f"source-{branch}",
            binding.identity,
            "interventional_predictive",
            ("mean",),
        )
        descriptor = CallbackDescriptor(obj, "numpy-lstsq-v1", "deterministic")
        t = np.tile([0.0, 1.0, 2.0], 10)
        requests.append(
            ExternalCallbackRequest(
                binding, spec, descriptor, {"t": t, "y": 3.0 + 0.5 * t}, branch=branch
            )
        )

        def callback(inputs, *, obj=obj, branch=branch):
            calls[branch].append(inputs)
            X = np.column_stack([np.ones(len(inputs.data["t"])), inputs.data["t"]])
            beta = np.linalg.lstsq(X, inputs.data["y"], rcond=None)[0]
            values = [float(beta @ [1.0, dose]) for dose in inputs.doses]
            return external.Response(
                obj,
                values,
                attested_by="test-lab",
                support=tuple(
                    "supported"
                    if not selected_providers or index == branch + 1
                    else "outside_empirical_support"
                    for index in range(3)
                ),
            )

        providers[branch] = CallbackProvider(callback, descriptor)
    quantities = (
        binding.expected_quantities if hasattr(binding, "expected_quantities") else spec.quantities
    )
    contract = decision.Contract(
        tuple(
            decision.Action(name, (q,), decision.x(0) - cost)
            for name, q, cost in zip(
                ("wait", "treat", "extend"), quantities, (0, 1, 3), strict=True
            )
        ),
        "net benefit",
        decision.Criterion.expected_utility(),
        "target",
    )
    request = CompositeRequest(
        native_request,
        binding,
        tuple(requests),
        contract,
        input_order=("external.0", "external.1", "native")
        if selected_providers
        else ("native", "external.0", "external.1"),
    )
    return native, CompositeSession(native, native_request), request, providers, calls


def test_shared_receipt_native_truth_utility_and_one_foreign_branch_full_rerun():
    native, session, request, providers, calls = fixture()
    session.plan(request)
    assert calls == [[], []]
    out = session.execute(request, providers=providers)
    assert [row.expected_utility for row in out.decision.outcomes] == pytest.approx([2.8, 3.0, 2.2])
    assert out.decision.verdict.selected == "treat"
    assert all(row.input_id == "native" for row in out.decision.dispositions)
    assert out.receipt.totals.external_invocations == 2
    assert out.receipt.totals.factor_builds == out.receipt.totals.identifications == 0
    assert out.receipt.totals.decisions == 1
    same = session.execute(request, providers={})
    assert same.receipt.totals.total == 0
    contract = replace(
        request.decision_contract,
        actions=(
            request.decision_contract.actions[0],
            replace(request.decision_contract.actions[1], utility=decision.x(0) - 2),
            request.decision_contract.actions[2],
        ),
    )
    utility = session.execute(replace(request, decision_contract=contract), providers={})
    assert utility.decision.verdict.selected == "wait"
    assert (
        utility.receipt.totals.external_invocations == 0 and utility.receipt.totals.decisions == 1
    )
    foreign = replace(
        request.externals[0],
        data={
            "t": request.externals[0].data["t"],
            "y": np.asarray(request.externals[0].data["y"]) + 7,
        },
    )
    changed = replace(request, externals=(foreign, request.externals[1]))
    update = session.execute(changed, providers=providers)
    assert update.receipt.totals.external_invocations == 1
    assert update.receipt.totals.decisions == 1
    assert [len(x) for x in calls] == [2, 1]
    fresh = CompositeSession(native, request.native).execute(changed, providers=providers)
    assert fresh.decision == update.decision
    assert fresh.receipt.totals.external_invocations == 2
    assert RecalcReceipt.consume(update.receipt.export()).totals.external_invocations == 1
    source = session.native_response
    assert source.program_identification.status == native.response.program_identification.status
    # Both callable outputs remain original attested envelopes and replay through actual providers.
    replay = ExternalCallbackSession.resume(session.export_callback(0), foreign).execute(
        foreign, provider=providers[0]
    )
    np.testing.assert_allclose(replay.claim.values, [10, 10.5, 11], atol=1e-12)
    assert replay.receipt.totals.external_invocations == 1


def test_native_data_update_full_rerun_without_foreign_calls():
    native, session, request, providers, calls = fixture()
    session.execute(request, providers=providers)
    changed_native = replace(
        request.native, data={**request.native.data, "y": np.asarray(request.native.data["y"]) + 1}
    )
    updated = session.execute(replace(request, native=changed_native), providers={})
    assert updated.receipt.totals.factor_builds > 0 and updated.receipt.totals.integrations > 0
    assert updated.receipt.totals.external_invocations == 0 and [len(x) for x in calls] == [1, 1]
    assert [row.expected_utility for row in updated.decision.outcomes] == pytest.approx(
        [3.8, 4.0, 3.2]
    )
    rerun_native = StaticResponseSession()
    rerun_native.execute(changed_native)
    rerun = CompositeSession(rerun_native, changed_native).execute(
        replace(request, native=changed_native), providers=providers
    )
    assert rerun.decision == updated.decision
    assert session.native_response.data_snapshot_id != native.response.data_snapshot_id


def test_failed_second_provider_reports_actual_branch_and_preserves_published_state():
    _, session, request, providers, calls = fixture()
    before = session.execute(request, providers=providers)
    changed = replace(
        request,
        externals=(
            request.externals[0],
            replace(
                request.externals[1],
                data={
                    "t": request.externals[1].data["t"],
                    "y": np.asarray(request.externals[1].data["y"]) + 1,
                },
            ),
        ),
    )

    def failed(_):
        raise ValueError("second provider failed")

    bad = CallbackProvider(failed, request.externals[1].descriptor)
    with pytest.raises(ExternalCallbackRefusal) as refused:
        session.execute(changed, providers={1: bad})
    assert refused.value.stage == "provider_request.1"
    assert refused.value.attempt["invocations"] == 1
    after = session.execute(request, providers={})
    assert after.receipt.totals.total == 0 and after.decision == before.decision


def test_shared_receipt_fresh_process_is_history_not_native_or_provider_state(tmp_path):
    _, session, request, providers, _ = fixture()
    out = session.execute(request, providers=providers)
    path = tmp_path / "receipt.ant"
    path.write_bytes(out.receipt.export())
    code = "from antecedent.recalc import RecalcReceipt; from pathlib import Path; import sys; r=RecalcReceipt.consume(Path(sys.argv[1]).read_bytes()); assert r.totals.external_invocations==2; assert r.totals.factor_builds==0; assert r.totals.decisions==1"
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert child.returncode == 0, child.stderr


def test_fresh_process_explicit_raw_native_and_supplied_providers_revalidate_and_execute(tmp_path):
    _, session, request, providers, _ = fixture()
    out = session.execute(request, providers=providers)
    (tmp_path / "native.ant").write_bytes(session.export_native())
    (tmp_path / "callback0.ant").write_bytes(session.export_callback(0))
    (tmp_path / "callback1.ant").write_bytes(session.export_callback(1))
    (tmp_path / "receipt.ant").write_bytes(out.receipt.export())
    script = """
from pathlib import Path
import sys
import antecedent as ac
from test_recalc_composite import fixture
from antecedent.recalc import RecalcReceipt
from antecedent.recalc_static import StaticResponseSession
from antecedent.recalc_composite import CompositeSession
from antecedent.recalc_external import ExternalCallbackSession
p=Path(sys.argv[1])
assert ac.artifacts.accept((p/'native.ant').read_bytes())['accepts_as_verified_program']=='true'
assert RecalcReceipt.consume((p/'receipt.ant').read_bytes()).totals.external_invocations==2
# Explicit executable recipe/raw requests and actual callables are supplied in this fresh process.
_, _, req, providers, calls=fixture()
native=StaticResponseSession()
produced=native.execute(req.native,seed=7)
assert produced.receipt.totals.factor_builds>0 and produced.receipt.totals.identifications>0
for branch in (0,1):
    replay=ExternalCallbackSession.resume((p/f'callback{branch}.ant').read_bytes(),req.externals[branch])
    assert not replay.is_live
    bound=replay.execute(req.externals[branch],provider=providers[branch])
    assert bound.receipt.totals.external_invocations==1
    assert all(abs(a-b)<1e-12 for a,b in zip(bound.claim.values,[3,3.5,4]))
actual=CompositeSession(native,req.native).execute(req,providers=providers)
assert actual.receipt.totals.external_invocations==2 and actual.receipt.totals.factor_builds==0
assert actual.decision.verdict.selected=='treat'
assert all(abs(a.expected_utility-b)<1e-12 for a,b in zip(actual.decision.outcomes,[2.8,3,2.2]))
assert [len(x) for x in calls]==[2,2]
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(__import__("pathlib").Path(__file__).parent), *sys.path]
        ),
    }
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env
    )
    assert child.returncode == 0, child.stderr


def test_all_three_sources_selected_and_each_numeric_update_is_independent():
    native, session, req, providers, calls = fixture(selected_providers=True)
    initial = session.execute(req, providers=providers)
    assert [x.input_id for x in initial.decision.dispositions] == [
        "native",
        "external.0",
        "external.1",
    ]
    assert [x.expected_utility for x in initial.decision.outcomes] == pytest.approx([2.8, 2.5, 1.0])
    assert initial.decision.verdict.selected == "wait"
    # Declared ordering is terminal policy, with no scientific source recomputation.
    ordered = session.execute(
        replace(req, input_order=("native", "external.0", "external.1")), providers={}
    )
    assert ordered.receipt.totals.external_invocations == ordered.receipt.totals.factor_builds == 0
    assert ordered.receipt.totals.decisions == 1
    assert ordered.decision.verdict.selected == "treat"
    session.execute(req, providers={})
    for branch, chosen in ((0, "treat"), (1, "extend")):
        changed_external = replace(
            req.externals[branch],
            data={
                **req.externals[branch].data,
                "y": np.asarray(req.externals[branch].data["y"]) + 8,
            },
        )
        requests = list(req.externals)
        requests[branch] = changed_external
        changed = replace(req, externals=tuple(requests))
        update = session.execute(changed, providers=providers)
        expected = [2.8, 2.5, 1.0]
        expected[branch + 1] += 8
        assert [x.expected_utility for x in update.decision.outcomes] == pytest.approx(expected)
        assert update.decision.verdict.selected == chosen
        assert update.receipt.totals.external_invocations == 1
        assert update.receipt.totals.factor_builds == 0
        fresh = CompositeSession(native, req.native).execute(changed, providers=providers)
        assert fresh.decision == update.decision
        session.execute(req, providers=providers)
    changed_native = replace(
        req.native, data={**req.native.data, "y": np.asarray(req.native.data["y"]) - 8}
    )
    changed = replace(req, native=changed_native)
    update = session.execute(changed, providers={})
    assert update.receipt.totals.external_invocations == 0
    assert update.receipt.totals.factor_builds > 0
    assert [x.expected_utility for x in update.decision.outcomes] == pytest.approx([-5.2, 2.5, 1.0])
    assert update.decision.verdict.selected == "treat"
    rerun_native = StaticResponseSession()
    rerun_native.execute(changed_native)
    assert (
        CompositeSession(rerun_native, changed_native)
        .execute(changed, providers=providers)
        .decision
        == update.decision
    )
    with pytest.raises(ExternalCallbackRefusal):
        session.plan(replace(changed, input_order=("native", "native", "external.1")))


def test_selected_provider_fresh_child_scientifically_replays_actual_sources(tmp_path):
    _, session, req, providers, _ = fixture(selected_providers=True)
    out = session.execute(req, providers=providers)
    (tmp_path / "native.ant").write_bytes(session.export_native())
    for branch in (0, 1):
        (tmp_path / f"callback{branch}.ant").write_bytes(session.export_callback(branch))
    (tmp_path / "receipt.ant").write_bytes(out.receipt.export())
    script = """
from pathlib import Path
import sys
import antecedent as ac
from test_recalc_composite import fixture
from antecedent.recalc_static import StaticResponseSession
from antecedent.recalc_composite import CompositeSession
from antecedent.recalc_external import ExternalCallbackSession
p=Path(sys.argv[1])
assert ac.artifacts.accept((p/'native.ant').read_bytes())['accepts_as_verified_program']=='true'
_,_,req,providers,calls=fixture(selected_providers=True)
native=StaticResponseSession()
fresh=native.execute(req.native,seed=7)
assert fresh.receipt.totals.factor_builds>0
for branch in (0,1):
    replay=ExternalCallbackSession.resume((p/f'callback{branch}.ant').read_bytes(),req.externals[branch])
    result=replay.execute(req.externals[branch],provider=providers[branch])
    assert result.receipt.totals.external_invocations==1
out=CompositeSession(native,req.native).execute(req,providers=providers)
assert out.receipt.totals.external_invocations==2
assert [x.input_id for x in out.decision.dispositions]==['native','external.0','external.1']
assert all(abs(x.expected_utility-y)<1e-12 for x,y in zip(out.decision.outcomes,[2.8,2.5,1]))
assert out.decision.verdict.selected=='wait'
assert [len(x) for x in calls]==[2,2]
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(__import__("pathlib").Path(__file__).parent), *sys.path]
        ),
    }
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env
    )
    assert child.returncode == 0, child.stderr


def conditional_policy():
    return ConditionalStudyPolicy(
        "conditional-study-rule-v1",
        {
            action: (
                StructuralCandidate(f"{action}-insufficient-cheap", False, 0, 1),
                StructuralCandidate(f"{action}-sufficient-expensive", True, 20, 100),
                StructuralCandidate(f"{action}-sufficient-cheap", True, 10, 200),
            )
            for action in ("wait", "treat", "extend")
        },
    )


def test_actual_selected_decision_routes_original_structural_ranking_and_source_updates():
    _, session, req, providers, calls = fixture(selected_providers=True)
    policy = conditional_policy()
    with pytest.raises(ConditionalStudyRankingRefusal, match="source_unavailable"):
        session.rank_studies(policy)
    initial = session.execute(req, providers=providers)
    assert [x.input_id for x in initial.decision.dispositions] == [
        "native",
        "external.0",
        "external.1",
    ]
    ranked = session.rank_studies(policy)
    assert ranked.selected_action == "wait"
    assert ranked.entries == rank_structural(policy.branches["wait"]).entries
    assert [x.id for x in ranked.entries] == [
        "wait-sufficient-cheap",
        "wait-sufficient-expensive",
        "wait-insufficient-cheap",
    ]
    assert [len(x) for x in calls] == [1, 1]
    assert ranked.source["input_order"] == ["external.0", "external.1", "native"]
    assert len(ranked.source["callback_claims"]) == 2
    assert (
        ConditionalStudyRanking.consume(
            ranked.export(), session, expected_identity=ranked.identity
        ).entries
        == ranked.entries
    )
    # The original declared verdict remains caller owned, and the complete policy is bound.
    with pytest.raises(ConditionalStudyRankingRefusal, match="artifact_invalid"):
        ConditionalStudyRanking.consume(ranked.export()[:-1], session)
    with pytest.raises(ConditionalStudyRankingRefusal, match="artifact_invalid"):
        ConditionalStudyRanking.consume(ranked.export(), session, expected_identity="wrong")
    with pytest.raises(ConditionalStudyRankingRefusal, match="resource_limit"):
        ConditionalStudyRanking.consume(b" " * (4 * 1024 * 1024 + 1), session)
    changed_policy = replace(policy, policy_id="conditional-study-rule-v2")
    assert session.rank_studies(changed_policy).identity != ranked.identity
    for branch, action in ((0, "treat"), (1, "extend")):
        updated = list(req.externals)
        updated[branch] = replace(
            updated[branch], data={**updated[branch].data, "y": updated[branch].data["y"] + 8}
        )
        result = session.execute(replace(req, externals=tuple(updated)), providers=providers)
        assert result.receipt.totals.external_invocations == 1
        assert result.receipt.totals.factor_builds == 0
        projection = session.rank_studies(policy)
        assert projection.selected_action == action
        assert projection.entries == rank_structural(policy.branches[action]).entries
        assert projection.identity != ranked.identity
        with pytest.raises(ConditionalStudyRankingRefusal, match="source_mismatch"):
            ConditionalStudyRanking.consume(ranked.export(), session)
        session.execute(req, providers=providers)
    updated_native = replace(req.native, data={**req.native.data, "y": req.native.data["y"] - 8})
    result = session.execute(replace(req, native=updated_native), providers={})
    assert result.receipt.totals.external_invocations == 0
    assert result.receipt.totals.factor_builds > 0
    assert session.rank_studies(policy).selected_action == "treat"
    count = [len(x) for x in calls]
    # Declared resource bounds reject before any provider or native work.
    for oversized in (
        replace(policy, policy_id="x" * 257),
        replace(policy, branches={f"a{i}": policy.branches["wait"] for i in range(65)}),
        replace(policy, branches={"wait": policy.branches["wait"] * 43}),
    ):
        with pytest.raises(ac.errors.CausalValueError):
            session.rank_studies(oversized)
    with pytest.raises(ac.errors.CausalValueError):
        session._handle.rank_conditional_studies(" " * (256 * 1024 + 1))
    with pytest.raises(ConditionalStudyRankingRefusal, match="invalid_policy"):
        session.rank_studies(replace(policy, branches={"treat": policy.branches["treat"]}))
    assert [len(x) for x in calls] == count


def test_conditional_ranking_fresh_child_reexecutes_science_before_independent_consumption(
    tmp_path,
):
    _, session, req, providers, _ = fixture(selected_providers=True)
    session.execute(req, providers=providers)
    ranked = session.rank_studies(conditional_policy())
    (tmp_path / "conditional-ranking.json").write_bytes(ranked.export())
    script = """
from pathlib import Path
import sys
from test_recalc_composite import fixture, conditional_policy
from antecedent.recalc_composite import ConditionalStudyRanking, ConditionalStudyRankingRefusal
p=Path(sys.argv[1]); artifact=(p/'conditional-ranking.json').read_bytes()
_,session,req,providers,calls=fixture(selected_providers=True)
try:
    ConditionalStudyRanking.consume(artifact,session)
except ConditionalStudyRankingRefusal as e:
    assert e.detail=='conditional_study_ranking.source_unavailable'
else:
    raise AssertionError('artifact must not create an executing source')
out=session.execute(req,providers=providers)
assert out.receipt.totals.external_invocations==2
assert [len(x) for x in calls]==[1,1]
assert [x.input_id for x in out.decision.dispositions]==['native','external.0','external.1']
assert all(abs(x.expected_utility-y)<1e-12 for x,y in zip(out.decision.outcomes,[2.8,2.5,1]))
ranked=ConditionalStudyRanking.consume(artifact,session,expected_identity=sys.argv[2])
assert ranked.selected_action=='wait'
assert [x.id for x in ranked.entries]==['wait-sufficient-cheap','wait-sufficient-expensive','wait-insufficient-cheap']
assert ranked.identity==session.rank_studies(conditional_policy()).identity
assert [len(x) for x in calls]==[1,1]
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(__import__("pathlib").Path(__file__).parent), *sys.path]
        ),
    }
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), ranked.identity],
        capture_output=True,
        text=True,
        env=env,
    )
    assert child.returncode == 0, child.stderr
