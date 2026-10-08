"""A0 remainders (Python half): the full lifecycle, lineage of the study-ranking provider and
of sensitivity inputs, refusal families across the bridge, and cross-surface parity.

Oracles are written by hand and mirror ``crates/antecedent-design/tests/a0r_parity.rs``:

* closed-form response ``E[Y | do(a)] = 1 + 2a`` on the grid ``0, 1, 2`` (values 1, 3, 5), the
  utility ``2 * mean - 1`` (EU(wait) = 1, EU(treat) = 9);
* the frozen F14 guess decision: binary state, prior 1/2, signals of accuracy 3/4 and 5/8 have
  ``EVSI = 1/4`` and ``1/8``, ``EVPI = 1/2``, and with an identical utility-unit cost 1/10 the net
  values are ``3/20`` and ``1/40``;
* the invariant sensitivity surface: A is ``5 - gamma`` (5, 4, 3) against B = 1.

Lineage ids are the literal strings the Rust ``provenance_chain`` writes; the digests of both
surfaces come from the same native chain function, so equal rows mean equal digests.

Fixture hooks (skipped until the files exist, written only on request):
``A0R_WRITE_FIXTURES=1 python -m pytest python/tests/test_a0r_lifecycle.py -k
a0r_regenerate_python_fixtures`` writes ``conformance/cross_surface/py_a0r_*``; the Rust twin is
``ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test a0r_parity -- --ignored
a0r_regenerate_rust_fixtures``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import antecedent as ac
import pytest
from antecedent import _native, decision, external, repair
from antecedent import design as dr
from antecedent import inverse_query as iq
from antecedent import sensitivity_decision as sd
from antecedent.errors import CausalUnsupportedError
from antecedent.joint_distribution import ScientificQuantity

import test_inverse_query as inverse_fixtures
from _refusal import assert_registered_refusal
from test_design_ranking import GUESS, UTILITY_MAP, _guess_candidate, _quantity, _signal
from test_repair import backdoor

_E = TypeVar("_E", bound=BaseException)
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "conformance" / "cross_surface"
REGENERATE_PY = (
    "A0R_WRITE_FIXTURES=1 python -m pytest python/tests/test_a0r_lifecycle.py "
    "-k a0r_regenerate_python_fixtures"
)
REGENERATE_RUST = (
    "ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test a0r_parity "
    "-- --ignored a0r_regenerate_rust_fixtures"
)

GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]
REFUSAL_DETAIL = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")


# ------------------------------------------------------------------- the lifecycle example


def _spec() -> external.ExternalSpec:
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))
    return external.response(
        ident,
        outcome_units="mmHg",
        population="target",
        require_evidence=("factor:z",),
        require_assumptions=("ignorability",),
    )


def _claim(spec: external.ExternalSpec) -> external.BoundExternalClaim:
    provider = external.ProviderObject(
        provider_id="lab",
        object_id="curve",
        version="v3",
        snapshot="snap-9",
        request="req-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )
    return spec.bind(
        external.Response(
            provider=provider,
            values=[1.0, 3.0, 5.0],
            evidence=("factor:z",),
            assumptions=("ignorability",),
            attested_by="lab",
        )
    )


def _contract(claim: external.BoundExternalClaim) -> decision.Contract:
    wait, _, treat = claim.quantities
    utility = decision.x(0) * 2.0 - 1.0
    return decision.Contract(
        actions=(
            decision.Action("wait", inputs=(wait,), utility=utility),
            decision.Action("treat", inputs=(treat,), utility=utility),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _study_candidates() -> list[dr.Candidate]:
    return [
        dr.Candidate(
            cid,
            1,
            dr.ExternalSignal(
                "lab",
                f"signal-{cid}",
                "v1",
                "snap",
                "lab-qa",
                dr.ExternalLaw.posterior(
                    states=[0.0, 1.0],
                    statistics=[0.0, 1.0],
                    predictive=[0.5, 0.5],
                    posterior=[[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]],
                ),
            ),
            cost=0.1,
            cost_unit="utility",
        )
        for cid, accuracy in (("cand-1", 0.75), ("cand-2", 0.625))
    ]


def test_a0r_lifecycle_identify_bind_inspect_decide_and_rank_a_study():
    # compile / identify -> bind an external object.
    claim = _claim(_spec())
    inspection = claim.inspect()
    assert inspection.native is False
    assert [link.id for link in inspection.lineage][-1] == "claim"
    claim_bytes = claim.export()

    # decide on the bound claim: 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9.
    contract = _contract(claim)
    result = contract.evaluate(claim)
    assert result.selected == ("treat",)
    assert result.actions[0].expected_utility == pytest.approx(1.0)
    assert result.actions[1].expected_utility == pytest.approx(9.0)

    # rank a study. The ranking is for this contract and rests on the claim's value digest;
    # its state model is the hand-derived guess record (wait = 1 - theta, treat = theta).
    values_digest = claim.identity_fields["values_blake3"]
    ranked_decision = dr.DesignDecision(
        contract=contract,
        actions=(dr.ActionUtility("wait", 1.0, -1.0), dr.ActionUtility("treat", 0.0, 1.0)),
        prior=dr.StatePrior.draws([0.0, 1.0]),
    )
    signal = dr.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=(f"claim:{values_digest}",),
        rng_seed=3,
    )
    ranked = dr.rank_designs(
        _study_candidates(),
        decision=ranked_decision,
        signal=signal,
        cost_map=UTILITY_MAP,
        source_digests=(values_digest,),
    )
    assert ranked.decision_contract_identity == contract.identity
    assert [c.id for c in ranked.candidates] == ["cand-1", "cand-2"]
    first, second = ranked.candidates
    assert first.evsi == pytest.approx(0.25, abs=1e-12)
    assert second.evsi == pytest.approx(0.125, abs=1e-12)
    assert first.evpi == pytest.approx(0.5, abs=1e-12)
    assert first.net_value == pytest.approx(0.15, abs=1e-12)
    assert second.net_value == pytest.approx(0.025, abs=1e-12)
    assert first.provider_trust == "externally_attested"
    assert not first.natively_replayed
    assert ranked.calibration == "unmeasured"

    # The ranking names the same decision contract link the decision result does, with the
    # same digest, and every stage behind the ranked numbers.
    decision_link = f"decision:{contract.identity}"
    decision_digest = {link.id: link.digest for link in result.lineage}[decision_link]
    ranking_links = {link.id: link for link in ranked.lineage}
    assert ranking_links[decision_link].digest == decision_digest
    assert ranked.lineage[-1].id == dr.RESULT_LINK_ID == "design_ranking_result"
    assert ranked.stages_behind() == {
        "decision_contract",
        "distribution_artifact",
        "external_provider",
        "study_ranking_provider",
        "claim",
    }
    assert f"distribution:{values_digest}" in ranking_links
    for link in ranked.lineage:
        assert len(link.digest) == 64, link.id
        assert [ranking_links[p].digest for p in link.parents] == list(link.parent_digests)

    # A fresh process holds only the bytes and constants: it loads the claim under the
    # retained identity and consumes the ranking by recomputation under its own expectation.
    ranking_bytes = ranked.export()
    script = textwrap.dedent(
        """
        import json, sys
        import antecedent as ac
        from antecedent import design, external

        ident = ac.identify(
            graph=[("x", "a"), ("x", "y"), ("a", "y")],
            names=["x", "a", "y"],
            query=ac.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
        )
        spec = external.response(
            ident,
            outcome_units="mmHg",
            population="target",
            require_evidence=("factor:z",),
            require_assumptions=("ignorability",),
        )
        claim = spec.load(open(sys.argv[1], "rb").read(), expected_identity=sys.argv[3])
        digest = claim.identity_fields["values_blake3"]
        ranked = design.consume(
            open(sys.argv[2], "rb").read(),
            expected_identity=design.Expectation(
                decision_contract_identity=sys.argv[4],
                source_digests=[digest],
                cost_map=design.CostMap("utility", "utility", 1.0),
            ),
        )
        assert [e.id for e in ranked.entries] == ["cand-1", "cand-2"]
        assert abs(ranked.entries[0].net_value - 0.15) < 1e-12
        assert abs(ranked.entries[1].net_value - 0.025) < 1e-12
        assert not any(e.natively_replayed for e in ranked.entries)
        assert ranked.source_digests == (digest,)
        assert ranked.calibration == "unmeasured"
        """
    )
    with (
        tempfile.NamedTemporaryFile(suffix=".bin") as claim_file,
        tempfile.NamedTemporaryFile(suffix=".bin") as ranking_file,
    ):
        claim_file.write(claim_bytes)
        claim_file.flush()
        ranking_file.write(ranking_bytes)
        ranking_file.flush()
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                claim_file.name,
                ranking_file.name,
                claim.identity,
                contract.identity,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    assert done.returncode == 0, done.stderr

    # A different retained contract identity does not consume the same ranking.
    with pytest.raises(dr.DesignRankingRefusal):
        dr.consume(
            ranking_bytes,
            expected_identity=dr.Expectation(decision_contract_identity="another-contract"),
        )


# ------------------------------------------------------------------ lineage of the provider


def _guess_ranking(
    *,
    contract: str = "contract-1",
    digests: tuple[str, ...] = ("digest-b", "digest-a"),
    snapshot: str = "snap",
    accuracy_two: float = 0.625,
) -> dr.DesignRankingResult:
    decision_ = dr.DesignDecision(
        contract=contract,
        actions=GUESS.actions,
        prior=GUESS.prior,
        utility_units="utility",
    )
    candidates = [
        dr.Candidate(
            cid,
            1,
            dr.ExternalSignal(
                "lab",
                f"signal-{cid}",
                "v1",
                snapshot if cid == "cand-1" else "snap",
                "lab-qa",
                dr.ExternalLaw.posterior(
                    states=[0.0, 1.0],
                    statistics=[0.0, 1.0],
                    predictive=[0.5, 0.5],
                    posterior=[[a, 1.0 - a], [1.0 - a, a]],
                ),
            ),
            cost=0.1,
            cost_unit="utility",
        )
        for cid, a in (("cand-1", 0.75), ("cand-2", accuracy_two))
    ]
    return dr.rank_designs(
        candidates,
        decision=decision_,
        signal=_signal(),
        cost_map=UTILITY_MAP,
        source_digests=digests,
    )


def _digests(ranking: dr.DesignRankingResult) -> dict[str, str]:
    return {link.id: link.digest for link in ranking.lineage}


def test_a0r_ranking_lineage_has_the_literal_order_and_stages_rust_writes():
    ranking = _guess_ranking()
    s1 = ranking.candidate("cand-1").signal_identity
    s2 = ranking.candidate("cand-2").signal_identity
    assert [link.id for link in ranking.lineage] == [
        "decision:contract-1",
        "distribution:digest-a",
        "distribution:digest-b",
        "provider:signal:lab/signal-cand-1@v1#snap",
        f"signal:cand-1:{s1}",
        "provider:signal:lab/signal-cand-2@v1#snap",
        f"signal:cand-2:{s2}",
        "design_ranking_result",
    ]
    assert [link.stage for link in ranking.lineage] == [
        "decision_contract",
        "distribution_artifact",
        "distribution_artifact",
        "external_provider",
        "study_ranking_provider",
        "external_provider",
        "study_ranking_provider",
        "claim",
    ]
    assert ranking.stages_behind(f"signal:cand-1:{s1}") == {
        "decision_contract",
        "external_provider",
        "study_ranking_provider",
    }
    with pytest.raises(ValueError, match="unknown lineage link"):
        ranking.stages_behind("nope")


def test_a0r_ranking_lineage_digests_move_with_every_upstream_identity():
    base = _guess_ranking()
    base_digests = _digests(base)
    s1 = f"signal:cand-1:{base.candidate('cand-1').signal_identity}"
    s2 = f"signal:cand-2:{base.candidate('cand-2').signal_identity}"
    result = "design_ranking_result"

    # The same constants give the same chain, whatever order the digests are given in.
    again = _guess_ranking(digests=("digest-a", "digest-b"))
    assert [(link.id, link.digest) for link in again.lineage] == [
        (link.id, link.digest) for link in base.lineage
    ]

    # A different decision contract: the root, every signal and the result move.
    other = _digests(_guess_ranking(contract="contract-2"))
    assert "decision:contract-1" not in other
    assert other["decision:contract-2"] != base_digests["decision:contract-1"]
    assert other[s1] != base_digests[s1]
    assert other[result] != base_digests[result]

    # A different source digest moves the result only.
    other = _digests(_guess_ranking(digests=("digest-z", "digest-b")))
    assert "distribution:digest-a" not in other
    assert other["decision:contract-1"] == base_digests["decision:contract-1"]
    assert other[s1] == base_digests[s1]
    assert other[result] != base_digests[result]

    # A different signal law of one candidate moves that signal and the result only.
    changed = _guess_ranking(accuracy_two=0.6)
    other = _digests(changed)
    new_s2 = f"signal:cand-2:{changed.candidate('cand-2').signal_identity}"
    assert new_s2 != s2 and s2 not in other
    assert other[s1] == base_digests[s1]
    assert other[result] != base_digests[result]

    # A different provider snapshot of one candidate moves its provider, its signal and the
    # result; the other candidate's signal stays put.
    snapshot = _guess_ranking(snapshot="snap-2")
    other = _digests(snapshot)
    assert "provider:signal:lab/signal-cand-1@v1#snap" not in other
    assert "provider:signal:lab/signal-cand-1@v1#snap-2" in other
    new_s1 = f"signal:cand-1:{snapshot.candidate('cand-1').signal_identity}"
    assert other[new_s1] != base_digests[s1]
    assert other[s2] == base_digests[s2]
    assert other[result] != base_digests[result]


# ----------------------------------------------------------------- lineage of sensitivity


def _surface(
    *,
    snapshot: str = "snapshot-1",
    contract: str = "checked-contract",
    ua: tuple[float, ...] = (5.0, 4.0, 3.0),
) -> sd.SensitivityArtifact:
    def quantity(name: str) -> ScientificQuantity:
        return ScientificQuantity(
            variable_id=name,
            variable_name=name,
            role="outcome",
            units="utils",
            population_id="target",
            regime_id="do(a=1)",
            horizon=0,
            functional_id="sensitivity_surface",
        )

    return sd.SensitivityArtifact.from_surface(
        coordinate=sd.AssumptionCoordinate(
            "gamma", "sensitivity_parameter", "dimensionless", 0.0, 2.0
        ),
        grid=[0.0, 1.0, 2.0],
        quantities=[
            sd.SurfaceQuantity(quantity("ua"), ua),
            sd.SurfaceQuantity(quantity("ub"), (1.0, 1.0, 1.0)),
        ],
        actions=[
            sd.SensitivityAction("A", sd.quantity("ua")),
            sd.SensitivityAction("B", sd.quantity("ub")),
        ],
        provenance=sd.SurfaceProvenance(
            source_kind="supplied_surface",
            query_binding="a0r-test",
            provider_snapshot=snapshot,
            source_regime="regime:1",
            method="hand-derived surface",
            causal_contract_id=contract,
        ),
    )


def test_a0r_sensitivity_lineage_links_contract_snapshot_input_and_claim():
    artifact = _surface()
    digest = artifact.identity["digest"]
    links = artifact.lineage
    assert [link.id for link in links] == [
        "causal_contract:checked-contract",
        "snapshot:snapshot-1",
        f"sensitivity_input:{digest}",
        "sensitivity_claim",
    ]
    assert [link.stage for link in links] == [
        "causal_contract",
        "data",
        "sensitivity_input",
        "claim",
    ]
    assert artifact.stages_behind() == {"causal_contract", "data", "sensitivity_input", "claim"}
    by_id = {link.id: link for link in links}
    for link in links:
        assert len(link.digest) == 64
        assert [by_id[p].digest for p in link.parents] == list(link.parent_digests)
    # The same artifact bytes give the same lineage in a consumer.
    consumed = sd.SensitivityArtifact.consume(
        artifact.export(), expected_identity=artifact.identity
    )
    assert [(link.id, link.digest) for link in consumed.lineage] == [
        (link.id, link.digest) for link in links
    ]


def test_a0r_sensitivity_lineage_digests_move_with_contract_snapshot_and_numbers():
    base = {link.id: link.digest for link in _surface().lineage}
    claim = "sensitivity_claim"

    other = {link.id: link.digest for link in _surface(contract="other-contract").lineage}
    assert "causal_contract:checked-contract" not in other
    assert other[claim] != base[claim]

    other = {link.id: link.digest for link in _surface(snapshot="snapshot-2").lineage}
    assert "snapshot:snapshot-1" not in other
    assert other["causal_contract:checked-contract"] == base["causal_contract:checked-contract"]
    assert other[claim] != base[claim]

    changed = _surface(ua=(5.0, 4.0, 3.5))
    other = {link.id: link.digest for link in changed.lineage}
    assert other["snapshot:snapshot-1"] == base["snapshot:snapshot-1"]
    assert other[claim] != base[claim]
    assert changed.identity["digest"] != _surface().identity["digest"]


# --------------------------------------------------------------------- refusal families


def _joint_law_refusal() -> iq.InverseQueryRefusal:
    q = inverse_fixtures._q
    p, qq, safe = q("do(a=1)", variable="p"), q("do(a=1)", variable="q"), q("do(a=0)", variable="s")
    contract = decision.Contract(
        actions=(
            decision.Action("risky", inputs=(p, qq), utility=decision.x(0) * decision.x(1)),
            decision.Action("safe", inputs=(safe,), utility=decision.x(0), kind="policy"),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    columns = [
        (p, [1.0, 3.0, 2.0, 0.0]),
        (qq, [4.0, 0.0, 2.0, 6.0]),
        (safe, [3.0, 3.0, 3.0, 3.0]),
    ]
    query = iq.InverseQuery(contract, ("risky", "safe"), (iq.target_mean(2.5),))
    with pytest.raises(iq.InverseQueryRefusal) as caught:
        query.evaluate(inverse_fixtures._joint(columns, alignment="independent_marginals"))
    return caught.value


def _raised(call: Callable[[], object], kind: type[_E]) -> _E:
    with pytest.raises(kind) as caught:
        call()
    return caught.value


def test_a0r_refusal_families_keep_their_own_details_across_the_bridge():
    law = inverse_fixtures._law()
    joint = _joint_law_refusal()
    no_constraints = _raised(
        lambda: iq.InverseQuery(inverse_fixtures._contract(), inverse_fixtures.IDS, ()).evaluate(
            law
        ),
        iq.InverseQueryRefusal,
    )
    sample_size = _raised(
        lambda: dr.evsi(
            GUESS,
            dr.Candidate(
                "study",
                1,
                dr.ExternalSignal(
                    "lab",
                    "signal-object",
                    "v1",
                    "snap",
                    "lab-qa",
                    dr.ExternalLaw.posterior(
                        states=[0.0, 1.0],
                        statistics=[0.0, 1.0],
                        predictive=[0.5, 0.5],
                        posterior=[[0.75, 0.25], [0.25, 0.75]],
                    ),
                    attested_sample_size=3,
                ),
            ),
            signal=_signal(),
        ),
        dr.SignalProviderRefusal,
    )
    cost_units = _raised(
        lambda: dr.rank_designs(
            [_guess_candidate("study", 0.75, 0.1, "usd")],
            decision=GUESS,
            signal=_signal(),
            cost_map=UTILITY_MAP,
        ),
        dr.CostUnitsRefusal,
    )
    cost_map_required = _raised(
        lambda: dr.rank_designs(
            [_guess_candidate("study", 0.75, 0.1, "usd")],
            decision=GUESS,
            signal=_signal(),
            require_net_value=True,
        ),
        dr.CostUnitsRefusal,
    )
    overlap = _raised(
        lambda: dr.rank_designs(
            [_guess_candidate("cand-1", 0.75, reused_observations=("obs-prior",))],
            decision=GUESS,
            signal=_signal(),
            cost_map=UTILITY_MAP,
            prior_observations=("obs-prior",),
        ),
        dr.SourceOverlapRefusal,
    )
    ranking_cost_map = _raised(
        lambda: dr.consume(
            _guess_ranking().export(),
            expected_identity=dr.Expectation(cost_map=dr.CostMap("utility", "utility", 2.0)),
        ),
        dr.CostUnitsRefusal,
    )

    # (error, registered code, stage, detail), the same literals as the Rust table.
    table = [
        (joint, "joint_law_required", "inverse_query", "decision_evaluation.joint_law_required"),
        (no_constraints, "invalid_argument", "inverse_query", "inverse_query.no_constraints"),
        (sample_size, "design_signal_invalid", "declare", "signal_provider.sample_size_mismatch"),
        (cost_units, "design_cost_units_mismatch", "evaluate", "evsi.cost_units_mismatch"),
        (cost_map_required, "design_cost_units_mismatch", "evaluate", "evsi.cost_map_required"),
        (overlap, "design_signal_invalid", "evaluate", "evsi.source_overlap"),
        (
            ranking_cost_map,
            "design_cost_units_mismatch",
            "consume",
            "design_ranking.cost_units_mismatch",
        ),
    ]
    for error, code, stage, detail in table:
        assert isinstance(error, CausalUnsupportedError)
        assert_registered_refusal(error)
        assert error.reason_code == code, detail
        assert error.stage == stage, detail
        assert error.detail == detail
        assert REFUSAL_DETAIL.match(error.detail)
    assert len({error.detail for error, *_ in table}) == len(table)

    # Compared semantics and remedies survive the bridge.
    assert joint.expected == "joint" and joint.supplied == "independent_marginals"
    assert joint.remedy is not None
    assert {sample_size.expected, sample_size.supplied} == {"1", "3"}
    assert overlap.offending == "obs-prior"
    assert cost_units.remedy is not None
    assert ranking_cost_map.remedy is not None


def test_a0r_obligation_refusals_use_the_codes_and_details_rust_registers():
    contract = backdoor(
        [repair.UnresolvedAssumption("assume:positivity", "positivity given the adjustment set")]
    )
    candidate = repair.StudyCandidate.observation(
        "cohort",
        population="clinic",
        measured=["t", "y", "z1", "z2"],
        cost=20,
        sample_size=500,
        recruitment="consecutive patients",
        timing="baseline",
        unit="patient",
        cost_unit="USD",
    )
    result = repair.repair(contract, [candidate])
    assert result.reason_code == "transport_missing_evidence"
    assert result.detail == "evidence_obligations.wrong_contract"
    assert result.reason_code in set(_native.runtime_refusal_codes())


# ---------------------------------------------------------------- obligations and the facade


def test_a0r_obligation_facade_exposes_the_rust_fields_and_identity_shape():
    (obligation,) = repair.obligations(backdoor())
    assert obligation.kind == "provide_joint_law"
    assert obligation.scope
    assert set(obligation.variables) == {"t", "y", "z1", "z2"}
    assert obligation.population == "clinic"
    assert obligation.joint is True
    assert obligation.interventions == ()
    assert obligation.conditioned_on == ()
    assert obligation.family == "backdoor"
    assert obligation.proof_step == "backdoor.adjustment_set:0,1,2,3"
    assert obligation.satisfiable_by_study is True
    # `eo1:<kind>:` and the first 32 hex digits of BLAKE3 over the canonical content, which
    # Rust recomputes in `a0r_backdoor_obligation_has_the_derived_identity_*`.
    assert re.fullmatch(r"eo1:provide_joint_law:[0-9a-f]{32}", obligation.id)
    # The identity is stable across independently built contracts and reorderings.
    (again,) = repair.obligations(backdoor())
    assert again == obligation
    assumption = {
        o.kind: o
        for o in repair.obligations(
            backdoor([repair.UnresolvedAssumption("assume:positivity", "positivity given the set")])
        )
    }["establish_assumption"]
    assert assumption.satisfiable_by_study is False
    assert re.fullmatch(r"eo1:establish_assumption:[0-9a-f]{32}", assumption.id)
    assert assumption.id != obligation.id


# ------------------------------------------------------------- cross-surface fixture hooks


def _obligation_fixture_text() -> str:
    (obligation,) = repair.obligations(backdoor())
    return (
        json.dumps(
            {
                "id": obligation.id,
                "kind": obligation.kind,
                "proof_step": obligation.proof_step or "",
                "population": obligation.population or "",
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def test_a0r_regenerate_python_fixtures():
    if os.environ.get("A0R_WRITE_FIXTURES") != "1":
        pytest.skip(f"fixture writer; run {REGENERATE_PY}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "py_a0r_design_ranking.bin").write_bytes(_guess_ranking().export())
    (FIXTURE_DIR / "py_a0r_obligation.identity.json").write_text(_obligation_fixture_text())


def _fixture(name: str) -> bytes:
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip(f"missing cross-surface fixture {path}; generate it with: {REGENERATE_RUST}")
    return path.read_bytes()


def test_a0r_rust_built_design_ranking_and_obligation_are_consumed_by_python():
    ranked = dr.consume(
        _fixture("rust_a0r_design_ranking.bin"),
        expected_identity=dr.Expectation(
            decision_contract_identity="contract-1",
            source_digests=("digest-a", "digest-b"),
            cost_map=UTILITY_MAP,
        ),
    )
    assert [e.id for e in ranked.entries] == ["cand-1", "cand-2"]
    assert ranked.entries[0].net_value == pytest.approx(0.15, abs=1e-12)
    assert ranked.entries[1].net_value == pytest.approx(0.025, abs=1e-12)
    assert ranked.calibration == "unmeasured"
    assert not any(entry.natively_replayed for entry in ranked.entries)
    # Python derives the identity of the same hand contract Rust wrote.
    assert _fixture("rust_a0r_obligation.identity.json").decode() == _obligation_fixture_text()
