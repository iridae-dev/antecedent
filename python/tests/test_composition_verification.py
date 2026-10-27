"""Composition verification through the Python facade.

The graph is hand computable (``cohort`` feeds ``measurement`` (data); ``fit`` reuses
``mechanisms``' structural derivation and reads ``cohort`` (data); ``reported`` reads ``fit`` and
inherits its calibration record; ``transported`` reads ``fit`` and the ``target``'s support;
``decision`` reads ``reported``; ``sibling_claim`` reads ``sibling_data``):

* replacing the cohort's data recomputes ``measurement``, ``mechanisms``, ``fit``, ``reported``,
  ``transported`` and ``decision`` and leaves ``provider``, ``target`` and the sibling valid;
* a withdrawn provider trust withdraws what rests on the provider with no numerical reason;
* an unsupported target withdraws only the claim resting on its support;
* a record dropped upstream refuses the claim that inherited it, leaving its point value;
* a record established for one target is never inherited by a claim about another.

No test asserts an interval, a posterior or a coverage statement.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import pytest
from antecedent import composition_verification as cv
from antecedent import evidence_ledger as el
from antecedent.composition_verification import CompositionVerificationRefusal
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

D = cv.digest
RECORD = D("calibration record")
SPACE = D("coordinate", "result")


def node(key: str, kind: str, version: str = "v1", **fields) -> cv.Node:
    return cv.Node(key, kind, identity=D(key, version), **fields)


def build(
    *,
    cohort: str = "v1",
    fit: str = "v1",
    trust: str = "held",
    unsupported: tuple[str, ...] = (),
    fit_calibration: cv.Calibration | None = None,
) -> cv.CompositionGraph:
    held = fit_calibration or cv.Calibration.held(RECORD, SPACE)
    nodes = [
        node("cohort", "cohort_source", cohort),
        node("provider", "external_provider", trust=trust),
        node(
            "measurement",
            "measurement_model",
            structure=D("measurement structure"),
            representation="measurement_model/1",
            assumptions=("nondifferential error",),
        ),
        node(
            "mechanisms",
            "mechanism_composition",
            structure=D("mechanism structure"),
            assumptions=("causal sufficiency",),
        ),
        node(
            "fit",
            "ledger_result",
            fit,
            structure=D("fit structure"),
            coordinate=SPACE,
            calibration=held,
        ),
        node("target", "target_population", unsupported=unsupported),
        node(
            "reported",
            "claim",
            coordinate=SPACE,
            calibration=cv.Calibration.inherited("fit", RECORD, SPACE),
        ),
        node(
            "transported",
            "transported_result",
            coordinate=D("coordinate", "target"),
            transformation="transport_to_target",
            assumptions=("s-admissible adjustment set",),
        ),
        node("decision", "decision"),
        node("sibling_data", "data"),
        node("sibling_claim", "claim"),
    ]
    edges = [
        cv.Edge("measurement", "cohort", "data"),
        cv.Edge("measurement", "provider", "provider_trust"),
        cv.Edge("mechanisms", "measurement", "model_version"),
        cv.Edge("mechanisms", "cohort", "evidence"),
        cv.Edge("fit", "mechanisms", "structural_derivation"),
        cv.Edge("fit", "cohort", "data"),
        cv.Edge("reported", "fit", "data"),
        cv.Edge("reported", "fit", "calibration"),
        cv.Edge("transported", "fit", "data"),
        cv.Edge("transported", "target", "support"),
        cv.Edge("decision", "reported", "data"),
        cv.Edge("sibling_claim", "sibling_data", "data"),
    ]
    return cv.CompositionGraph(nodes=nodes, edges=edges)


ALL = [
    "cohort", "decision", "fit", "measurement", "mechanisms", "provider", "reported",
    "sibling_claim", "sibling_data", "target", "transported",
]  # fmt: skip


def refused(call) -> CompositionVerificationRefusal:
    with pytest.raises(CompositionVerificationRefusal) as caught:
        call()
    assert isinstance(caught.value, CausalUnsupportedError)
    assert_registered_refusal(caught.value)
    return caught.value


# --------------------------------------------------------------------------- classification


def test_an_unchanged_composition_is_still_valid_and_keeps_its_authority():
    graph = build()
    result = cv.verify_composition(graph, previous=graph)
    assert list(result.keys_of("still_valid")) == ALL
    assert all(c.authority_retained for c in result.claims.values())
    assert result.classify("reported").calibration.state == "inherited"
    assert result.classify("fit").calibration.state == "held"
    assert result.require_valid("decision").klass == "still_valid"


def test_a_change_leaves_the_sibling_valid_and_recomputes_every_dependent():
    before, after = build(), build(cohort="v2")
    result = cv.verify_composition(after, previous=before)
    assert result.keys_of("recompute") == (
        "cohort", "decision", "fit", "measurement", "mechanisms", "reported", "transported",
    )  # fmt: skip
    assert result.keys_of("still_valid") == (
        "provider", "sibling_claim", "sibling_data", "target",
    )  # fmt: skip
    assert result.keys_of("withdrawn") == ()
    for key in result.keys_of("recompute"):
        claim = result.classify(key)
        assert claim.withdrawals == ()
        assert not claim.authority_retained
        assert {r.origin for r in claim.numerical} == {"cohort"}
    error = refused(lambda: result.require_valid("decision"))
    assert error.reason_code == "claim_revalidation_required"
    assert error.detail == "composition_verification.recompute_required"
    assert error.remedy


def test_reusing_a_structural_derivation_after_data_replacement_retains_no_fit_authority():
    result = cv.verify_composition(build(cohort="v2"), previous=build())
    fit = result.classify("fit")
    causes = {(r.via_node, r.via_kind, r.cause) for r in fit.numerical}
    assert ("mechanisms", "structural_derivation", "structure_reused_fit_not_retained") in causes
    assert ("cohort", "data", "identity_changed") in causes
    assert fit.klass == "recompute"
    assert not fit.authority_retained
    # a refit under an unchanged structure
    refit = cv.verify_composition(build(fit="v2"), previous=build())
    own = refit.classify("fit")
    assert [(r.origin, r.via_node, r.cause) for r in own.numerical] == [
        ("fit", None, "structure_reused_fit_not_retained")
    ]
    assert refit.classify("cohort").klass == "still_valid"
    assert refit.classify("mechanisms").klass == "still_valid"
    assert refit.classify("decision").klass == "recompute"


def test_a_withdrawn_provider_trust_withdraws_dependents_without_stale_numbers():
    result = cv.verify_composition(build(trust="withdrawn"), previous=build())
    assert result.keys_of("withdrawn") == (
        "decision", "fit", "measurement", "mechanisms", "provider", "reported", "transported",
    )  # fmt: skip
    assert result.keys_of("still_valid") == ("cohort", "sibling_claim", "sibling_data", "target")
    assert all(not c.numerical for c in result.claims.values())
    assert result.classify("decision").withdrawn_for("provider_trust")
    error = refused(lambda: result.require_valid("decision"))
    assert error.reason_code == "claim_license_withdrawn"
    assert error.detail == "composition_verification.provider_trust_withdrawn"


def test_an_unsupported_target_withdraws_only_the_claim_resting_on_its_support():
    result = cv.verify_composition(build(unsupported=("income above 200k",)), previous=build())
    assert result.keys_of("withdrawn") == ("target", "transported")
    assert result.classify("decision").klass == "still_valid"
    (withdrawal,) = result.classify("transported").withdrawals
    assert (withdrawal.kind, withdrawal.origin, withdrawal.detail) == (
        "support", "target", "income above 200k",
    )  # fmt: skip
    error = refused(lambda: result.require_valid("transported"))
    assert error.detail == "composition_verification.support_withdrawn"


def test_calibration_inherited_from_an_object_that_no_longer_holds_it_is_refused():
    after = build(fit_calibration=cv.Calibration.point_only())
    result = cv.verify_composition(after, previous=build())
    reported = result.classify("reported")
    assert reported.klass == "withdrawn"
    assert reported.calibration.state == "lost"
    assert reported.calibration.newly_required
    assert reported.calibration.source == "fit"
    # the point value was not touched: withdrawn for calibration only
    assert reported.numerical == ()
    assert reported.withdrawn_for("calibration")
    error = refused(lambda: result.require_valid("reported"))
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "composition_verification.calibration_not_held"
    # a point-only dependent keeps its value
    assert result.classify("decision").klass == "still_valid"
    # another record is not the inherited one
    other = build(fit_calibration=cv.Calibration.held(D("another record"), SPACE))
    swapped = cv.verify_composition(other, previous=build())
    assert swapped.classify("reported").calibration.detail == "record_differs"


def _transport_pair(held_at: str, node_at: str, claim: str, claim_node: str):
    nodes = [
        cv.Node("transported", "transported_result", identity=D("t"), coordinate=D(node_at),
                calibration=cv.Calibration.held(RECORD, D(held_at))),
        cv.Node("retargeted", "transported_result", identity=D("r"), coordinate=D(claim_node),
                calibration=cv.Calibration.inherited("transported", RECORD, D(claim))),
    ]  # fmt: skip
    return cv.CompositionGraph(nodes, [cv.Edge("retargeted", "transported", "calibration")])


def test_transported_calibration_never_inherits_to_a_different_target():
    same = _transport_pair("t1", "t1", "t1", "t1")
    result = cv.verify_composition(same, previous=same)
    assert result.classify("retargeted").calibration.state == "inherited"

    other = _transport_pair("t1", "t1", "t1", "t2")
    lost = cv.verify_composition(other, previous=other).classify("retargeted")
    assert (lost.klass, lost.calibration.detail) == ("withdrawn", "coordinate_mismatch")
    claims_t2 = _transport_pair("t1", "t1", "t2", "t2")
    differs = cv.verify_composition(claims_t2, previous=claims_t2).classify("retargeted")
    assert differs.calibration.detail == "record_differs"

    # the target of a held record changes: the record is lost and newly required there
    moved = cv.verify_composition(
        _transport_pair("t1", "t2", "t1", "t1"), previous=_transport_pair("t1", "t1", "t1", "t1")
    )
    transported = moved.classify("transported")
    assert transported.klass == "withdrawn"
    assert transported.calibration.state == "lost" and transported.calibration.newly_required
    assert any(r.cause == "coordinate_changed" for r in transported.numerical)
    assert moved.classify("retargeted").klass == "withdrawn"
    error = refused(lambda: moved.require_valid("retargeted"))
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "composition_verification.calibration_not_held"
    other_target = refused(
        lambda: cv.verify_composition(other, previous=other).require_valid("retargeted")
    )
    assert other_target.reason_code == "cell_not_licensed"


def test_a_removed_input_and_a_new_node_are_classified():
    before = build()
    kept = [n for n in before.nodes if n.key != "target"]
    after = cv.CompositionGraph(kept, [e for e in before.edges if e.upstream != "target"])
    removed = cv.verify_composition(after, previous=before)
    assert removed.removed == ("target",)
    transported = removed.classify("transported")
    assert transported.klass == "withdrawn"
    assert (transported.withdrawals[0].kind, transported.withdrawals[0].detail) == (
        "retired", "removed_from_graph",
    )  # fmt: skip
    grown = cv.CompositionGraph(
        [*before.nodes, node("late", "claim")], [*before.edges, cv.Edge("late", "fit", "data")]
    )
    late = cv.verify_composition(grown, previous=before).classify("late")
    assert late.klass == "recompute"
    assert late.numerical[0].cause == "added"


def test_a_first_verification_reports_nothing_stale():
    result = cv.verify_composition(build(cohort="v9"))
    assert not result.has_previous
    assert result.keys_of("recompute") == ()


# --------------------------------------------------------------------------- invariance


def test_node_and_edge_declaration_order_never_changes_the_verification():
    base = build()
    after = build(cohort="v2")
    reference = cv.verify_composition(after, previous=base)
    for nodes_order in itertools.islice(
        itertools.permutations(range(len(after.nodes))), 0, 4000, 997
    ):
        shuffled = cv.CompositionGraph(
            [after.nodes[i] for i in nodes_order], list(reversed(after.edges))
        )
        previous = cv.CompositionGraph([base.nodes[i] for i in nodes_order], list(base.edges))
        got = cv.verify_composition(shuffled, previous=previous)
        assert got.identity == reference.identity
        assert {k: c.klass for k, c in got.claims.items()} == {
            k: c.klass for k, c in reference.claims.items()
        }


# --------------------------------------------------------------------------- lineage


def test_the_view_reports_lineage_boundaries_versions_and_unsupported_coordinates():
    graph = build(unsupported=("income above 200k",))
    result = cv.verify_composition(graph, previous=graph)
    view = result.view("transported")
    assert [e.key for e in view.upstream] == [
        "cohort", "provider", "measurement", "mechanisms", "fit", "target",
    ]  # fmt: skip
    assert [t[2] for t in view.transformations] == [
        "measurement_model", "mechanism_composition", "transport_to_target",
    ]  # fmt: skip
    assert [a[1] for a in view.assumptions] == [
        "nondifferential error", "causal sufficiency", "s-admissible adjustment set",
    ]  # fmt: skip
    assert view.versions == (("measurement", "measurement_model/1"),)
    assert view.unsupported == (("target", "income above 200k"),)
    assert all(not e.key.startswith("sibling") for e in view.upstream)
    reported = result.view("reported")
    assert reported.claim.calibration.state == "inherited"
    assert reported.claim.calibration.source == "fit"
    refused(lambda: result.view("absent"))


# --------------------------------------------------------------------------- ledger


def _ledger_world():
    ledger = el.EvidenceLedger()

    def study(label, lo, hi):
        return ledger.add_source(
            el.SourceDeclaration(
                label, "observation_set", observations=el.rows("cohort", range(lo, hi))
            )
        )

    s1, s2, s3 = study("s1", 0, 10), study("s2", 10, 20), study("s3", 20, 30)
    r12 = require_success(
        ledger.pool_summaries(
            "s1_s2", [el.StudySummary(s1, 1.0, 1.0), el.StudySummary(s2, 3.0, 4.0)]
        )
    ).result
    r23 = require_success(
        ledger.pool_summaries(
            "s2_s3", [el.StudySummary(s2, 1.0, 1.0), el.StudySummary(s3, 3.0, 4.0)]
        )
    ).result
    rall = require_success(
        ledger.pool_summaries(
            "all",
            [
                el.StudySummary(s1, 1.0, 1.0),
                el.StudySummary(s2, 3.0, 4.0),
                el.StudySummary(s3, 2.0, 2.0),
            ],
        )
    ).result
    return ledger, (s1, s2, s3), (r12, r23, rall)


def test_ledger_standings_drive_the_classification_and_agree_with_the_ledger():
    ledger, (s1, _s2, _s3), (r12, r23, rall) = _ledger_world()
    graph = cv.CompositionGraph(
        [
            node("p12", "ledger_result"), node("p23", "ledger_result"),
            node("pall", "ledger_result"), node("d12", "decision"),
            node("d23", "decision"), node("dall", "decision"),
        ],
        [
            cv.Edge("d12", "p12", "evidence"), cv.Edge("d23", "p23", "evidence"),
            cv.Edge("dall", "pall", "evidence"),
        ],
    )  # fmt: skip
    bindings = {"p12": r12, "p23": r23, "pall": rall}
    assert cv.ledger_effects(ledger, bindings) == ()
    require_success(
        ledger.revise_source(require_success(el.SourceRevision.withdraw(s1, "consent withdrawn")))
    )
    effects = cv.ledger_effects(ledger, bindings)
    result = cv.verify_composition(graph, previous=graph, effects=effects)
    # pooling two studies loses one member of two; three minus one recompute; one is untouched
    assert result.classify("p12").klass == "withdrawn"
    assert result.classify("d12").klass == "withdrawn"
    assert result.classify("pall").klass == "recompute"
    assert result.classify("dall").klass == "recompute"
    assert result.classify("p23").klass == "still_valid"
    for key, rid in bindings.items():
        assert result.classify(key).klass == ledger.standing(rid)
    with pytest.raises(CausalValueError):
        cv.ledger_effects(ledger, {"p12": "0" * 64})


def test_shared_data_comes_from_the_ledger_and_unknown_is_never_independent():
    ledger = el.EvidenceLedger()
    a = ledger.add_source(
        el.SourceDeclaration("a", "observation_set", observations=el.rows("cohort", range(0, 100)))
    )
    b = ledger.add_source(
        el.SourceDeclaration("b", "observation_set", observations=el.rows("cohort", range(50, 150)))
    )
    c = ledger.add_source(
        el.SourceDeclaration(
            "c", "observation_set", observations=el.rows("cohort", range(200, 210))
        )
    )
    v = ledger.add_source(el.SourceDeclaration("vendor", "external_claim"))
    nodes = [
        cv.Node(key, "cohort_source", identity=D(key), evidence=(src,))
        for key, src in (("na", a), ("nb", b), ("nc", c), ("nv", v))
    ]
    nodes.append(node("pooled", "ledger_result"))
    graph = cv.CompositionGraph(
        nodes, [cv.Edge("pooled", k, "evidence") for k in ("na", "nb", "nc", "nv")]
    )
    relations = cv.shared_data_from_ledger(graph, ledger)
    assert [(r.first, r.second, r.relation, r.shared_observations) for r in relations] == [
        ("na", "nb", "shared_data", 50),
        ("na", "nv", "unknown_dependence", 0),
        ("nb", "nv", "unknown_dependence", 0),
        ("nc", "nv", "unknown_dependence", 0),
    ]
    result = cv.verify_composition(graph, ledger=ledger)
    assert result.ledger_state == ledger.state_digest
    assert result.view("pooled").shared_data == relations
    result.verify_ledger(graph, ledger)
    # the ledger moves on: the verification was read from another state
    require_success(
        ledger.revise_source(require_success(el.SourceRevision.withdraw(a, "consent withdrawn")))
    )
    error = refused(lambda: result.verify_ledger(graph, ledger))
    assert error.detail == "composition_verification.ledger_mismatch"


# --------------------------------------------------------------------------- artifact


def test_a_fresh_process_consumes_the_export_and_refuses_tampering(tmp_path):
    result = cv.verify_composition(build(cohort="v2"), previous=build())
    exported = result.export("composition-python")
    consumed = cv.consume_composition_verification_artifact(
        exported.data, exported.expectation.identity
    )
    assert consumed.identity == exported.expectation.identity == result.identity
    assert consumed.keys_of("recompute") == result.keys_of("recompute")
    assert consumed.view("decision").identity == result.view("decision").identity
    assert refused(lambda: consumed.require_valid("decision")).reason_code == (
        "claim_revalidation_required"
    )

    path = tmp_path / "composition.bin"
    path.write_bytes(exported.data)
    script = (
        "import json, sys\n"
        "from antecedent import composition_verification as cv\n"
        "c = cv.consume_composition_verification_artifact(open(sys.argv[1], 'rb').read(), "
        "sys.argv[2])\n"
        "print(json.dumps({'identity': c.identity, 'recompute': list(c.keys_of('recompute')), "
        "'valid': list(c.keys_of('still_valid'))}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), exported.expectation.identity],
        capture_output=True, text=True, cwd=tmp_path, check=True,
    )  # fmt: skip
    wire = json.loads(done.stdout)
    assert wire["identity"] == exported.expectation.identity
    assert wire["recompute"] == list(result.keys_of("recompute"))
    assert wire["valid"] == ["provider", "sibling_claim", "sibling_data", "target"]
    wrong = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True, text=True, cwd=tmp_path,
    )  # fmt: skip
    assert wrong.returncode != 0
    assert "identity_mismatch" in wrong.stderr


def test_a_mutated_artifact_is_refused():
    exported = cv.verify_composition(build(cohort="v2"), previous=build()).export()
    data = exported.data
    for position in range(0, len(data), max(1, len(data) // 25)):
        mutated = bytearray(data)
        mutated[position] ^= 0x55
        error = refused(
            lambda m=bytes(mutated): cv.consume_composition_verification_artifact(
                m, exported.expectation.identity
            )
        )
        assert error.reason_code in {"invalid_argument", "route_not_supported"}
    error = refused(lambda: cv.consume_composition_verification_artifact(data[: len(data) // 2]))
    assert error.reason_code == "invalid_argument"
    other = cv.verify_composition(build(cohort="v3"), previous=build()).export()
    error = refused(
        lambda: cv.consume_composition_verification_artifact(
            other.data, exported.expectation.identity
        )
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "composition_verification.identity_mismatch"
    assert error.remedy


# --------------------------------------------------------------------------- declarations


def test_malformed_declarations_are_refused_or_rejected():
    with pytest.raises(CausalValueError):
        cv.Node("a", "no_such_kind", identity=D("a"))
    with pytest.raises(CausalValueError):
        cv.Node("a", "claim", identity="abc")
    with pytest.raises(CausalValueError):
        cv.Edge("a", "b", "teleport")
    with pytest.raises(CausalTypeError):
        cv.verify_composition("graph")  # type: ignore[arg-type]
    twice = cv.CompositionGraph([node("a", "claim"), node("a", "claim")])
    assert refused(lambda: cv.verify_composition(twice)).detail == (
        "composition_verification.graph_invalid"
    )
    cyclic = cv.CompositionGraph(
        [node("a", "claim"), node("b", "claim")],
        [cv.Edge("a", "b", "data"), cv.Edge("b", "a", "data")],
    )
    assert refused(lambda: cv.verify_composition(cyclic)).detail == (
        "composition_verification.graph_invalid"
    )
    no_edge = cv.CompositionGraph(
        [
            node("a", "claim"),
            node("b", "claim", calibration=cv.Calibration.inherited("a", RECORD, SPACE)),
        ]
    )
    assert refused(lambda: cv.verify_composition(no_edge)).detail == (
        "composition_verification.graph_invalid"
    )


# -- shipped coverage records ---------------------------------------------------------------


def _shipped_record_id() -> str:
    """The id of a coverage record that currently governs a measured pooled interval."""
    ledger = el.EvidenceLedger()
    sources = [
        ledger.add_source(
            el.SourceDeclaration(
                f"study_{i}",
                "observation_set",
                observations=el.rows("cohort", range(i * 100, (i + 1) * 100)),
            )
        )
        for i in range(5)
    ]
    summaries = [
        el.StudySummary(sources[i], 0.5 + 0.01 * (i - 2), 0.01 + 0.002 * i) for i in range(5)
    ]
    pooled = require_success(ledger.pool_summaries("fixed", summaries)).result
    record_id = ledger.pooled_interval(pooled)["calibration"]["record_id"]
    assert record_id
    return str(record_id)


def test_a_declared_record_is_backed_only_by_a_currently_attesting_shipped_record():
    shipped = cv.registry_record_identity(_shipped_record_id())
    assert len(shipped) == 64
    assert cv.registry_record_identity("cov.no_such_record") != shipped
    assert cv.registry_record_attests(shipped) is True
    # A digest that names no shipped record backs nothing.
    assert cv.registry_record_attests(RECORD) is False
    assert cv.registry_record_attests(cv.registry_record_identity("cov.no_such_record")) is False
    with pytest.raises(CausalValueError):
        cv.registry_record_attests("not a digest")

    backed = cv.Calibration.held(shipped, SPACE)
    assert cv.registry_resolved_calibration(backed) == backed
    unbacked = cv.Calibration.held(RECORD, SPACE)
    assert cv.registry_resolved_calibration(unbacked) == cv.Calibration.required()
    inherited = cv.Calibration.inherited("fit", RECORD, SPACE)
    assert cv.registry_resolved_calibration(inherited) == cv.Calibration.required()
    # A point-only declaration and a requirement are kept as they are.
    for kept in (cv.Calibration.point_only(), cv.Calibration.required()):
        assert cv.registry_resolved_calibration(kept) == kept
    with pytest.raises(CausalTypeError):
        cv.registry_resolved_calibration({"kind": "held"})


def test_the_nodes_whose_calibration_no_shipped_record_backs_are_named():
    # `fit` holds and `reported` inherits a record that is not a shipped one.
    assert sorted(cv.unresolved_registry_calibrations(build())) == ["fit", "reported"]
    shipped = cv.registry_record_identity(_shipped_record_id())
    backed = build(fit_calibration=cv.Calibration.held(shipped, SPACE))
    assert cv.unresolved_registry_calibrations(backed) == ("reported",)
    none_held = build(fit_calibration=cv.Calibration.point_only())
    assert cv.unresolved_registry_calibrations(none_held) == ("reported",)
    with pytest.raises(CausalTypeError):
        cv.unresolved_registry_calibrations("graph")
    duplicate = cv.CompositionGraph(nodes=[node("a", "data"), node("a", "data")])
    error = refused(lambda: cv.unresolved_registry_calibrations(duplicate))
    assert error.reason_code == "invalid_argument"


# --------------------------------------------------------------------------- observed effects

_PLAN_STAGES = (
    "graph",
    "query",
    "regime",
    "evidence",
    "source_population",
    "target_population",
    "data_snapshot",
    "row_design",
    "treatment_grid",
    "learner_folds_rng",
    "utility",
    "identification",
    "score_artifact",
    "law",
    "decision",
)


def _declared(**versions: str) -> dict[str, str]:
    from antecedent import recalc

    return {s: recalc.stage_identity(s, versions.get(s, "v1")) for s in _PLAN_STAGES}


def _plan_bindings(law_standing: cv.Standing | None = None) -> list[cv.StageBinding]:
    return [
        cv.StageBinding("identification", "claim", "identification"),
        cv.StageBinding("data", "data", "data_snapshot"),
        cv.StageBinding("target", "target_population", "target_population"),
        cv.StageBinding("score_fit", "ledger_result", "score_artifact"),
        cv.StageBinding("law", "transported_result", "law", law_standing),
        cv.StageBinding("decision", "decision", "decision"),
    ]


def _plan_graph(
    declaration: cv.PlanDeclaration, calibration: cv.Calibration
) -> cv.CompositionGraph:
    from dataclasses import replace

    n = declaration.node
    nodes = [
        n("identification"),
        n("data"),
        n("target"),
        replace(n("score_fit"), structure=n("identification").identity),
        replace(n("law"), coordinate=n("target").identity, calibration=calibration),
        n("decision"),
    ]
    edges = [
        cv.Edge("score_fit", "identification", "structural_derivation"),
        cv.Edge("score_fit", "data", "data"),
        cv.Edge("law", "score_fit", "data"),
        cv.Edge("law", "target", "support"),
        cv.Edge("decision", "law", "data"),
    ]
    return cv.CompositionGraph(nodes=nodes, edges=edges)


def test_plan_stages_give_the_nodes_and_effects_of_a_data_replacement():
    from antecedent import recalc

    previous, requested = _declared(), _declared(data_snapshot="v2")
    before = cv.plan_declaration(previous, previous, _plan_bindings())
    after = cv.plan_declaration(previous, requested, _plan_bindings())
    # a node's identity is the effective identity the plan assigned to its stage
    plan = recalc.plan_recalculation(previous, requested)
    assert (
        after.node("data").identity
        == {e.stage.value: e.identity for e in plan.entries}["data_snapshot"]
    )
    assert after.node("data").identity != before.node("data").identity
    assert after.node("identification").identity == before.node("identification").identity
    assert after.node("target").identity == before.node("target").identity
    for key in ("score_fit", "law", "decision"):
        assert after.node(key).identity != before.node(key).identity
    # planner oracle: identification is reused, everything computed from the data recomputes
    assert {e.key for e in after.effects} == {"data", "score_fit", "law", "decision"}
    assert all(e.kind == "recomputed" for e in after.effects)
    assert before.effects == ()

    result = cv.verify_composition(
        _plan_graph(after, cv.Calibration.point_only()),
        previous=_plan_graph(before, cv.Calibration.point_only()),
        effects=after.effects,
    )
    assert result.keys_of("recompute") == ("data", "decision", "law", "score_fit")
    assert result.keys_of("still_valid") == ("identification", "target")
    # the structural derivation was reused; the old fit and its uncertainty were not retained
    fit = result.classify("score_fit")
    assert "structure_reused_fit_not_retained" in {r.cause for r in fit.numerical}
    assert fit.authority_retained is False
    assert result.classify("identification").authority_retained is True
    error = refused(lambda: result.require_valid("decision"))
    assert error.reason_code == "claim_revalidation_required"
    assert error.detail == "composition_verification.recompute_required"
    assert error.remedy
    assert result.require_valid("identification").klass == "still_valid"


def test_a_plan_target_change_never_inherits_a_calibration_record_held_for_the_old_target():
    previous, requested = _declared(), _declared(target_population="v2")
    before = cv.plan_declaration(previous, previous, _plan_bindings())
    after = cv.plan_declaration(previous, requested, _plan_bindings())
    # the score artifact does not read the target (retarget licensed); the law does
    assert {e.key for e in after.effects} == {"target", "law", "decision"}
    held = cv.Calibration.held(RECORD, before.node("target").identity)
    result = cv.verify_composition(
        _plan_graph(after, held), previous=_plan_graph(before, held), effects=after.effects
    )
    law = result.classify("law")
    assert law.klass == "withdrawn"
    assert law.calibration.state == "lost" and law.calibration.newly_required
    assert law.withdrawn_for("calibration")
    assert refused(lambda: result.require_valid("law")).reason_code == "cell_not_licensed"
    assert result.classify("score_fit").klass == "still_valid"
    # the decision rests on a law that must be recomputed; its point value is not withdrawn
    assert result.classify("decision").klass == "recompute"


def test_a_stage_standing_is_stated_on_its_node_and_a_stage_outside_the_plan_is_refused():
    standing = cv.Standing(unsupported=("age>80",), trust="held")
    declared = cv.plan_declaration(_declared(), _declared(), _plan_bindings(standing))
    law = declared.node("law")
    assert law.unsupported == ("age>80",) and law.trust == "held"
    assert (
        declared.node("data").unsupported == () and declared.node("data").trust == "not_applicable"
    )
    with pytest.raises(CausalValueError):
        cv.Standing(trust="maybe")
    # `calibration` is a standing input the requested workflow does not declare
    missing = refused(
        lambda: cv.plan_declaration(
            _declared(), _declared(), [cv.StageBinding("cal", "claim", "calibration")]
        )
    )
    assert missing.reason_code == "invalid_argument"
    assert missing.detail == "composition_verification.stage_not_in_plan"
    with pytest.raises(CausalValueError):
        cv.plan_declaration(
            _declared(), _declared(), [cv.StageBinding("x", "claim", "no_such_stage")]
        )
    with pytest.raises(CausalTypeError):
        cv.plan_declaration(_declared(), _declared(), [])


def _claim_pair():
    from antecedent.joint_distribution import ScientificQuantity
    from antecedent.scientific_bridge import SourceClaim

    q = ScientificQuantity.outcome(
        "trial:effect",
        units="mmol_per_l",
        population="adults",
        regime="do(t=1)",
        functional="average_effect",
    )
    return q, SourceClaim


def test_claim_freshness_is_withdrawal_supersession_or_a_moved_snapshot_and_offline_is_silent():
    q, SourceClaim = _claim_pair()
    claim = SourceClaim.declared("trial", 0.5, q, data_snapshot="s1")
    snapshot = claim.identities["data_snapshot"]

    def one(**fields):
        return cv.freshness_effects([cv.ClaimBinding("model", claim, **fields)])

    assert one() == ()  # offline: nothing learned
    assert one(current_snapshot=snapshot) == ()  # the snapshot still matches
    assert one(withdrawn=True) == (cv.Effect.withdrawn("model", "claim.withdrawn"),)
    assert one(superseded_by=D("successor")) == (cv.Effect.recomputed("model", "claim.superseded"),)
    assert one(current_snapshot=D("another snapshot")) == (
        cv.Effect.recomputed("model", "claim.snapshot_changed"),
    )
    # withdrawal outranks supersession
    both = one(withdrawn=True, superseded_by=D("successor"))
    assert both[0].kind == "withdrawn"

    graph = cv.CompositionGraph(
        nodes=[node("model", "mechanism_composition"), node("report", "claim")],
        edges=[cv.Edge("report", "model", "model_version")],
    )
    result = cv.verify_composition(graph, previous=graph, effects=one(withdrawn=True))
    assert result.classify("report").klass == "withdrawn"
    error = refused(lambda: result.require_valid("report"))
    assert error.reason_code == "claim_license_withdrawn"
    assert error.detail == "composition_verification.input_retired"
    moved = cv.verify_composition(
        graph, previous=graph, effects=one(current_snapshot=D("another snapshot"))
    )
    assert moved.classify("report").klass == "recompute"


def test_contract_identities_name_the_layers_that_changed_even_at_an_unchanged_snapshot():
    q, SourceClaim = _claim_pair()
    claim = SourceClaim.declared("trial", 0.5, q, data_snapshot="s1")

    def effects(current):
        return cv.contract_effects([cv.ClaimBinding("model", claim, current=current)])

    assert effects(SourceClaim.declared("trial", 0.5, q, data_snapshot="s1")) == ()
    # the same trial on other data: the observation and the data snapshot layers moved
    other_data = effects(SourceClaim.declared("trial", 0.5, q, data_snapshot="s2"))
    assert other_data == (
        cv.Effect.recomputed("model", "claim.contract_changed:observation,data_snapshot"),
    )
    # another analysis of the same data: identification and inference binding moved, the data
    # snapshot did not
    other_analysis = effects(SourceClaim.declared("other", 0.5, q, data_snapshot="s1"))
    assert other_analysis == (
        cv.Effect.recomputed("model", "claim.contract_changed:identification,inference_binding"),
    )
    assert (
        claim.identities["data_snapshot"]
        == SourceClaim.declared("other", 0.5, q, data_snapshot="s1").identities["data_snapshot"]
    )
    # the changed layers drive the classification: what rests on the claim must be recomputed
    graph = cv.CompositionGraph(
        nodes=[node("model", "mechanism_composition"), node("report", "claim")],
        edges=[cv.Edge("report", "model", "model_version")],
    )
    stale = cv.verify_composition(graph, previous=graph, effects=other_data)
    assert stale.classify("report").klass == "recompute"
    error = refused(lambda: stale.require_valid("report"))
    assert error.reason_code == "claim_revalidation_required"
    # no current identities: the freshness reading stands
    assert cv.contract_effects([cv.ClaimBinding("model", claim, withdrawn=True)]) == (
        cv.Effect.withdrawn("model", "claim.withdrawn"),
    )
    with pytest.raises(CausalTypeError):
        cv.ClaimBinding("model", {"claim_id": "x"})
    with pytest.raises(CausalValueError):
        cv.ClaimBinding("model", claim, current_snapshot="not hex")


def _log_entries():
    from antecedent import recalc

    data = recalc.stage_identity("data_snapshot", "rows 1")
    premise = recalc.stage_identity("graph", "constraint c1")
    return (
        cv.InvalidationEntry(2, cv.InvalidationTarget.all_results(), "append_data", ("data_snapshot", "modified", data)),
        cv.InvalidationEntry(3, cv.InvalidationTarget.all_results(), "remove_constraint", ("graph", "removed", premise)),
        cv.InvalidationEntry(4, cv.InvalidationTarget.query_result(8), "new rows arrived"),
        cv.InvalidationEntry(5, cv.InvalidationTarget.model(1), "refit"),
    ), data, premise  # fmt: skip


def test_the_invalidation_log_classifies_in_three_states_and_the_most_severe_entry_wins():
    log, data, premise = _log_entries()
    bound = {
        "report": cv.InvalidationTarget.query_result(7),
        "other": cv.InvalidationTarget.query_result(8),
        "model": cv.InvalidationTarget.model(1),
    }
    # only the entries after the observer's version count
    assert cv.invalidation_effects(log, 5, bound) == ()
    assert cv.query_revalidation(log, 7, 5) == "still_valid"
    # version 4: only the entry for query 8 is after it
    after_four = cv.invalidation_effects(log, 4, bound)
    assert [(e.key, e.kind) for e in after_four] == [("model", "recomputed")]
    assert cv.query_revalidation(log, 7, 4) == "still_valid"
    assert cv.query_revalidation(log, 8, 3) == "recompute"
    # after version 1 the all-results data change recomputes, the removed premise withdraws
    effects = {e.key: e for e in cv.invalidation_effects(log, 1, bound)}
    assert effects["report"].kind == "withdrawn"
    assert effects["report"].text.startswith("invalidation.at_version:3:")
    assert premise in effects["report"].text
    assert effects["other"].kind == "withdrawn"  # the more severe all-results entry still wins
    assert effects["model"].kind == "withdrawn"
    assert cv.query_revalidation(log, 7, 1) == "withdrawn"
    # after version 2 the data change is already known: the removal is the only news
    after_two = {e.key: e for e in cv.invalidation_effects(log, 2, bound)}
    assert after_two["report"].kind == "withdrawn"
    # the data change alone recomputes every bound node
    only_data = log[:1]
    assert cv.query_revalidation(only_data, 7, 1) == "recompute"
    assert [e.kind for e in cv.invalidation_effects(only_data, 1, bound)] == [
        "recomputed",
        "recomputed",
        "recomputed",
    ]
    assert data in cv.invalidation_effects(only_data, 1, {"report": bound["report"]})[0].text

    graph = cv.CompositionGraph(
        nodes=[node("query_result", "ledger_result"), node("report", "claim")],
        edges=[cv.Edge("report", "query_result", "evidence")],
    )
    recomputes = cv.invalidation_effects(
        only_data, 1, {"query_result": cv.InvalidationTarget.query_result(7)}
    )
    verified = cv.verify_composition(graph, previous=graph, effects=recomputes)
    assert verified.classify("report").klass == "recompute"
    removes = cv.invalidation_effects(
        log, 1, {"query_result": cv.InvalidationTarget.query_result(7)}
    )
    assert (
        cv.verify_composition(graph, previous=graph, effects=removes).classify("report").klass
        == "withdrawn"
    )


def test_invalid_invalidation_declarations_raise():
    with pytest.raises(CausalValueError):
        cv.InvalidationTarget("query_result")
    with pytest.raises(CausalValueError):
        cv.InvalidationTarget("all_results", 3)
    with pytest.raises(CausalValueError):
        cv.InvalidationEntry(
            1, cv.InvalidationTarget.all_results(), "x", ("graph", "vanished", D("i"))
        )
    with pytest.raises(CausalTypeError):
        cv.invalidation_effects([], 0, {"report": "query 7"})


def test_ledger_effects_are_the_ledgers_own_standings_and_a_foreign_ledger_is_refused():
    ledger, (s1, _s2, _s3), (r12, r23, rall) = _ledger_world()
    bindings = {"p12": r12, "p23": r23, "pall": rall}
    assert cv.ledger_effects(ledger, bindings) == ()
    require_success(
        ledger.revise_source(require_success(el.SourceRevision.withdraw(s1, "consent withdrawn")))
    )
    effects = {e.key: e for e in cv.ledger_effects(ledger, bindings)}
    assert set(effects) == {"p12", "pall"}
    assert effects["p12"].kind == "withdrawn" and effects["pall"].kind == "recomputed"
    assert all(e.text.startswith("ledger.") for e in effects.values())
    with pytest.raises(CausalTypeError):
        cv.ledger_effects(object(), bindings)
    with pytest.raises(CausalValueError):
        cv.ledger_effects(ledger, {"p12": "not hex"})
