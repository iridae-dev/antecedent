"""2.3 B2 ordered-response recovery row through the Python facade.

The truth is the enumerated binary SCM of the Rust core tests (``b2_recovery_chain_truth.rs``):
``P(X1 = 1) = 0.37``, ``P(X2 = 1 | X1) = (0.2, 0.7)``, head response ``P(R_h = 1) = 0.65`` and
tail response ``P(R_t = 1 | R_h, X_h) = [[0.4, 0.55], [0.3, 0.8]]`` (dependent) or
``P(R_t = 1 | R_h) = (0.3, 0.8)`` (independent). The pattern law and the target law are
marginals of the enumerated 16-cell joint. The nonrecoverable fixture adds the self-censoring
edge ``X2 -> R2``; its witness numbers are exact integers over ``60**4``.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.graph import Admg
from antecedent.recovery_chain import (
    ChainRoles,
    PatternLaw,
    RecoveryChainQuery,
    RecoveryChainRefusal,
    consume_recovery_chain_artifact,
    decide_chain,
    recover_chain,
)

from _refusal import assert_registered_refusal

NAMES = ["x1", "x2", "r1", "r2", "p1", "p2"]
QUERY = RecoveryChainQuery(ChainRoles("x1", "r1", "p1"), ChainRoles("x2", "r2", "p2"))


def edges(head: int, dependent: bool, extra=()):
    out = [("x1", "x2"), ("x1", "p1"), ("r1", "p1"), ("x2", "p2"), ("r2", "p2")]
    if head == 0:
        out.append(("r1", "r2"))
        if dependent:
            out.append(("x1", "r2"))
    else:
        out.append(("r2", "r1"))
        if dependent:
            out.append(("x2", "r1"))
    return [*out, *extra]


def graph(head=0, dependent=True, extra=(), nodes=NAMES):
    return {"nodes": list(nodes), "edges": edges(head, dependent, extra)}


def scm(head: int, dependent: bool):
    """Enumerated SCM: (pattern law, truth), axes in declaration order."""
    tail = 1 - head
    both = np.zeros((2, 2))
    only = np.zeros((2, 2))
    neither = 0.0
    truth = np.zeros((2, 2))
    for cell in range(16):
        x = [(cell >> 3) & 1, (cell >> 2) & 1]
        r = [(cell >> 1) & 1, cell & 1]
        p_x1 = 0.37 if x[0] == 1 else 0.63
        p_x2_one = 0.7 if x[0] == 1 else 0.2
        p_x2 = p_x2_one if x[1] == 1 else 1.0 - p_x2_one
        p_head = 0.65 if r[head] == 1 else 0.35
        tail_one = [[0.4, 0.55], [0.3, 0.8]][r[head]][x[head]] if dependent else [0.3, 0.8][r[head]]
        p_tail = tail_one if r[tail] == 1 else 1.0 - tail_one
        mass = p_x1 * p_x2 * p_head * p_tail
        truth[x[0], x[1]] += mass
        if r == [1, 1]:
            both[x[0], x[1]] += mass
        elif r == [1, 0]:
            only[0, x[0]] += mass
        elif r == [0, 1]:
            only[1, x[1]] += mass
        else:
            neither += mass
    law = PatternLaw(both=both.tolist(), only_first=only[0], only_second=only[1], neither=neither)
    return law, truth


def refused(call, detail: str, code: str) -> RecoveryChainRefusal:
    with pytest.raises(RecoveryChainRefusal) as info:
        call()
    error = info.value
    assert isinstance(error, CausalUnsupportedError)
    assert error.detail == detail
    assert error.reason_code == code
    assert_registered_refusal(error)
    return error


@pytest.mark.parametrize("head", [0, 1])
@pytest.mark.parametrize("dependent", [True, False])
def test_b2_chain_recovery_equals_the_enumerated_truth(head: int, dependent: bool) -> None:
    law, truth = scm(head, dependent)
    result = recover_chain(graph(head, dependent), QUERY, law)
    np.testing.assert_allclose(result.law, truth, atol=1e-12)
    assert result.plan.head == head
    assert result.plan.tail_depends_on_head_variable is dependent
    assert result.rule_version == "b2.recovery_chain.v1"
    assert result.law.sum() == pytest.approx(1.0, abs=1e-12)
    assert result.to_dict()["interval"] == {"available": False, "status": "point_only"}


def test_b2_recovery_is_not_complete_case_analysis() -> None:
    law, truth = scm(0, True)
    complete_case = np.asarray(law.both) / np.sum(law.both)
    assert abs(complete_case[0, 0] - truth[0, 0]) > 1e-3
    result = recover_chain(graph(0, True), QUERY, law)
    np.testing.assert_allclose(result.law, truth, atol=1e-12)


def test_b2_answer_is_invariant_to_node_order_and_labels() -> None:
    law, truth = scm(0, True)
    reference = recover_chain(graph(0, True), QUERY, law)
    reversed_nodes = list(reversed(NAMES))
    permuted = recover_chain(graph(0, True, nodes=reversed_nodes), QUERY, law)
    np.testing.assert_allclose(permuted.law, reference.law, atol=1e-15)
    rename = {n: f"v_{n}_z" for n in NAMES}
    renamed_query = RecoveryChainQuery(
        ChainRoles(rename["x1"], rename["r1"], rename["p1"]),
        ChainRoles(rename["x2"], rename["r2"], rename["p2"]),
    )
    renamed_graph = {
        "nodes": [rename[n] for n in NAMES],
        "edges": [(rename[a], rename[b]) for a, b in edges(0, True)],
    }
    renamed = recover_chain(renamed_graph, renamed_query, law)
    np.testing.assert_allclose(renamed.law, truth, atol=1e-12)
    # An Admg is accepted as the same graph.
    admg = Admg.from_edges(NAMES, edges(0, True))
    np.testing.assert_allclose(recover_chain(admg, QUERY, law).law, truth, atol=1e-12)


def test_b2_decide_reports_the_checked_plan_without_a_law() -> None:
    decision = decide_chain(graph(1, True), QUERY)
    assert decision.outcome == "recovered"
    assert decision.witness is None and decision.plan is not None
    assert decision.plan.head == 1 and decision.plan.tail_depends_on_head_variable
    assert "P(R_h=1)" in decision.plan.formula
    assert decision.plan.operations_consumed > 0
    assert any(p.startswith("response_chain:") for p in decision.plan.premises)
    with pytest.raises(RecoveryChainRefusal):
        decision.export()


def test_b2_self_censoring_edge_refuses_with_a_verified_exact_witness() -> None:
    law, _ = scm(0, True)
    error = refused(
        lambda: recover_chain(graph(0, True, extra=[("x2", "r2")]), QUERY, law),
        "recovery_chain.nonrecoverable_witness",
        "transport_proven_non_transportable",
    )
    witness = error.witness
    assert witness is not None
    assert witness.self_censoring_edge == ("x2", "r2")
    assert witness.observed_cells_equal == 9
    assert witness.denominator == 60**4
    assert witness.differing_target_cell == (0, 0)
    # P(X1 = 0) P(X2 = 0) over 60^4: (30 / 60)(30 / 60) against (30 / 60)(36 / 60).
    assert witness.target_masses == (30 * 30 * 3600, 30 * 36 * 3600)
    assert [m["node"] for m in witness.first_model] == ["x1", "x2", "r1", "r2"]
    assert error.artifact is not None
    # The decision form returns the same witness as a result.
    decision = decide_chain(graph(0, True, extra=[("x2", "r2")]), QUERY)
    assert decision.outcome == "nonrecoverable" and decision.plan is None
    assert decision.witness == witness
    assert decision.export() == error.artifact


def test_b2_head_self_censoring_with_an_independent_tail_is_also_witnessed() -> None:
    decision = decide_chain(graph(0, False, extra=[("x1", "r1")]), QUERY)
    assert decision.outcome == "nonrecoverable"
    assert decision.witness is not None
    assert decision.witness.self_censoring_edge == ("x1", "r1")
    assert decision.witness.target_masses[0] != decision.witness.target_masses[1]


def test_b2_graphs_outside_the_row_are_not_supported_never_nonrecoverable() -> None:
    law, _ = scm(0, True)
    unsupported = "recovery_chain.unsupported_mechanism"
    # A response drives the other variable's proxy.
    refused(
        lambda: recover_chain(graph(0, True, extra=[("r1", "p2")]), QUERY, law),
        unsupported,
        "route_not_supported",
    )
    # The tail variable drives the head response.
    refused(
        lambda: recover_chain(graph(0, True, extra=[("x2", "r1")]), QUERY, law),
        unsupported,
        "route_not_supported",
    )
    # No response chain: the 2.2 route's graph.
    no_chain = {
        "nodes": NAMES,
        "edges": [("x1", "x2"), ("x1", "p1"), ("r1", "p1"), ("x2", "p2"), ("r2", "p2")],
    }
    refused(lambda: recover_chain(no_chain, QUERY, law), unsupported, "route_not_supported")
    # Unmeasured confounding.
    confounded = {**graph(0, True), "bidirected": [("x1", "x2")]}
    refused(lambda: recover_chain(confounded, QUERY, law), unsupported, "route_not_supported")
    # An unsupported edge outranks a self-censoring edge: no witness is claimed.
    both = graph(0, True, extra=[("x2", "r2"), ("x2", "r1")])
    error = refused(lambda: recover_chain(both, QUERY, law), unsupported, "route_not_supported")
    assert error.witness is None


def test_b2_malformed_roles_are_invalid_queries() -> None:
    law, _ = scm(0, True)
    broken = graph(0, True)
    broken["edges"] = [e for e in broken["edges"] if e != ("r2", "p2")]
    refused(
        lambda: recover_chain(broken, QUERY, law),
        "recovery_chain.invalid_query",
        "invalid_argument",
    )
    with pytest.raises(ValueError, match="three distinct"):
        ChainRoles("x1", "x1", "p1")
    with pytest.raises(TypeError, match="graph must be an Admg"):
        recover_chain("graph", QUERY, law)  # type: ignore[arg-type]


def test_b2_zero_mass_cell_and_invalid_laws_refuse() -> None:
    law, _ = scm(0, True)
    both = [list(row) for row in law.both]
    moved = both[1][1]
    both[1][1] = 0.0
    broken = PatternLaw(
        both=both,
        only_first=law.only_first,
        only_second=law.only_second,
        neither=law.neither + moved,
    )
    refused(
        lambda: recover_chain(graph(0, True), QUERY, broken),
        "recovery_chain.positivity",
        "transport_support_failure",
    )
    not_normalized = PatternLaw(
        both=law.both, only_first=law.only_first, only_second=law.only_second, neither=0.9
    )
    refused(
        lambda: recover_chain(graph(0, True), QUERY, not_normalized),
        "recovery_chain.invalid_observed_law",
        "invalid_argument",
    )
    with_nan = PatternLaw(
        both=law.both, only_first=law.only_first, only_second=law.only_second, neither=float("nan")
    )
    refused(
        lambda: recover_chain(graph(0, True), QUERY, with_nan),
        "recovery_chain.invalid_observed_law",
        "invalid_argument",
    )
    with pytest.raises(ValueError, match="2 x 2"):
        PatternLaw(both=[[0.1, 0.1]], only_first=[0.1, 0.1], only_second=[0.1, 0.1], neither=0.5)


def test_b2_export_consume_round_trip_recomputes_the_recovered_law() -> None:
    law, truth = scm(1, True)
    result = recover_chain(graph(1, True), QUERY, law)
    artifact = result.export()
    assert isinstance(artifact, bytes)
    fresh = consume_recovery_chain_artifact(artifact)
    np.testing.assert_allclose(fresh.law, truth, atol=1e-12)
    assert fresh.cells == result.cells
    assert fresh.plan == result.plan
    assert fresh.premises_digest == result.premises_digest
    assert fresh.data_digest == result.data_digest
    assert fresh.export() == artifact
    # A mapping is accepted for the law.
    mapping = {
        "both": [list(row) for row in law.both],
        "only_first": list(law.only_first),
        "only_second": list(law.only_second),
        "neither": law.neither,
    }
    assert recover_chain(graph(1, True), QUERY, mapping).cells == result.cells


def test_b2_nonrecoverable_artifact_reverifies_its_witness() -> None:
    law, _ = scm(0, True)
    with pytest.raises(RecoveryChainRefusal) as info:
        recover_chain(graph(0, True, extra=[("x2", "r2")]), QUERY, law)
    artifact = info.value.artifact
    assert artifact is not None
    error = refused(
        lambda: consume_recovery_chain_artifact(artifact),
        "recovery_chain.nonrecoverable_witness",
        "transport_proven_non_transportable",
    )
    assert error.witness == info.value.witness


def test_b2_corrupt_truncated_and_foreign_artifacts_refuse() -> None:
    law, _ = scm(0, True)
    artifact = recover_chain(graph(0, True), QUERY, law).export()
    with pytest.raises(CausalSerializationError):
        consume_recovery_chain_artifact(artifact[:-5])
    with pytest.raises(CausalSerializationError):
        consume_recovery_chain_artifact(b"not an artifact")
    corrupt = bytearray(artifact)
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises((CausalSerializationError, RecoveryChainRefusal)):
        consume_recovery_chain_artifact(bytes(corrupt))
    with pytest.raises(TypeError, match="artifact must be bytes"):
        consume_recovery_chain_artifact("text")  # type: ignore[arg-type]
