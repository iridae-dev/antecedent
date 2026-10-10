"""DAG completions of a CPDAG as one finite scenario set (2.3A X2).

The chain ``a - b - c`` has three completions: ``a -> b -> c``, ``a <- b -> c`` and
``a <- b <- c``. One target joint law is enumerated from the finite SCM
``a -> b -> c``, and the query is ``P(c | do(b = 1))``. Expected values are computed
here from the law: the first two completions give ``P(c=1 | b=1)`` (the SCM makes
``c`` independent of ``a`` given ``b``) and the third gives ``P(c=1)`` because ``c``
is not a descendant of ``b``.
"""

import antecedent
import pytest
from antecedent import Admg, Cpdag
from antecedent.errors import (
    CausalResourceError,
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

NAMES = ["a", "b", "c"]
FORWARD = frozenset({("a", "b"), ("b", "c")})
FORK = frozenset({("b", "a"), ("b", "c")})
BACKWARD = frozenset({("b", "a"), ("c", "b")})
PREFIX = b"ANTECEDENT-CPDAG-SCENARIOS\x01"


def joint(a, b, c):
    def p(one, p_one):
        return p_one if one else 1.0 - p_one

    return p(a == 1, 0.4) * p(b == 1, (0.2, 0.7)[a]) * p(c == 1, (0.1, 0.8)[b])


TABLE = tuple(joint(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))


def prob(keep):
    return sum(joint(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if keep(a, b, c))


def truth(edges):
    edges = frozenset(edges)
    if edges == FORWARD:
        return sum(
            prob(lambda x, y, z, k=k: x == k)
            * prob(lambda x, y, z, k=k: x == k and y == 1 and z == 1)
            / prob(lambda x, y, z, k=k: x == k and y == 1)
            for k in (0, 1)
        )
    if edges == FORK:
        return prob(lambda x, y, z: y == 1 and z == 1) / prob(lambda x, y, z: y == 1)
    if edges == BACKWARD:
        return prob(lambda x, y, z: z == 1)
    raise AssertionError(edges)


def chain(undirected=(("a", "b"), ("b", "c"))):
    return Cpdag.from_directed_undirected(NAMES, [], list(undirected))


def coordinates(names=NAMES):
    return [transport.VariableCoordinate(n, "binary") for n in names]


def catalog(measured=NAMES):
    return transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("obs", "target", measured=list(measured))]
    )


def laws():
    axes = tuple((n, (0.0, 1.0)) for n in NAMES)
    return transport.ExactTransportData(
        (transport.ExactDiscreteLaw("target", "obs", axes, TABLE, "target"),)
    )


def shared(identity="target-law"):
    return transport.CompletionEvidence(catalog(), identity)


def run(graph=None, evidence=None, **overrides):
    options = {
        "outcomes": ["c"],
        "treatments": ["b"],
        "source": "src_pop",
        "target": "target",
        "coordinates": coordinates(),
        "evidence": shared() if evidence is None else evidence,
        "laws": laws(),
        "at": {"b": 1.0},
    }
    options.update(overrides)
    return transport.cpdag_completion_scenarios(graph or chain(), **options)


def bind(edges, identity, measured=NAMES, certified_for=None):
    return transport.CompletionEvidence(
        catalog(measured), identity, completion=edges, certified_for=certified_for
    )


def test_x2_enumerated_truth_effects_and_envelope():
    result = run()
    assert result.scope == "cpdag_completion_scenarios_structural_envelope"
    assert len(result.completions) == 3 and result.complete and result.exportable
    assert result.counts == transport.CompletionCounts(
        identified=3, unidentified=0, unevaluated=0, not_enumerated=0, total=3
    )
    assert result.status_counts["identified"] == 3
    assert {frozenset(c.edges) for c in result.completions} == {FORWARD, FORK, BACKWARD}
    for completion in result.completions:
        assert completion.identified and completion.identification_status == "identified"
        assert completion.evidence_identity == "target-law"
        assert len(completion.id) == 64
        assert completion.mean("c") == pytest.approx(truth(completion.edges), abs=1e-12)
    assert truth(FORWARD) == pytest.approx(0.8) and truth(BACKWARD) == pytest.approx(0.38)
    # The completions disagree and the envelope spans all of them: a range, not a probability.
    assert result.envelope.mean_range("c") == pytest.approx((0.38, 0.8))
    assert len(result.envelope.completions) == 3
    assert "not_a_confidence_interval" in result.envelope.interpretation
    lower = result.envelope.means[0].lower_completion
    assert result.by_id(lower).edges and frozenset(result.by_id(lower).edges) == BACKWARD
    assert result.completion(sorted(FORK)).mean("c") == pytest.approx(0.8)
    # Nothing is weighted or renormalized, and no aggregate interval exists.
    assert not hasattr(result, "weights") and not hasattr(result, "mass")
    with pytest.raises(transport.CpdagScenarioRefusal) as refused:
        result.aggregate_interval()
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    assert refused.value.detail == "scenarios.shared_data_aggregate"
    assert isinstance(refused.value, CausalUnsupportedError)


def test_x2_missing_evidence_is_counted_not_redistributed():
    evidence = [
        bind(sorted(FORWARD), "ev-forward"),
        bind(sorted(FORK), "ev-fork"),
        bind(sorted(BACKWARD), "ev-backward", measured=["a", "b"]),
    ]
    result = run(evidence=evidence)
    status = {frozenset(c.edges): c.status for c in result.completions}
    assert status == {FORWARD: "identified", FORK: "identified", BACKWARD: "missing_evidence"}
    assert (result.counts.identified, result.counts.unidentified, result.counts.unevaluated) == (
        2,
        1,
        0,
    )
    assert result.status_counts["missing_evidence"] == 1 and len(result.completions) == 3
    backward = result.completion(sorted(BACKWARD))
    assert backward.point is None and backward.detail
    assert backward.evidence_identity == "ev-backward"
    with pytest.raises(CausalValueError):
        backward.mean("c")
    # The envelope ranges over the two identified completions only: 0.38 is absent.
    assert result.envelope.mean_range("c") == pytest.approx((0.8, 0.8))
    assert len(result.envelope.completions) == 2
    # A completion with no evidence bound to it is missing evidence as well.
    only_one = run(evidence=[bind(sorted(FORWARD), "ev-forward")])
    assert (only_one.counts.identified, only_one.counts.unidentified) == (1, 2)
    assert only_one.status_counts["missing_evidence"] == 2
    assert sum(1 for c in only_one.completions if c.evidence_identity is None) == 2


def test_x2_completion_identity_ignores_edge_insertion_order():
    canonical = run()
    reversed_edges = run(chain((("c", "b"), ("b", "a"))))
    assert canonical.cpdag_identity == reversed_edges.cpdag_identity
    assert [c.id for c in canonical.completions] == [c.id for c in reversed_edges.completions]
    assert canonical.completions == reversed_edges.completions
    assert canonical.envelope == reversed_edges.envelope
    assert canonical.counts == reversed_edges.counts
    # Evidence bound by edges or by canonical id is the same binding.
    by_id = run(
        evidence=[
            transport.CompletionEvidence(catalog(), "ev", completion=c.id)
            for c in canonical.completions
        ]
    )
    assert [c.evidence_identity for c in by_id.completions] == ["ev"] * 3
    assert by_id.envelope == canonical.envelope


def test_x2_budget_stop_retains_explored_and_unevaluated():
    # Enumeration stops after one of three completions (three orientation attempts).
    partial = run(max_steps=3)
    assert not partial.complete and partial.exportable
    assert len(partial.completions) == 1 and partial.counts.not_enumerated == 3
    assert partial.completions[0].status == "unevaluated"
    assert (partial.counts.identified, partial.counts.unidentified) == (0, 0)
    assert partial.counts.unevaluated == 4 and partial.counts.total == 4
    assert partial.receipt.stop == "search.operations"
    assert partial.receipt.explored == (partial.completions[0].id,)
    assert partial.receipt.unevaluated == ("cpdag_completions_not_enumerated_upper_bound:3",)
    assert not partial.counts.exact
    assert partial.receipt.not_enumerated == 3
    assert partial.envelope is None
    # The recorded limit replays: the consumer reproduces the identical prefix.
    replayed = transport.consume_cpdag_scenarios_artifact(partial.export())
    assert replayed.completions == partial.completions
    assert replayed.counts == partial.counts and replayed.receipt == partial.receipt

    # Enumeration takes six orientations and three stored completions; all nine operations are spent, so no decision runs.
    stopped = run(max_steps=9)
    assert len(stopped.completions) == 3 and stopped.counts.not_enumerated == 0
    assert all(c.status == "unevaluated" for c in stopped.completions)
    assert stopped.counts.unevaluated == 3 and stopped.counts.identified == 0
    assert len(stopped.receipt.unevaluated) == 3 and stopped.envelope is None

    # The depth bound must reach the number of undirected edges.
    shallow = run(max_depth=0)
    assert shallow.receipt.stop == "search.depth"
    assert shallow.completions == () and shallow.counts.not_enumerated == 4

    # The unconstrained run is complete.
    assert run().complete


def test_x2_cancelled_report_is_not_exportable():
    token = antecedent.state.CancellationToken()
    token.cancel()
    cancelled = run(cancel=token)
    assert cancelled.completions == () and cancelled.counts.not_enumerated == 4
    assert cancelled.counts.unevaluated == 4 and cancelled.envelope is None
    assert cancelled.receipt.stop == "search.cancelled" and not cancelled.exportable
    with pytest.raises(transport.CpdagScenarioRefusal) as refused:
        cancelled.export()
    assert refused.value.reason_code == "cancelled_no_claim"
    assert refused.value.detail == "cpdag_scenarios.cancelled_not_exportable"


def test_x2_export_and_fresh_consume_reproduce_the_report():
    result = run()
    artifact = result.export()
    assert artifact.startswith(PREFIX)
    consumed = transport.consume_cpdag_scenarios_artifact(artifact)
    assert consumed.cpdag_identity == result.cpdag_identity
    assert consumed.completions == result.completions
    assert consumed.counts == result.counts and consumed.envelope == result.envelope
    assert consumed.premises_digest and consumed.data_digest
    assert consumed.export() == artifact
    # Per-completion evidence with a missing completion replays as well.
    partial = run(evidence=[bind(sorted(FORWARD), "ev-forward")])
    again = transport.consume_cpdag_scenarios_artifact(partial.export())
    assert again.completions == partial.completions and again.counts == partial.counts


def _head(buf, i):
    """One CBOR item head: (major type, argument, next offset)."""
    major, info = buf[i] >> 5, buf[i] & 31
    if info < 24:
        return major, info, i + 1
    width = {24: 1, 25: 2, 26: 4, 27: 8}[info]
    return major, int.from_bytes(buf[i + 1 : i + 1 + width], "big"), i + 1 + width


def _encode_head(major, n):
    if n < 24:
        return bytes([major << 5 | n])
    for info, width in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if n < 1 << (8 * width):
            return bytes([major << 5 | info]) + n.to_bytes(width, "big")
    raise ValueError(n)


def _unframe(artifact):
    """The frame is a CBOR pair: the variable names and the artifact bytes."""
    body = artifact[len(PREFIX) :]
    _, _, i = _head(body, 0)
    _, count, i = _head(body, i)
    names = []
    for _ in range(count):
        _, length, i = _head(body, i)
        names.append(body[i : i + length].decode())
        i += length
    major, length, i = _head(body, i)
    if major == 2:
        return names, bytearray(body[i : i + length])
    inner = bytearray()
    for _ in range(length):
        _, value, i = _head(body, i)
        inner.append(value)
    return names, inner


def _frame(names, inner):
    out = bytearray(PREFIX) + _encode_head(4, 2) + _encode_head(4, len(names))
    for name in names:
        out += _encode_head(3, len(name.encode())) + name.encode()
    out += _encode_head(4, len(inner))
    for value in inner:
        out += _encode_head(0, value)
    return bytes(out)


def test_x2_tampered_artifacts_refuse_with_typed_details():
    artifact = run().export()
    names, inner = _unframe(artifact)
    assert names == NAMES and _frame(names, inner) == artifact
    # Evidence identity is data: the data digest refuses first.
    changed = bytes(inner).replace(b"target-law", b"target-lax")
    assert changed != bytes(inner)
    with pytest.raises(transport.CpdagScenarioRefusal) as refused:
        transport.consume_cpdag_scenarios_artifact(_frame(names, changed))
    assert refused.value.reason_code == "invalid_argument"
    assert refused.value.detail == "cpdag_scenarios.data_identity_mismatch"
    # The question is a premise: the premises digest refuses.
    premises = _frame(names, bytes(inner).replace(b"src_pop", b"src_poq"))
    assert premises != artifact
    with pytest.raises(transport.CpdagScenarioRefusal) as refused:
        transport.consume_cpdag_scenarios_artifact(premises)
    assert refused.value.detail == "cpdag_scenarios.premises_mismatch"
    # Damaged or foreign bytes are serialization failures, not refusals.
    with pytest.raises(CausalSerializationError):
        transport.consume_cpdag_scenarios_artifact(artifact[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(CausalSerializationError):
        transport.consume_cpdag_scenarios_artifact(b"not an artifact")
    with pytest.raises(CausalTypeError):
        transport.consume_cpdag_scenarios_artifact("text")
    # The names the report is read with are bound by the coordinate schema.
    with pytest.raises(CausalSerializationError, match="coordinate schema"):
        transport.consume_cpdag_scenarios_artifact(_frame(["c", "b", "a"], inner))
    # The consumer's own limits bound the replay: the artifact cannot raise them.
    with pytest.raises(CausalResourceError):
        transport.consume_cpdag_scenarios_artifact(artifact, max_steps=1)


def test_x2_refusals_carry_exact_code_and_detail():
    def refusal(call):
        with pytest.raises(transport.CpdagScenarioRefusal) as refused:
            call()
        assert isinstance(refused.value, CausalUnsupportedError)
        return refused.value.reason_code, refused.value.detail

    # A graph that is not maximally oriented: a -> b - c compels b -> c.
    compelled = Cpdag.from_directed_undirected(NAMES, [("a", "b")], [("b", "c")])
    assert refusal(lambda: run(compelled)) == ("invalid_argument", "cpdag_scenarios.not_a_cpdag")
    # More than six nodes.
    wide = [f"n{i}" for i in range(7)]
    seven = Cpdag.from_directed_undirected(wide, [], [("n0", "n1"), ("n1", "n2")])
    over = {"coordinates": coordinates(wide), "outcomes": ["n2"], "treatments": ["n1"]}
    over["at"] = {"n1": 1.0}
    over["evidence"] = transport.CompletionEvidence(catalog(wide[:3]), "ev")
    over["laws"] = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                tuple((n, (0.0, 1.0)) for n in wide[:3]),
                TABLE,
                "target",
            ),
        )
    )
    assert refusal(lambda: run(seven, **over)) == (
        "route_not_supported",
        "cpdag_scenarios.bounds_exceeded",
    )
    # Evidence for a completion this CPDAG does not have (a collider), or twice for one.
    collider = bind([("a", "b"), ("c", "b")], "ev-collider")
    mismatch = ("invalid_argument", "cpdag_scenarios.evidence_identity_mismatch")
    assert refusal(lambda: run(evidence=[collider])) == mismatch
    twice = [bind(sorted(FORWARD), "ev-1"), bind(sorted(FORWARD), "ev-2")]
    assert refusal(lambda: run(evidence=twice)) == mismatch
    unknown = transport.CompletionEvidence(catalog(), "ev", completion="0" * 64)
    assert refusal(lambda: run(evidence=[unknown])) == mismatch
    # Evidence certified for another completion never satisfies this one.
    ids = {frozenset(c.edges): c.id for c in run().completions}
    forged = bind(sorted(FORWARD), "ev", certified_for=ids[FORK])
    assert refusal(lambda: run(evidence=[forged])) == mismatch


def test_x2_inputs_are_checked_before_any_search():
    graph = Admg.from_edges(NAMES, [("a", "b"), ("b", "c")], [("a", "c")])
    with pytest.raises(CausalUnsupportedError, match="cpdag_scenarios.selection_or_latent") as e:
        run(graph)
    assert e.value.reason_code == "route_not_supported"
    with pytest.raises(CausalTypeError):
        run("a - b - c")
    with pytest.raises(CausalTypeError):
        run(evidence="ev")
    # One shared declaration names no completion; per-completion evidence must name one.
    with pytest.raises(CausalValueError):
        run(evidence=bind(sorted(FORWARD), "ev"))
    with pytest.raises(CausalValueError):
        run(evidence=[shared()])
    # A coordinate schema naming an unknown variable is a schema mismatch.
    bad = [*coordinates(), transport.VariableCoordinate("q", "binary")]
    with pytest.raises(CausalValueError, match="scenarios.coordinate_mismatch") as e:
        run(coordinates=bad)
    assert e.value.reason_code == "schema_mismatch"
