"""ADMG conditional transport (2.2B B1) from Python.

The graph is ``x -> y -> w`` with ``x <-> y``; the target's ``w`` mechanism
differs from the source's. Every law and the conditional truth
``P*(y | do(x), w)`` are enumerated here from one latent binary SCM,
independently of the library. ``w`` is a descendant of ``y`` and cannot be moved
by rule 2, so the answer is the reduced joint ``P*(y, w | do(x))`` normalized at
``w``: ``y``'s confounded factor comes from a source experiment, ``w``'s selected
mechanism from the target law.
"""

import itertools
import json
import math
import struct

import pytest
from antecedent import Admg
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

NAMES = ["x", "y", "w"]


def mechanisms(shift):
    return {
        "x": lambda v, u: 0.2 + 0.5 * u,
        "y": lambda v, u: 0.1 + 0.4 * v["x"] + 0.3 * u + shift,
        "w": lambda v, u: 0.2 + 0.5 * v["y"],
    }


def law_table(target, do, measured, shift=0.0, w_zero=False):
    """Exact joint over ``measured`` (first most significant) under ``do``.

    ``w_zero`` makes the target's ``w`` mechanism put no mass on ``w = 1``.
    """
    table = [0.0] * (1 << len(measured))
    mech = mechanisms(shift)
    for u in (0, 1):
        for bits in itertools.product((0, 1), repeat=len(NAMES)):
            values = dict(zip(NAMES, bits, strict=True))
            if any(values[k] != level for k, level in do.items()):
                continue
            weight = 0.5
            for name in NAMES:
                if name in do:
                    continue
                p = mech[name](values, u)
                if name == "w" and target:
                    p = 0.0 if w_zero else p + 0.25
                weight *= p if values[name] else 1.0 - p
            index = 0
            for name in measured:
                index = (index << 1) | values[name]
            table[index] += weight
    return table


def truth(x, w, shift=0.0):
    """P*(y = 1 | do(x), w) in the target."""
    joint = law_table(True, {"x": x}, ["y", "w"], shift)
    return joint[2 + w] / (joint[w] + joint[2 + w])


def graph():
    return Admg.from_edges(NAMES, [("x", "y"), ("y", "w")], [("x", "y")])


def subsets():
    for r in range(len(NAMES) + 1):
        yield from itertools.combinations(NAMES, r)


def rid(do):
    return "s-" + ("".join(do) or "obs")


def catalog(selected=("w",), drop=()):
    """Every source experiment (except those intervening on a name in ``drop``)
    and the target observational joint."""
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in NAMES)
    regimes = [
        transport.EvidenceRegime(
            rid(do),
            "source",
            kind="experimental" if do else "observational",
            interventions=list(do),
            measured=[n for n in NAMES if n not in do],
            distribution="joint",
        )
        for do in subsets()
        if not set(do) & set(drop)
    ]
    regimes.append(
        transport.EvidenceRegime(
            "t-obs", "target", kind="observational", measured=NAMES, distribution="joint"
        )
    )
    return transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates, selection_targets=tuple(selected)),
            transport.Environment("target", coordinates),
        ),
        regimes=tuple(regimes),
        bindings=tuple(
            transport.RegimeBinding(r.id, f"snap-{r.id}", sampling="independent") for r in regimes
        ),
    )


def laws(shift=0.0, w_zero=False):
    out = []
    for do in subsets():
        measured = [n for n in NAMES if n not in do]
        for levels in itertools.product((0, 1), repeat=len(do)):
            world = dict(zip(do, levels, strict=True))
            out.append(
                transport.ExactDiscreteLaw(
                    "source",
                    rid(do),
                    tuple((name, (0.0, 1.0)) for name in measured),
                    tuple(law_table(False, world, measured, shift)),
                    f"snap-{rid(do)}",
                    interventions=tuple((k, float(v)) for k, v in world.items()),
                )
            )
    out.append(
        transport.ExactDiscreteLaw(
            "target",
            "t-obs",
            tuple((name, (0.0, 1.0)) for name in NAMES),
            tuple(law_table(True, {}, NAMES, shift, w_zero)),
            "snap-t-obs",
        )
    )
    return tuple(out)


def query(selected=("w",)):
    return transport.ConditionalTransportQuery(
        diagram=transport.SelectionDiagram("source", "target", list(selected)),
        outcomes=["y"],
        treatments=["x"],
        conditioned_on=["w"],
    )


def decide(selected=("w",), **limits):
    return transport.identify_admg_conditional_transport(
        graph=graph(), query=query(selected), catalog=catalog(selected), **limits
    )


REQUESTS = [{"x": float(x), "w": float(w)} for x in (0, 1) for w in (0, 1)]


def risk(point):
    return sum(
        p for atom, p in zip(point["atoms"], point["probabilities"], strict=True) if atom == [1.0]
    )


def test_conditional_transport_identifies_and_matches_the_enumerated_truth():
    builder = decide()
    assert builder.outcome == "identified", builder.decision()
    decision = builder.decision()
    assert decision["moved"] == [] and decision["remaining"] == ["w"]
    assert decision["reduced_query"] == {"outcomes": ["y", "w"], "treatments": ["x"]}
    populations = {leaf["population"] for leaf in decision["cited_leaves"]}
    assert populations == {"source", "target"}
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    result = json.loads(prepared.estimate())
    assert prepared.plan()["compiled_plans"] == len(REQUESTS)
    for request, point in zip(REQUESTS, result["requests"], strict=True):
        expected = truth(int(request["x"]), int(request["w"]))
        assert risk(point) == pytest.approx(expected, abs=1e-12)
    # Conditioning is not a no-op: the two w levels differ.
    assert abs(risk(result["requests"][0]) - risk(result["requests"][1])) > 1e-3


def test_prepare_exact_compiles_a_plan_after_query_disposal():
    builder = decide()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    plan = prepared.plan()
    assert plan["compiled_plans"] == 4
    assert plan["remaining"] == ["w"] and plan["reduced_treatments"] == ["x"]
    assert risk(json.loads(prepared.estimate())) == pytest.approx(truth(0, 0), abs=1e-12)
    # A request that does not bind the conditioned variable is refused by name.
    with pytest.raises(CausalUnsupportedError, match="admg_transport.invalid_request") as info:
        decide().prepare_exact(laws(), {"x": 1.0})
    assert info.value.reason_code == "invalid_argument"


def test_estimate_returns_exact_points_only():
    builder = decide()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    result = json.loads(prepared.estimate(seed=3))
    assert prepared.plan()["compiled_plans"] == 4
    assert result["interval"] == {"available": False, "status": "point_only"}
    assert result["scope"] == "admg_conditional_transport_sound_incomplete"
    assert result["seed"] == 3
    assert risk(result) == pytest.approx(truth(0, 0), abs=1e-12)


def test_refresh_keeps_the_proof_and_moves_the_point():
    builder = decide()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    before = json.loads(prepared.estimate())
    plan = prepared.plan()
    prepared.refresh(laws(shift=0.1))
    assert prepared.plan() == plan
    after = json.loads(prepared.estimate())
    for request, point in zip(REQUESTS, after["requests"], strict=True):
        expected = truth(int(request["x"]), int(request["w"]), shift=0.1)
        assert risk(point) == pytest.approx(expected, abs=1e-12)
    assert risk(before) != pytest.approx(risk(after), abs=1e-6)
    # Refresh clears the last result: nothing is exported until the plan runs again.
    prepared.refresh(laws())
    with pytest.raises(CausalUnsupportedError, match="not_executed"):
        prepared.export()


def test_export_requires_an_estimate_and_frames_the_names():
    builder = decide()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    with pytest.raises(CausalUnsupportedError, match="not_executed"):
        prepared.export()
    prepared.estimate()
    assert prepared.plan()["compiled_plans"] == 4
    artifact = prepared.export()
    assert artifact.startswith(b"ANTECEDENT-ADMG-CONDITIONAL\x01")
    # The frame carries the names (CBOR text "x", "y", "w").
    assert b"\x83axayaw" in artifact


def test_exported_result_is_recomputed_by_the_artifact_consumer():
    builder = decide()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    produced = json.loads(prepared.estimate())
    assert prepared.plan()["remaining"] == ["w"]
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_admg_conditional_transport_artifact(artifact))
    assert consumed["probabilities"] == produced["probabilities"]
    for a, b in zip(consumed["requests"], produced["requests"], strict=True):
        assert a["probabilities"] == b["probabilities"]
    assert consumed["proof"]["remaining"] == [2]
    assert consumed["premises_digest"] and consumed["data_digest"]
    # A consumer whose search limit is below the producer's refuses before any work.
    with pytest.raises(CausalUnsupportedError, match="consumer_limits") as info:
        transport.consume_admg_conditional_transport_artifact(artifact, max_search_operations=100)
    assert info.value.reason_code == "route_not_supported"
    # Relabelled names in the frame (same byte length) are refused.
    swapped = artifact.replace(b"\x83axayaw", b"\x83ayaxaw", 1)
    assert swapped != artifact
    with pytest.raises(CausalUnsupportedError, match="admg_transport.invalid_artifact") as info:
        transport.consume_admg_conditional_transport_artifact(swapped)
    assert info.value.reason_code == "invalid_argument"

    # A renamed law snapshot fails the data-identity digest. The framed payload is
    # a CBOR byte array: each ASCII byte is `0x18 <byte>`.
    def framed(text):
        return b"".join(b"\x18" + bytes([c]) for c in text.encode())

    renamed = artifact.replace(framed("snap-t-obs"), framed("snap-t-obz"))
    assert renamed != artifact
    with pytest.raises(CausalUnsupportedError, match="data_identity_mismatch") as info:
        transport.consume_admg_conditional_transport_artifact(renamed)
    assert info.value.reason_code == "transport_not_certified"
    # A flipped payload byte is refused with a typed error.
    tampered = bytearray(artifact)
    tampered[-10] ^= 0xFF
    with pytest.raises((CausalSerializationError, CausalUnsupportedError)):
        transport.consume_admg_conditional_transport_artifact(bytes(tampered))


def test_counted_laws_are_not_licensed_on_the_conditional_route():
    stage = decide()
    with pytest.raises(CausalUnsupportedError, match="admg_transport.interval_withheld") as info:
        stage.prepare_empirical(laws(), REQUESTS)
    assert info.value.reason_code == "cell_not_licensed"


def witness_holds(witness, names):
    """Independent exact check of a two-model witness (fractions, names): both
    models agree on every source experiment and the target observational law,
    and ``P*(y | do(x), w)`` differs at the recorded level."""
    from fractions import Fraction

    def mass(model, target, do, values):
        cards = [len(latent["probabilities"]) for latent in model["latents"]]
        total = Fraction(0)
        for levels in itertools.product(*(range(c) for c in cards)):
            m = Fraction(1)
            for latent, level in zip(model["latents"], levels, strict=True):
                m *= Fraction(latent["probabilities"][level])
            for name in names:
                if name in do:
                    continue
                kernels = {k["node"]: k for k in model["source"]}
                if target:
                    kernels.update({k["node"]: k for k in model["target"]})
                kernel = kernels[name]
                row = 0
                for parent in kernel["parents"]:
                    row = row * 2 + values[parent]
                for edge in kernel["latents"]:
                    e = [latent["edge"] for latent in model["latents"]].index(edge)
                    row = row * cards[e] + levels[e]
                p = Fraction(kernel["p_one"][row])
                m *= p if values[name] else 1 - p
            total += m
        return total

    first, second = witness["first_model"], witness["second_model"]
    for do in subsets():
        for bits in itertools.product((0, 1), repeat=len(names)):
            values = dict(zip(names, bits, strict=True))
            if mass(first, False, do, values) != mass(second, False, do, values):
                return False
    for bits in itertools.product((0, 1), repeat=len(names)):
        values = dict(zip(names, bits, strict=True))
        if mass(first, True, (), values) != mass(second, True, (), values):
            return False

    def value(model):
        x, w, y = (
            witness["treatment_level"][0],
            witness["conditioned_level"][0],
            witness["outcome_level"][0],
        )
        num = den = Fraction(0)
        for bits in itertools.product((0, 1), repeat=len(names)):
            values = dict(zip(names, bits, strict=True))
            if values["x"] != x or values["w"] != w:
                continue
            p = mass(model, True, ("x",), values)
            den += p
            if values["y"] == y:
                num += p
        return num / den

    a, b = value(first), value(second)
    return (
        a != b and a == Fraction(witness["first_value"]) and b == Fraction(witness["second_value"])
    )


def test_a_proven_obstruction_bounds_and_invalid_queries_carry_reason_codes():
    # Selection on the confounded outcome: the reduced joint has an s-hedge and
    # an exactly verified two-model witness proves the query non-transportable.
    stage = decide(selected=("y",))
    assert stage.outcome == "proven_non_transportable"
    assert stage.identification_status == "proven_non_transportable"
    decision = stage.decision()
    assert decision["reason_code"] == "transport_proven_non_transportable"
    assert decision["detail"] == "admg_transport.proven_non_transportable"
    candidate = decision["candidate"]
    assert candidate["proof"] is True
    assert candidate["reduced_query"] == {"outcomes": ["y", "w"], "treatments": ["x"]}
    witness = decision["witness"]
    assert witness["verified"] is True
    assert witness["first_value"] != witness["second_value"]
    assert witness_holds(witness, NAMES)
    with pytest.raises(
        CausalUnsupportedError, match="admg_transport.proven_non_transportable"
    ) as info:
        stage.prepare_exact(laws(), REQUESTS)
    assert info.value.reason_code == "transport_proven_non_transportable"
    # The obstruction exports and the consumer re-verifies the witness.
    artifact = stage.export_obstruction()
    consumed = json.loads(transport.consume_admg_conditional_obstruction_artifact(artifact))
    assert consumed["identification_status"] == "proven_non_transportable"
    assert consumed["witness"] == witness
    assert consumed["remaining"] == ["w"]
    tampered = bytearray(artifact)
    tampered[-10] ^= 0xFF
    with pytest.raises((CausalSerializationError, CausalUnsupportedError)):
        transport.consume_admg_conditional_obstruction_artifact(bytes(tampered))
    # An identified stage has no obstruction to export.
    with pytest.raises(CausalUnsupportedError):
        decide().export_obstruction()
    # A budget stop is a receipt, never a verdict.
    exhausted = decide(max_operations=2)
    assert exhausted.outcome == "exhausted"
    assert exhausted.identification_status == "budget_cancel"
    assert exhausted.decision()["identification_status"] == "budget_cancel"
    receipt = exhausted.decision()["receipt"]
    assert receipt["stop"] == "search.operations" and receipt["operations_consumed"] == 2
    # Limits above the frozen maxima refuse as bounds_exceeded.
    with pytest.raises(CausalUnsupportedError, match="admg_transport.bounds_exceeded") as info:
        decide(max_operations=4097)
    assert info.value.reason_code == "route_not_supported"
    # Overlapping roles are refused before native code.
    with pytest.raises(CausalValueError, match="distinct and disjoint"):
        transport.ConditionalTransportQuery(
            diagram=transport.SelectionDiagram("source", "target", ["w"]),
            outcomes=["y"],
            treatments=["y"],
            conditioned_on=["w"],
        )
    with pytest.raises(CausalTypeError, match="ConditionalTransportQuery"):
        transport.identify_admg_conditional_transport(
            graph=graph(), query="P(y|do(x),w)", catalog=catalog()
        )


def test_an_s_hedge_without_a_witness_stays_not_certified():
    # x -> y -> w, x <-> y, selection on y, plus a district over w, a, b, c
    # whose six bidirected edges put every witness model above the search
    # bound: the reduced joint's s-hedge is an inspection-only candidate.
    names = ["x", "y", "w", "a", "b", "c"]
    district = [("w", "a"), ("w", "b"), ("w", "c"), ("a", "b"), ("a", "c"), ("b", "c")]
    wide = Admg.from_edges(names, [("x", "y"), ("y", "w")], [("x", "y"), *district])
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes = [
        transport.EvidenceRegime(
            "s-" + ("".join(do) or "obs"),
            "source",
            kind="experimental" if do else "observational",
            interventions=list(do),
            measured=[n for n in names if n not in do],
            distribution="joint",
        )
        for r in range(len(names) + 1)
        for do in itertools.combinations(names, r)
    ]
    regimes.append(
        transport.EvidenceRegime(
            "t-obs", "target", kind="observational", measured=names, distribution="joint"
        )
    )
    wide_catalog = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates, selection_targets=("y",)),
            transport.Environment("target", coordinates),
        ),
        regimes=tuple(regimes),
        bindings=tuple(
            transport.RegimeBinding(r.id, f"snap-{r.id}", sampling="independent") for r in regimes
        ),
    )
    stage = transport.identify_admg_conditional_transport(
        graph=wide, query=query(("y",)), catalog=wide_catalog
    )
    assert stage.outcome == "not_certified"
    decision = stage.decision()
    assert decision["reason_code"] == "transport_not_certified"
    assert decision["detail"] == "admg_transport.not_certified"
    assert decision["stages"][-1] == {"stage": "conditional_witness", "outcome": "out_of_scope"}
    assert decision["candidate"]["proof"] is False
    with pytest.raises(CausalUnsupportedError, match="admg_transport.not_certified") as info:
        stage.prepare_exact(laws(), REQUESTS)
    assert info.value.reason_code == "transport_not_certified"


def test_an_unknown_variable_reaches_native_code_as_an_invalid_query():
    unknown = transport.ConditionalTransportQuery(
        diagram=transport.SelectionDiagram("source", "target", ["w"]),
        outcomes=["z"],
        treatments=["x"],
        conditioned_on=["w"],
    )
    with pytest.raises(CausalUnsupportedError, match="admg_transport.invalid_query") as info:
        transport.identify_admg_conditional_transport(
            graph=graph(), query=unknown, catalog=catalog()
        )
    assert info.value.reason_code == "invalid_argument"
    assert "not a variable of the graph" in str(info.value)


def test_a_catalog_disagreeing_with_the_diagram_is_an_invalid_catalog():
    # The source environment declares selection on y; the diagram selects w.
    with pytest.raises(CausalUnsupportedError, match="admg_transport.invalid_catalog") as info:
        transport.identify_admg_conditional_transport(
            graph=graph(), query=query(("w",)), catalog=catalog(selected=("y",))
        )
    assert info.value.reason_code == "invalid_argument"
    assert "source selections disagree with the diagram" in str(info.value)


def test_a_missing_source_experiment_is_a_missing_evidence_stage():
    # y's confounded factor needs a source experiment on x; drop all of them.
    stage = transport.identify_admg_conditional_transport(
        graph=graph(), query=query(), catalog=catalog(drop=("x",))
    )
    assert stage.outcome == "missing_evidence"
    assert stage.identification_status == "missing_evidence"
    decision = stage.decision()
    assert decision["identification_status"] == "missing_evidence"
    assert decision["reason_code"] == "transport_missing_evidence"
    assert decision["detail"] == "admg_transport.missing_evidence"
    assert decision["remaining"] == ["w"]
    assert any("source" in o for o in decision["obligations"]), decision["obligations"]
    with pytest.raises(CausalUnsupportedError, match="admg_transport.missing_evidence") as info:
        stage.prepare_exact(laws(), REQUESTS)
    assert info.value.reason_code == "transport_missing_evidence"


def test_a_zero_mass_conditioning_event_is_a_support_failure():
    # The target's w mechanism puts no mass on w = 1: P*(w = 1 | do(x)) = 0.
    prepared = decide().prepare_exact(laws(w_zero=True), [{"x": 0.0, "w": 0.0}])
    point = json.loads(prepared.estimate())
    assert risk(point) == pytest.approx(truth_w_zero(0), abs=1e-12)
    prepared = decide().prepare_exact(laws(w_zero=True), [{"x": 0.0, "w": 1.0}])
    with pytest.raises(CausalUnsupportedError, match="admg_transport.support_failure") as info:
        prepared.estimate()
    assert info.value.reason_code == "transport_support_failure"
    assert "zero mass" in str(info.value)


def truth_w_zero(x):
    """P*(y = 1 | do(x), w = 0) when the target never has w = 1."""
    joint = law_table(True, {"x": x}, ["y", "w"], w_zero=True)
    return joint[2] / (joint[0] + joint[2])


def payload(raw):
    """``raw`` artifact bytes as they appear in the Python frame, whose payload
    is a CBOR array of unsigned integers (one per byte)."""
    return b"".join(bytes([c]) if c < 24 else b"\x18" + bytes([c]) for c in raw)


def edit_payload(artifact, old, new):
    """Replace the one occurrence of the raw bytes ``old`` with ``new``."""
    assert len(old) == len(new)
    assert artifact.count(payload(old)) == 1, old
    return artifact.replace(payload(old), payload(new))


def test_consumer_mismatches_carry_their_reason_codes():
    prepared = decide().prepare_exact(laws(), REQUESTS)
    produced = json.loads(prepared.estimate())
    artifact = prepared.export()
    # A stored premise (the search operation limit, 4096 -> 4095, CBOR uint16)
    # edited without re-sealing: the premises digest no longer matches.
    key = b"\x71search_operations"
    edited = edit_payload(artifact, key + b"\x19\x10\x00", key + b"\x19\x0f\xff")
    with pytest.raises(CausalUnsupportedError, match="admg_transport.premises_mismatch") as info:
        transport.consume_admg_conditional_transport_artifact(edited)
    assert info.value.reason_code == "transport_not_certified"
    assert "premises digest mismatch" in str(info.value)
    # A stored point (outside both digests) moved by one ulp: the recomputed
    # point differs bit for bit.
    p = produced["requests"][0]["probabilities"][0]
    old = b"\xfb" + struct.pack(">d", p)
    new = b"\xfb" + struct.pack(">d", math.nextafter(p, 1.0))
    edited = edit_payload(artifact, old, new)
    with pytest.raises(CausalUnsupportedError, match="admg_transport.replay_mismatch") as info:
        transport.consume_admg_conditional_transport_artifact(edited)
    assert info.value.reason_code == "transport_not_certified"
    assert "a point does not replay" in str(info.value)
    # The unedited artifact still replays.
    consumed = json.loads(transport.consume_admg_conditional_transport_artifact(artifact))
    assert consumed["probabilities"] == produced["probabilities"]


def scenario_set():
    """The fixture graph (``chain``), a graph where rule 2 moves ``w`` (``parent``:
    ``w -> y``) and the fixture graph with ``y``'s mechanism selected instead
    (``shifted``: the reduced joint has an s-hedge)."""
    coordinates = [transport.VariableCoordinate(name, "binary") for name in NAMES]
    parent = Admg.from_edges(NAMES, [("x", "y"), ("w", "y")], [("x", "y")])
    return transport.TransportScenarioSet(
        [
            transport.TransportScenario("chain", graph(), ["w"]),
            transport.TransportScenario("parent", parent, ["w"]),
            transport.TransportScenario("shifted", graph(), ["y"]),
        ],
        coordinates,
    )


def prepare_scenarios(at, **options):
    return transport.prepare_transport_scenarios(
        scenario_set(),
        outcomes=["y"],
        treatments=["x"],
        conditioned_on=["w"],
        source="source",
        target="target",
        catalog=catalog(selected=()),
        laws=laws(),
        at=at,
        **options,
    )


def test_conditional_questions_enter_the_scenario_envelope():
    prepared = prepare_scenarios({"x": 1.0, "w": 1.0})
    report = json.loads(prepared.estimate())
    rows = {row["name"]: row for row in report["scenarios"]}
    negative = rows["shifted"]["status"]
    assert {name: row["status"] for name, row in rows.items()} == {
        "chain": "identified",
        "parent": "identified",
        "shifted": negative,
    }
    # The s-hedge scenario is not certified (inspection-only candidate) unless a
    # verified two-model witness proves it; it is never identified.
    assert rows["chain"]["identification_status"] == "identified"
    if negative == "not_certified":
        assert rows["shifted"]["identification_status"] == "not_certified"
        assert rows["shifted"]["detail"].startswith("admg_transport.not_certified")
        assert "inspection-only candidate (not a proof)" in rows["shifted"]["detail"]
    else:
        assert negative == "structurally_unidentified"
        assert rows["shifted"]["identification_status"] == "proven_non_transportable"
        assert rows["shifted"]["detail"].startswith("admg_transport.proven_non_transportable")
    assert rows["shifted"]["point"] is None
    # The true scenario's point is the enumerated truth, inside the envelope.
    expected = truth(1, 1)
    assert rows["chain"]["point"]["means"]["y"] == pytest.approx(expected, abs=1e-12)
    envelope = report["envelope"]
    assert envelope["scenarios"] == ["chain", "parent"]
    mean = envelope["means"][0]
    assert mean["lower"] <= expected <= mean["upper"]
    counts = {m["status"]: m["count"] for m in report["masses"]}
    assert counts[negative] == 1 and counts["identified"] == 2
    # The artifact replays independently.
    artifact = prepared.export()
    consumed = json.loads(transport.consume_transport_scenarios_artifact(artifact))
    assert [row["status"] for row in consumed["scenarios"]] == [
        row["status"] for row in report["scenarios"]
    ]
    with pytest.raises(CausalUnsupportedError, match="scenarios.shared_data_aggregate"):
        prepared.aggregate_interval()


def test_conditional_scenario_requests_bind_the_conditioned_variables():
    with pytest.raises((CausalValueError, CausalUnsupportedError)) as info:
        prepare_scenarios({"x": 1.0})
    assert info.value.reason_code == "invalid_argument"
    assert "admg_transport.invalid_request" in str(info.value)
    with pytest.raises(CausalTypeError):
        transport.prepare_transport_scenarios(
            scenario_set(),
            outcomes=["y"],
            treatments=["x"],
            conditioned_on="w",
            source="source",
            target="target",
            catalog=catalog(selected=()),
            laws=laws(),
            at={"x": 1.0, "w": 1.0},
        )


def test_a_cancelled_decision_is_a_receipt_never_a_verdict():
    from antecedent.state import CancellationToken

    token = CancellationToken()
    token.cancel()
    stage = decide(cancel=token)
    assert stage.outcome == "exhausted"
    assert stage.identification_status == "budget_cancel"
    receipt = stage.decision()["receipt"]
    assert receipt["stop"] == "search.cancelled"
    assert receipt["operations_consumed"] is None
