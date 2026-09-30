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


def law_table(target, do, measured, shift=0.0):
    """Exact joint over ``measured`` (first most significant) under ``do``."""
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
                    p += 0.25
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


def catalog(selected=("w",)):
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


def laws(shift=0.0):
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
            tuple(law_table(True, {}, NAMES, shift)),
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


def test_exported_result_is_recomputed_by_an_independent_consumer():
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


def test_not_certified_bounds_and_invalid_queries_carry_reason_codes():
    # Selection on the confounded outcome: the reduced joint has an s-hedge.
    stage = decide(selected=("y",))
    assert stage.outcome == "not_certified"
    decision = stage.decision()
    assert decision["reason_code"] == "transport_not_certified"
    assert decision["detail"] == "admg_transport.not_certified"
    candidate = decision["candidate"]
    assert candidate["proof"] is False
    assert candidate["reduced_query"] == {"outcomes": ["y", "w"], "treatments": ["x"]}
    with pytest.raises(CausalUnsupportedError, match="admg_transport.not_certified") as info:
        stage.prepare_exact(laws(), REQUESTS)
    assert info.value.reason_code == "transport_not_certified"
    # A budget stop is a receipt, never a verdict.
    exhausted = decide(max_operations=2)
    assert exhausted.outcome == "exhausted"
    receipt = exhausted.decision()["receipt"]
    assert receipt["stop"] == "search.operations" and receipt["operations_consumed"] == 2
    # Limits above the frozen maxima refuse as bounds_exceeded.
    with pytest.raises(CausalUnsupportedError, match="admg_transport.bounds_exceeded") as info:
        decide(max_operations=4097)
    assert info.value.reason_code == "route_not_supported"
    # An unknown variable and overlapping roles are refused before native code.
    with pytest.raises(CausalValueError):
        transport.ConditionalTransportQuery(
            diagram=transport.SelectionDiagram("source", "target", ["w"]),
            outcomes=["y"],
            treatments=["y"],
            conditioned_on=["w"],
        )
    with pytest.raises(CausalTypeError):
        transport.identify_admg_conditional_transport(
            graph=graph(), query="P(y|do(x),w)", catalog=catalog()
        )
