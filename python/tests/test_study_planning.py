"""X6 study planning over the X9 mixed-source catalog, through the Python surface.

The fixture is the front door ``x -> z -> y`` with ``x <-> y``: an observational
study of ``(x, z)`` and a surrogate trial on ``z`` that published only separate
marginals of ``x`` and ``y``. Every law is enumerated from one structural model,
so an executed arrival is checked against the model's interventional truth.
"""

from __future__ import annotations

import itertools
import json

import pytest
from antecedent import Admg
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

NAMES = ["x", "z", "y"]
EXO = [0.4, 0.8, 0.3, 0.7, 0.2, 0.5]
FRONTDOOR = {
    "x": lambda v, e: e[5] ^ (e[0] & e[1]),
    "z": lambda v, e: e[1] if v["x"] else e[2],
    "y": lambda v, e: (e[3] if v["z"] else e[4]) ^ (e[5] & (1 - v["z"])),
}


def law_table(do, measured):
    table = [0.0] * (1 << len(measured))
    for bits in itertools.product((0, 1), repeat=len(EXO)):
        weight = 1.0
        for bit, p in zip(bits, EXO, strict=True):
            weight *= p if bit else 1.0 - p
        values = {}
        for name in NAMES:
            values[name] = do[name] if name in do else FRONTDOOR[name](values, bits)
        index = 0
        for name in measured:
            index = (index << 1) | values[name]
        table[index] += weight
    return table


def truth(x):
    return law_table({"x": x}, ["y"])[1]


def graph():
    return Admg.from_edges(NAMES, [("x", "z"), ("z", "y")], [("x", "y")])


def query():
    return transport.MixedSourceQuery(target="target", outcomes=["y"], treatments=["x"])


# (regime id, study, intervened, measured, distribution, snapshot)
BASE = [
    ("obs", "observational", (), ("x", "z"), "joint", "snap-obs"),
    ("trial", "trial", ("z",), ("x", "y"), "separate_marginals", "snap-trial"),
]


def catalog(studies):
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in NAMES)
    return transport.EvidenceCatalog(
        environments=(transport.Environment("target", coordinates),),
        regimes=tuple(
            transport.EvidenceRegime(
                rid,
                "target",
                kind="experimental" if do else "observational",
                interventions=list(do),
                measured=list(measured),
                distribution=distribution,
                study=study,
            )
            for rid, study, do, measured, distribution, _snapshot in studies
        ),
        bindings=tuple(
            transport.RegimeBinding(rid, snapshot, sampling="independent")
            for rid, *_rest, snapshot in studies
        ),
    )


def laws(studies):
    out = []
    for rid, _study, do, measured, _distribution, snapshot in studies:
        for levels in itertools.product((0, 1), repeat=len(do)):
            world = dict(zip(do, levels, strict=True))
            out.append(
                transport.ExactDiscreteLaw(
                    "target",
                    rid,
                    tuple((name, (0.0, 1.0)) for name in measured),
                    tuple(law_table(world, list(measured))),
                    snapshot,
                    interventions=tuple((k, float(v)) for k, v in world.items()),
                )
            )
    return tuple(out)


CANDIDATES = [
    transport.StudyCandidate(
        "trial_xy", "target", ["x", "y"], 3, "randomise z", interventions=["z"]
    ),
    transport.StudyCandidate("trial_y", "target", ["y"], 2, "randomise z", interventions=["z"]),
    transport.StudyCandidate("full_observational", "target", ["x", "z", "y"], 5, "cohort"),
]


def plan():
    return transport.plan_studies(
        graph=graph(), query=query(), catalog=catalog(BASE), candidates=CANDIDATES
    )


def arrival(deliver, snapshot="arrival"):
    """The base studies plus every delivered regime, bound to ``snapshot``."""
    delivered = [
        (
            d["regime"],
            d["study"],
            tuple(d["interventions"]),
            tuple(d["measured"]),
            "joint",
            snapshot,
        )
        for d in deliver
    ]
    return BASE + delivered, delivered


def risk(result):
    return sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )


def execute_arrival(deliver):
    """The arriving catalog identifies through the public route; its retained plan
    executes after the builder is discarded and matches the enumerated truth."""
    studies, delivered = arrival(deliver)
    builder = transport.identify_mixed_source_transport(
        graph=graph(), query=query(), catalog=catalog(studies)
    )
    assert builder.outcome == "identified", builder.decision()
    prepared = builder.prepare_exact(laws(BASE[:1] + delivered), [{"x": 0.0}, {"x": 1.0}])
    del builder
    result = json.loads(prepared.estimate())
    program = prepared.plan()
    assert [s["source"]["regime"] for s in program["steps"] if s["source"]] == [
        "obs",
        deliver[0]["regime"],
    ]
    assert [risk(r) for r in result["requests"]] == pytest.approx([truth(0), truth(1)], abs=1e-12)


def test_the_cheapest_sufficient_study_arrives_and_executes_its_plan():
    stage = plan()
    assert stage.outcome == "sufficient"
    decided = stage.plan()
    assert decided["inference_claim"] == "none"
    assert decided["failure"]["detail"] == "mixed_search.missing_joint"
    ranked = [(p["candidates"], p["cost_units"], p["stage"]) for p in decided["proposals"]]
    assert ranked == [
        (["trial_y"], 2, "mixed_source:rule_search"),
        (["trial_xy"], 3, "mixed_source:rule_search"),
        (["full_observational"], 5, "mixed_source:named:target_first_sid"),
    ]
    assert decided["minimal"] is True
    (repair,) = decided["proposals"][0]["repairs"]
    assert repair["required_margin"] == ["y"] and repair["intervened"] == ["z"]
    assert repair["proof_steps"]
    deliver = decided["proposals"][0]["deliver"]
    assert [(d["regime"], d["interventions"], d["measured"]) for d in deliver] == [
        ("trial_y#0", ["z"], ["y"])
    ]
    studies, _ = arrival(deliver)
    received = stage.receive(0, catalog(studies), "arrival")
    assert received["outcome"] == "identified"
    assert received["cited_regimes"] == ["obs", "trial_y#0"]
    execute_arrival(deliver)


def test_arrival_of_another_shape_is_refused_and_the_right_shape_executes():
    stage = plan()
    deliver = stage.plan()["proposals"][0]["deliver"]
    wrong = [{**deliver[0], "measured": ["x", "y"]}]
    studies, _ = arrival(wrong)
    with pytest.raises(CausalUnsupportedError, match="study_plan.arrival_mismatch") as refused:
        stage.receive(0, catalog(studies), "arrival")
    assert refused.value.reason_code == "invalid_argument"
    assert_registered_refusal(refused.value)
    studies, _ = arrival(deliver)
    with pytest.raises(CausalUnsupportedError, match="study_plan.arrival_mismatch"):
        stage.receive(0, catalog(studies), "another-snapshot")
    assert stage.receive(0, catalog(studies), "arrival")["outcome"] == "identified"
    execute_arrival(deliver)


def test_export_binds_the_plan_and_a_consumer_refuses_what_it_cannot_afford():
    stage = plan()
    artifact = stage.export()
    assert artifact.startswith(b"ANTECEDENT-STUDY-PLAN\x01")
    with pytest.raises(CausalUnsupportedError, match="study_plan.bounds_exceeded") as refused:
        transport.replay_study_plan(artifact, max_operations=1_000, max_depth=16)
    assert refused.value.reason_code == "route_not_supported"
    with pytest.raises(CausalSerializationError):
        transport.replay_study_plan(b"not a plan")
    tampered = bytearray(artifact)
    tampered[-3] ^= 0x01
    with pytest.raises((CausalUnsupportedError, CausalSerializationError)):
        transport.replay_study_plan(bytes(tampered))
    execute_arrival(stage.plan()["proposals"][0]["deliver"])


def test_an_exported_study_plan_is_replayed_by_an_independent_consumer():
    stage = plan()
    live = stage.plan()
    artifact = stage.export()
    del stage
    replayed = json.loads(transport.replay_study_plan(artifact))
    for key in ("failure", "subsets", "proposals", "minimal", "operations_consumed", "limits"):
        assert replayed[key] == live[key], key
    assert len(replayed["plan_digest"]) == 64
    # The consumer's plan says what to deliver; that arrival executes.
    execute_arrival(replayed["proposals"][0]["deliver"])


def test_invalid_designs_and_a_base_that_identifies_are_refused_by_reason():
    restricted = transport.StudyCandidate(
        "trial_z1", "target", ["y"], 1, "randomise z", interventions=["z"], levels=[{"z": 1.0}]
    )
    with pytest.raises(CausalUnsupportedError, match="study_plan.invalid_candidate") as refused:
        transport.plan_studies(
            graph=graph(), query=query(), catalog=catalog(BASE), candidates=[restricted]
        )
    assert refused.value.reason_code == "invalid_argument"
    elsewhere = transport.StudyCandidate("abroad", "elsewhere", ["y"], 1, "cohort")
    with pytest.raises(CausalUnsupportedError, match="study_plan.invalid_candidate"):
        transport.plan_studies(
            graph=graph(), query=query(), catalog=catalog(BASE), candidates=[elsewhere]
        )
    complete = BASE[:1] + [("trial", "trial", ("z",), ("x", "y"), "joint", "snap-trial")]
    with pytest.raises(CausalUnsupportedError, match="study_plan.no_failure_to_repair"):
        transport.plan_studies(
            graph=graph(), query=query(), catalog=catalog(complete), candidates=CANDIDATES
        )
    with pytest.raises(CausalUnsupportedError, match="study_plan.bounds_exceeded") as refused:
        transport.plan_studies(
            graph=graph(), query=query(), catalog=catalog(BASE), candidates=CANDIDATES, max_depth=17
        )
    assert_registered_refusal(refused.value)


def test_a_budget_stop_is_a_receipt_never_a_verdict():
    stage = transport.plan_studies(
        graph=graph(),
        query=query(),
        catalog=catalog(BASE),
        candidates=CANDIDATES,
        max_operations=plan().plan()["operations_consumed"] - 1,
    )
    decided = stage.plan()
    assert decided["stop"]["kind"] == "budget"
    assert decided["stop"]["stop"] == "search.operations"
    assert decided["stop"]["unevaluated"]
    assert decided["proposals"], "the proposals verified before the stop are kept"
    with pytest.raises(CausalUnsupportedError, match="study_plan.budget") as refused:
        transport.plan_studies(
            graph=graph(),
            query=query(),
            catalog=catalog(BASE),
            candidates=CANDIDATES,
            max_operations=2,
        )
    assert refused.value.reason_code == "transport_budget_cancel"
