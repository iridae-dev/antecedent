"""Bounded mixed-source proof search from Python.

Every law and the truth are enumerated here from binary structural equations,
independently of the library. The chain ``x -> z -> y`` is measured by two
studies, one of ``{x, z}`` and one of ``{z, y}``; the confounded front-door graph
``x -> z -> y``, ``x <-> y`` is measured by an observational study of ``{x, z}``
and a trial of ``do(z)`` measuring ``{x, y}``. No study alone, and no named
theorem route, identifies ``P(y | do(x))``; the studies together do.
"""

import itertools
import json

import pytest
from antecedent import Admg
from antecedent.errors import (
    CausalResourceError,
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

NAMES = ["x", "z", "y"]

# e[0..2] private noise, e[3..4] extra mechanism bits, e[5] the latent x <-> y bit.
EXO = [0.4, 0.8, 0.3, 0.7, 0.2, 0.5]

CHAIN = {
    "x": lambda v, e: e[0],
    "z": lambda v, e: e[1] if v["x"] else e[2],
    "y": lambda v, e: e[3] if v["z"] else e[4],
}
FRONTDOOR = {
    "x": lambda v, e: e[5] ^ (e[0] & e[1]),
    "z": lambda v, e: e[1] if v["x"] else e[2],
    "y": lambda v, e: (e[3] if v["z"] else e[4]) ^ (e[5] & (1 - v["z"])),
}


def law_table(mechanisms, do, measured):
    """Exact joint over ``measured`` (first most significant) under ``do``."""
    table = [0.0] * (1 << len(measured))
    for bits in itertools.product((0, 1), repeat=len(EXO)):
        weight = 1.0
        for bit, p in zip(bits, EXO, strict=True):
            weight *= p if bit else 1.0 - p
        values = {}
        for name in NAMES:
            values[name] = do[name] if name in do else mechanisms[name](values, bits)
        index = 0
        for name in measured:
            index = (index << 1) | values[name]
        table[index] += weight
    return table


def truth(mechanisms, x):
    return law_table(mechanisms, {"x": x}, ["y"])[1]


def chain_graph():
    return Admg.from_edges(NAMES, [("x", "z"), ("z", "y")], [])


def frontdoor_graph():
    return Admg.from_edges(NAMES, [("x", "z"), ("z", "y")], [("x", "y")])


def bow_graph():
    return Admg.from_edges(["x", "y"], [("x", "y")], [("x", "y")])


def query():
    return transport.MixedSourceQuery(target="target", outcomes=["y"], treatments=["x"])


# (regime id, study, intervened, measured, distribution)
CHAIN_STUDIES = [
    ("s1", "study-1", (), ("x", "z"), "joint"),
    ("s2", "study-2", (), ("z", "y"), "joint"),
]
FRONTDOOR_STUDIES = [
    ("obs", "observational", (), ("x", "z"), "joint"),
    ("trial", "trial", ("z",), ("x", "y"), "joint"),
]


def catalog(studies, names=NAMES, **fields):
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    regimes = tuple(
        transport.EvidenceRegime(
            rid,
            fields.get("population", "target"),
            kind="experimental" if do else "observational",
            interventions=list(do),
            measured=list(measured),
            distribution=distribution,
            study=study,
        )
        for rid, study, do, measured, distribution in studies
    )
    bindings = tuple(
        transport.RegimeBinding(rid, f"snap-{rid}", sampling="independent") for rid, *_ in studies
    )
    return transport.EvidenceCatalog(
        environments=(transport.Environment("target", coordinates),),
        regimes=regimes,
        bindings=bindings,
    )


def laws(mechanisms, studies, snapshot_suffix=""):
    out = []
    for rid, _study, do, measured, _distribution in studies:
        for levels in itertools.product((0, 1), repeat=len(do)):
            world = dict(zip(do, levels, strict=True))
            out.append(
                transport.ExactDiscreteLaw(
                    "target",
                    rid,
                    tuple((name, (0.0, 1.0)) for name in measured),
                    tuple(law_table(mechanisms, world, list(measured))),
                    f"snap-{rid}{snapshot_suffix}",
                    interventions=tuple((k, float(v)) for k, v in world.items()),
                )
            )
    return tuple(out)


def decide(graph, studies, **limits):
    return transport.identify_mixed_source_transport(
        graph=graph, query=query(), catalog=catalog(studies), **limits
    )


def risk(result):
    return sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )


def test_complementary_studies_identify_what_no_single_study_can():
    stage = decide(chain_graph(), CHAIN_STUDIES)
    assert stage.outcome == "identified", stage.decision()
    decision = stage.decision()
    assert decision["rule_set"] == "x9.rules.v1"
    assert decision["stages"] == [{"stage": "target_first_sid", "outcome": "missing_evidence"}]
    # Every step records its premises; the leaves name the study and regime that supply them.
    steps = decision["steps"]
    assert all(all(p < s["step"] for p in s["premises"]) for s in steps)
    leaves = {s["source"]["regime"]: s["source"]["study"] for s in steps if s["source"]}
    assert leaves == {"s1": "study-1", "s2": "study-2"}
    assert decision["cited_regimes"] == ["s1", "s2"]
    assert decision["alternatives"] == []
    # The compact proof graph renders every step and marks the target.
    graph_lines = decision["proof_graph"]
    assert len(graph_lines) == len(steps)
    assert graph_lines[-1].endswith("<- target") and "P(y | do(x))" in graph_lines[-1]
    for x in (0, 1):
        prepared = stage.prepare_exact(laws(CHAIN, CHAIN_STUDIES), {"x": float(x)})
        result = json.loads(prepared.estimate())
        assert result["scope"] == "mixed_source_proof_search_sound_incomplete"
        assert risk(result) == pytest.approx(truth(CHAIN, x), abs=1e-12)
        assert result["interval"] == {"available": False, "status": "point_only"}
        assert prepared.cited_studies == ["study-1", "study-2"]
    # No single study identifies it.
    for alone in CHAIN_STUDIES:
        assert decide(chain_graph(), [alone]).outcome == "not_certified"


def test_a_surrogate_trial_corrects_the_confounded_observational_conditional():
    stage = decide(frontdoor_graph(), FRONTDOOR_STUDIES)
    assert stage.outcome == "identified", stage.decision()
    rules = {s["rule"] for s in stage.decision()["steps"]}
    assert {"rule3_insert", "rule2_to_observation", "rule2_to_do"} <= rules
    prepared = stage.prepare_exact(laws(FRONTDOOR, FRONTDOOR_STUDIES), [{"x": 0.0}, {"x": 1.0}])
    result = json.loads(prepared.estimate())
    assert [risk(r) for r in result["requests"]] == pytest.approx(
        [truth(FRONTDOOR, 0), truth(FRONTDOOR, 1)], abs=1e-12
    )
    # The fixture is not trivial: the observational conditional is confounded.
    joint = law_table(FRONTDOOR, {}, ["x", "y"])
    assert abs(joint[3] / (joint[2] + joint[3]) - truth(FRONTDOOR, 1)) > 1e-2
    for alone in FRONTDOOR_STUDIES:
        assert decide(frontdoor_graph(), [alone]).outcome == "not_certified"


def test_a_missing_joint_names_the_exact_leaf():
    marginals = [CHAIN_STUDIES[0], ("s2", "study-2", (), ("z", "y"), "separate_marginals")]
    stage = decide(chain_graph(), marginals)
    assert stage.outcome == "missing_evidence"
    decision = stage.decision()
    assert decision["reason"] == "transport_missing_evidence"
    (leaf,) = decision["missing_leaves"]
    assert leaf["regime"] == "s2" and leaf["study"] == "study-2"
    assert leaf["variables"] == ["z", "y"]
    assert leaf["supplied_as_separate_marginals"] == ["z", "y"]
    marked = [line for line in decision["proof_graph"] if "MISSING" in line]
    assert len(marked) == 1 and "P(z, y)" in marked[0]
    with pytest.raises(CausalUnsupportedError, match="transport_missing_evidence"):
        stage.prepare_exact(laws(CHAIN, CHAIN_STUDIES), {"x": 1.0})


def test_a_bounded_unsuccessful_search_stays_unresolved():
    # The bow arc is not identifiable; the search still only reports what it explored.
    stage = transport.identify_mixed_source_transport(
        graph=bow_graph(),
        query=query(),
        catalog=catalog([("s", "study", (), ("x", "y"), "joint")], names=["x", "y"]),
    )
    assert stage.outcome == "not_certified"
    decision = stage.decision()
    assert decision["reason"] == "transport_not_certified"
    assert decision["goal"]["text"] == "P(y | do(x))"
    assert decision["frontier"] and decision["generations"] > 0
    assert decision["stages"][-1] == {"stage": "rule_search", "outcome": "not_certified"}
    with pytest.raises(CausalUnsupportedError, match="transport_not_certified"):
        stage.prepare_exact((), {"x": 1.0})
    # A budget stop is a resource outcome with a receipt, never a verdict.
    exhausted = decide(chain_graph(), CHAIN_STUDIES, max_operations=30)
    assert exhausted.outcome == "exhausted"
    assert exhausted.identification_status == "budget_cancel"
    assert exhausted.decision()["identification_status"] == "budget_cancel"
    receipt = exhausted.decision()["limits_receipt"]
    assert receipt["stop"] == "search.operations" and receipt["operations_consumed"] == 30
    assert "stage:target_first_sid" in receipt["explored"]
    assert "stage:rule_search" in receipt["unevaluated"]
    with pytest.raises(Exception, match="transport_budget_cancel"):
        exhausted.prepare_exact(laws(CHAIN, CHAIN_STUDIES), {"x": 1.0})
    depth = decide(chain_graph(), CHAIN_STUDIES, max_depth=1)
    assert depth.decision()["limits_receipt"]["stop"] == "search.depth"


def test_a_named_route_that_solves_the_query_is_never_searched():
    stage = decide(chain_graph(), [("full", "study", (), ("x", "z", "y"), "joint")])
    assert stage.outcome == "named_route"
    # A theorem-scoped route identifies it: the canonical status says so.
    assert stage.identification_status == "identified"
    decision = stage.decision()
    assert decision["identification_status"] == "identified"
    assert decision["route"] == "target_first_sid" and decision["reason"] == "route_not_supported"
    with pytest.raises(CausalUnsupportedError, match="route_not_supported"):
        stage.prepare_exact((), {"x": 1.0})


def test_alternative_derivations_appear_only_when_found():
    trials = [
        ("a", "trial-a", ("x",), ("y",), "joint"),
        ("b", "trial-b", ("x",), ("y",), "joint"),
    ]
    decision = decide(chain_graph(), trials).decision()
    assert decision["cited_regimes"] == ["a"]
    (alternative,) = decision["alternatives"]
    assert alternative["cited_regimes"] == ["b"]


def test_bounds_and_model_artifacts_are_refused_by_name():
    with pytest.raises(CausalUnsupportedError, match="bounds_exceeded"):
        decide(chain_graph(), CHAIN_STUDIES, max_operations=20_001)
    posterior = transport.EvidenceCatalog(
        environments=(
            transport.Environment(
                "target", tuple(transport.VariableCoordinate(n, "binary") for n in NAMES)
            ),
        ),
        regimes=(
            transport.EvidenceRegime("s1", "target", measured=["x", "z"], study="study-1"),
            transport.EvidenceRegime(
                "post",
                "target",
                kind="experimental",
                interventions=["z"],
                measured=["x", "y"],
                model_artifact="posterior-1",
            ),
        ),
        bindings=tuple(
            transport.RegimeBinding(rid, f"snap-{rid}", sampling="independent")
            for rid in ("s1", "post")
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="posterior_as_law"):
        transport.identify_mixed_source_transport(
            graph=chain_graph(), query=query(), catalog=posterior
        )


def test_a_value_restricted_trial_is_excluded_not_identified_for_every_level():
    def trial_catalog(values):
        return transport.EvidenceCatalog(
            environments=(
                transport.Environment(
                    "target", tuple(transport.VariableCoordinate(n, "binary") for n in NAMES)
                ),
            ),
            regimes=(
                transport.EvidenceRegime(
                    "t",
                    "target",
                    kind="experimental",
                    interventions=["x"],
                    measured=["y"],
                    intervention_values=values,
                    study="trial",
                ),
            ),
            bindings=(transport.RegimeBinding("t", "snap-t", sampling="independent"),),
        )

    family = transport.identify_mixed_source_transport(
        graph=chain_graph(), query=query(), catalog=trial_catalog({})
    )
    assert family.outcome == "identified"
    restricted = transport.identify_mixed_source_transport(
        graph=chain_graph(), query=query(), catalog=trial_catalog({"x": 1.0})
    )
    # A trial of do(x = 1) only supplies one level: nothing is identified for every level.
    assert restricted.outcome == "not_certified"
    assert restricted.decision()["excluded"] == [{"regime": "t", "reason": "restricted_levels"}]
    with pytest.raises(CausalUnsupportedError, match="transport_not_certified"):
        restricted.prepare_exact((), {"x": 1.0})


def test_counted_laws_are_not_licensed_on_the_mixed_source_route():
    stage = decide(chain_graph(), CHAIN_STUDIES)
    with pytest.raises(CausalUnsupportedError, match="cell_not_licensed"):
        stage.prepare_empirical(laws(CHAIN, CHAIN_STUDIES), {"x": 1.0})


def test_exported_result_is_recomputed_by_an_independent_consumer():
    prepared = decide(frontdoor_graph(), FRONTDOOR_STUDIES).prepare_exact(
        laws(FRONTDOOR, FRONTDOOR_STUDIES), {"x": 1.0}
    )
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    consumed = json.loads(transport.consume_mixed_source_artifact(artifact))
    assert consumed["probabilities"] == live["probabilities"]
    assert consumed["proof"]["rule_set"] == "x9.rules.v1"
    assert {s["study"] for s in consumed["cited_sources"]} == {"observational", "trial"}
    assert consumed["premises_digest"] and consumed["data_digest"]
    with pytest.raises(CausalSerializationError):
        transport.consume_mixed_source_artifact(artifact[:-3] + b"\x00\x00\x00")


def test_export_binds_names_limits_and_snapshots_into_the_verified_identity():
    builder = decide(frontdoor_graph(), FRONTDOOR_STUDIES)
    prepared = builder.prepare_exact(laws(FRONTDOOR, FRONTDOOR_STUDIES), {"x": 1.0})
    del builder
    json.loads(prepared.estimate())
    assert prepared.plan()["compiled_plans"] == 1
    artifact = prepared.export()
    # Relabelling coordinates in the frame (same byte length) is refused.
    swapped = artifact.replace(b"\x83axazay", b"\x83azaxay", 1)
    assert swapped != artifact
    with pytest.raises(CausalSerializationError):
        transport.consume_mixed_source_artifact(swapped)

    # Renaming a snapshot is caught by a digest (the proof's source leaves bind the snapshot
    # into the premises digest, the catalog and laws into the data digest). The framed
    # payload is a CBOR byte array: each ASCII byte is `0x18 <byte>`.
    def framed(text):
        return b"".join(b"\x18" + bytes([c]) for c in text.encode())

    renamed = artifact.replace(framed("snap-obs"), framed("snap-obz"))
    assert renamed != artifact
    with pytest.raises(CausalSerializationError, match="digest mismatch"):
        transport.consume_mixed_source_artifact(renamed)
    # A consumer with smaller limits than the producer refuses.
    with pytest.raises(CausalResourceError):
        transport.consume_mixed_source_artifact(artifact, max_search_operations=10)


def test_refresh_rebinds_snapshots_and_re_executes_the_retained_plan():
    builder = decide(chain_graph(), CHAIN_STUDIES)
    prepared = builder.prepare_exact(laws(CHAIN, CHAIN_STUDIES), {"x": 1.0})
    del builder
    before = json.loads(prepared.estimate())
    plan = prepared.plan()
    shifted = {**CHAIN, "z": lambda v, e: e[2] if v["x"] else e[1]}
    prepared.refresh(laws(shifted, CHAIN_STUDIES))
    assert prepared.plan() == plan
    after = json.loads(prepared.estimate())
    assert after["probabilities"] != before["probabilities"]
    with pytest.raises(CausalUnsupportedError, match="snapshot does not match catalog binding"):
        prepared.refresh(laws(CHAIN, CHAIN_STUDIES, snapshot_suffix="-new"))
    # Refresh clears the last claim: nothing is exported until the plan runs again.
    prepared.refresh(laws(CHAIN, CHAIN_STUDIES))
    with pytest.raises(Exception, match="not_executed"):
        prepared.export()


def test_prepare_exact_compiles_a_checked_plan_after_builder_disposal():
    builder = decide(chain_graph(), CHAIN_STUDIES)
    prepared = builder.prepare_exact(laws(CHAIN, CHAIN_STUDIES), [{"x": 0.0}, {"x": 1.0}])
    del builder
    plan = prepared.plan()
    assert plan["compiled_plans"] == 2 and plan["requests"] == [{"x": 0.0}, {"x": 1.0}]
    assert plan["cited_regimes"] == ["s1", "s2"] and plan["rule_set"] == "x9.rules.v1"
    result = json.loads(prepared.estimate())
    assert [risk(r) for r in result["requests"]] == pytest.approx(
        [truth(CHAIN, 0), truth(CHAIN, 1)], abs=1e-12
    )
    assert [r["means"]["y"] for r in result["requests"]] == pytest.approx(
        [truth(CHAIN, 0), truth(CHAIN, 1)], abs=1e-12
    )


def test_query_contract_is_validated_before_search():
    with pytest.raises(CausalValueError):
        transport.MixedSourceQuery(target="target", outcomes=["y"], treatments=["y"])
    with pytest.raises(CausalValueError):
        transport.MixedSourceQuery(target="", outcomes=["y"], treatments=["x"])
    with pytest.raises(CausalTypeError):
        transport.MixedSourceQuery(target="target", outcomes=["y"], treatments=["x"], sources=[1])
    with pytest.raises(CausalTypeError):
        transport.identify_mixed_source_transport(graph=chain_graph(), query=query(), catalog={})
    with pytest.raises(CausalTypeError):
        transport.consume_mixed_source_artifact("not bytes")


def test_mixed_source_identification_executes_its_plan_after_query_disposal():
    """The decision is taken once; the stage keeps its proof after the query is gone."""
    builder = decide(chain_graph(), CHAIN_STUDIES)
    assert builder.outcome == "identified"
    program = builder.decision()
    prepared = builder.prepare_exact(laws(CHAIN, CHAIN_STUDIES), {"x": 1.0})
    del builder
    assert program["rule_set"] == "x9.rules.v1" and program["steps"]
    assert prepared.plan()["cited_regimes"] == program["cited_regimes"]
    assert risk(json.loads(prepared.estimate())) == pytest.approx(truth(CHAIN, 1), abs=1e-12)


def test_estimate_executes_every_request_from_the_retained_plan():
    builder = decide(chain_graph(), CHAIN_STUDIES)
    prepared = builder.prepare_exact(laws(CHAIN, CHAIN_STUDIES), [{"x": 0.0}, {"x": 1.0}])
    del builder
    result = json.loads(prepared.estimate())
    assert result["probabilities"] == result["requests"][0]["probabilities"]
    assert [r["assignment"] for r in result["requests"]] == [{"x": 0.0}, {"x": 1.0}]
    plan = prepared.plan()
    assert plan["compiled_plans"] == len(result["requests"])
    assert json.loads(prepared.estimate())["requests"] == result["requests"]


def test_consumer_recomputes_points_after_builder_disposal():
    builder = decide(chain_graph(), CHAIN_STUDIES)
    prepared = builder.prepare_exact(laws(CHAIN, CHAIN_STUDIES), [{"x": 0.0}, {"x": 1.0}])
    del builder
    live = json.loads(prepared.estimate())
    plan = prepared.plan()
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_mixed_source_artifact(artifact))
    assert consumed["requests"] == live["requests"]
    assert [s["source"]["regime"] for s in plan["steps"] if s["source"]] == ["s1", "s2"]
    assert {c["study"] for c in consumed["cited_sources"]} == {"study-1", "study-2"}
