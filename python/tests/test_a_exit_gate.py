"""2.2 A exit gate: the six end-to-end user stories through the Python surfaces.

Each story runs start to finish in one test and builds its own graph, catalog and evidence
from scratch (independent of the unit tests' fixtures): prepare once, estimate, refresh where
the cell supports it, export the artifact, drop every producer object, then consume the
artifact bytes alone and require the consumer's recomputed answers to equal the producer's
bit for bit and the exactly enumerated truth. Each story also asserts its named refusal or
incomplete side.

Story 3 says "calibrated". Calibration is measured separately; this file makes no calibration
claim and asserts that no interval is ever attached. ``scripts/gate_a_exit.sh`` reports that
part as ``PENDING_CALIBRATION`` until the coverage records exist.
"""

import itertools
import json

import numpy as np
import pytest
from antecedent import Admg, Dag
from antecedent import transport as tr
from antecedent.cross_world import (
    EdgeIntervention,
    consume_cross_world_artifact,
    path_specific_effect,
)
from antecedent.errors import CausalUnsupportedError
from antecedent.learners import Linear
from antecedent.transport import advanced as transport


def hexes(values):
    """Exact bit patterns of a float sequence (round-tripped through JSON)."""
    return [float(v).hex() for v in values]


def risk(result):
    return sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )


def enumerate_law(mechanisms, names, p, do, measured):
    """Exact joint over ``measured`` (first most significant) under ``do``."""
    table = [0.0] * (1 << len(measured))
    for bits in itertools.product((0, 1), repeat=len(p)):
        weight = 1.0
        for b, pr in zip(bits, p, strict=True):
            weight *= pr if b else 1.0 - pr
        values = {}
        for name in names:
            values[name] = do[name] if name in do else mechanisms[name](values, bits)
        index = 0
        for name in measured:
            index = (index << 1) | values[name]
        table[index] += weight
    return [min(1.0, max(0.0, t)) for t in table]


# --------------------------------------------------------------------------------------
# Story 1 (X1): multi-source limited-experiment effect requiring complementary sources.
# --------------------------------------------------------------------------------------

S1_NAMES = ["z1", "x", "z2", "y"]
S1_P = [0.4, 0.55, 0.5, 0.35, 0.65, 0.7, 0.45]
S1_P_REFRESHED = [0.4, 0.55, 0.5, 0.35, 0.65, 0.7, 0.6]


def _s1_x(v, e):
    return v["z1"] ^ (e[0] & e[4])


def _s1_z2(v, e):
    return (v["x"] & e[5]) | ((1 - v["x"]) & e[1])


def _s1_y(v, e):
    return (v["z2"] & e[6]) | (e[2] & (1 - e[6]))


S1_MECHANISMS = {
    "target": {
        "z1": lambda v, e: (e[0] & e[3]) | (e[1] & e[2]),
        "x": _s1_x,
        "z2": _s1_z2,
        "y": _s1_y,
    },
    "a": {
        "z1": lambda v, e: e[3] | e[1],
        "x": _s1_x,
        "z2": lambda v, e: v["x"] ^ (e[1] & e[5]),
        "y": _s1_y,
    },
    "b": {
        "z1": lambda v, e: e[0] & e[2],
        "x": _s1_x,
        "z2": _s1_z2,
        "y": lambda v, e: v["z2"] ^ (e[2] & e[6]),
    },
}
S1_REGIMES = [
    ("obs", "target", {}),
    ("a_z2_0", "a", {"z2": 0}),
    ("a_z2_1", "a", {"z2": 1}),
    ("b_z1_0", "b", {"z1": 0}),
]


def s1_truth(p, x):
    return enumerate_law(S1_MECHANISMS["target"], S1_NAMES, p, {"x": x}, ["y"])[1]


def s1_catalog():
    coordinates = tuple(transport.VariableCoordinate(n, "binary") for n in S1_NAMES)
    regimes = tuple(
        transport.EvidenceRegime(
            rid,
            population,
            kind="experimental" if do else "observational",
            interventions=list(do),
            intervention_values={k: float(v) for k, v in do.items()},
            measured=[n for n in S1_NAMES if n not in do],
        )
        for rid, population, do in S1_REGIMES
    )
    bindings = tuple(
        transport.RegimeBinding(
            rid,
            f"snap-{rid}",
            schema_names=[n for n in S1_NAMES if n not in do],
            sampling="independent",
            dependence="independent_studies",
        )
        for rid, _, do in S1_REGIMES
    )
    environments = tuple(transport.Environment(pop, coordinates) for pop in ("target", "a", "b"))
    return transport.EvidenceCatalog(environments=environments, regimes=regimes, bindings=bindings)


def s1_laws(p, *, sample_size=None):
    out = []
    for rid, population, do in S1_REGIMES:
        measured = [n for n in S1_NAMES if n not in do]
        probabilities = enumerate_law(S1_MECHANISMS[population], S1_NAMES, p, do, measured)
        counts = None
        if sample_size is not None:
            counts = tuple(round(q * sample_size) for q in probabilities)
            probabilities = [c / sum(counts) for c in counts]
        out.append(
            transport.ExactDiscreteLaw(
                population,
                rid,
                tuple((name, (0.0, 1.0)) for name in measured),
                tuple(probabilities),
                f"snap-{rid}",
                interventions=tuple((k, float(v)) for k, v in do.items()),
                empirical_counts=counts,
            )
        )
    return tuple(out)


def s1_graph():
    return Admg.from_edges(
        S1_NAMES,
        [("z1", "x"), ("x", "z2"), ("z2", "y")],
        [("z1", "x"), ("z1", "z2"), ("z1", "y")],
    )


def s1_query(*sources):
    return transport.MultiSourceZTransportQuery(
        target="target", outcomes=["y"], treatments=["x"], sources=list(sources)
    )


def s1_producer():
    """Everything the producer holds is local to this function."""
    source_a = transport.ZTransportSource("a", controllable=["z2"], selections=["z1", "z2"])
    source_b = transport.ZTransportSource(
        "b", controllable=["z1"], selections=["z1", "y"], experiment_assignment={"z1": 0.0}
    )
    useless = transport.ZTransportSource("u", controllable=["x"], selections=S1_NAMES)
    # Each source alone (with an unhelpful one) is a checked obstruction.
    for alone in (source_a, source_b):
        stage = transport.identify_multi_source_z_transport(
            graph=s1_graph(), query=s1_query(alone, useless), catalog=s1_catalog()
        )
        assert stage.outcome == "proven_non_transportable"
        assert stage.decision()["reason"] == "transport_proven_non_transportable"
    # Together they identify the query.
    stage = transport.identify_multi_source_z_transport(
        graph=s1_graph(), query=s1_query(source_a, source_b), catalog=s1_catalog()
    )
    assert stage.outcome == "identified", stage.decision()
    assert stage.decision()["route"] == "combined"
    requests = [{"x": 0.0}, {"x": 1.0}]
    prepared = stage.prepare_exact(s1_laws(S1_P), requests)
    del stage
    live = json.loads(prepared.estimate())
    assert [risk(r) for r in live["requests"]] == pytest.approx(
        [s1_truth(S1_P, 0), s1_truth(S1_P, 1)], abs=1e-12
    )
    (contrast,) = live["contrasts"]
    assert contrast["estimate"] == pytest.approx(s1_truth(S1_P, 1) - s1_truth(S1_P, 0), abs=1e-12)
    assert live["interval"]["status"] == "point_only"
    before = (prepared.export(), live)

    prepared.refresh(s1_laws(S1_P_REFRESHED))
    moved = json.loads(prepared.estimate())
    assert [risk(r) for r in moved["requests"]] == pytest.approx(
        [s1_truth(S1_P_REFRESHED, 0), s1_truth(S1_P_REFRESHED, 1)], abs=1e-12
    )
    assert moved["contrasts"] != live["contrasts"]
    after = (prepared.export(), moved)

    # Counted laws: the point is returned, the unmeasured interval route stays closed.
    counted = transport.identify_multi_source_z_transport(
        graph=s1_graph(), query=s1_query(source_a, source_b), catalog=s1_catalog()
    ).prepare_empirical(s1_laws(S1_P, sample_size=40_000), {"x": 1.0}, seed=3)
    counted_live = json.loads(counted.estimate())
    assert risk(counted_live) == pytest.approx(s1_truth(S1_P, 1), abs=1e-2)
    interval = counted_live["interval"]
    assert (interval["available"], interval["status"], interval["reason"]) == (
        False,
        "withheld",
        "cell_not_licensed",
    )
    assert interval["mean_intervals"] == [] and interval["seed"] is None
    return before, after, (counted.export(), counted_live)


def test_story_1_multi_source_limited_experiment_effect_requiring_complementary_sources():
    before, after, counted = s1_producer()
    for (artifact, live), p in ((before, S1_P), (after, S1_P_REFRESHED)):
        consumed = json.loads(transport.consume_multi_source_z_transport_artifact(artifact))
        for c, lv in zip(consumed["requests"], live["requests"], strict=True):
            assert hexes(c["probabilities"]) == hexes(lv["probabilities"])
        assert hexes(x["estimate"] for x in consumed["contrasts"]) == hexes(
            x["estimate"] for x in live["contrasts"]
        )
        assert [risk(r) for r in consumed["requests"]] == pytest.approx(
            [s1_truth(p, 0), s1_truth(p, 1)], abs=1e-12
        )
        assert consumed["cited_sources"] == ["a", "b"]
    artifact, live = counted
    consumed = json.loads(transport.consume_multi_source_z_transport_artifact(artifact))
    assert hexes(consumed["probabilities"]) == hexes(live["probabilities"])
    assert consumed["interval"] == live["interval"]


# --------------------------------------------------------------------------------------
# Story 2 (X2): finite structural scenario set: identified, unidentified, unevaluated.
# --------------------------------------------------------------------------------------

S2_NAMES = ["z", "x", "y"]
S2_SOURCE = (0.28, 0.12, 0.10, 0.50)  # do(x=1) over (z, y); P(z=1) = 0.6
S2_TARGET = (0.28, 0.14, 0.14, 0.14, 0.06, 0.06, 0.09, 0.09)  # (z, x, y); P*(z=1) = 0.3
S2_STANDARDIZED = 0.7 * 0.3 + 0.3 * (0.5 / 0.6)
S2_DIRECT = 0.12 + 0.50


def s2_graph():
    return Admg.from_edges(S2_NAMES, [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])


def s2_coordinates():
    return [transport.VariableCoordinate(n, "binary") for n in S2_NAMES]


def s2_scenarios(weighted=True):
    w = (0.3, 0.2, 0.4) if weighted else (None, None, None)
    return transport.TransportScenarioSet(
        [
            transport.TransportScenario("standardize", s2_graph(), ["z"], w[0]),
            transport.TransportScenario("direct", s2_graph(), [], w[1]),
            transport.TransportScenario("outcome_shift", s2_graph(), ["y"], w[2]),
        ],
        s2_coordinates(),
    )


def s2_catalog():
    return transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]),
        ]
    )


def s2_laws(source=S2_SOURCE):
    return transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "trial",
                (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
                source,
                "trial",
                interventions=(("x", 1.0),),
            ),
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("z", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
                S2_TARGET,
                "target",
            ),
        )
    )


def s2_prepare(**limits):
    return transport.prepare_transport_scenarios(
        s2_scenarios(),
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=s2_catalog(),
        laws=s2_laws(),
        at={"x": 1.0},
        **limits,
    )


def s2_by_name(report):
    return {s["name"]: s for s in report["scenarios"]}


def s2_producer():
    def statuses(steps):
        return {
            s["name"]: s["status"]
            for s in json.loads(s2_prepare(max_steps=steps).estimate())["scenarios"]
        }

    budget = next(
        steps
        for steps in range(1, 10_000)
        if statuses(steps)["direct"] == "identified"
        and statuses(steps)["outcome_shift"] != "unevaluated"
    )
    prepared = s2_prepare(max_steps=budget)
    report = json.loads(prepared.estimate())
    by = s2_by_name(report)
    assert by["direct"]["status"] == "identified"
    assert by["outcome_shift"]["status"] == "structurally_unidentified"
    assert by["standardize"]["status"] == "unevaluated"
    assert by["outcome_shift"]["point"] is None and by["standardize"]["point"] is None
    assert by["direct"]["point"]["means"]["y"] == pytest.approx(S2_DIRECT, abs=1e-12)
    assert by["standardize"]["detail"] == "scenarios.unevaluated_budget: search.operations"
    assert report["receipt"]["stop"] == "search.operations"
    assert report["receipt"]["unevaluated"] == ["standardize"]
    masses = {m["status"]: m["mass"] for m in report["masses"]}
    assert masses["identified"] == pytest.approx(0.2)
    assert masses["structurally_unidentified"] == pytest.approx(0.4)
    assert masses["unevaluated"] == pytest.approx(0.3)
    assert report["residual_mass"] == pytest.approx(0.1)
    weighted = report["weighted"]
    assert weighted["identified_mass"] == pytest.approx(0.2)
    assert weighted["unaccounted_mass"] == pytest.approx(0.8)
    total = weighted["identified_weighted_sums"]["y"]
    assert total == pytest.approx(0.2 * S2_DIRECT, abs=1e-12)  # not renormalized by 0.2
    assert abs(total - S2_DIRECT) > 0.1
    rng = weighted["ranges"][0]
    assert (rng["lower"], rng["upper"]) == pytest.approx((total, total + 0.8))
    assert report["envelope"]["scenarios"] == ["direct"]
    with pytest.raises(CausalUnsupportedError, match="scenarios.shared_data_aggregate") as refused:
        prepared.aggregate_interval()
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    prepared.refresh(s2_laws(source=(0.1, 0.3, 0.3, 0.3)))
    moved = json.loads(prepared.estimate())
    assert [s["status"] for s in moved["scenarios"]] == [s["status"] for s in report["scenarios"]]
    assert s2_by_name(moved)["direct"]["point"]["means"]["y"] == pytest.approx(0.6)
    prepared.refresh(s2_laws())
    final = json.loads(prepared.estimate())
    return prepared.export(), final


def test_story_2_finite_scenario_set_retains_identified_unidentified_and_unevaluated_members():
    artifact, live = s2_producer()
    consumed = json.loads(transport.consume_transport_scenarios_artifact(artifact))
    assert consumed["scenarios"] == live["scenarios"]
    assert consumed["weighted"] == live["weighted"]
    assert consumed["masses"] == live["masses"]
    assert consumed["residual_mass"] == live["residual_mass"]
    assert consumed["receipt"] == live["receipt"]
    assert {s["status"] for s in consumed["scenarios"]} == {
        "identified",
        "structurally_unidentified",
        "unevaluated",
    }
    # With a sufficient budget the same set is fully evaluated and standardize is exact.
    full = json.loads(s2_prepare().estimate())
    assert full["receipt"] is None
    assert s2_by_name(full)["standardize"]["point"]["means"]["y"] == pytest.approx(
        S2_STANDARDIZED, abs=1e-12
    )


# --------------------------------------------------------------------------------------
# Story 3 (X4): overlap-supported learned transport estimate and analytic interval.
# --------------------------------------------------------------------------------------


def s3_fixture(design="independent_samples", shift=0.4, n_trial=800, n_target=500, seed=3):
    """Trial z ~ N(0, 1), target z ~ N(shift, 1); y = 1.5 z + a (1 + 0.5 z) + e; truth 1 + shift/2."""
    rng = np.random.default_rng(seed)
    total = n_trial + n_target
    if design == "nested_cohort":
        source = rng.random(total) < n_trial / total
    else:
        source = np.arange(total) < n_trial
    z = np.where(source, rng.normal(size=total), shift + rng.normal(size=total))
    a = rng.random(total) < 0.5
    y = 1.5 * z + a * (1.0 + 0.5 * z) + rng.normal(size=total)
    data = tr.TrialAipwData(
        {"z": z},
        np.where(source, y, 0.0),
        [bool(v) for v in (a & source)],
        [bool(v) for v in source],
        [0.5] * total,
        design,
    )
    graph = Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    query = transport.TrialAipwQuery(
        graph, transport.SelectionDiagram("trial", "target", ["z"]), "a", "y"
    )
    return query, data


def s3_options():
    return transport.LearnedContinuousOptions(outcome=Linear(), folds=3)


def s3_producer(design):
    query, data = s3_fixture(design)
    study = transport.prepare_learned_continuous(query, data, options=s3_options(), seed=11)
    del query, data
    result = study.estimate()
    assert result.estimate == pytest.approx(1.2, abs=0.3)
    assert result.interval is None and result.uncertainty["status"] == "point_only"
    assert result.overlap["selection"]["probability_min"] > 0.05
    assert result.overlap["treatment"]["probability_min"] == pytest.approx(0.5)
    assert len(result.provenance) == 9
    interval = study.interval()
    assert interval.uncertainty["status"] == "available"
    assert interval.standard_error > 0
    assert interval.interval[0] < interval.estimate < interval.interval[1]
    # Refresh keeps the certificate and moves the point.
    _, fresh = s3_fixture(design, seed=9)
    study.refresh(fresh)
    moved = study.estimate()
    assert moved.execution_id != result.execution_id and moved.certificate == result.certificate
    assert moved.estimate == pytest.approx(1.2, abs=0.3)
    return study.export(), moved


@pytest.mark.parametrize("design", ["nested_cohort", "independent_samples"])
def test_story_3_learned_continuous_transport_estimate_with_overlap_refusal(design):
    artifact, live = s3_producer(design)
    consumed = transport.consume_learned_continuous(artifact)
    assert float(consumed.estimate).hex() == float(live.estimate).hex()
    assert consumed.execution_id == live.execution_id
    assert consumed.premises_digest == live.premises_digest
    assert consumed.provenance == live.provenance
    assert consumed.overlap == live.overlap and consumed.folds == live.folds
    assert consumed.interval is None and consumed.uncertainty == live.uncertainty
    assert consumed.estimate == pytest.approx(1.2, abs=0.3)
    # Overlap refusal: the target is outside what the trial can represent.
    weak_query, weak_data = s3_fixture(design, shift=3.0, seed=5)
    weak = transport.prepare_learned_continuous(
        weak_query, weak_data, options=s3_options(), seed=11
    )
    with pytest.raises(CausalUnsupportedError, match="learned_transport.membership_overlap") as e:
        weak.estimate()
    assert e.value.reason_code == "transport_support_failure"
    query, data = s3_fixture(design, seed=6)
    import dataclasses

    low = dataclasses.replace(data, randomization=[0.01] * len(data.randomization))
    with pytest.raises(CausalUnsupportedError, match="learned_transport.treatment_overlap") as e:
        transport.prepare_learned_continuous(query, low, options=s3_options())
    assert e.value.reason_code == "transport_support_failure"


# --------------------------------------------------------------------------------------
# Story 4 (X5): two-step discrete temporal transported intervention, history-support refusal.
# --------------------------------------------------------------------------------------

S4_NAMES = ["b", "l1", "a1", "l2", "a2", "y"]
B, L1, A1, L2, A2, Y = range(6)
S4_DIRECTED = [
    ("b", "l1"), ("l1", "a1"), ("b", "l2"), ("a1", "l2"), ("l2", "a2"), ("a1", "a2"),
    ("b", "y"), ("l1", "y"), ("a1", "y"), ("l2", "y"), ("a2", "y"),
]  # fmt: skip
S4_SOURCE_P = [0.3, 0.35, 0.4, 0.55, 0.3, 0.45, 0.65, 0.4]
S4_TARGET_P = [0.75, 0.35, 0.4, 0.55, 0.55, 0.45, 0.65, 0.4]


def s4_mechanisms(population):
    def l2(v, e):
        if population == "source":
            return v[A1] ^ e[4]
        return int(v[A1] == 1 and v[B] == 0) ^ e[4]

    return [
        lambda v, e: e[0],
        lambda v, e: v[B] ^ e[1],
        lambda v, e: e[2] ^ int(v[L1] == 1 and e[3] == 1),
        l2,
        lambda v, e: int(v[L2] == 1 and e[6] == 1) ^ e[5],
        lambda v, e: (
            int(v[A1] == 1 and e[2] == 0)
            ^ int(v[A2] == 1 and v[L2] == 1)
            ^ int(v[B] == 1 and e[7] == 1)
            ^ int(e[5] == 1 and v[L1] == 1)
        ),
    ]


def s4_law(population, do, measured, p):
    mechanisms = s4_mechanisms(population)
    out = [0.0] * (1 << len(measured))
    for mask in range(1 << len(p)):
        e = [(mask >> bit) & 1 for bit in range(len(p))]
        weight = 1.0
        for bit, pr in enumerate(p):
            weight *= pr if e[bit] else 1.0 - pr
        v = [0] * 6
        for i in range(6):
            v[i] = do[i] if i in do else mechanisms[i](v, e)
        index = 0
        for m in measured:
            index = (index << 1) | v[m]
        out[index] += weight
    return [min(1.0, max(0.0, q)) for q in out]


def s4_truth(p, sequence):
    return s4_law("target", {A1: int(sequence[0]), A2: int(sequence[1])}, [Y], p)[1]


def s4_spec():
    return transport.TemporalSequenceSpec(
        Admg.from_edges(S4_NAMES, S4_DIRECTED, [("a1", "y"), ("a2", "y")]),
        baseline=["b"],
        covariates=[["l1"], ["l2"]],
        actions=["a1", "a2"],
        outcome="y",
        coordinates=[transport.VariableCoordinate(n, "binary") for n in S4_NAMES],
        selections=["b", "l2"],
        horizon=2,
    )


def s4_catalog():
    return transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime("obs", "target", measured=S4_NAMES),
            transport.EvidenceRegime(
                "history",
                "source",
                kind="experimental",
                interventions=["b", "l1", "a1", "l2", "a2"],
                measured=["y"],
            ),
        ]
    )


def s4_laws(source_p, target_p, skip=()):
    out = [
        transport.ExactDiscreteLaw(
            "target",
            "obs",
            tuple((n, (0.0, 1.0)) for n in S4_NAMES),
            s4_law("target", {}, list(range(6)), target_p),
            "target",
        )
    ]
    for h in range(32):
        bits = [(h >> k) & 1 for k in (4, 3, 2, 1, 0)]
        if (bits[0], bits[1], bits[3]) in skip:
            continue
        do = dict(zip([B, L1, A1, L2, A2], bits, strict=True))
        out.append(
            transport.ExactDiscreteLaw(
                "source",
                "history",
                (("y", (0.0, 1.0)),),
                s4_law("source", do, [Y], source_p),
                "source",
                interventions=tuple(
                    zip(["b", "l1", "a1", "l2", "a2"], map(float, bits), strict=True)
                ),
            )
        )
    return transport.ExactTransportData(tuple(out))


def s4_prepare(sequence, data):
    return transport.prepare_temporal_transport_sequence(
        s4_spec(),
        sequence=list(sequence),
        source="source",
        target="target",
        catalog=s4_catalog(),
        laws=data,
    )


def s4_producer():
    moved_source, moved_target = list(S4_SOURCE_P), list(S4_TARGET_P)
    moved_source[7] = moved_target[7] = 0.55
    artifacts = []
    for sequence in ((1.0, 0.0), (0.0, 1.0)):
        prepared = s4_prepare(sequence, s4_laws(S4_SOURCE_P, S4_TARGET_P))
        live = json.loads(prepared.estimate())
        assert live["mean"] == pytest.approx(s4_truth(S4_TARGET_P, sequence), abs=1e-12)
        assert live["inference_claim"] == "point_only" and live["horizon"] == 2
        assert live["time_varying_confounders"] == ["l2"]
        assert {r["status"] for r in live["support"]["rows"]} == {"supported"}
        artifacts.append((prepared.export(), live, S4_TARGET_P, sequence))
        prepared.refresh(s4_laws(moved_source, moved_target))
        moved = json.loads(prepared.estimate())
        assert moved["mean"] == pytest.approx(s4_truth(moved_target, sequence), abs=1e-12)
        assert abs(moved["mean"] - live["mean"]) > 1e-4
        artifacts.append((prepared.export(), moved, moved_target, sequence))
    # History-support refusal: typed, and local to the offending history.
    messages = []
    for skip in ((1, 0, 1), (0, 1, 0)):
        with pytest.raises(
            CausalUnsupportedError, match="temporal_transport.history_outside_support"
        ) as refused:
            s4_prepare((1.0, 1.0), s4_laws(S4_SOURCE_P, S4_TARGET_P, skip=(skip,)))
        assert refused.value.reason_code == "transport_support_failure"
        text = str(refused.value)
        assert f"b={skip[0]}" in text and f"l1={skip[1]}" in text and f"l2={skip[2]}" in text
        messages.append(text)
    assert messages[0] != messages[1]
    prepared = s4_prepare((1.0, 0.0), s4_laws(S4_SOURCE_P, S4_TARGET_P))
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.interval_requested"
    ) as refused:
        prepared.interval()
    assert refused.value.reason_code == "estimator_inference_mismatch"
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.horizon") as three:
        s4_prepare((1.0, 0.0, 1.0), s4_laws(S4_SOURCE_P, S4_TARGET_P))
    assert three.value.reason_code == "route_not_supported"
    return artifacts


def test_story_4_two_step_temporal_transport_with_history_support_refusal():
    for artifact, live, p, sequence in s4_producer():
        consumed = json.loads(transport.consume_temporal_transport_artifact(artifact))
        assert float(consumed["mean"]).hex() == float(live["mean"]).hex()
        assert hexes(consumed["point"]["probabilities"]) == hexes(live["point"]["probabilities"])
        assert consumed["support"] == live["support"]
        assert consumed["invariances"] == live["invariances"]
        assert consumed["sequence"] == list(sequence) and consumed["horizon"] == 2
        assert consumed["mean"] == pytest.approx(s4_truth(p, sequence), abs=1e-12)


# --------------------------------------------------------------------------------------
# Story 5 (X8): second fixed-DAG cross-world query: shared abduction and typed refusal.
# --------------------------------------------------------------------------------------

S5_EDGES = [("x", "m"), ("x", "y"), ("m", "y")]
CONTROL, ACTIVE = -1.0, 2.0


def s5_dag():
    return Dag.from_edges(["x", "m", "y"], S5_EDGES)


def s5_linear(n=240):
    """M = -0.6 X + U, Y = 2.2 X + 3 M with U orthogonal to the constant and X."""
    i = np.arange(n, dtype=float)
    x = np.sin(i * 0.53)
    raw = np.cos(i * 1.07)
    design = np.column_stack([np.ones(n), x])
    u = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    m = -0.6 * x + u
    return {"x": x, "m": m, "y": 2.2 * x + 3.0 * m}


def s5_outcome(x, m):
    return 1.1 * x + 0.4 * m + 0.8 * x * m * m


def s5_non_separable(n=600):
    i = np.arange(n, dtype=float)
    x = np.sin(i * 0.41) * 3.0
    u = np.cos(i * 0.83)
    m = 0.7 * x + u
    return {"x": x, "m": m, "y": s5_outcome(x, m)}, u


def s5_producer():
    indirect = EdgeIntervention("x", "y", CONTROL, ACTIVE, (("x", "m"), ("m", "y")))
    linear = path_specific_effect(s5_dag(), s5_linear(), indirect)
    assert linear.point == pytest.approx(3.0 * -0.6 * (ACTIVE - CONTROL), abs=1e-9)
    data, u = s5_non_separable()
    direct = EdgeIntervention.natural_direct("x", "m", "y", control=CONTROL, active=ACTIVE)
    shared = path_specific_effect(s5_dag(), data, direct, mechanism="non_separable_basis")
    m0 = 0.7 * CONTROL + u
    per_unit = float(np.mean(s5_outcome(ACTIVE, m0) - s5_outcome(CONTROL, m0)))
    plug_in = float(s5_outcome(ACTIVE, m0.mean()) - s5_outcome(CONTROL, m0.mean()))
    assert abs(per_unit - plug_in) > 1.0
    assert abs(shared.point - per_unit) < 0.25 and abs(shared.point - plug_in) > 1.0
    separable = path_specific_effect(s5_dag(), data, direct).point
    assert abs(separable - per_unit) > 1.0
    # Typed refusals.
    edges = [("x", "w"), ("w", "m"), ("w", "y"), ("m", "y")]
    recanting = Dag.from_edges(["x", "w", "m", "y"], edges)
    i = np.arange(80, dtype=float)
    table = {"x": np.sin(i), "w": np.cos(i * 1.3), "m": np.sin(i * 0.7), "y": np.cos(i * 2.1)}
    with pytest.raises(CausalUnsupportedError) as raised:
        path_specific_effect(
            recanting, table, EdgeIntervention("x", "y", 0.0, 1.0, (("x", "w"), ("w", "y")))
        )
    assert raised.value.reason_code == "cross_world_not_identified"
    assert "cross_world.recanting_witness" in str(raised.value)
    with pytest.raises(CausalUnsupportedError) as raised:
        path_specific_effect(s5_dag(), s5_linear(60), direct, uncertainty="interval")
    assert raised.value.reason_code == "estimator_inference_mismatch"
    assert "cross_world.interval_requested" in str(raised.value)
    return linear, shared, per_unit


def test_story_5_second_fixed_dag_cross_world_query_with_shared_abduction_and_typed_refusal():
    linear, shared, per_unit = s5_producer()
    artifacts = (
        (linear.artifact, linear.point, linear.witness),
        (shared.artifact, shared.point, None),
    )
    del linear, shared
    for artifact, point, witness in artifacts:
        consumed = consume_cross_world_artifact(artifact)
        assert float(consumed.point).hex() == float(point).hex()
        if witness is not None:
            assert consumed.witness == witness
            assert consumed.witness["intervened_edges"] == [[0, 1], [1, 2]]
    assert abs(consumed.point - per_unit) < 0.25
    with pytest.raises(Exception, match=".") as bad:
        consume_cross_world_artifact(b"not an artifact")
    assert type(bad.value).__name__ == "CausalSerializationError"


# --------------------------------------------------------------------------------------
# Story 6 (X9): mixed-source proof by bounded search; incomplete search stays unresolved.
# --------------------------------------------------------------------------------------

S6_NAMES = ["x", "z", "y"]
S6_P = [0.55, 0.75, 0.25, 0.65, 0.15, 0.5]  # e[5] is the latent x <-> y bit: keep 0.5
S6_P_REFRESHED = [0.55, 0.6, 0.25, 0.65, 0.15, 0.5]
S6_FRONTDOOR = {
    "x": lambda v, e: e[5] ^ (e[0] & e[1]),
    "z": lambda v, e: e[1] if v["x"] else e[2],
    "y": lambda v, e: (e[3] if v["z"] else e[4]) ^ (e[5] & (1 - v["z"])),
}
S6_STUDIES = [
    ("obs", "observational", (), ("x", "z")),
    ("trial", "trial", ("z",), ("x", "y")),
]


def s6_truth(p, x):
    return enumerate_law(S6_FRONTDOOR, S6_NAMES, p, {"x": x}, ["y"])[1]


def s6_catalog(studies, names=S6_NAMES):
    return transport.EvidenceCatalog(
        environments=(
            transport.Environment(
                "target", tuple(transport.VariableCoordinate(n, "binary") for n in names)
            ),
        ),
        regimes=tuple(
            transport.EvidenceRegime(
                rid,
                "target",
                kind="experimental" if do else "observational",
                interventions=list(do),
                measured=list(measured),
                study=study,
            )
            for rid, study, do, measured in studies
        ),
        bindings=tuple(
            transport.RegimeBinding(rid, f"snap-{rid}", sampling="independent")
            for rid, *_ in studies
        ),
    )


def s6_laws(p):
    out = []
    for rid, _study, do, measured in S6_STUDIES:
        for levels in itertools.product((0, 1), repeat=len(do)):
            world = dict(zip(do, levels, strict=True))
            out.append(
                transport.ExactDiscreteLaw(
                    "target",
                    rid,
                    tuple((n, (0.0, 1.0)) for n in measured),
                    tuple(enumerate_law(S6_FRONTDOOR, S6_NAMES, p, world, list(measured))),
                    f"snap-{rid}",
                    interventions=tuple((k, float(v)) for k, v in world.items()),
                )
            )
    return tuple(out)


def s6_query(names=("x", "y")):
    return transport.MixedSourceQuery(target="target", outcomes=[names[1]], treatments=[names[0]])


def s6_graph():
    return Admg.from_edges(S6_NAMES, [("x", "z"), ("z", "y")], [("x", "y")])


def s6_producer():
    # No single study identifies the effect: the search reports it as not certified.
    for alone in S6_STUDIES:
        stage = transport.identify_mixed_source_transport(
            graph=s6_graph(), query=s6_query(), catalog=s6_catalog([alone])
        )
        assert stage.outcome == "not_certified"
    stage = transport.identify_mixed_source_transport(
        graph=s6_graph(), query=s6_query(), catalog=s6_catalog(S6_STUDIES)
    )
    assert stage.outcome == "identified", stage.decision()
    decision = stage.decision()
    assert decision["rule_set"] == "x9.rules.v1" and decision["alternatives"] == []
    assert decision["cited_regimes"] == ["obs", "trial"]
    leaves = {s["source"]["regime"]: s["source"]["study"] for s in decision["steps"] if s["source"]}
    assert leaves == {"obs": "observational", "trial": "trial"}
    prepared = stage.prepare_exact(s6_laws(S6_P), [{"x": 0.0}, {"x": 1.0}])
    del stage
    live = json.loads(prepared.estimate())
    assert [risk(r) for r in live["requests"]] == pytest.approx(
        [s6_truth(S6_P, 0), s6_truth(S6_P, 1)], abs=1e-12
    )
    joint = enumerate_law(S6_FRONTDOOR, S6_NAMES, S6_P, {}, ["x", "y"])
    assert abs(joint[3] / (joint[2] + joint[3]) - s6_truth(S6_P, 1)) > 1e-2
    before = (prepared.export(), live, S6_P)
    prepared.refresh(s6_laws(S6_P_REFRESHED))
    moved = json.loads(prepared.estimate())
    assert [risk(r) for r in moved["requests"]] == pytest.approx(
        [s6_truth(S6_P_REFRESHED, 0), s6_truth(S6_P_REFRESHED, 1)], abs=1e-12
    )
    after = (prepared.export(), moved, S6_P_REFRESHED)

    # Incomplete side 1: the bow arc is not identifiable; the search only reports what it
    # explored and never claims non-identification.
    bow = transport.identify_mixed_source_transport(
        graph=Admg.from_edges(["x", "y"], [("x", "y")], [("x", "y")]),
        query=s6_query(),
        catalog=s6_catalog([("s", "study", (), ("x", "y"))], names=["x", "y"]),
    )
    assert bow.outcome == "not_certified"
    assert bow.decision()["reason"] == "transport_not_certified"
    assert bow.decision()["frontier"] and bow.decision()["generations"] > 0
    with pytest.raises(CausalUnsupportedError, match="transport_not_certified"):
        bow.prepare_exact((), {"x": 1.0})
    # Incomplete side 2: a budget stop is a resource outcome with a receipt, never a verdict.
    stopped = transport.identify_mixed_source_transport(
        graph=s6_graph(),
        query=s6_query(),
        catalog=s6_catalog(S6_STUDIES),
        max_operations=30,
    )
    assert stopped.outcome == "exhausted"
    receipt = stopped.decision()["limits_receipt"]
    assert receipt["stop"] == "search.operations" and receipt["operations_consumed"] == 30
    assert "stage:rule_search" in receipt["unevaluated"]
    with pytest.raises(Exception, match="transport_budget_cancel"):
        stopped.prepare_exact(s6_laws(S6_P), {"x": 1.0})
    return before, after


def test_story_6_mixed_source_proof_by_bounded_search_and_incomplete_search_stays_unresolved():
    before, after = s6_producer()
    for artifact, live, p in (before, after):
        consumed = json.loads(transport.consume_mixed_source_artifact(artifact))
        for c, lv in zip(consumed["requests"], live["requests"], strict=True):
            assert hexes(c["probabilities"]) == hexes(lv["probabilities"])
        assert [risk(r) for r in consumed["requests"]] == pytest.approx(
            [s6_truth(p, 0), s6_truth(p, 1)], abs=1e-12
        )
        assert {c["study"] for c in consumed["cited_sources"]} == {"observational", "trial"}
        assert consumed["proof"]["rule_set"] == "x9.rules.v1"
