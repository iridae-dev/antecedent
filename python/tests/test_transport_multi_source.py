"""Bounded multi-source limited-experiment (mz) transport from Python.

The fixture is Bareinboim & Pearl (NeurIPS 2014, R-443) Figure 1(c,d):
``z1 -> x -> z2 -> y`` with ``z1 <-> x``, ``z1 <-> z2``, ``z1 <-> y``. Source ``a``
changes the ``z1`` and ``z2`` mechanisms and experiments on ``z2``; source ``b``
changes ``z1`` and ``y`` and experiments on ``z1``. Neither identifies
``P*(y | do(x))`` alone; together they do. Every law and the truth are enumerated
here from each population's structural equations, independently of the library.
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

NAMES = ["z1", "x", "z2", "y"]
EXO = [0.35, 0.6, 0.45, 0.3, 0.7, 0.65, 0.4]  # e0=z1<->x, e1=z1<->z2, e2=z1<->y, e3..e6 private


def _x(v, e):
    return v["z1"] ^ (e[0] & e[4])


def _z2_target(v, e):
    return (v["x"] & e[5]) | ((1 - v["x"]) & e[1])


def _y_target(v, e):
    return (v["z2"] & e[6]) | (e[2] & (1 - e[6]))


MECHANISMS = {
    "target": {
        "z1": lambda v, e: (e[0] & e[3]) | (e[1] & e[2]),
        "x": _x,
        "z2": _z2_target,
        "y": _y_target,
    },
    "a": {
        "z1": lambda v, e: e[3] | e[1],
        "x": _x,
        "z2": lambda v, e: v["x"] ^ (e[1] & e[5]),
        "y": _y_target,
    },
    "b": {
        "z1": lambda v, e: e[0] & e[2],
        "x": _x,
        "z2": _z2_target,
        "y": lambda v, e: v["z2"] ^ (e[2] & e[6]),
    },
}


def law_table(population, do, measured):
    """Exact joint over ``measured`` (first most significant) under ``do``."""
    table = [0.0] * (1 << len(measured))
    for bits in itertools.product((0, 1), repeat=len(EXO)):
        weight = 1.0
        for bit, p in zip(bits, EXO, strict=True):
            weight *= p if bit else 1.0 - p
        values = {}
        for name in NAMES:
            values[name] = do[name] if name in do else MECHANISMS[population][name](values, bits)
        index = 0
        for name in measured:
            index = (index << 1) | values[name]
        table[index] += weight
    return table


def truth(x):
    return law_table("target", {"x": x}, ["y"])[1]


def graph():
    return Admg.from_edges(
        NAMES,
        [("z1", "x"), ("x", "z2"), ("z2", "y")],
        [("z1", "x"), ("z1", "z2"), ("z1", "y")],
    )


def source_a():
    return transport.ZTransportSource("a", controllable=["z2"], selections=["z1", "z2"])


def source_b():
    return transport.ZTransportSource(
        "b", controllable=["z1"], selections=["z1", "y"], experiment_assignment={"z1": 0.0}
    )


def unhelpful():
    return transport.ZTransportSource("unhelpful", controllable=["x"], selections=NAMES)


def query(*sources):
    return transport.MultiSourceZTransportQuery(
        target="target", outcomes=["y"], treatments=["x"], sources=list(sources)
    )


# (regime id, population, do assignment)
REGIMES = [
    ("obs", "target", {}),
    ("a_z2_0", "a", {"z2": 0}),
    ("a_z2_1", "a", {"z2": 1}),
    ("b_z1_0", "b", {"z1": 0}),
]


def catalog(*, studies=False, omit=(), snapshot_suffix=""):
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in NAMES)
    kept = [r for r in REGIMES if r[0] not in omit]
    regimes = tuple(
        transport.EvidenceRegime(
            rid,
            population,
            kind="experimental" if do else "observational",
            interventions=list(do),
            intervention_values={k: float(v) for k, v in do.items()},
            measured=[n for n in NAMES if n not in do],
            study=f"study-{rid}" if studies else None,
        )
        for rid, population, do in kept
    )
    bindings = tuple(
        transport.RegimeBinding(
            rid,
            f"snap-{rid}{snapshot_suffix}",
            schema_names=[n for n in NAMES if n not in do],
            sampling="independent",
            dependence="independent_studies",
        )
        for rid, _, do in kept
    )
    environments = tuple(
        transport.Environment(population, coordinates) for population in ("target", "a", "b")
    )
    return transport.EvidenceCatalog(environments=environments, regimes=regimes, bindings=bindings)


def laws(*, sample_size=None, omit=(), snapshot_suffix=""):
    out = []
    for rid, population, do in REGIMES:
        if rid in omit:
            continue
        measured = [n for n in NAMES if n not in do]
        probabilities = law_table(population, do, measured)
        counts = None
        if sample_size is not None:
            counts = tuple(round(p * sample_size) for p in probabilities)
            total = sum(counts)
            probabilities = [c / total for c in counts]
        out.append(
            transport.ExactDiscreteLaw(
                population,
                rid,
                tuple((name, (0.0, 1.0)) for name in measured),
                tuple(probabilities),
                f"snap-{rid}{snapshot_suffix}",
                interventions=tuple((k, float(v)) for k, v in do.items()),
                empirical_counts=counts,
            )
        )
    return tuple(out)


def identified_stage(evidence=None, *sources):
    stage = transport.identify_multi_source_z_transport(
        graph=graph(),
        query=query(*(sources or (source_a(), source_b()))),
        catalog=evidence or catalog(),
    )
    assert stage.outcome == "identified", stage.decision()
    return stage


def risk(result):
    return sum(
        p for atom, p in zip(result["atoms"], result["probabilities"], strict=True) if atom == [1.0]
    )


def test_complementary_sources_identify_what_neither_can_alone():
    stage = identified_stage()
    decision = stage.decision()
    assert decision["route"] == "combined"
    assert decision["sources"] == ["a", "b"]
    assert set(decision["cited_regimes"]) == {"a_z2_0", "a_z2_1", "b_z1_0"}
    assert {"stage": "source:a", "outcome": "obstruction"} in decision["stages"]
    for x in (0.0, 1.0):
        prepared = stage.prepare_exact(laws(), {"x": x})
        result = json.loads(prepared.estimate())
        assert result["scope"] == "multi_source_z_transport_cited_joints_sound_incomplete"
        assert risk(result) == pytest.approx(truth(int(x)), abs=1e-12)
        assert result["interval"]["status"] == "point_only"
        assert prepared.cited_sources == ["a", "b"]
    # The target's own observational conditional is not the answer.
    joint = law_table("target", {}, ["x", "y"])
    assert abs(joint[3] / (joint[2] + joint[3]) - truth(1)) > 1e-3


def test_source_order_does_not_change_the_decision():
    forward = identified_stage(None, source_a(), source_b()).decision()
    reverse = identified_stage(None, source_b(), source_a()).decision()
    assert forward == reverse


def test_each_source_alone_is_a_checked_obstruction():
    for alone, c0 in ((source_a(), ["z2"]), (source_b(), ["y"])):
        stage = transport.identify_multi_source_z_transport(
            graph=graph(), query=query(alone, unhelpful()), catalog=catalog()
        )
        assert stage.outcome == "proven_non_transportable"
        decision = stage.decision()
        assert decision["reason"] == "transport_proven_non_transportable"
        assert decision["c0"] == c0
        with pytest.raises(Exception, match="transport_not_certified"):
            stage.prepare_exact(laws(), {"x": 1.0})


def test_an_unsupplied_regime_is_missing_evidence_not_an_obstruction():
    stage = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(omit=("b_z1_0",))
    )
    assert stage.outcome == "missing_evidence"
    decision = stage.decision()
    assert decision["reason"] == "transport_missing_evidence"
    assert decision["formula_certified"] is True


def test_an_exhausted_search_returns_a_receipt():
    stage = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(), max_operations=3
    )
    assert stage.outcome == "exhausted"
    receipt = stage.decision()["limits_receipt"]
    assert receipt["stop"] == "search.operations"
    assert "multi_source" in receipt["unevaluated"]


def test_exported_result_is_recomputed_by_an_independent_consumer():
    prepared = identified_stage().prepare_exact(laws(), {"x": 1.0})
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    consumed = json.loads(transport.consume_multi_source_z_transport_artifact(artifact))
    assert consumed["probabilities"] == live["probabilities"]
    assert consumed["cited_sources"] == ["a", "b"]
    assert consumed["proof"]["route"] == "combined"
    with pytest.raises(CausalSerializationError):
        transport.consume_multi_source_z_transport_artifact(artifact[:-3] + b"\x00\x00\x00")


def test_empirical_interval_requires_declared_independence():
    with_studies = identified_stage(catalog(studies=True)).prepare_empirical(
        laws(sample_size=40_000), {"x": 1.0}, seed=3
    )
    result = json.loads(with_studies.estimate())
    interval = result["interval"]
    assert interval["status"] == "nominal_interval"
    assert interval["reason"] == "estimator_grid_not_measured"
    assert risk(result) == pytest.approx(truth(1), abs=1e-2)
    consumed = json.loads(
        transport.consume_multi_source_z_transport_artifact(with_studies.export())
    )
    assert consumed["interval"]["status"] == "nominal_interval"
    # Without study identities the two arms of source a are not known to be
    # independent: the interval is withheld, the point is not.
    undeclared = identified_stage().prepare_empirical(laws(sample_size=40_000), {"x": 1.0})
    withheld = json.loads(undeclared.estimate())
    assert withheld["interval"]["status"] == "withheld"
    assert withheld["interval"]["reason"] == "sampling_dependence_unknown"
    assert risk(withheld) == pytest.approx(risk(result))


def test_refresh_keeps_the_proof_and_refuses_other_snapshots():
    prepared = identified_stage().prepare_empirical(laws(sample_size=40_000), {"x": 1.0})
    before = json.loads(prepared.estimate())
    prepared.refresh(laws(sample_size=5_000))
    after = json.loads(prepared.estimate())
    assert after["probabilities"] != before["probabilities"]
    # Evidence under a snapshot the frozen catalog never bound needs a new preparation.
    with pytest.raises(CausalUnsupportedError, match="snapshot does not match catalog binding"):
        prepared.refresh(laws(sample_size=5_000, snapshot_suffix="-new"))


def test_query_contract_is_validated_before_search():
    with pytest.raises(CausalValueError):
        query(source_a())
    with pytest.raises(CausalValueError):
        query(source_a(), transport.ZTransportSource("a", controllable=["z1"]))
    with pytest.raises(CausalValueError):
        transport.ZTransportSource("a", controllable=["z2"], experiment_assignment={"x": 0.0})
    with pytest.raises(CausalTypeError):
        transport.identify_multi_source_z_transport(
            graph=graph(), query=query(source_a(), source_b()), catalog={}
        )
