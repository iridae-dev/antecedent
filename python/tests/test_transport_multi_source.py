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
    CausalResourceError,
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


def catalog(*, studies=False, omit=(), snapshot_suffix="", regime_fields=None):
    """``studies`` is ``True`` (one study per regime), ``False``, or a mapping
    from regime id to study; ``regime_fields`` adds EvidenceRegime fields per id."""
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in NAMES)
    kept = [r for r in REGIMES if r[0] not in omit]

    def study(rid):
        if isinstance(studies, dict):
            return studies.get(rid)
        return f"study-{rid}" if studies else None

    regimes = tuple(
        transport.EvidenceRegime(
            rid,
            population,
            kind="experimental" if do else "observational",
            interventions=list(do),
            intervention_values={k: float(v) for k, v in do.items()},
            measured=[n for n in NAMES if n not in do],
            study=study(rid),
            **(regime_fields or {}).get(rid, {}),
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
        with pytest.raises(CausalUnsupportedError, match="transport_proven_non_transportable"):
            stage.prepare_exact(laws(), {"x": 1.0})


def test_an_unsupplied_regime_is_missing_evidence_not_an_obstruction():
    stage = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(omit=("b_z1_0",))
    )
    assert stage.outcome == "missing_evidence"
    decision = stage.decision()
    assert decision["reason"] == "transport_missing_evidence"
    assert decision["formula_certified"] is True
    with pytest.raises(CausalUnsupportedError, match="transport_missing_evidence"):
        stage.prepare_exact(laws(omit=("b_z1_0",)), {"x": 1.0})


def test_an_exhausted_search_returns_a_receipt():
    stage = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(), max_operations=3
    )
    assert stage.outcome == "exhausted"
    assert stage.identification_status == "budget_cancel"
    assert stage.decision()["identification_status"] == "budget_cancel"
    receipt = stage.decision()["limits_receipt"]
    assert receipt["stop"] == "search.operations"
    assert "stage:multi_source" in receipt["unevaluated"]
    with pytest.raises(Exception, match="transport_budget_cancel"):
        stage.prepare_exact(laws(), {"x": 1.0})


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


def test_the_unmeasured_interval_route_refuses_with_cell_not_licensed():
    """Counted laws return the point; the interval route is closed until its
    coverage records exist, so no interval is ever attached (live or consumed)."""
    with_studies = identified_stage(catalog(studies=True)).prepare_empirical(
        laws(sample_size=40_000), {"x": 1.0}, seed=3
    )
    result = json.loads(with_studies.estimate())
    assert risk(result) == pytest.approx(truth(1), abs=1e-2)
    interval = result["interval"]
    assert interval["available"] is False
    assert interval["status"] == "withheld"
    assert interval["reason"] == "cell_not_licensed"
    assert interval["dependence_reason"] is None
    assert interval["mean_intervals"] == [] and interval["contrast_intervals"] == []
    assert interval["seed"] is None and interval["replicates_requested"] == 0
    consumed = json.loads(
        transport.consume_multi_source_z_transport_artifact(with_studies.export())
    )
    assert consumed["interval"] == interval
    assert consumed["probabilities"] == result["probabilities"]
    # Without study identities the two arms of source a are not known to be
    # independent: that reason is reported beside the closed-route reason.
    undeclared = identified_stage().prepare_empirical(laws(sample_size=40_000), {"x": 1.0})
    withheld = json.loads(undeclared.estimate())
    assert withheld["interval"]["available"] is False
    assert withheld["interval"]["reason"] == "cell_not_licensed"
    assert withheld["interval"]["dependence_reason"] == "sampling_dependence_unknown"
    assert risk(withheld) == pytest.approx(risk(result))
    # No public Python route builds or yields an interval.
    for owner in (with_studies, transport):
        assert not [
            name
            for name in dir(owner)
            if "bootstrap" in name.lower() or name.lower().startswith("interval")
        ]
    # An artifact that claims an interval is refused as the closed route by the
    # consumer: `cell_not_licensed` with the `mz_transport.interval_withheld` detail.
    exact = identified_stage().prepare_exact(laws(), {"x": 1.0})
    exact.estimate()
    artifact = exact.export()
    claimed = _edit_inner(
        artifact,
        lambda inner: _replace_once(inner, b"\x6apoint_only", b"\x70nominal_interval"),
    )
    assert claimed != artifact
    with pytest.raises(CausalUnsupportedError) as refused:
        transport.consume_multi_source_z_transport_artifact(claimed)
    assert refused.value.reason_code == "cell_not_licensed"
    assert "mz_transport.interval_withheld" in str(refused.value)


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


REQUESTS = [{"x": 0.0}, {"x": 1.0}]


def test_multi_source_identification_executes_its_plan_after_query_disposal():
    """The decision is taken once; the stage keeps its proof after the query is gone."""
    builder = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog()
    )
    assert builder.outcome == "identified"
    program = builder.decision()
    prepared = builder.prepare_exact(laws(), {"x": 1.0})
    del builder
    assert program["route"] == "combined" and program["rules"]
    result = json.loads(prepared.estimate())
    assert risk(result) == pytest.approx(truth(1), abs=1e-12)
    assert prepared.cited_sources == program["sources"]


def test_prepare_exact_compiles_a_checked_plan_after_builder_disposal():
    builder = identified_stage()
    retained_plan = builder.prepare_exact(laws(), REQUESTS)
    del builder
    plan = retained_plan.plan()
    assert plan["compiled_plans"] == 2 and plan["requests"] == REQUESTS
    assert set(plan["cited_regimes"]) == {"a_z2_0", "a_z2_1", "b_z1_0"} and plan["rules"]
    result = json.loads(retained_plan.estimate())
    assert result["interval"]["status"] == "point_only"
    assert result["interval"]["available"] is False
    assert [risk(r) for r in result["requests"]] == pytest.approx([truth(0), truth(1)], abs=1e-12)
    assert retained_plan.cited_sources == ["a", "b"]


def test_prepare_empirical_compiles_a_counted_plan_after_builder_disposal():
    builder = identified_stage(catalog(studies=True))
    with pytest.raises(Exception, match="empirical_counts_required"):
        builder.prepare_empirical(laws(), {"x": 1.0})
    retained_plan = builder.prepare_empirical(laws(sample_size=40_000), REQUESTS, seed=5)
    del builder
    plan = retained_plan.plan()
    assert plan["compiled_plans"] == 2
    result = json.loads(retained_plan.estimate())
    assert [risk(r) for r in result["requests"]] == pytest.approx([truth(0), truth(1)], abs=1e-2)
    assert result["interval"]["reason"] == "cell_not_licensed"
    assert retained_plan.seed == 5


def test_estimate_executes_every_request_and_contrast_from_the_retained_plan():
    builder = identified_stage()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    result = json.loads(prepared.estimate())
    # Request 0 stays at the top level; every request is listed with its means.
    assert result["probabilities"] == result["requests"][0]["probabilities"]
    assert [r["assignment"] for r in result["requests"]] == REQUESTS
    means = [r["means"]["y"] for r in result["requests"]]
    assert means == pytest.approx([truth(0), truth(1)], abs=1e-12)
    (contrast,) = result["contrasts"]
    assert (contrast["request"], contrast["baseline"], contrast["outcome"]) == (1, 0, "y")
    assert contrast["estimate"] == pytest.approx(truth(1) - truth(0), abs=1e-12)
    # The retained plan re-executes to the same numbers.
    plan = prepared.plan()
    assert plan["compiled_plans"] == len(result["requests"])
    again = json.loads(prepared.estimate())
    assert again["contrasts"] == result["contrasts"]


def test_refresh_rebinds_snapshots_and_re_executes_the_retained_plan():
    builder = identified_stage()
    prepared = builder.prepare_empirical(laws(sample_size=40_000), REQUESTS)
    del builder
    before = json.loads(prepared.estimate())
    plan = prepared.plan()
    prepared.refresh(laws(sample_size=5_000))
    # The proof and requests of the plan are unchanged; only the laws moved.
    assert prepared.plan() == plan
    after = json.loads(prepared.estimate())
    assert len(after["requests"]) == 2 and len(after["contrasts"]) == 1
    assert after["contrasts"][0]["estimate"] != before["contrasts"][0]["estimate"]
    assert after["contrasts"][0]["estimate"] == pytest.approx(truth(1) - truth(0), abs=5e-2)
    # Refresh clears the last claim: nothing is exported until the plan runs again.
    prepared.refresh(laws(sample_size=5_000))
    with pytest.raises(Exception, match="not_executed"):
        prepared.export()


_MZ_PREFIX = b"ANTECEDENT-MZ-TRANSPORT\x01"


def _cbor_head(buf, i):
    initial = buf[i]
    major, info = initial >> 5, initial & 31
    i += 1
    if info < 24:
        return major, info, i
    width = {24: 1, 25: 2, 26: 4, 27: 8}[info]
    return major, int.from_bytes(buf[i : i + width], "big"), i + width


def _edit_inner(artifact, edit):
    """Apply ``edit`` to the inner CBOR bytes of a framed artifact and re-frame.

    The frame is the magic prefix and the CBOR array ``[names, [byte, ...]]``.
    """
    assert artifact.startswith(_MZ_PREFIX)
    body = artifact[len(_MZ_PREFIX) :]
    _, _, i = _cbor_head(body, 0)  # the two-element frame
    _, count, i = _cbor_head(body, i)  # names
    for _ in range(count):
        _, length, i = _cbor_head(body, i)
        i += length
    names_end = i
    _, n, i = _cbor_head(body, i)
    inner = bytearray()
    for _ in range(n):
        _, value, i = _cbor_head(body, i)
        inner.append(value)
    assert i == len(body)
    edited = bytes(edit(bytes(inner)))
    out = bytearray(_MZ_PREFIX) + body[:names_end]
    out += _cbor_uint(4, len(edited))
    for byte in edited:
        out += _cbor_uint(0, byte)
    return bytes(out)


def _cbor_uint(major, value):
    if value < 24:
        return bytes([major << 5 | value])
    if value < 256:
        return bytes([major << 5 | 24, value])
    if value < 65536:
        return bytes([major << 5 | 25]) + value.to_bytes(2, "big")
    return bytes([major << 5 | 26]) + value.to_bytes(4, "big")


def _replace_once(inner, old, new):
    assert inner.count(old) == 1, (old, inner.count(old))
    return inner.replace(old, new)


def _swap_first(data, a, b):
    i, j = data.index(a), data.index(b)
    out = bytearray(data)
    out[i : i + len(a)], out[j : j + len(b)] = b, a
    return bytes(out)


def test_export_binds_names_limits_and_snapshots_into_the_verified_identity():
    builder = identified_stage()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    json.loads(prepared.estimate())
    plan = prepared.plan()
    artifact = prepared.export()
    consumed = json.loads(transport.consume_multi_source_z_transport_artifact(artifact))
    assert consumed["proof"]["rules"] == plan["rules"]
    # Relabelling coordinates in the frame (z1 <-> z2, same byte length) is refused:
    # the names are bound into the artifact's verified identity.
    relabelled = _swap_first(artifact, b"bz1", b"bz2")
    assert relabelled != artifact
    with pytest.raises(CausalSerializationError, match="variable names do not match"):
        transport.consume_multi_source_z_transport_artifact(relabelled)

    # Renaming every snapshot consistently is caught by the data-identity digest.
    # The framed payload is a CBOR byte array: each ASCII byte is `0x18 <byte>`.
    def framed(text):
        return b"".join(b"\x18" + bytes([c]) for c in text.encode())

    renamed = artifact.replace(framed("snap-obs"), framed("snap-obz"))
    assert renamed != artifact
    with pytest.raises(CausalSerializationError, match="data identity digest mismatch"):
        transport.consume_multi_source_z_transport_artifact(renamed)
    # A consumer with smaller limits than the producer refuses ...
    with pytest.raises(CausalResourceError):
        transport.consume_multi_source_z_transport_artifact(artifact, max_search_operations=10)
    # ... and the limits the artifact itself stores are bound too: editing the
    # stored search limits in the frame (not just the consumer's argument) is refused.
    ops = _cbor_uint(0, 4096)
    stored_ops = _edit_inner(
        artifact,
        lambda inner: _replace_once(
            inner, b"search_operations" + ops, b"search_operations" + _cbor_uint(0, 4095)
        ),
    )
    assert stored_ops != artifact
    with pytest.raises(CausalSerializationError, match="premises digest mismatch"):
        transport.consume_multi_source_z_transport_artifact(stored_ops)
    stored_depth = _edit_inner(
        artifact,
        lambda inner: _replace_once(
            inner, b"search_depth" + _cbor_uint(0, 24), b"search_depth" + _cbor_uint(0, 23)
        ),
    )
    with pytest.raises(CausalSerializationError, match="premises digest mismatch"):
        transport.consume_multi_source_z_transport_artifact(stored_depth)
    # The operation and evaluation limits are bound the same way.
    stored_eval = _edit_inner(
        artifact,
        lambda inner: _replace_once(
            inner, b"depth_limit" + _cbor_uint(0, 256), b"depth_limit" + _cbor_uint(0, 255)
        ),
    )
    with pytest.raises(CausalSerializationError, match="premises digest mismatch"):
        transport.consume_multi_source_z_transport_artifact(stored_eval)
    # The search receipt is stored with the proof and reported on consumption.
    receipt = consumed["proof"]["search"]
    assert receipt["operations_limit"] == 4096 and receipt["depth_limit"] == 24
    assert receipt["memory_limit_bytes"] == 512 * 1024 * 1024
    assert receipt["operations_consumed"] > 0 and receipt["depth_reached"] > 0
    assert receipt["explored"][-1] == "stage:multi_source" and receipt["unevaluated"] == []
    # A consumer whose own memory limit is below the stored cap refuses up front
    # with a limits error the caller can retry with a larger budget, not an
    # invalid-proof error.
    with pytest.raises(CausalResourceError, match="search memory limit"):
        transport.consume_multi_source_z_transport_artifact(artifact, memory_bytes=1 << 20)
    # With the stored cap (or more) the same artifact verifies.
    affordable = json.loads(
        transport.consume_multi_source_z_transport_artifact(
            artifact, memory_bytes=512 * 1024 * 1024
        )
    )
    assert affordable["proof"]["search"]["memory_limit_bytes"] == 512 * 1024 * 1024


def test_consumer_recomputes_points_and_contrasts_after_builder_disposal():
    builder = identified_stage()
    prepared = builder.prepare_exact(laws(), REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    plan = prepared.plan()
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_multi_source_z_transport_artifact(artifact))
    assert consumed["proof"]["rules"] == plan["rules"]
    assert consumed["requests"] == live["requests"]
    assert consumed["contrasts"] == live["contrasts"]
    assert consumed["interval"] == live["interval"]
    assert consumed["cited_sources"] == ["a", "b"]
    assert consumed["premises_digest"] and consumed["data_digest"]


def test_a_fabricated_cross_source_joint_is_refused():
    """R-443 Figure 1(e,f): one c-factor would need do(z1, z2) jointly, split across
    two sources. The search never fabricates that joint."""
    names = ["z1", "x", "z2", "y"]
    split_graph = Admg.from_edges(
        names,
        [("z1", "x"), ("z2", "x"), ("x", "y")],
        [("z1", "x"), ("z1", "y"), ("z2", "x"), ("z2", "y")],
    )
    split = transport.MultiSourceZTransportQuery(
        target="target",
        outcomes=["y"],
        treatments=["x"],
        sources=[
            transport.ZTransportSource(
                "a", controllable=["z2"], selections=["z1"], experiment_assignment={"z2": 0.0}
            ),
            transport.ZTransportSource(
                "b", controllable=["z1"], selections=["z2"], experiment_assignment={"z1": 0.0}
            ),
        ],
    )
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    evidence = transport.EvidenceCatalog(
        environments=tuple(transport.Environment(p, coordinates) for p in ("target", "a", "b")),
        regimes=(
            transport.EvidenceRegime("obs", "target", measured=names),
            transport.EvidenceRegime(
                "a_z2",
                "a",
                kind="experimental",
                interventions=["z2"],
                measured=["z1", "x", "y"],
            ),
            transport.EvidenceRegime(
                "b_z1",
                "b",
                kind="experimental",
                interventions=["z1"],
                measured=["x", "z2", "y"],
            ),
        ),
        bindings=tuple(
            transport.RegimeBinding(rid, f"snap-{rid}", sampling="independent")
            for rid in ("obs", "a_z2", "b_z1")
        ),
    )
    stage = transport.identify_multi_source_z_transport(
        graph=split_graph, query=split, catalog=evidence
    )
    assert stage.outcome == "not_certified"
    decision = stage.decision()
    assert decision["reason"] == "transport_not_certified"
    assert {"stage": "multi_source", "outcome": "not_certified"} in decision["stages"]
    with pytest.raises(CausalUnsupportedError, match="transport_not_certified") as raised:
        stage.prepare_exact((), {"x": 1.0})
    assert _pair(raised) == ("transport_not_certified", "mz_transport.fabricated_joint")


def test_model_artifact_and_selected_sample_regimes_never_satisfy_a_factor():
    for fields in ({"model_artifact": "simulator-v1"}, {"selected_on": ["z2"]}):
        stage = transport.identify_multi_source_z_transport(
            graph=graph(),
            query=query(source_a(), source_b()),
            catalog=catalog(regime_fields={"b_z1_0": fields}),
        )
        assert stage.outcome != "identified", (fields, stage.decision())
        with pytest.raises(CausalUnsupportedError):
            stage.prepare_exact(laws(), {"x": 1.0})


def test_distinct_studies_are_the_declared_independence_the_interval_reports():
    counted = laws(sample_size=40_000)
    # One study per regime: independence is declared, nothing else withholds.
    distinct = identified_stage(catalog(studies=True)).prepare_empirical(counted, {"x": 1.0})
    assert json.loads(distinct.estimate())["interval"]["dependence_reason"] is None
    # Both arms of source a in one study (no shared dataset declared): unknown.
    shared = {"obs": "s-obs", "a_z2_0": "trial-a", "a_z2_1": "trial-a", "b_z1_0": "s-b"}
    same_study = identified_stage(catalog(studies=shared)).prepare_empirical(counted, {"x": 1.0})
    interval = json.loads(same_study.estimate())["interval"]
    assert interval["dependence_reason"] == "sampling_dependence_unknown"
    assert interval["reason"] == "cell_not_licensed"


def _pair(exc):
    """The (reason code, mz_transport detail) pair a raised refusal carries."""
    message = str(exc.value)
    detail = next(w.strip(":") for w in message.split() if w.startswith("mz_transport."))
    return exc.value.reason_code, detail


def test_every_reachable_detail_pairs_with_its_recorded_reason_code():
    # Non-identified decisions refuse preparation with their own frozen pair.
    missing = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(omit=("b_z1_0",))
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        missing.prepare_exact(laws(omit=("b_z1_0",)), {"x": 1.0})
    assert _pair(raised) == ("transport_missing_evidence", "mz_transport.missing_joint_regime")
    obstruction = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), unhelpful()), catalog=catalog()
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        obstruction.prepare_exact(laws(), {"x": 1.0})
    assert _pair(raised) == (
        "transport_proven_non_transportable",
        "mz_transport.checked_obstruction",
    )
    exhausted = transport.identify_multi_source_z_transport(
        graph=graph(), query=query(source_a(), source_b()), catalog=catalog(), max_operations=3
    )
    with pytest.raises(Exception, match="mz_transport.budget") as raised:
        exhausted.prepare_exact(laws(), {"x": 1.0})
    assert raised.value.reason_code == "transport_budget_cancel"
    # A bound: route_not_supported.
    with pytest.raises(CausalUnsupportedError) as raised:
        transport.identify_multi_source_z_transport(
            graph=graph(),
            query=query(source_a(), source_b()),
            catalog=catalog(),
            max_operations=4097,
        )
    assert _pair(raised) == ("route_not_supported", "mz_transport.bounds_exceeded")
    # An invalid catalog (the target supplies an experiment): invalid_argument.
    bad = catalog()
    experimental = transport.EvidenceRegime(
        "target_do",
        "target",
        kind="experimental",
        interventions=["z1"],
        intervention_values={"z1": 0.0},
        measured=["x", "z2", "y"],
    )
    bad = transport.EvidenceCatalog(
        environments=bad.environments,
        regimes=(*bad.regimes, experimental),
        bindings=(
            *bad.bindings,
            transport.RegimeBinding(
                "target_do",
                "snap-target_do",
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    with pytest.raises(Exception, match="mz_transport.invalid_catalog") as raised:
        transport.identify_multi_source_z_transport(
            graph=graph(), query=query(source_a(), source_b()), catalog=bad
        )
    assert _pair(raised) == ("invalid_argument", "mz_transport.invalid_catalog")
    # Empirical preparation over laws without counts.
    with pytest.raises(Exception, match="mz_transport.empirical_counts_required") as raised:
        identified_stage().prepare_empirical(laws(), {"x": 1.0})
    assert raised.value.reason_code == "transport_missing_provider"
