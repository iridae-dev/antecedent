"""One finite two-step temporal transport sequence, exact and point-only.

Coordinates in time order: ``b`` (baseline), ``l1`` (step-1 covariate), ``a1``,
``l2`` (step-2 covariate, affected by ``a1``, a time-varying confounder of ``a2``
and ``y``), ``a2``, ``y``; two latent bits confound each action with ``y``. The
source and target differ at the initial state (``b``) and at the step-2 covariate
mechanism (``l2``); ``y`` is invariant. Every law is enumerated below from each
population's structural model, independently of the formula under test.
"""

import json

import antecedent
import pytest
from antecedent import Admg
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

NAMES = ["b", "l1", "a1", "l2", "a2", "y"]
B, L1, A1, L2, A2, Y = range(6)
DIRECTED = [
    ("b", "l1"), ("l1", "a1"), ("b", "l2"), ("a1", "l2"), ("l2", "a2"), ("a1", "a2"),
    ("b", "y"), ("l1", "y"), ("a1", "y"), ("l2", "y"), ("a2", "y"),
]  # fmt: skip
SOURCE_P = [0.35, 0.3, 0.45, 0.6, 0.25, 0.4, 0.7, 0.35]
TARGET_P = [0.8, 0.3, 0.45, 0.6, 0.6, 0.4, 0.7, 0.35]


def _mechanisms(population):
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


def enumerate_law(population, do, measured, exo=None):
    """Exact law of `measured` under `do` by enumerating the exogenous bits."""
    exo = exo or (SOURCE_P if population == "source" else TARGET_P)
    mechanisms = _mechanisms(population)
    out = [0.0] * (1 << len(measured))
    for mask in range(1 << len(exo)):
        e = [(mask >> bit) & 1 for bit in range(len(exo))]
        weight = 1.0
        for bit, p in enumerate(exo):
            weight *= p if e[bit] else 1.0 - p
        v = [0] * 6
        for i in range(6):
            v[i] = do[i] if i in do else mechanisms[i](v, e)
        index = 0
        for m in measured:
            index = (index << 1) | v[m]
        out[index] += weight
    return [min(1.0, max(0.0, p)) for p in out]


def truth(population, sequence):
    return enumerate_law(population, {A1: int(sequence[0]), A2: int(sequence[1])}, [Y])[1]


def graph():
    return Admg.from_edges(NAMES, DIRECTED, [("a1", "y"), ("a2", "y")])


def spec(selections=("b", "l2"), *, directed=None, actions=None, horizon=2):
    g = (
        graph()
        if directed is None
        else Admg.from_edges(NAMES, directed, [("a1", "y"), ("a2", "y")])
    )
    domain = "binary" if actions is None else "categorical"
    return transport.TemporalSequenceSpec(
        g,
        baseline=["b"],
        covariates=[["l1"], ["l2"]],
        actions=["a1", "a2"],
        outcome="y",
        coordinates=[
            transport.VariableCoordinate(
                n,
                domain if n in ("a1", "a2") else "binary",
                cardinality=actions if n in ("a1", "a2") and actions else None,
            )
            for n in NAMES
        ],
        selections=list(selections),
        horizon=horizon,
    )


def catalog(with_history=True, extra=()):
    regimes = [transport.EvidenceRegime("obs", "target", measured=NAMES)]
    if with_history:
        regimes.append(
            transport.EvidenceRegime(
                "history",
                "source",
                kind="experimental",
                interventions=["b", "l1", "a1", "l2", "a2"],
                measured=["y"],
            )
        )
    regimes.extend(extra)
    return transport.EvidenceCatalog(regimes=regimes)


def laws(source_p=None, target_p=None, skip=()):
    source_p, target_p = source_p or SOURCE_P, target_p or TARGET_P
    out = [
        transport.ExactDiscreteLaw(
            "target",
            "obs",
            tuple((n, (0.0, 1.0)) for n in NAMES),
            enumerate_law("target", {}, list(range(6)), target_p),
            "target",
        )
    ]
    for h in range(32):
        bits = [(h >> k) & 1 for k in (4, 3, 2, 1, 0)]
        b, l1, _a1, l2, _a2 = bits
        if (b, l1, l2) in skip:
            continue
        do = dict(zip([B, L1, A1, L2, A2], bits, strict=True))
        out.append(
            transport.ExactDiscreteLaw(
                "source",
                "history",
                (("y", (0.0, 1.0)),),
                enumerate_law("source", do, [Y], source_p),
                "source",
                interventions=tuple(
                    zip(["b", "l1", "a1", "l2", "a2"], map(float, bits), strict=True)
                ),
            )
        )
    return transport.ExactTransportData(tuple(out))


def prepare(sequence=(1.0, 0.0), *, the_spec=None, cat=None, data=None, **limits):
    return transport.prepare_temporal_transport_sequence(
        the_spec or spec(),
        sequence=list(sequence),
        source="source",
        target="target",
        catalog=cat or catalog(),
        laws=data or laws(),
        **limits,
    )


def test_the_whole_sequence_matches_the_target_interventional_truth():
    for a1 in (0.0, 1.0):
        for a2 in (0.0, 1.0):
            report = json.loads(prepare((a1, a2)).estimate())
            assert report["scope"] == "finite_two_step_temporal_transport_sequence_point_only"
            assert report["inference_claim"] == "point_only"
            assert report["horizon"] == 2 and report["sequence"] == [a1, a2]
            assert report["mean"] == pytest.approx(truth("target", (a1, a2)), abs=1e-12)
            assert report["point"]["means"]["y"] == pytest.approx(report["mean"])
            # The mechanism change matters: the source's own answer differs.
            assert abs(truth("source", (a1, a2)) - report["mean"]) > 0.01


def test_sequence_order_matters():
    ab = json.loads(prepare((0.0, 1.0)).estimate())
    ba = json.loads(prepare((1.0, 0.0)).estimate())
    assert ab["mean"] == pytest.approx(truth("target", (0, 1)), abs=1e-12)
    assert ba["mean"] == pytest.approx(truth("target", (1, 0)), abs=1e-12)
    assert abs(ab["mean"] - ba["mean"]) > 0.01


def test_time_varying_confounding_and_every_invariance_are_explicit():
    report = json.loads(prepare().estimate())
    assert report["time_varying_confounders"] == ["l2"]
    by = {i["variable"]: i for i in report["invariances"]}
    assert (by["b"]["slice"], by["b"]["assumption"]) == (0, "differs_by_selection")
    assert (by["l1"]["slice"], by["l1"]["assumption"]) == (1, "invariant")
    assert (by["l2"]["slice"], by["l2"]["assumption"]) == (2, "differs_by_selection")
    assert (by["y"]["slice"], by["y"]["assumption"]) == (2, "invariant")
    assert by["y"]["borrowed_from_source"] and not by["l2"]["borrowed_from_source"]
    assert "a1" not in by and "a2" not in by
    populations = {e["population"] for e in report["evidence"] if e["slice"] == 2}
    assert populations == {"source", "target"}
    # Without l2 -> a2 the second action has no time-varying confounder.
    plain = [e for e in DIRECTED if e != ("l2", "a2")]
    without = json.loads(prepare(the_spec=spec(directed=plain)).estimate())
    assert without["time_varying_confounders"] == []


def test_the_support_report_is_local_to_each_history_and_step():
    support = json.loads(prepare((0.0, 1.0)).estimate())["support"]
    assert support["initial_coordinates"] == ["b", "l1"]
    assert support["history_coordinates"] == ["b", "l1", "l2"]
    steps = [r["step"] for r in support["rows"]]
    assert steps.count(1) == 4 and steps.count(2) == 8
    assert {r["status"] for r in support["rows"]} == {"supported"}


def test_a_reached_history_without_source_support_refuses():
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.history_outside_support"
    ) as refused:
        prepare((1.0, 1.0), data=laws(skip=((1, 0, 1),)))
    assert refused.value.reason_code == "transport_support_failure"
    assert "b=1" in str(refused.value) and "l2=1" in str(refused.value)


def test_invalid_horizon_sequence_and_bounds_refuse():
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.horizon") as three:
        spec(horizon=3)
    assert three.value.reason_code == "route_not_supported"
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.horizon") as longer:
        prepare((1.0, 0.0, 1.0))
    assert longer.value.reason_code == "route_not_supported"
    with pytest.raises(CausalValueError, match="temporal_transport.invalid_sequence") as short:
        prepare((1.0,))
    assert short.value.reason_code == "invalid_argument"
    with pytest.raises(CausalValueError, match="temporal_transport.invalid_sequence"):
        prepare((1.0, 2.0))
    with pytest.raises(CausalValueError, match="temporal_transport.invalid_spec") as action:
        prepare(the_spec=spec(selections=("a1",)))
    assert action.value.reason_code == "invalid_argument"
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.bounds_exceeded") as wide:
        prepare(the_spec=spec(actions=9))
    assert wide.value.reason_code == "route_not_supported"


def test_one_shared_budget_stops_are_receipts_never_verdicts():
    with pytest.raises(CausalResourceError, match="temporal_transport.history_budget") as steps:
        prepare(max_steps=3)
    assert "search.operations" in str(steps.value)
    with pytest.raises(CausalResourceError, match="temporal_transport.history_budget"):
        prepare(max_depth=1)
    with pytest.raises(CausalResourceError, match="temporal_transport.history_budget"):
        prepare(memory_bytes=64)
    token = antecedent.state.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="temporal_transport.history_budget"):
        prepare(cancel=token)
    # The same premises decide under a modest budget: the stop said nothing about identification.
    assert json.loads(prepare(max_steps=1000).estimate())["status"] == "available"


def test_a_structural_obstruction_and_missing_evidence_refuse_with_their_own_reasons():
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.checked_obstruction"
    ) as shifted:
        prepare(the_spec=spec(selections=("y",)))
    assert shifted.value.reason_code == "transport_proven_non_transportable"
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.missing_evidence"
    ) as missing:
        prepare(
            cat=catalog(with_history=False),
            data=transport.ExactTransportData(tuple(laws().laws[:1])),
        )
    assert missing.value.reason_code == "transport_missing_evidence"


def test_an_interval_request_refuses_with_estimator_inference_mismatch():
    prepared = prepare()
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.interval_requested"
    ) as refused:
        prepared.interval()
    assert refused.value.reason_code == "estimator_inference_mismatch"


def test_same_window_refresh_re_estimates_and_a_new_window_needs_a_new_preparation():
    prepared = prepare((1.0, 0.0))
    before = json.loads(prepared.estimate())
    moved = list(SOURCE_P)
    moved[7] = 0.5
    target = list(TARGET_P)
    target[7] = 0.5
    prepared.refresh(laws(source_p=moved, target_p=target))
    after = json.loads(prepared.estimate())
    assert abs(after["mean"] - before["mean"]) > 1e-4
    assert after["mean"] == pytest.approx(
        enumerate_law("target", {A1: 1, A2: 0}, [Y], target)[1], abs=1e-12
    )
    # Without the source's history experiments the window is another window.
    only_target = transport.ExactTransportData(tuple(laws().laws[:1]))
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.horizon") as changed:
        prepared.refresh(only_target)
    assert changed.value.reason_code == "route_not_supported"
    # A dropped history experiment is a history outside support, not a new window.
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.history_outside_support"):
        prepared.refresh(laws(skip=((1, 1, 1),)))


PREFIX = b"ANTECEDENT-TEMPORAL-TRANSPORT\x01"


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


def test_artifact_round_trip_is_recomputed_bit_for_bit_and_tampering_fails():
    prepared = prepare((1.0, 0.0))
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    consumed = json.loads(transport.consume_temporal_transport_artifact(artifact))
    assert consumed["mean"] == live["mean"]
    assert consumed["point"]["probabilities"] == live["point"]["probabilities"]
    assert consumed["support"] == live["support"] and consumed["invariances"] == live["invariances"]
    assert consumed["sequence"] == [1.0, 0.0] and consumed["horizon"] == 2
    # Another sequence, another identity.
    other = prepare((0.0, 1.0))
    other.estimate()
    other_digest = json.loads(transport.consume_temporal_transport_artifact(other.export()))
    assert other_digest["premises_digest"] != consumed["premises_digest"]
    names, inner = _unframe(artifact)
    assert names == NAMES and _frame(names, inner) == artifact
    # A frame that relabels the variables cannot silently relabel the report: the
    # names are bound into the identity through the coordinate schema.
    swapped = ["l1", "b", *NAMES[2:]]
    with pytest.raises(CausalSerializationError, match="coordinate schema"):
        transport.consume_temporal_transport_artifact(_frame(swapped, inner))
    # One recorded search limit changed: the premises digest binds it.
    limit = b"\x1a\x00\x01\x86\xa0"
    changed = bytes(inner).replace(limit, b"\x1a\x00\x01\x86\x9f", 1)
    assert changed != bytes(inner)
    with pytest.raises(CausalSerializationError, match="premises digest"):
        transport.consume_temporal_transport_artifact(_frame(names, changed))
    # The consumer's own limits bound the recorded ones.
    with pytest.raises(CausalResourceError):
        transport.consume_temporal_transport_artifact(artifact, max_steps=5)
    with pytest.raises(CausalSerializationError):
        transport.consume_temporal_transport_artifact(artifact[:-3])
    with pytest.raises(antecedent.errors.CausalTypeError):
        transport.consume_temporal_transport_artifact("not bytes")


def test_export_needs_an_estimate_first():
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        prepare().export()


# Checked execution: each licensed Python route runs from the retained, frozen plan
# with no builder alive, and the test inspects that retained plan.


def _retained_plan(sequence=(1.0, 0.0)):
    builder = spec()
    retained_plan = prepare(sequence, the_spec=builder)
    del builder
    return retained_plan


def test_prepare_compiles_a_checked_plan_after_builder_disposal():
    builder = spec()
    retained_plan = prepare((0.0, 1.0), the_spec=builder)
    del builder
    plan = json.loads(retained_plan.plan())
    assert plan["compiled"] and plan["horizon"] == 2 and plan["sequence"] == [0.0, 1.0]
    assert plan["rules"] and plan["histories"] == {"initial": 4, "complete": 8}
    assert {population for population, _ in plan["cited"]} <= {"source", "target"}
    assert json.loads(retained_plan.estimate())["mean"] == pytest.approx(
        truth("target", (0, 1)), abs=1e-12
    )


def test_estimate_executes_the_retained_plan_after_builder_disposal():
    retained_plan = _retained_plan()
    plan = json.loads(retained_plan.plan())
    assert plan["compiled"]
    first = json.loads(retained_plan.estimate())
    second = json.loads(retained_plan.estimate())
    assert first["mean"] == second["mean"] == pytest.approx(truth("target", (1, 0)), abs=1e-12)


def test_refresh_re_estimates_from_the_retained_plan_after_builder_disposal():
    retained_plan = _retained_plan()
    before = json.loads(retained_plan.plan())
    moved = list(SOURCE_P)
    moved[7] = 0.5
    retained_plan.refresh(laws(source_p=moved))
    after_plan = json.loads(retained_plan.plan())
    assert after_plan["rules"] == before["rules"]
    assert after_plan["cited"] == before["cited"]
    assert json.loads(retained_plan.estimate())["status"] == "available"


def test_export_replays_the_retained_plan_after_builder_disposal():
    retained_plan = _retained_plan()
    assert json.loads(retained_plan.plan())["compiled"]
    live = json.loads(retained_plan.estimate())
    artifact = retained_plan.export()
    consumed = json.loads(transport.consume_temporal_transport_artifact(artifact))
    assert consumed["mean"] == live["mean"] and consumed["support"] == live["support"]


def test_consume_recomputes_from_the_retained_plan_after_builder_disposal():
    retained_plan = _retained_plan((0.0, 1.0))
    plan = json.loads(retained_plan.plan())
    live = json.loads(retained_plan.estimate())
    consumed = json.loads(transport.consume_temporal_transport_artifact(retained_plan.export()))
    assert consumed["sequence"] == plan["sequence"] == [0.0, 1.0]
    assert consumed["point"]["probabilities"] == live["point"]["probabilities"]
    assert consumed["mean"] == pytest.approx(truth("target", (0, 1)), abs=1e-12)


def _source_obs():
    return transport.EvidenceRegime("src_obs", "source", measured=NAMES)


def _source_actions():
    return transport.EvidenceRegime(
        "actions",
        "source",
        kind="experimental",
        interventions=["a1", "a2"],
        measured=["b", "l1", "l2", "y"],
    )


def test_a_source_law_that_does_not_serve_the_cited_leaf_supports_no_history():
    # The proof cites the source's outcome under every complete history. An
    # observational source law, or an experiment on the actions alone, is not that
    # leaf at any history: the support refusal is typed, never vacuously satisfied.
    obs_law = transport.ExactDiscreteLaw(
        "source",
        "src_obs",
        tuple((n, (0.0, 1.0)) for n in NAMES),
        enumerate_law("source", {}, list(range(6))),
        "source-obs",
    )
    only_obs = transport.ExactTransportData((laws().laws[0], obs_law))
    with pytest.raises(
        CausalUnsupportedError, match="temporal_transport.history_outside_support"
    ) as refused:
        prepare(cat=catalog(extra=[_source_obs()]), data=only_obs)
    assert refused.value.reason_code == "transport_support_failure"
    assert "8 of 12 histories" in str(refused.value)
    actions_law = transport.ExactDiscreteLaw(
        "source",
        "actions",
        tuple((n, (0.0, 1.0)) for n in ("b", "l1", "l2", "y")),
        enumerate_law("source", {A1: 1, A2: 0}, [B, L1, L2, Y]),
        "source-actions",
        interventions=(("a1", 1.0), ("a2", 0.0)),
    )
    only_actions = transport.ExactTransportData((laws().laws[0], actions_law))
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.history_outside_support"):
        prepare(cat=catalog(extra=[_source_actions()]), data=only_actions)


def test_evidence_is_marked_cited_only_when_the_derivation_reads_it():
    report = json.loads(prepare(cat=catalog(extra=[_source_obs()])).estimate())
    cited = {}
    for row in report["evidence"]:
        cited.setdefault(row["regime"], set()).add(row["cited_by_derivation"])
    # Regimes 0 (target observational) and 1 (history experiments) are read; 2 is not.
    assert cited == {0: {True}, 1: {True}, 2: {False}}


def test_a_refresh_that_moves_the_fixed_initial_state_needs_a_new_preparation():
    prepared = prepare((1.0, 0.0))
    moved = list(TARGET_P)
    moved[0] = 0.5
    with pytest.raises(CausalUnsupportedError, match="temporal_transport.horizon") as refused:
        prepared.refresh(laws(target_p=moved))
    assert refused.value.reason_code == "route_not_supported"
    assert "initial-state" in str(refused.value)
    # Other target mechanisms keep the same initial state: a same-window refresh.
    later = list(TARGET_P)
    later[7] = 0.4
    prepared.refresh(laws(target_p=later))
    assert json.loads(prepared.estimate())["status"] == "available"
