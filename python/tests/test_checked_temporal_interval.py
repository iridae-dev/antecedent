"""Original checked temporal source/proof lifecycle; no coverage measurements."""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalCancelledError, CausalError, CausalTypeError, CausalValueError
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success
from antecedent.recalc import RecalcRefusal, Utility
from antecedent.recalc_temporal import (
    CheckedTemporalIntervalCandidate as Candidate,
)
from antecedent.recalc_temporal import (
    CheckedTemporalIntervalConfig as Config,
)
from antecedent.recalc_temporal import (
    CheckedTemporalIntervalIdentity as Identity,
)
from antecedent.recalc_temporal import (
    TemporalEffect,
    TemporalHistory,
    TemporalRequest,
    TemporalResponse,
    TemporalSession,
    TemporalUnit,
)

MASK = 2**64 - 1
GAMMA = 0x9E3779B97F4A7C15


def mix(value):
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def draw(n, replicate_id):
    state = mix((replicate_id + GAMMA) & MASK)
    indexes = []
    for _ in range(n):
        state = (state + GAMMA) & MASK
        indexes.append((mix(state) * n) >> 64)
    return indexes


def request(functional):
    rng = np.random.default_rng(921713)
    units = []
    for unit in range(80):
        shock = 0.1 if rng.random() < 0.5 else -0.1
        histories = []
        for s0, a1, l2, a2 in itertools.product(range(2), repeat=4):
            mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2) + shock
            histories.append(
                TemporalHistory(3 * len(histories), s0, a1, l2, a2, float(rng.random() < mean))
            )
        units.append(TemporalUnit(unit, tuple(histories)))
    return TemporalRequest(
        tuple(itertools.combinations(range(5), 2)),
        tuple(units),
        (0, 49),
        "checked-binary-shared-unit-shock",
        (0.3, 0.7),
        "retained-target-law",
        functional,
        Utility(1, 0),
    )


def scores(req):
    def response(unit, sequence):
        return sum(
            req.initial_state[h.s0] * 0.5 * h.y for h in unit.histories if (h.a1, h.a2) == sequence
        )

    if isinstance(req.functional, TemporalResponse):
        return np.array([response(unit, req.functional.sequence) for unit in req.units])
    return np.array(
        [
            response(unit, req.functional.active) - response(unit, req.functional.control)
            for unit in req.units
        ]
    )


def identity():
    return Identity(
        seal="a" * 64,
        source_data_digest="b" * 64,
        source_premises_digest="c" * 64,
        snapshot_id="original",
        initial_state_id="target",
        functional=TemporalEffect((1, 1), (0, 0)),
        producing_seed=41,
        config=Config(method="studentized"),
        panel_digest="d" * 16,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("replicates", 19),
        ("replicates", 2001),
        ("level", 0),
        ("level", 1),
        ("level", float("nan")),
        ("min_units", 1),
        ("min_units", 4097),
        ("bootstrap_seed", -1),
        ("max_failed_fraction", 1),
        ("method", "wild"),
    ],
)
def test_configuration_rejects_outside_declared_bounds(field, value):
    with pytest.raises(CausalValueError):
        Config(**({"method": "studentized"} | {field: value}))


@pytest.mark.parametrize(
    "field,value",
    [("replicates", True), ("bootstrap_seed", 1.5), ("level", True), ("min_units", "20")],
)
def test_configuration_rejects_wrong_types(field, value):
    with pytest.raises(CausalTypeError):
        Config(**({"method": "studentized"} | {field: value}))


def test_identity_is_exact_config_and_paired_functional_and_immutable():
    original = identity()
    restored = Identity._from_wire(json.loads(json.dumps(original._wire())))
    assert restored == original
    assert restored.config.bootstrap_seed != restored.producing_seed
    assert restored.functional == TemporalEffect((1, 1), (0, 0))
    with pytest.raises(FrozenInstanceError):
        restored.seal = "e" * 64
    with pytest.raises(CausalValueError):
        replace(original, panel_digest="d" * 64)
    with pytest.raises(CausalValueError):
        replace(original, functional=TemporalResponse((1, 2)))


def test_candidate_constructor_and_foreign_callback_cannot_mint_native_authority():
    with pytest.raises(CausalTypeError):
        Candidate()
    calls = []
    foreign = SimpleNamespace(payload=lambda: calls.append(1), export=lambda: calls.append(2))
    with pytest.raises(CausalTypeError, match="native authority"):
        Candidate._from_native(foreign)
    assert calls == []


def test_load_rejects_bounds_before_dispatch_and_normal_route_is_exactly_closed(monkeypatch):
    monkeypatch.delattr(_native, "consume_checked_temporal_interval_candidate", raising=False)
    with pytest.raises(CausalValueError):
        Candidate.load(b"", expected=identity(), max_replicates=2001)
    with pytest.raises(CausalTypeError):
        Candidate.load(b"", expected=identity(), max_units=True)
    with pytest.raises(CausalTypeError):
        Candidate.load(b"", expected=identity(), execution=Execution(cancel=object()))
    with pytest.raises(RecalcRefusal) as caught:
        Candidate.load(b"", expected=identity())
    assert (caught.value.code, caught.value.detail, caught.value.stage) == (
        "cell_not_licensed",
        "temporal_interval.route_frozen",
        "inference",
    )


INTERNAL = hasattr(_native, "consume_checked_temporal_interval_candidate")


@pytest.mark.skipif(
    not INTERNAL, reason="original checked lifecycle requires isolated calibration-internal wheel"
)
@pytest.mark.parametrize(
    ("functional", "method"),
    [
        (TemporalResponse((0, 0)), "studentized"),
        (TemporalEffect((1, 1), (0, 0)), "studentized"),
        (TemporalEffect((1, 1), (0, 0)), "percentile"),
        (TemporalEffect((1, 1), (0, 0)), "basic"),
    ],
)
def test_original_checked_source_score_draws_intervals_and_fresh_consumer(
    functional, method, tmp_path
):
    req = request(functional)
    session = TemporalSession()
    point = require_success(session.execute(req, execution=Execution(seed=41)))
    config = Config(method=method, replicates=500, bootstrap_seed=9171, max_failed_fraction=0)
    candidate = session.interval_candidate(config=config)
    payload = candidate.inspect()
    values = scores(req)
    assert candidate.point == pytest.approx(values.mean(), abs=1e-12)
    assert point.law.ate == pytest.approx(candidate.point, abs=1e-12)
    assert candidate.functional == functional
    assert candidate.config == config
    assert candidate.calibration == "unmeasured"
    assert candidate.expected_identity.snapshot_id == req.snapshot_id
    assert candidate.expected_identity.initial_state_id == req.initial_state_id
    assert candidate.expected_identity.producing_seed == 41
    # The panel and replicate identities are BLAKE3 canonical digests of the panel's content,
    # which this file does not re-derive: it checks that they are carried consistently and
    # that each replicate id drives the documented unit draw.
    assert re.fullmatch(r"[0-9a-f]{16}", candidate.expected_identity.panel_digest)
    assert payload["result"]["panel_digest"] == candidate.expected_identity.panel_digest
    records = payload["result"]["replicates"]
    assert len({record["replicate_id"] for record in records}) == config.replicates
    selected = np.array([values[draw(len(values), record["replicate_id"])] for record in records])
    points = selected.mean(axis=1)
    assert payload["result"]["failed"] == 0
    assert [record["point"] for record in records] == pytest.approx(points, abs=1e-12)
    if method == "studentized":
        se = values.std(ddof=1) / len(values) ** 0.5
        replicate_se = selected.std(axis=1, ddof=1) / len(values) ** 0.5
        pivots = (points - values.mean()) / replicate_se
        assert candidate.unit_scores == pytest.approx(values, abs=1e-12)
        assert candidate.standard_error == pytest.approx(se, abs=1e-12)
        assert payload["result"]["studentization"]["replicate_standard_errors"] == pytest.approx(
            replicate_se, abs=1e-12
        )
        assert payload["result"]["studentization"]["pivots"] == pytest.approx(pivots, abs=1e-10)
        tails = np.quantile(pivots, [0.025, 0.975])
        expected = (values.mean() - tails[1] * se, values.mean() - tails[0] * se)
    else:
        low, high = np.quantile(points, [0.025, 0.975])
        if method == "percentile":
            # The checked percentile effect interval reports bias-corrected and accelerated
            # endpoints (Efron 1987): bias from the replicate mass below the point, acceleration
            # from the delete-one-unit jackknife of the unit-score mean, type-7 quantiles.
            from statistics import NormalDist

            normal = NormalDist()
            # Exact ties with the point estimate set the bias constant, and re-summing the draws
            # in numpy differs from the native sums in the last place. Count against the
            # native-reported replicate points and the point the candidate reports.
            reported = np.array([record["point"] for record in records])
            below = np.sum(reported < candidate.point)
            ties = np.sum(reported == candidate.point)
            fraction = (below + 0.5 * ties) / len(points)
            bias = normal.inv_cdf(fraction)
            jackknife = (values.sum() - values) / (len(values) - 1)
            deltas = jackknife.mean() - jackknife
            acceleration = np.sum(deltas**3) / (6 * np.sum(deltas**2) ** 1.5)

            def adjusted(tail):
                shifted = bias + normal.inv_cdf(tail)
                return normal.cdf(bias + shifted / (1 - acceleration * shifted))

            expected = tuple(np.quantile(points, [adjusted(0.025), adjusted(0.975)]))
        elif method == "basic":
            expected = (2 * values.mean() - high, 2 * values.mean() - low)
        else:
            raise AssertionError(method)
        assert candidate.standard_error is None and candidate.unit_scores is None
    assert candidate.interval == pytest.approx(expected, abs=1e-12)
    artifact = tmp_path / "original-checked-interval.cbor"
    artifact.write_bytes(bytes(candidate.export()))
    replayed = Candidate.load(artifact.read_bytes(), expected=candidate.expected_identity)
    assert replayed.inspect() == candidate.inspect()
    script = """
import json,pathlib,sys
from antecedent.recalc_temporal import CheckedTemporalIntervalCandidate as C,CheckedTemporalIntervalIdentity as I
r=C.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    observed = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                script,
                str(artifact),
                json.dumps(candidate.expected_identity._wire()),
            ],
            text=True,
        )
    )
    assert observed == candidate.inspect()
    payload["result"]["point"] = 12345
    assert candidate.point == pytest.approx(values.mean(), abs=1e-12)
    with pytest.raises(CausalTypeError):
        replace(candidate, _native=object())
    for changed in [
        replace(candidate.expected_identity, producing_seed=42),
        replace(candidate.expected_identity, snapshot_id="changed"),
        replace(candidate.expected_identity, initial_state_id="changed"),
        replace(candidate.expected_identity, source_data_digest="0" * 64),
        replace(candidate.expected_identity, config=replace(config, bootstrap_seed=9172)),
        replace(candidate.expected_identity, functional=TemporalResponse((1, 0))),
    ]:
        with pytest.raises(CausalError):
            Candidate.load(bytes(candidate.export()), expected=changed)
    for bounds in [
        {"max_units": 79},
        {"max_histories": 1279},
        {"max_replicates": 79},
    ]:
        with pytest.raises(CausalError):
            Candidate.load(
                bytes(candidate.export()), expected=candidate.expected_identity, **bounds
            )
    with pytest.raises(CausalError):
        Candidate.load(
            bytes(candidate.export()),
            expected=candidate.expected_identity,
            execution=Execution(limits=ResourceLimits(memory_bytes=0)),
        )
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        Candidate.load(
            bytes(candidate.export()),
            expected=candidate.expected_identity,
            execution=Execution(cancel=cancelled),
        )
    with pytest.raises(CausalError):
        Candidate.load(b"invalid", expected=candidate.expected_identity)


def test_configuration_numeric_overflow_and_identity_unicode_are_typed_refusals():
    with pytest.raises(CausalValueError):
        Config(method="studentized", level=10**1000)
    with pytest.raises(CausalValueError):
        replace(identity(), snapshot_id="\ud800")


def test_duck_typed_session_callback_is_refused_before_invocation():
    calls = []
    session = TemporalSession.__new__(TemporalSession)
    session._handle = SimpleNamespace(interval_candidate=lambda *args, **kwargs: calls.append(1))
    with pytest.raises(CausalTypeError, match="original checked temporal session"):
        session.interval_candidate(config=Config(method="studentized"))
    assert calls == []


def test_live_original_source_without_optional_hook_keeps_exact_frozen_refusal(monkeypatch):
    import builtins

    import antecedent.recalc_temporal as temporal

    req = request(TemporalEffect((1, 1), (0, 0)))
    session = TemporalSession()
    require_success(session.execute(req, execution=Execution(seed=41)))
    identities = session.identities

    def without_optional(obj, name, default=None):
        return None if name == "interval_candidate" else builtins.getattr(obj, name, default)

    monkeypatch.setattr(temporal, "getattr", without_optional, raising=False)
    with pytest.raises(RecalcRefusal) as caught:
        session.interval_candidate(config=Config(method="studentized"))
    assert (caught.value.code, caught.value.detail, caught.value.stage) == (
        "cell_not_licensed",
        "temporal_interval.route_frozen",
        "inference",
    )
    assert session.identities == identities
    assert (
        require_success(session.execute(req, execution=Execution(seed=41))).receipt.totals.total
        == 0
    )


@pytest.mark.skipif(
    not INTERNAL, reason="original checked lifecycle requires isolated calibration-internal wheel"
)
def test_original_checked_source_reversed_nonconsecutive_ids_fresh_replay(tmp_path):
    original = request(TemporalEffect((1, 1), (0, 0)))
    req = replace(
        original,
        units=tuple(
            TemporalUnit(1009 + 17 * index, unit.histories)
            for index, unit in reversed(tuple(enumerate(original.units)))
        ),
    )
    assert [unit.unit_id for unit in req.units] == sorted(
        [unit.unit_id for unit in req.units], reverse=True
    )
    session = TemporalSession()
    point = require_success(session.execute(req, execution=Execution(seed=41)))
    config = Config(
        method="studentized", replicates=500, bootstrap_seed=9171, max_failed_fraction=0
    )
    candidate = session.interval_candidate(config=config)
    values = scores(req)
    payload = candidate.inspect()
    assert candidate.unit_scores == pytest.approx(values, abs=1e-12)
    assert candidate.point == pytest.approx(values.mean(), abs=1e-12)
    assert point.law.ate == pytest.approx(values.mean(), abs=1e-12)
    assert re.fullmatch(r"[0-9a-f]{16}", candidate.expected_identity.panel_digest)
    selected = np.array(
        [values[draw(len(values), row["replicate_id"])] for row in payload["result"]["replicates"]]
    )
    assert [row["point"] for row in payload["result"]["replicates"]] == pytest.approx(
        selected.mean(axis=1), abs=1e-12
    )
    artifact = tmp_path / "reversed-source-checked-interval.cbor"
    artifact.write_bytes(bytes(candidate.export()))
    replayed = Candidate.load(artifact.read_bytes(), expected=candidate.expected_identity)
    assert replayed.inspect() == payload
    script = """
import json,pathlib,sys
from antecedent.recalc_temporal import CheckedTemporalIntervalCandidate as C,CheckedTemporalIntervalIdentity as I
r=C.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    observed = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                script,
                str(artifact),
                json.dumps(candidate.expected_identity._wire()),
            ],
            text=True,
        )
    )
    assert observed == payload
