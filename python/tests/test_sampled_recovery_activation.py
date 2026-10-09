"""Prepared public candidate lifecycle, with independent whole-row recovery algebra.

These are release-acceptance preparations in a calibration-internal wheel. They
do not measure coverage or activate a default release route.
"""

import itertools
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import (
    CausalError,
    CausalSerializationError,
    CausalTypeError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

from test_observation_recovery import catalog, graph, query

pytestmark = pytest.mark.skipif(
    not hasattr(_native, "consume_sampled_recovery_candidate"),
    reason="candidate requires calibration-internal wheel; normal release route stays closed",
)


def rows():
    """Frozen confounded missingness SCM, probabilities enumerated independently."""
    probabilities = Counter()
    bern = lambda p, value: p if value else 1 - p  # noqa: E731
    for t, y, z, rt, ry in itertools.product((0, 1), repeat=5):
        p = (
            bern(0.4, z)
            * bern((0.3, 0.7)[z], t)
            * bern(((0.2, 0.5), (0.6, 0.8))[t][z], y)
            * bern((0.5, 0.8)[z], rt)
            * bern((0.6, 0.9)[t], ry)
        )
        probabilities[(rt | (ry << 1), (rt & t) | ((ry & y) << 1), z)] += p
    counts = {pattern: round(2000 * p) for pattern, p in sorted(probabilities.items())}
    counts[max(counts, key=counts.get)] += 2000 - sum(counts.values())
    result = []
    for pattern, n in counts.items():
        offset = len(result)
        result.extend((offset + i, *pattern) for i in range(n))
    return result


def oracle(sample):
    """Empirical complete-case cell / observed response ratios, then backdoor mean."""
    counts = Counter(tuple(row[1:]) for row in sample)
    n = len(sample)
    rt_by_z = [
        sum(c for (r, _, z0), c in counts.items() if z0 == z and r & 1)
        / sum(c for (_, _, z0), c in counts.items() if z0 == z)
        for z in (0, 1)
    ]
    ry_by_t = [
        sum(c for (r, p, _), c in counts.items() if r & 1 and p & 1 == t and r & 2)
        / sum(c for (r, p, _), c in counts.items() if r & 1 and p & 1 == t)
        for t in (0, 1)
    ]
    law = np.array(
        [
            counts[(3, t | (y << 1), z)] / n / rt_by_z[z] / ry_by_t[t]
            for t, y, z in itertools.product((0, 1), repeat=3)
        ]
    ).reshape(2, 2, 2)
    z_mass = law.sum(axis=(0, 1)) / law.sum()
    y_given_tz = law[:, 1, :] / law.sum(axis=1)
    effect = np.dot(z_mass, y_given_tz[1] - y_given_tz[0])
    return law.ravel(), effect


def mix(value):
    """Reference SplitMix64 finalizer; only the primitive RNG is shared mathematically."""
    mask = (1 << 64) - 1
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def resample(sample, replicate, seed):
    gamma, mask = 0x9E3779B97F4A7C15, (1 << 64) - 1
    state = mix(((seed ^ mix(replicate + 1)) + gamma) & mask)
    result = []
    for _ in sample:
        state = (state + gamma) & mask
        result.append(sample[(mix(state) * len(sample)) >> 64])
    return result


def produce(*, seed=7, sample=None, declaration=None, snapshot="snap-1"):
    stage = transport.identify_observation_recovery(
        graph=graph(),
        query=query(),
        catalog=catalog(),
        effect_outcomes=["y"],
        effect_treatments=["t"],
    )
    return transport.sampled_observation_recovery(
        stage=stage,
        query=query() if declaration is None else declaration,
        rows=rows() if sample is None else sample,
        snapshot=snapshot,
        replicates=500,
        seed=seed,
    )


def test_public_sampled_recovery_whole_method_matches_independent_oracle(tmp_path):
    sample = rows()
    result = produce(sample=sample)
    law, point = oracle(sample)
    evidence = result.inspect()
    assert result.calibration == "unmeasured"
    assert result.level == 0.95
    assert result.effect == pytest.approx(point, abs=1e-12)
    assert result.effect == pytest.approx(0.36, abs=0.01)
    assert evidence["result"]["recovered"]["probabilities"] == pytest.approx(law, abs=1e-12)
    replicate_laws, effects = zip(
        *(oracle(resample(sample, r, 7)) for r in range(500)), strict=True
    )
    assert evidence["result"]["failed_replicates"] == 0
    assert evidence["result"]["replicate_effects"] == pytest.approx(effects, abs=1e-12)
    assert result.interval == pytest.approx(np.quantile(effects, [0.025, 0.975]), abs=1e-12)
    assert result.effect_standard_error == pytest.approx(np.std(effects, ddof=1), abs=1e-12)
    assert evidence["result"]["recovered_cell_covariance"] == pytest.approx(
        np.cov(np.array(replicate_laws).T, ddof=1).ravel(), abs=1e-12
    )
    artifact = tmp_path / "recovery.cbor"
    artifact.write_bytes(result.export())
    identity = result.expected_identity
    script = """
import json, pathlib, sys
from antecedent.transport.advanced import SampledRecoveryCandidate, SampledRecoveryIdentity
result=SampledRecoveryCandidate.load(pathlib.Path(sys.argv[1]).read_bytes(),
    expected=SampledRecoveryIdentity(sys.argv[2],sys.argv[3]))
print(json.dumps(result.inspect(),sort_keys=True))
"""
    replayed = subprocess.check_output(
        [
            sys.executable,
            "-c",
            script,
            str(artifact),
            identity.premises_digest,
            identity.data_digest,
        ],
        text=True,
    )
    assert json.loads(replayed) == evidence


def test_changed_seed_or_rows_cannot_replace_retained_artifact_identity():
    original = produce()
    sample = rows()
    sample[0] = (sample[0][0], 3, 0, 0)
    for changed in (produce(seed=8), produce(sample=sample)):
        with pytest.raises(CausalValueError, match="identity"):
            transport.SampledRecoveryCandidate.load(
                changed.export(), expected=original.expected_identity
            )


def test_query_and_fake_stage_do_not_gain_native_candidate_authority():
    changed = transport.ObservationRecoveryQuery(
        "clinic", "observed", query().partially_observed, fully_observed=["t"]
    )
    with pytest.raises(CausalValueError, match="query differs"):
        produce(declaration=changed)
    with pytest.raises(CausalTypeError, match="use sampled_observation_recovery"):
        transport.SampledRecoveryCandidate()


def test_reordered_query_roles_and_bits_preserve_original_canonical_authority():
    original = produce()
    swapped = [
        (row_id, ((r & 1) << 1) | ((r >> 1) & 1), ((p & 1) << 1) | ((p >> 1) & 1), full)
        for row_id, r, p, full in rows()
    ]
    declaration = query(partially=list(reversed(query().partially_observed)))
    reordered = produce(sample=swapped, declaration=declaration)
    assert reordered.expected_identity == original.expected_identity
    assert reordered.export() == original.export()
    assert reordered.inspect() == original.inspect()


def test_fake_stage_callback_cannot_issue_candidate_authority():
    calls = []
    fake = SimpleNamespace(outcome="recovered", sampled_candidate=lambda *args: calls.append(args))
    with pytest.raises(CausalTypeError, match="original native recovery stage"):
        transport.sampled_observation_recovery(
            stage=fake,
            query=query(),
            rows=rows(),
            snapshot="snap-1",
            replicates=500,
            seed=7,
        )
    assert calls == []


def test_oversized_row_sequence_refuses_before_materializing_patterns():
    class OversizedRows(Sequence):
        def __len__(self):
            return 100001

        def __getitem__(self, index):
            raise AssertionError("oversized rows must never be materialized")

    with pytest.raises(CausalValueError, match="bounds_exceeded") as caught:
        produce(sample=OversizedRows())
    assert caught.value.reason_code == "invalid_argument"


@pytest.mark.parametrize("change", ["duplicate_ids", "invalid_proxy", "oversized_id", "boolean_id"])
def test_malformed_rows_refuse_before_original_candidate_execution(change):
    sample = rows()
    row_id, responses, proxies, fully = sample[0]
    if change == "duplicate_ids":
        row_id = sample[1][0]
    elif change == "invalid_proxy":
        proxies |= 4
    elif change == "oversized_id":
        row_id = 2**64
    else:
        row_id = True
    sample[0] = (row_id, responses, proxies, fully)
    with pytest.raises((CausalTypeError, CausalValueError)):
        produce(sample=sample)


def test_changed_snapshot_and_oversized_seed_do_not_bypass_retained_input_contract():
    with pytest.raises(CausalValueError, match="snapshot"):
        produce(snapshot="changed-snapshot")
    with pytest.raises(CausalValueError, match="64-bit"):
        produce(seed=2**64)
    with pytest.raises(CausalTypeError):
        produce(seed=True)


@pytest.mark.parametrize("digest", ["", "0" * 32, "A" * 64, "g" * 64])
def test_malformed_retained_recovery_identity_is_typed(digest):
    with pytest.raises(CausalValueError, match="64 lowercase hex"):
        transport.SampledRecoveryIdentity(digest, "0" * 64)


def test_original_recovery_consumer_enforces_identity_bytes_and_work_budgets():
    result = produce()
    expected = result.expected_identity
    with pytest.raises(CausalValueError, match="identity"):
        transport.SampledRecoveryCandidate.load(
            result.export(),
            expected=transport.SampledRecoveryIdentity("0" * 64, "0" * 64),
        )
    with pytest.raises(CausalSerializationError):
        transport.SampledRecoveryCandidate.load(b"not-cbor", expected=expected)
    for limits in ({"max_rows": 1999}, {"max_replicates": 499}):
        with pytest.raises(CausalValueError, match="count") as caught:
            transport.SampledRecoveryCandidate.load(result.export(), expected=expected, **limits)
        assert caught.value.reason_code == "invalid_argument"
    with pytest.raises(CausalError, match="budget") as caught:
        transport.SampledRecoveryCandidate.load(result.export(), expected=expected, memory_bytes=1)
    assert caught.value.reason_code == "transport_budget_cancel"
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalError, match="budget") as caught:
        transport.SampledRecoveryCandidate.load(result.export(), expected=expected, cancel=token)
    assert caught.value.reason_code == "transport_budget_cancel"
    for limits in (
        {"max_rows": True},
        {"max_rows": 100001},
        {"max_replicates": 2001},
        {"memory_bytes": True},
        {"memory_bytes": 2**64},
    ):
        with pytest.raises(CausalValueError):
            transport.SampledRecoveryCandidate.load(result.export(), expected=expected, **limits)


def test_candidate_scalar_reads_reuse_immutable_original_report(monkeypatch):
    import antecedent.transport._sampled_candidate as candidate_module

    result = produce()
    artifact = result.export()
    expected = (
        result.effect,
        result.effect_standard_error,
        result.interval,
        result.level,
        result.expected_identity,
    )
    report = result.inspect()
    report["result"]["effect"] = 999.0
    report["result"]["interval"]["lower"] = 999.0
    with pytest.raises(TypeError, match="does not support item assignment"):
        result._body["result"]["effect"] = 999.0

    def no_reparse(*args, **kwargs):
        raise AssertionError("scalar reads must not reparse the full native report")

    monkeypatch.setattr(candidate_module.json, "loads", no_reparse)
    for _ in range(3):
        assert (
            result.effect,
            result.effect_standard_error,
            result.interval,
            result.level,
            result.expected_identity,
        ) == expected
        assert result.inspect()["result"]["effect"] == expected[0]
    assert result.export() == artifact
