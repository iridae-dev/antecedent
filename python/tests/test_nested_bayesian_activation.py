"""Original continuous posterior lifecycle; no repeated-sampling measurement."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.graph import Admg
from antecedent.transport._closed_pilots import _nested_posterior_candidate as binary_nested_markov
from antecedent.transport.advanced import (
    NestedMarkovPosteriorCandidate,
    NestedMarkovPrior,
)

_INTERNAL = hasattr(_native, "nested_markov_posterior_candidate")
_CANDIDATE = pytest.mark.skipif(
    not _INTERNAL, reason="requires calibration-internal acceptance wheel"
)


def graph():
    return Admg.from_edges(
        ["X1", "X2", "X3", "X4"], [("X1", "X2"), ("X2", "X3"), ("X3", "X4")], [("X2", "X4")]
    )


def produce(prior=None):
    return binary_nested_markov(
        graph=graph(), regimes=[{"counts": [1] * 16}], prior=prior, seed=817
    )


def beta_quantile(p, shape):
    """Independent integer-Beta CDF via binomial polynomial, bisection inversion."""
    n = 2 * shape - 1
    lo, hi = 0.0, 1.0
    for _ in range(64):
        x = (lo + hi) / 2
        cdf = sum(math.comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(shape, n + 1))
        if cdf < p:
            lo = x
        else:
            hi = x
    return (lo + hi) / 2


def test_nested_bayesian_named_prior_is_immutable_and_has_domain_errors():
    prior = NestedMarkovPrior(q40=(4, 1), q41=(4, 1))
    assert prior.q40 == (4.0, 1.0)
    assert prior.coordinates == (
        "a",
        "c0",
        "c1",
        "q20",
        "q21",
        "q40",
        "q41",
        "g00",
        "g01",
        "g10",
        "g11",
    )
    with pytest.raises(FrozenInstanceError):
        prior.a = (2.0, 2.0)
    for pair in ((False, 1), (1 + 0j, 1), "bad", (1,)):
        with pytest.raises(CausalTypeError) as caught:
            NestedMarkovPrior(g11=pair)
        assert caught.value.reason_code == "invalid_argument"
    for pair in ((0.5, 1), (float("nan"), 1), (10**400, 1), (1_000_001, 1)):
        with pytest.raises(CausalValueError) as caught:
            NestedMarkovPrior(q20=pair)
        assert caught.value.reason_code == "invalid_argument"


@pytest.mark.skipif(_INTERNAL, reason="normal released wheel refusal boundary")
def test_nested_bayesian_default_producer_and_consumer_remain_frozen():
    with pytest.raises(CausalUnsupportedError) as caught:
        produce()
    assert caught.value.reason_code == "cell_not_licensed"
    with pytest.raises(CausalUnsupportedError) as caught:
        NestedMarkovPosteriorCandidate.load(b"unlicensed", expected_identity="absent")
    assert caught.value.reason_code == "cell_not_licensed"


@_CANDIDATE
def test_nested_bayesian_public_original_posterior_matches_conjugate_reference_and_fresh_consumer(
    tmp_path,
):
    result = produce()
    assert isinstance(result, NestedMarkovPosteriorCandidate)
    with pytest.raises(CausalTypeError):
        NestedMarkovPosteriorCandidate()
    with pytest.raises(CausalTypeError):
        replace(result, values=(0.0, 0.0, 0.0))
    with pytest.raises(CausalTypeError):
        NestedMarkovPosteriorCandidate._from_native(
            type("Fake", (), {"payload": result._native.payload})()
        )
    for array in (result.samples, result.covariance):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    assert result.calibration == "unmeasured"
    assert result.inference == "posterior_candidate_withheld_calibration_unmeasured"
    assert result.identification == "nonparametrically_identified"
    assert result.coordinate_order[:11] == result.prior.coordinates
    assert result.samples.shape == (4, 4096, 14)
    assert result.covariance.shape == (14, 14)
    assert not result.samples.flags.writeable and not result.covariance.flags.writeable
    for k, shape in enumerate((9, 5, 5)):
        assert result.parameter_mean[k] == pytest.approx(0.5, abs=0.012)
        assert result.covariance[k, k] == pytest.approx(1 / (4 * (2 * shape + 1)), abs=0.002)
        expected = (beta_quantile(0.025, shape), beta_quantile(0.975, shape))
        assert result.credible_intervals[k] == pytest.approx(expected, abs=0.014)
    np.testing.assert_allclose(
        result.samples[:, :, 13],
        result.samples[:, :, 12] - result.samples[:, :, 11],
        atol=1e-15,
        rtol=0,
    )
    flattened = result.samples.reshape(-1, 14)
    np.testing.assert_allclose(np.cov(flattened.T), result.covariance, atol=1e-14, rtol=0)
    assert all(
        max(d.rank_rhat, d.folded_rhat) <= 1.01 and min(d.bulk_ess, d.tail_ess) >= 400
        for d in result.diagnostics
    )
    payload = result.to_dict()
    assert payload["options"] == {
        "chains": 4,
        "warmup": 2048,
        "draws": 4096,
        "max_proposals": 5_000_000,
        "seed": 817,
        "credible_mass": 0.95,
    }
    artifact = tmp_path / "continuous-verma.cbor"
    artifact.write_bytes(result.export())
    code = """
import json,sys
from pathlib import Path
from antecedent.transport.advanced import NestedMarkovPosteriorCandidate
r=NestedMarkovPosteriorCandidate.load(Path(sys.argv[1]).read_bytes(),expected_identity=sys.argv[2])
print(json.dumps(r.to_dict()))
"""
    fresh = subprocess.run(
        [sys.executable, "-c", code, str(artifact), result.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(fresh.stdout) == payload


@_CANDIDATE
def test_nested_bayesian_prior_changes_actual_posterior_and_bound_identity():
    baseline = produce()
    changed = produce(NestedMarkovPrior(q40=(4, 1), q41=(4, 1)))
    assert changed.prior.q40 == (4, 1)
    assert baseline.values[0] - changed.values[0] > 0.04
    assert baseline.identity != changed.identity
    with pytest.raises(CausalValueError) as caught:
        NestedMarkovPosteriorCandidate.load(changed.export(), expected_identity=baseline.identity)
    assert caught.value.reason_code == "invalid_argument"
    assert "bayesian_identity_mismatch" in str(caught.value)
    fractional = [1.0] * 16
    fractional[0] = 1.5
    with pytest.raises(CausalValueError) as caught:
        binary_nested_markov(graph=graph(), regimes=[{"counts": fractional}])
    assert caught.value.reason_code == "invalid_argument"
    assert "integer multinomial" in str(caught.value)


def test_nested_pilot_declines_oversized_sequences_before_materializing():
    reads = []

    class Oversized(Sequence):
        def __len__(self):
            return 1_000_000

        def __getitem__(self, index):
            reads.append(index)
            raise IndexError(index)

    for regimes in (
        Oversized(),
        [{"counts": Oversized()}],
        [{"counts": [1] * 16, "levels": Oversized()}],
    ):
        with pytest.raises(CausalUnsupportedError) as caught:
            binary_nested_markov(graph=graph(), regimes=regimes)
        assert caught.value.reason_code == "route_not_supported"
        assert "outside_binary_pilot" in str(caught.value)
    assert reads == []
