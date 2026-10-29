"""Transport across a discovered class refined by a randomized binary source trial.

Oracles (by hand): the source is a completely randomized trial of 200 rows, the first 100 control
and the last 100 treated. The outcome is 1 for rows 0-19 and 100-179, so the control mean is
20/100 = 0.2 and the treated mean 80/100 = 0.8: a strong dependence between assignment and
outcome (the exact no-effect test rejects at any usual level). FCI over two dependent binary
variables leaves the edge unoriented, a class of several members; the trial retains only the
members that permit a directed effect of the treatment on the outcome, and every retained member
standardizes to the same two means. Discovering instead on independent columns gives a class
with no edge, whose only member permits no directed effect: the trial rejects it, so the
whole-procedure set is the declared logical support ``[a - b, b - a]`` and never a point.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.transport import discovered
from antecedent.transport import trial_compatible as tc

ROWS = 200
TREATMENT = np.array([float(i >= 100) for i in range(ROWS)])
OUTCOME = np.array([float(i < 20 or 100 <= i < 180) for i in range(ROWS)])
SOURCE = {"x": TREATMENT, "y": OUTCOME}
INDEPENDENT = {"x": TREATMENT, "y": np.array([float(i % 2 == 0) for i in range(ROWS)])}
CONTRACT = sd.StaticContract(
    variables=("x", "y"), algorithm="fci", ci_test="gsquared", fdr=None, seed=71
)
COMMON = {
    "outcome": "y",
    "treatment": "x",
    "selection_targets": [],
    "source_population": "trial",
    "target_population": "target",
}


def _estimate(**overrides):
    arguments = {**COMMON, "grid": [0.0, 1.0], "alpha": 0.01, **overrides}
    return tc.estimate_trial_compatible_discovered_transport(CONTRACT, SOURCE, SOURCE, **arguments)


def test_a_real_trial_refines_an_ambiguous_class_and_replays_every_decision():
    refined = _estimate()
    proof = refined.compatibility
    assert len(proof.members) > 1
    assert any(not member["retained"] for member in proof.members)
    assert proof.p_value < 0.01
    assert proof.alpha == 0.01
    assert proof.retained_members == tuple(
        index for index, member in enumerate(proof.members) if member["retained"]
    )
    assert refined.answer.kind == "point"
    assert refined.answer.levels[0][1] == pytest.approx(0.2, abs=1e-12)
    assert refined.answer.levels[1][1] == pytest.approx(0.8, abs=1e-12)
    assert len(refined.members) == len(proof.retained_members)
    assert len(refined.report_identity) == 64

    verified = tc.verify_trial_compatible_transport_estimate(bytes(refined.export()))
    assert verified.members == proof.retained_members
    assert verified.compatibility == proof
    assert verified.answer == refined.answer
    assert verified.report_identity == refined.report_identity


def test_a_forged_or_truncated_artifact_does_not_replay():
    artifact = bytes(_estimate().export())
    with pytest.raises(CausalValueError, match="UnexpectedEof"):
        tc.verify_trial_compatible_transport_estimate(artifact[:-1])
    with pytest.raises(CausalValueError, match="expected map"):
        tc.verify_trial_compatible_transport_estimate(b"not an artifact")
    flipped = bytearray(artifact)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises(CausalValueError, match="checksum mismatch"):
        tc.verify_trial_compatible_transport_estimate(bytes(flipped))
    with pytest.raises(CausalTypeError, match="bytes or ExportedArtifact"):
        tc.verify_trial_compatible_transport_estimate("artifact")  # type: ignore[arg-type]


def test_a_trial_that_rejects_every_member_of_the_class_is_refused_as_an_estimate():
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        tc.estimate_trial_compatible_discovered_transport(
            CONTRACT,
            SOURCE,
            SOURCE,
            grid=[0.0, 1.0],
            alpha=0.01,
            discovery=INDEPENDENT,
            **COMMON,
        )
    assert info.value.detail == "discovered_transport.trial_rejects_every_member"


def test_the_test_level_is_declared_in_the_open_interval():
    for alpha in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(CausalValueError, match="strictly between 0 and 1"):
            _estimate(alpha=alpha)
    with pytest.raises(CausalTypeError, match="alpha must be a real number"):
        _estimate(alpha="0.01")


def test_the_identified_branch_is_the_sampling_interval_of_the_refined_estimate_and_replays():
    interval = _estimate(bootstrap_replicates=64).answer.interval
    assert interval is not None
    arguments = {
        **COMMON,
        "alpha": 0.01,
        "support_lower": 0.0,
        "support_upper": 1.0,
        "bootstrap_replicates": 64,
    }
    result = tc.estimate_trial_compatible_transport_confidence_set(
        CONTRACT, SOURCE, SOURCE, **arguments
    )
    assert result.kind == "identified_sampling_interval"
    assert result.logical_reason is None
    assert (result.lower, result.upper) == (interval["low"], interval["high"])
    assert result.calibration == "not_bound_to_this_execution"
    assert result.outcome_support == (0.0, 1.0)
    assert result.estimate is not None
    replay = tc.verify_trial_compatible_transport_confidence_set_against(
        result.export(), CONTRACT, SOURCE, SOURCE, **arguments
    )
    assert replay.identity == result.identity
    assert (replay.lower, replay.upper) == (result.lower, result.upper)


def test_a_rejected_class_returns_the_declared_support_not_a_point_and_replays():
    arguments = {
        **COMMON,
        "alpha": 0.01,
        "support_lower": -2.0,
        "support_upper": 3.0,
        "bootstrap_replicates": 64,
        "discovery": INDEPENDENT,
    }
    result = tc.estimate_trial_compatible_transport_confidence_set(
        CONTRACT, SOURCE, SOURCE, **arguments
    )
    assert result.kind == "logical_support_bound"
    assert result.logical_reason == "trial_rejects_every_member"
    assert (result.lower, result.upper) == (-5.0, 5.0)
    assert result.estimate is None
    assert result.compatibility.retained_members == ()
    replay = tc.verify_trial_compatible_transport_confidence_set_against(
        result.export(), CONTRACT, SOURCE, SOURCE, **arguments
    )
    assert replay.identity == result.identity


def test_a_receipt_does_not_replay_against_other_inputs():
    arguments = {
        **COMMON,
        "alpha": 0.01,
        "support_lower": -2.0,
        "support_upper": 3.0,
        "bootstrap_replicates": 64,
        "discovery": INDEPENDENT,
    }
    receipt = tc.estimate_trial_compatible_transport_confidence_set(
        CONTRACT, SOURCE, SOURCE, **arguments
    ).export()
    with pytest.raises(CausalValueError, match="does not replay against supplied inputs"):
        tc.verify_trial_compatible_transport_confidence_set_against(
            receipt, CONTRACT, SOURCE, SOURCE, **{**arguments, "support_upper": 4.0}
        )
    with pytest.raises(CausalValueError, match="does not replay against supplied inputs"):
        tc.verify_trial_compatible_transport_confidence_set_against(
            receipt, CONTRACT, SOURCE, SOURCE, **{**arguments, "alpha": 0.05}
        )
    with pytest.raises(CausalValueError, match="UnexpectedEof"):
        tc.verify_trial_compatible_transport_confidence_set_against(
            bytes(receipt)[:-1], CONTRACT, SOURCE, SOURCE, **arguments
        )


def test_an_empty_or_inverted_support_is_refused_before_any_work():
    arguments = {**COMMON, "alpha": 0.01}
    with pytest.raises(CausalValueError, match="support_lower must be below support_upper"):
        tc.estimate_trial_compatible_transport_confidence_set(
            CONTRACT, SOURCE, SOURCE, support_lower=1.0, support_upper=1.0, **arguments
        )
    with pytest.raises(CausalTypeError, match="support_lower must be a real number"):
        tc.estimate_trial_compatible_transport_confidence_set(
            CONTRACT,
            SOURCE,
            SOURCE,
            support_lower="0",
            support_upper=1.0,
            **arguments,  # type: ignore[arg-type]
        )
