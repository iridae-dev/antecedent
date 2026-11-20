"""Time-ordered path transport from Python: exact laws, sampled counts and a continuous chain.

The finite fixture is the mediation model with a covariate shift in ``Z`` (``Z`` at
time 0, ``X`` at 1, ``M`` at 2, ``Y`` at 3)::

    Z = Uz            X = Z xor Ux       M = (X + Z + Um) mod 3
    Y = [M + X*Z >= 2] xor Uy            Uz ~ Bern(1/4 source, 3/4 target)
    Ux ~ Bern(1/4)    Um ~ (1/4, 1/4, 1/2)    Uy ~ Bern(1/8)

With the ``X -> Y`` edge active the active world is ``Y_{x=1, M_{x=0}}`` and the
baseline world ``Y_{x=0, M_{x=0}}``. Both truths are enumerated here from the
target's own structural equations with exact rationals: 1/2 and 23/64, so the
contrast is the natural direct effect 9/64. The library only sees the source joint
law of ``(Z, X, M, Y)`` and the target marginal of ``Z``. The sampled fixture is the
same laws as integer counts out of 4096 (every probability is a multiple of 1/512),
so the plug-in contrast is exact.

The continuous chain is the Rust route's fixture: 200 randomized source units with two
sequential mediators whose linear mechanisms put a treatment-to-first-mediator effect
of 2 and a total of 7 on the outcome.
"""

from __future__ import annotations

import itertools
from fractions import Fraction as F

import pytest
from antecedent import Admg, Execution
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced, temporal_paths

from _refusal import assert_registered_refusal

FOREIGN = "0" * 64
TOLERANCE = 1e-12
NAMES = ("Z", "X", "M", "Y")
LEVELS = {"Z": (0.0, 1.0), "X": (0.0, 1.0), "M": (0.0, 1.0, 2.0), "Y": (0.0, 1.0)}
HALF, QUARTER, EIGHTH = F(1, 2), F(1, 4), F(1, 8)
UX = ((0, 1 - QUARTER), (1, QUARTER))
UM = ((0, QUARTER), (1, QUARTER), (2, HALF))
UY = ((0, 1 - EIGHTH), (1, EIGHTH))


def states(uz: F):
    """Every exogenous state ``(z, ux, um, uy)`` of a population and its probability."""
    for (z, pz), (ux, pux), (um, pum), (uy, puy) in itertools.product(
        ((0, 1 - uz), (1, uz)), UX, UM, UY
    ):
        yield z, ux, um, uy, pz * pux * pum * puy


def source_joint() -> list[F]:
    """``P(Z, X, M, Y)`` of the source, row-major with ``Y`` fastest."""
    table = [F(0)] * 24
    for z, ux, um, uy, p in states(QUARTER):
        x = z ^ ux
        m = (x + z + um) % 3
        y = int(m + x * z >= 2) ^ uy
        table[((z * 2 + x) * 3 + m) * 2 + y] += p
    return table


def nested_truth(x: int, x_mediator: int) -> F:
    """``P*(Y_{x, M_{x'}} = 1)`` enumerated from the target's structural equations."""
    total = F(0)
    for z, _, um, uy, p in states(1 - QUARTER):
        m = (x_mediator + z + um) % 3
        total += p * (int(m + x * z >= 2) ^ uy)
    return total


def finite_problem() -> dict:
    return {
        "graph": Admg.from_edges(
            list(NAMES),
            [("Z", "X"), ("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")],
        ),
        "levels": LEVELS,
        "target": "target",
        "sources": [advanced.CounterfactualTransportSource("source", selections=["Z"])],
        "catalog": advanced.EvidenceCatalog(
            regimes=(
                advanced.EvidenceRegime("source|obs", "source", measured=list(NAMES)),
                advanced.EvidenceRegime("target|obs", "target", measured=["Z"]),
            )
        ),
    }


def exact_laws() -> list[advanced.ExactDiscreteLaw]:
    return [
        advanced.ExactDiscreteLaw(
            "source",
            "source|obs",
            tuple((name, LEVELS[name]) for name in NAMES),
            tuple(float(p) for p in source_joint()),
            "snapshot|source",
        ),
        advanced.ExactDiscreteLaw(
            "target", "target|obs", (("Z", LEVELS["Z"]),), (0.25, 0.75), "snapshot|target"
        ),
    ]


def count_tables() -> list[advanced.SampledCountTable]:
    tables = []
    for law in exact_laws():
        counts = [int(p * 4096) for p in law.probabilities]
        assert sum(counts) == 4096
        tables.append(
            advanced.SampledCountTable(
                law.population, law.regime, law.axes, counts, law.snapshot_identity
            )
        )
    return tables


def direct_path(**changes) -> temporal_paths.TemporalPath:
    declaration = {
        "treatment": "X",
        "outcome": "Y",
        "times": {"Z": 0, "X": 1, "M": 2, "Y": 3},
        "active_edges": [("X", "Y")],
    }
    return temporal_paths.TemporalPath(**(declaration | changes))


def refused_foreign(consume) -> None:
    with pytest.raises(CausalUnsupportedError) as refused:
        consume(FOREIGN)
    assert refused.value.reason_code == "external_binding_mismatch"
    assert_registered_refusal(refused.value)


def test_the_enumerated_truths_are_the_hand_derived_values():
    assert sum(source_joint()) == 1
    assert nested_truth(1, 0) == F(1, 2)
    assert nested_truth(0, 0) == F(23, 64)


def test_exact_path_contrast_matches_the_enumerated_direct_effect_and_replays():
    compiled = temporal_paths.compile_temporal_path_transport(
        direct_path(), exact_laws(), **finite_problem()
    )
    assert compiled.kind == "exact"
    assert len(compiled.identity) == 64
    assert compiled.plan
    result = compiled.run()
    assert result.program_identity == compiled.identity
    assert result.active == pytest.approx(float(nested_truth(1, 0)), abs=TOLERANCE)
    assert result.baseline == pytest.approx(float(nested_truth(0, 0)), abs=TOLERANCE)
    assert result.contrast == pytest.approx(9 / 64, abs=TOLERANCE)
    assert result.interval is None and result.standard_error is None
    assert result.worlds["active"]["probability"] == result.active
    data = result.export()
    replay = temporal_paths.consume_temporal_path_transport(
        data, kind=compiled.kind, expected_identity=compiled.identity
    )
    assert replay.independently_verified
    assert replay.acceptance.replayed
    assert replay.contrast == result.contrast
    refused_foreign(
        lambda identity: temporal_paths.consume_temporal_path_transport(
            data, kind=compiled.kind, expected_identity=identity
        )
    )


def test_an_arrow_that_does_not_advance_time_is_refused():
    with pytest.raises(CausalValueError) as refused:
        temporal_paths.compile_temporal_path_transport(
            direct_path(times={"Z": 0, "X": 1, "M": 1, "Y": 3}), exact_laws(), **finite_problem()
        )
    assert refused.value.reason_code == "invalid_argument"


def test_sampled_path_contrast_is_the_plug_in_with_an_uncalibrated_error_and_replays():
    compiled = temporal_paths.compile_temporal_path_transport(
        direct_path(),
        count_tables(),
        overlap=advanced.SamplingOverlap.all_independent(),
        limits=advanced.SampledTransportLimits(max_operations=40_000_000),
        **finite_problem(),
    )
    assert compiled.kind == "sampled"
    result = compiled.run()
    assert result.contrast == pytest.approx(9 / 64, abs=1e-9)
    assert result.active - result.baseline == pytest.approx(result.contrast, abs=TOLERANCE)
    assert result.standard_error is not None and result.standard_error > 0
    assert result.calibration == "not_measured"
    assert result.sampled_units == 2 * 4096
    data = result.export()
    replay = temporal_paths.consume_temporal_path_transport(
        data, kind=compiled.kind, expected_identity=compiled.identity
    )
    assert replay.contrast == result.contrast
    assert replay.standard_error == result.standard_error
    assert replay.evidence_identity == result.evidence_identity
    refused_foreign(
        lambda identity: temporal_paths.consume_temporal_path_transport(
            data, kind=compiled.kind, expected_identity=identity
        )
    )


def chain_sample() -> temporal_paths.ContinuousChainSample:
    baseline, treated, first, second, outcome = [], [], [], [], []
    for i in range(200):
        x = (i % 25) / 25
        a = 1.0 if i >= 100 else 0.0
        m1 = x + a + (-1.0 if (i // 25) % 2 == 0 else 1.0)
        m2 = 2.0 * m1 + 3.0 * a + (-1.0 if (i // 50) % 2 == 0 else 1.0)
        baseline.append(x)
        treated.append(i >= 100)
        first.append(m1)
        second.append(m2)
        outcome.append(3.0 + x + m2 + 4.0 * a)
    return temporal_paths.ContinuousChainSample(
        baseline=baseline,
        treated=treated,
        mediators=[first, second],
        outcome=outcome,
        target_baseline=[i / 25 for i in range(25)],
    )


def chain_path(active_edges) -> temporal_paths.SequentialEdgePath:
    return temporal_paths.SequentialEdgePath(
        mediator_parents=[[], [0], [0, 1]],
        treatment_edges=[True, True, True],
        active_edges=active_edges,
        bootstrap_replicates=100,
    )


def test_continuous_chain_contrast_and_bootstrap_schedule_replay_under_their_seed():
    execution = Execution(seed=991)
    compiled = temporal_paths.compile_temporal_path_transport(
        chain_path([True, False, False]), chain_sample(), execution=execution
    )
    assert compiled.kind == "continuous_chain"
    result = compiled.run()
    assert result.contrast == pytest.approx(2.0, abs=1e-10)
    assert result.interval is not None and result.interval[0] <= result.interval[1]
    assert result.calibration == "not_measured"
    data = result.export()
    replay = temporal_paths.consume_temporal_path_transport(
        data, kind=compiled.kind, expected_identity=compiled.identity, execution=execution
    )
    assert replay.contrast == result.contrast
    assert replay.interval == result.interval
    alternate = temporal_paths.compile_temporal_path_transport(
        chain_path([False, True, True]), chain_sample(), execution=execution
    )
    assert alternate.identity != compiled.identity
    assert alternate.run().contrast == pytest.approx(7.0, abs=1e-10)
    refused_foreign(
        lambda identity: temporal_paths.consume_temporal_path_transport(
            data, kind=compiled.kind, expected_identity=identity, execution=execution
        )
    )
    with pytest.raises(CausalUnsupportedError) as reseeded:
        temporal_paths.consume_temporal_path_transport(
            data,
            kind=compiled.kind,
            expected_identity=compiled.identity,
            execution=Execution(seed=992),
        )
    assert reseeded.value.reason_code == "external_binding_mismatch"


def test_a_continuous_chain_parent_must_precede_its_node():
    path = temporal_paths.SequentialEdgePath(
        mediator_parents=[[], [1], [0, 1]],
        treatment_edges=[True, True, True],
        active_edges=[True, False, False],
        bootstrap_replicates=100,
    )
    with pytest.raises(CausalValueError) as refused:
        temporal_paths.compile_temporal_path_transport(path, chain_sample())
    assert refused.value.reason_code == "invalid_argument"
