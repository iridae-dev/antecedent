"""The whole-chain bootstrap union of an assumption envelope, under every resampling scheme.

The chain adds a sampling estimate and an interval-valued bias, ``answer = estimate + bias`` with
``bias`` in ``[0, 1]``, so its enclosure is ``[estimate, estimate + 1]`` row by row. Each scheme
compiles one program, reruns the whole chain on 100 resamples, and its sealed artifact is
consumed independently under the retained program identity and the producing seed.
"""

from __future__ import annotations

import pytest
from antecedent import CancellationToken, Execution
from antecedent import propagation_chain as pc
from antecedent import sampling_assumption_union as sau
from antecedent.errors import CausalCancelledError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

QUANTITY = ScientificQuantity(
    variable_id="study.estimate_plus_bias",
    variable_name="Estimate plus bias",
    role="outcome",
    units="points",
    population_id="target",
    regime_id="observed",
    horizon=1,
    functional_id="mean",
)
STAGES = [pc.binary("answer", "derived_quantity", ["estimate", "bias"], "sum")]
WRONG_IDENTITY = "0" * 64


def _sources(values: list[float]) -> list[pc.ChainSource]:
    return [
        pc.draws("estimate", values, meaning="sampling", alignment="joint-sampling-table"),
        pc.interval("bias", 0.0, 1.0),
    ]


def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}-{i}" for i in range(n)]


# seed, sampling column, scheme keywords, compiled route
SCHEMES: dict[str, tuple[int, list[float], dict[str, object], str]] = {
    "declared": (
        11,
        [1.0, 2.0, 3.0],
        {
            "bootstrap": [
                _sources([1.0 + shift, 2.0 + shift, 3.0 + shift])
                for shift in ((i - 49.5) * 0.002 for i in range(100))
            ],
            "replicate_ids": [f"resample-{i:03}" for i in range(100)],
            "resampling_design": "fixed-row-bootstrap-seed-11",
        },
        "SamplingAssumptionUnion",
    ),
    "iid": (
        31,
        [1.0 + i / 40 for i in range(40)],
        {"unit_ids": _ids("unit", 40), "replicates": 100},
        "IidSamplingAssumptionUnion",
    ),
    "cluster": (
        37,
        [1.0 + (i // 2) / 20 for i in range(40)],
        {
            "unit_ids": _ids("unit", 40),
            "cluster_ids": [f"cluster-{i // 2}" for i in range(40)],
            "replicates": 100,
        },
        "ClusterSamplingAssumptionUnion",
    ),
    "stratified_cluster": (
        43,
        [i / 40 for i in range(40)],
        {
            "unit_ids": _ids("unit", 40),
            "cluster_ids": [f"cluster-{i // 2}" for i in range(40)],
            "stratum_ids": [f"stratum-{i // 20}" for i in range(40)],
            "replicates": 100,
        },
        "ClusterSamplingAssumptionUnion",
    ),
    "moving_block": (
        41,
        [1.0 + i / 40 for i in range(40)],
        {"unit_ids": _ids("time", 40), "block_length": 5, "replicates": 100},
        "MovingBlockSamplingAssumptionUnion",
    ),
    "stratified_moving_block": (
        47,
        [i / 40 for i in range(40)],
        {
            "unit_ids": _ids("row", 40),
            "stratum_ids": [f"series-{i % 2}" for i in range(40)],
            "block_length": 4,
            "replicates": 100,
        },
        "MovingBlockSamplingAssumptionUnion",
    ),
    "panel_block": (
        53,
        [i / 200 for i in range(200)],
        {
            "unit_ids": _ids("row", 200),
            "panel_ids": [f"panel-{i % 10:02}" for i in range(200)],
            "block_length": 4,
            "replicates": 100,
        },
        "MovingBlockSamplingAssumptionUnion",
    ),
}


def _union(scheme: str, **changes: object) -> sau.SamplingAssumptionUnion:
    seed, values, design, _ = SCHEMES[scheme]
    keywords = {**design, **changes}
    return require_success(
        sau.sampling_assumption_union(
            _sources(values),
            STAGES,
            output="answer",
            output_quantity=QUANTITY,
            scheme=scheme,  # type: ignore[arg-type]
            execution=Execution(seed=seed),
            **keywords,  # type: ignore[arg-type]
        )
    )


@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_every_scheme_compiles_runs_and_replays_at_the_final_coordinate(scheme: str) -> None:
    seed, _, _, route = SCHEMES[scheme]
    union = _union(scheme)
    assert union.scheme == scheme
    assert union.route == route
    assert union.calibration == "not_measured"
    assert union.level == 0.95
    assert len(union.chain_identities) == 101
    assert union.original_lower <= union.original_upper
    assert union.union_lower <= union.union_upper

    artifact = union.export()
    assert artifact.kind.endswith("sampling_assumption_union_v1")
    assert artifact.expectation is not None
    assert artifact.expectation.identity == union.program_identity
    replayed = require_success(
        sau.consume_sampling_assumption_union(
            artifact, expected_identity=union.program_identity, execution=Execution(seed=seed)
        )
    )
    assert replayed == union
    assert replayed.route is None
    assert replayed.acceptance is not None
    assert replayed.acceptance.replayed and replayed.acceptance.identity_bound
    assert bytes(replayed.export()) == bytes(artifact)


@pytest.mark.parametrize("scheme", ["declared", "iid"])
def test_the_union_widens_the_plug_in_endpoints(scheme: str) -> None:
    union = _union(scheme)
    assert union.union_lower <= union.original_lower
    assert union.union_upper >= union.original_upper


def test_another_identity_or_seed_is_refused_on_replay() -> None:
    seed = SCHEMES["moving_block"][0]
    union = _union("moving_block")
    other = _union("moving_block", block_length=6)
    assert other.program_identity != union.program_identity
    artifact = union.export()
    for identity, replay_seed in (
        (other.program_identity, seed),
        (WRONG_IDENTITY, seed),
        (union.program_identity, seed + 1),
    ):
        with pytest.raises(sau.SamplingAssumptionUnionRefusal) as refused:
            require_success(
                sau.consume_sampling_assumption_union(
                    artifact, expected_identity=identity, execution=Execution(seed=replay_seed)
                )
            )
        assert refused.value.reason_code == "external_binding_mismatch"


def test_a_schedule_the_compiler_cannot_resample_is_refused() -> None:
    with pytest.raises(sau.SamplingAssumptionUnionRefusal) as too_few:
        _union("iid", replicates=99)
    assert too_few.value.reason_code == "invalid_argument"
    with pytest.raises(sau.SamplingAssumptionUnionRefusal) as short_block:
        _union("moving_block", block_length=1)
    assert short_block.value.reason_code == "invalid_argument"


def test_a_scheme_takes_exactly_its_own_keywords() -> None:
    with pytest.raises(CausalValueError, match="does not take block_length"):
        _union("iid", block_length=4)
    with pytest.raises(CausalValueError, match="needs cluster_ids"):
        _union("cluster", cluster_ids=None)
    with pytest.raises(CausalValueError, match="scheme must be one of"):
        require_success(
            sau.sampling_assumption_union(
                _sources([1.0 + i / 40 for i in range(40)]),
                STAGES,
                output="answer",
                output_quantity=QUANTITY,
                scheme="bootstrap",  # type: ignore[arg-type]
                unit_ids=_ids("unit", 40),
                replicates=100,
            )
        )


def _cancelled(seed: int) -> Execution:
    token = CancellationToken()
    token.cancel()
    return Execution(seed=seed, cancel=token)


@pytest.mark.parametrize("scheme", ["declared", "iid", "cluster"])
def test_a_cancelled_union_raises_the_cancellation_not_a_refusal(scheme: str) -> None:
    # A cancelled run is an operational stop, never an invalid declaration: it raises the
    # cancellation, on compile, on export and on replay, and no refusal outcome is returned.
    seed, values, design, _ = SCHEMES[scheme]
    with pytest.raises(CausalCancelledError):
        sau.sampling_assumption_union(
            _sources(values),
            STAGES,
            output="answer",
            output_quantity=QUANTITY,
            scheme=scheme,  # type: ignore[arg-type]
            execution=_cancelled(seed),
            **design,  # type: ignore[arg-type]
        )
    union = _union(scheme)
    with pytest.raises(CausalCancelledError):
        union.export(execution=_cancelled(seed))
    artifact = union.export()
    with pytest.raises(CausalCancelledError):
        sau.consume_sampling_assumption_union(
            artifact, expected_identity=union.program_identity, execution=_cancelled(seed)
        )
