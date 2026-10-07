"""Shared-data covariance of scenario estimates over one row sample (2.3A X2).

The units are six complete rows ``(z, x, y)`` of a tiny discrete SCM sample. Every
scenario estimator is a plug-in functional of the resampled rows' cell proportions.
The oracles never use the library's enumerator: the linear pair has the closed-form
covariance of the empirical proportions under multinomial resampling,
``Cov(a, b) = (E[ab] - E[a]E[b]) / n``, and the adjustment pair is enumerated by
walking all ``n^n`` equally likely row-index tuples.
"""

import itertools

import antecedent
import pytest
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

ROWS = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0)]
N = len(ROWS)
PREFIX = b"ANTECEDENT-SCENARIO-COVARIANCE\x01"


def table(rows=ROWS, units=None):
    return transport.RowTable(["z", "x", "y"], rows, units or [f"u{i}" for i in range(len(rows))])


def estimator(name, **overrides):
    scores = {
        "a": [(1.0, {"x": 1, "y": 1}), (-1.0, {"x": 0, "y": 1})],
        "b": [(1.0, {"x": 1, "y": 1}), (-1.0, {"x": 1, "y": 0})],
        "c": [(1.0, {"z": 1, "y": 1})],
    }
    if name in scores:
        functional = transport.LinearScore([transport.ScoreTerm(c, w) for c, w in scores[name]])
    elif name == "adjusted":
        functional = transport.AdjustedContrast("x", "y", adjustment=["z"])
    else:
        functional = transport.AdjustedContrast("x", "y")
    return transport.ScenarioEstimator(name, functional, **overrides)


def family(*names, **overrides):
    return [estimator(n, **overrides) for n in names]


def exact(scenarios, **options):
    options = {"method": "exact_enumeration", "max_compositions": 10_000, **options}
    return transport.scenario_shared_covariance(table(), scenarios, **options)


def bootstrap(scenarios, seed=20_260_101, **options):
    options = {"replicates": 2000, **options}
    return transport.scenario_shared_covariance(table(), scenarios, seed=seed, **options)


def score(row, name):
    z, x, y = row
    if name == "a":
        return {(1, 1): 1.0, (0, 1): -1.0}.get((x, y), 0.0)
    if name == "b":
        return {(1, 1): 1.0, (1, 0): -1.0}.get((x, y), 0.0)
    return 1.0 if (z, y) == (1, 1) else 0.0


def closed_form(first, second):
    a = [score(r, first) for r in ROWS]
    b = [score(r, second) for r in ROWS]
    mean = lambda v: sum(v) / N  # noqa: E731
    return (mean([p * q for p, q in zip(a, b, strict=True)]) - mean(a) * mean(b)) / N


def cells(counts):
    out = {}
    for row, k in zip(ROWS, counts, strict=True):
        out[row] = out.get(row, 0) + k
    return out


def oracle_adjusted(counts):
    c = cells(counts)
    effect = 0.0
    for z in (0, 1):
        n1 = c.get((z, 1, 0), 0) + c.get((z, 1, 1), 0)
        n0 = c.get((z, 0, 0), 0) + c.get((z, 0, 1), 0)
        if n1 == 0 or n0 == 0:
            return None
        effect += (n1 + n0) / N * (c.get((z, 1, 1), 0) / n1 - c.get((z, 0, 1), 0) / n0)
    return effect


def oracle_crude(counts):
    c = cells(counts)
    n1 = sum(c.get((z, 1, y), 0) for z in (0, 1) for y in (0, 1))
    n0 = sum(c.get((z, 0, y), 0) for z in (0, 1) for y in (0, 1))
    if n1 == 0 or n0 == 0:
        return None
    y1 = sum(c.get((z, 1, 1), 0) for z in (0, 1))
    y0 = sum(c.get((z, 0, 1), 0) for z in (0, 1))
    return y1 / n1 - y0 / n0


def enumerate_tuples():
    kept = []
    total = N**N
    for tup in itertools.product(range(N), repeat=N):
        counts = [0] * N
        for i in tup:
            counts[i] += 1
        a, b = oracle_adjusted(counts), oracle_crude(counts)
        if a is not None and b is not None:
            kept.append((a, b))
    m = len(kept)
    ma = sum(p[0] for p in kept) / m
    mb = sum(p[1] for p in kept) / m
    caa = sum((p[0] - ma) ** 2 for p in kept) / m
    cbb = sum((p[1] - mb) ** 2 for p in kept) / m
    cab = sum((p[0] - ma) * (p[1] - mb) for p in kept) / m
    return [[caa, cab], [cab, cbb]], 1.0 - m / total


def refusal(call):
    with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
        call()
    assert isinstance(refused.value, CausalUnsupportedError)
    return refused.value.reason_code, refused.value.detail


def test_x2_shared_rows_exact_covariance_matches_closed_form_and_is_point_only():
    result = exact(family("a", "b"))
    assert result.method == "exact_enumeration" and result.claim == "point_only"
    assert result.scope == "shared_row_covariance_point_only"
    assert "not_an_interval" in result.interpretation
    assert result.replicates_total == 462 and result.replicates_used == 462
    assert result.failed_replicates == 0 and result.n_rows == N
    assert result.scenario_ids == ("a", "b")
    for first in ("a", "b"):
        for second in ("a", "b"):
            assert result.entry(first, second) == pytest.approx(
                closed_form(first, second), abs=1e-12
            )
    assert result.entry("a", "a") == pytest.approx(17 / 216) and result.entry(
        "a", "b"
    ) == pytest.approx(11 / 216)
    assert result.entry("a", "b") == result.entry("b", "a")
    assert result.means == pytest.approx((1 / 6, 1 / 6))
    assert abs(result.entry("a", "b")) > 0.05, "shared rows give a nonzero off-diagonal"
    assert result.correlation("a", "b") == pytest.approx(11 / 17)
    assert result.as_array().shape == (2, 2)
    # There is no interval anywhere on the result, and none can be derived.
    assert not hasattr(result, "interval") and not hasattr(result, "lower")
    with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
        result.aggregate_interval()
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    assert refused.value.detail == "scenarios.shared_data_aggregate"


def test_x2_bootstrap_shares_one_replicate_selection_across_scenarios():
    result = bootstrap(family("a", "b"))
    assert result.method == "shared_row_bootstrap" and result.seed == 20_260_101
    assert result.replicates_total == 2000 and result.failed_replicates == 0
    for first in ("a", "b"):
        for second in ("a", "b"):
            want = closed_form(first, second)
            scale = (closed_form(first, first) * closed_form(second, second)) ** 0.5
            assert abs(result.entry(first, second) - want) < 0.15 * scale
    assert result.entry("a", "b") > 0.03, "the shared selection recovers the dependence"
    # Bit-reproducible under the seed, sensitive to it.
    again = bootstrap(family("a", "b"))
    assert (
        again.covariance == result.covariance and again.replicate_digest == result.replicate_digest
    )
    other = bootstrap(family("a", "b"), seed=7)
    assert other.replicate_digest != result.replicate_digest


def test_x2_adjustment_pair_with_joint_dropping_matches_tuple_enumeration():
    oracle, failed_fraction = enumerate_tuples()
    assert failed_fraction > 0
    result = exact(family("adjusted", "crude"), max_failure_mass=0.9)
    assert result.failed_replicates > 0
    assert result.replicates_used + result.failed_replicates == 462
    assert result.failed_mass == pytest.approx(failed_fraction, abs=1e-12)
    ids = ("adjusted", "crude")
    for i, first in enumerate(ids):
        for j, second in enumerate(ids):
            assert result.entry(first, second) == pytest.approx(oracle[i][j], abs=1e-12)
    assert abs(result.entry("adjusted", "crude")) > 1e-4


def test_x2_scenario_permutation_reorders_rows_and_columns_identically():
    first = exact(family("a", "b", "c"))
    second = exact(family("c", "a", "b"))
    assert first.scenario_ids == ("a", "b", "c") and second.scenario_ids == ("c", "a", "b")
    for x in "abc":
        for y in "abc":
            assert first.entry(x, y) == pytest.approx(second.entry(x, y), abs=1e-15)
        assert (
            first.means[first.scenario_ids.index(x)] == second.means[second.scenario_ids.index(x)]
        )
    assert first.entry("a", "c") == pytest.approx(closed_form("a", "c"), abs=1e-12)
    assert first.replicate_digest != second.replicate_digest, "scenario order is identity"
    boot_a = bootstrap(family("a", "b", "c"), replicates=200)
    boot_b = bootstrap(family("c", "a", "b"), replicates=200)
    for x in "abc":
        for y in "abc":
            assert boot_a.entry(x, y) == pytest.approx(boot_b.entry(x, y), abs=1e-15)


def test_x2_independent_or_incompatible_rows_refuse_unknown_dependence():
    unknown = ("route_not_supported", "scenario_covariance.unknown_dependence")
    different_snapshot = [estimator("a", snapshot="snap:one"), estimator("b", snapshot="snap:two")]
    assert refusal(lambda: exact(different_snapshot)) == unknown
    assert refusal(lambda: bootstrap(different_snapshot, replicates=10)) == unknown
    for dependence in ("independent_sample", "unknown"):
        declared = [estimator("a"), estimator("b", dependence=dependence)]
        assert refusal(lambda declared=declared: exact(declared)) == unknown
    other_units = [estimator("a"), estimator("b", unit_ids=[f"w{i}" for i in range(N)])]
    assert refusal(lambda: exact(other_units)) == unknown
    reordered = [f"u{i}" for i in range(N)]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert refusal(lambda: exact([estimator("a"), estimator("b", unit_ids=reordered)])) == unknown
    duplicate_units = table(units=["u0", "u1", "u2", "u2", "u4", "u5"])
    assert (
        refusal(lambda: transport.scenario_shared_covariance(duplicate_units, family("a", "b")))
        == unknown
    )
    # Consistent custom snapshot labels are fine and change the replicate identity.
    labelled = exact(family("a", "b", snapshot="snap:custom"))
    assert labelled.snapshot_digest == "snap:custom"
    assert labelled.replicate_digest != exact(family("a", "b")).replicate_digest


def test_x2_bounds_arguments_and_failures_refuse_with_typed_details():
    two = family("a", "b")
    over = refusal(lambda: bootstrap(two, replicates=2001))
    assert over == ("cell_not_licensed", "scenario_covariance.too_many_replicates")
    under = refusal(lambda: bootstrap(two, replicates=1))
    assert under == ("invalid_argument", "scenario_covariance.too_few_replicates")
    capped = refusal(lambda: exact(two, max_compositions=461))
    assert capped == ("cell_not_licensed", "scenario_covariance.exact_enumeration_cap")
    many = [transport.ScenarioEstimator(f"s{i}", estimator("a").functional) for i in range(65)]
    assert refusal(lambda: exact(many)) == (
        "cell_not_licensed",
        "scenario_covariance.too_many_scenarios",
    )
    twins = [estimator("a"), estimator("a")]
    assert refusal(lambda: exact(twins)) == (
        "invalid_argument",
        "scenario_covariance.duplicate_scenario_id",
    )
    assert refusal(lambda: exact([])) == ("invalid_argument", "scenario_covariance.no_scenarios")
    one_row = transport.RowTable(["z", "x", "y"], [(0, 0, 0)], ["only"])
    assert refusal(lambda: transport.scenario_shared_covariance(one_row, [estimator("a")])) == (
        "invalid_argument",
        "scenario_covariance.too_few_rows",
    )
    unknown_column = transport.ScenarioEstimator(
        "q", transport.LinearScore([transport.ScoreTerm(1.0, {"nope": 1})])
    )
    assert refusal(lambda: exact([unknown_column])) == (
        "invalid_argument",
        "scenario_covariance.invalid_functional",
    )
    # An estimator that fails in every replicate is refused, never dropped silently.
    never_treated = transport.ScenarioEstimator(
        "never", transport.AdjustedContrast("x", "y", treated=5)
    )
    failing = [estimator("a"), never_treated]
    assert refusal(lambda: bootstrap(failing, replicates=50, max_failure_fraction=0.1)) == (
        "transport_numerical_failure",
        "scenario_covariance.too_many_failed_replicates",
    )
    token = antecedent.state.CancellationToken()
    token.cancel()
    assert refusal(lambda: exact(two, cancel=token)) == (
        "cancelled_no_claim",
        "scenario_covariance.cancelled",
    )


def test_x2_inputs_are_typed_before_native_code():
    with pytest.raises(CausalTypeError):
        transport.scenario_shared_covariance("rows", family("a"))
    with pytest.raises(CausalTypeError):
        transport.scenario_shared_covariance(table(), ["a"])
    with pytest.raises(CausalTypeError):
        transport.RowTable(["z"], [(0.5,), (1.0,)], ["u0", "u1"])
    with pytest.raises(CausalValueError):
        transport.ScenarioEstimator("a", family("a")[0].functional, dependence="mostly")
    with pytest.raises(CausalValueError):
        transport.scenario_shared_covariance(table(), family("a"), method="jackknife")
    with pytest.raises(CausalValueError):
        transport.RowTable.from_columns({"z": [0, 1], "x": [0]})
    columns = transport.RowTable.from_columns({"z": [0, 1], "x": [1, 0]})
    assert columns.unit_ids == ("u0", "u1") and columns.rows == ((0, 1), (1, 0))


def test_x2_export_and_fresh_consume_reproduce_the_matrix():
    result = bootstrap(family("a", "b"), replicates=300)
    artifact = result.export()
    assert artifact.startswith(PREFIX)
    consumed = transport.consume_scenario_covariance_artifact(artifact)
    assert consumed.covariance == result.covariance and consumed.means == result.means
    assert consumed.scenario_ids == result.scenario_ids and consumed.method == result.method
    assert consumed.replicate_digest == result.replicate_digest
    assert consumed.row_identity_digest == result.row_identity_digest
    assert consumed.snapshot_digest == result.snapshot_digest and consumed.seed == result.seed
    assert consumed.claim == "point_only" and consumed.premises_digest and consumed.data_digest
    assert consumed.export() == artifact
    exact_result = exact(family("a", "b", "c"))
    assert transport.consume_scenario_covariance_artifact(exact_result.export()).covariance == (
        exact_result.covariance
    )
    # A fresh consumer needs only the bytes: the rows travel inside the artifact.
    assert b"u5" in artifact


def test_x2_tampered_covariance_artifacts_refuse_with_typed_details():
    result = exact(family("a", "b"))
    artifact = result.export()
    # A changed unit id is part of the stored row snapshot: the data digest refuses.
    unit = b"\x62u0"
    assert unit in artifact
    changed = artifact.replace(unit, b"\x62v0", 1)
    with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
        transport.consume_scenario_covariance_artifact(changed)
    assert refused.value.reason_code == "invalid_argument"
    assert refused.value.detail == "scenario_covariance.data_identity_mismatch"
    # A changed declared coefficient (-1.0 -> -2.0) is a premise: the premises digest refuses.
    # CBOR stores the shortest exact float: -1.0 is the half-float f9 bc00, -2.0 is f9 c000.
    coefficient = b"\xf9\xbc\x00"
    assert coefficient in artifact
    reweighted = artifact.replace(coefficient, b"\xf9\xc0\x00", 1)
    with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
        transport.consume_scenario_covariance_artifact(reweighted)
    assert refused.value.detail == "scenario_covariance.premises_mismatch"
    # Damaged or foreign bytes are serialization failures, not refusals.
    with pytest.raises(CausalSerializationError):
        transport.consume_scenario_covariance_artifact(artifact[:-4])
    with pytest.raises(CausalSerializationError):
        transport.consume_scenario_covariance_artifact(b"not an artifact")
    with pytest.raises(CausalTypeError):
        transport.consume_scenario_covariance_artifact("text")
    # The consumer's own limits bound the recomputation.
    for limits in ({"max_rows": N - 1}, {"max_columns": 2}):
        with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
            transport.consume_scenario_covariance_artifact(artifact, **limits)
        assert refused.value.reason_code == "cell_not_licensed"
        assert refused.value.detail == "scenario_covariance.consumer_limit_exceeded"
    boot = bootstrap(family("a", "b"), replicates=300).export()
    with pytest.raises(transport.ScenarioCovarianceRefusal) as refused:
        transport.consume_scenario_covariance_artifact(boot, max_replicates=299)
    assert refused.value.detail == "scenario_covariance.consumer_limit_exceeded"
