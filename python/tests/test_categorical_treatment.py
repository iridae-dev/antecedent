"""B4 categorical treatments through the Python facade.

Frozen dataset A (rows interleaved on purpose): ``a = [1,3]`` (mean 2), ``b = [4,6,5]`` (mean 5),
``c = [9,7,11,9]`` (mean 9); ``RSS = 12``, ``sigma^2 = 2``. With reference ``a``: ``b_b = 3``,
``b_c = 7`` and ``V = [[5/3, 1], [1, 3/2]]``. Frozen dataset M (ordered ``lo < mid < hi``, three
rows each): ``lo = [3,2,1]``, ``mid = [2,1,0]``, ``hi = [1,0,-1]``, so ``sigma^2 = 1``, every
adjacent step is ``-1`` with variance ``2/3``. Calibration (Type I error, coverage, power) is
deliberately NOT measured here, and the result says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import categorical_treatment
from antecedent.categorical_treatment import (
    CategoricalTreatmentRefusal,
    Direction,
    categorical_effects,
    consume_categorical_effects,
)
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)

from _refusal import assert_registered_refusal

ROWS = [
    ("c", 9.0),
    ("a", 1.0),
    ("b", 4.0),
    ("c", 7.0),
    ("b", 6.0),
    ("a", 3.0),
    ("c", 11.0),
    ("b", 5.0),
    ("c", 9.0),
]
LEVELS = [r[0] for r in ROWS]
OUTCOME = [r[1] for r in ROWS]
MONO_LEVELS = ["lo"] * 3 + ["mid"] * 3 + ["hi"] * 3
MONO_OUTCOME = [3.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, -1.0]
ORDER = ["lo", "mid", "hi"]


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def abc(**kwargs):
    kwargs.setdefault("categories", ["a", "b", "c"])
    kwargs.setdefault("reference", "a")
    return categorical_effects(OUTCOME, LEVELS, **kwargs)


def monotone(**kwargs):
    kwargs.setdefault("categories", ORDER)
    kwargs.setdefault("ordered", True)
    kwargs.setdefault("reference", "lo")
    kwargs.setdefault("monotonicity", "non_decreasing")
    return categorical_effects(MONO_OUTCOME, MONO_LEVELS, **kwargs)


def test_b4_categorical_dummy_regression_matches_group_mean_algebra() -> None:
    result = abc()
    assert result.reference == "a"
    assert result.ordered is False
    assert result.level_order == ("a", "b", "c")
    assert dict(result.counts) == {"a": 2, "b": 3, "c": 4}
    assert result.coefficient_names == ("level:b", "level:c")
    assert result.effect("b").estimate == pytest.approx(3.0, abs=1e-9)
    assert result.effect("c").estimate == pytest.approx(7.0, abs=1e-9)
    assert result.effect("b").reference == "a"
    assert result.covariance == pytest.approx(np.array([[5 / 3, 1.0], [1.0, 1.5]]), abs=1e-9)
    assert not result.covariance.flags.writeable
    assert result.effect("c").standard_error == pytest.approx(math.sqrt(1.5), abs=1e-9)
    assert result.effect("c").se == result.effect("c").standard_error
    assert result.effect("c").z == pytest.approx(7.0 / math.sqrt(1.5), abs=1e-9)
    assert result.effect("c").p_value == pytest.approx(normal_p(7.0 / math.sqrt(1.5)), abs=1e-9)
    assert result.monotonicity is None
    assert result.replay == "summary"
    with pytest.raises(KeyError):
        result.effect("a")  # the reference's effect is zero by definition


def test_b4_categorical_omnibus_and_holm_family_match_closed_forms() -> None:
    result = abc(pairs=[("b", "c")])
    statistic = 319.0 / 9.0
    assert result.omnibus.statistic == pytest.approx(statistic, abs=1e-9)
    assert result.omnibus.degrees_of_freedom == 2
    assert result.omnibus.p_value == pytest.approx(math.exp(-statistic / 2.0), abs=1e-9)
    assert result.omnibus.null == "every level has the reference level's effect"
    assert result.family_size == 3
    pair = result.pair("b", "c")
    assert pair.estimate == pytest.approx(4.0, abs=1e-9)
    # The pair variance needs the off-diagonal: 5/3 + 3/2 - 2 = 7/6 (the naive sum is 19/6).
    assert pair.standard_error**2 == pytest.approx(7.0 / 6.0, abs=1e-9)
    p_b = normal_p(3.0 / math.sqrt(5.0 / 3.0))
    p_c = normal_p(7.0 / math.sqrt(1.5))
    p_pair = normal_p(4.0 / math.sqrt(7.0 / 6.0))
    holm_c = min(1.0, 3.0 * p_c)
    holm_pair = max(holm_c, min(1.0, 2.0 * p_pair))
    holm_b = max(holm_pair, p_b)
    assert result.effect("c").p_holm == pytest.approx(holm_c, abs=1e-9)
    assert pair.p_holm == pytest.approx(holm_pair, abs=1e-9)
    assert result.effect("b").p_holm == pytest.approx(holm_b, abs=1e-9)


def test_b4_categorical_unordered_level_permutation_is_invariant() -> None:
    forward = abc()
    permuted = abc(categories=["c", "a", "b"])
    assert permuted.level_order == ("a", "b", "c")
    assert permuted.identity == forward.identity
    assert np.array_equal(permuted.covariance, forward.covariance)
    assert permuted.effects == forward.effects


def test_b4_categorical_ordered_scale_is_part_of_the_estimand() -> None:
    unordered = abc()
    ordered = abc(ordered=True)
    reversed_scale = abc(ordered=True, categories=["c", "b", "a"])
    assert ordered.level_order == ("a", "b", "c")
    assert reversed_scale.level_order == ("c", "b", "a")
    assert ordered.identity.level_scale_id != unordered.identity.level_scale_id
    assert reversed_scale.identity.level_scale_id != ordered.identity.level_scale_id
    with pytest.raises(CausalValueError):
        categorical_effects(OUTCOME, LEVELS, ordered=True)  # the order is the scale


def test_b4_categorical_monotonicity_test_and_its_null() -> None:
    result = monotone()
    mono = result.monotonicity
    assert mono is not None
    assert mono.direction is Direction.NON_DECREASING
    assert mono.null == (
        "every adjacent step of the adjusted level effects is non-negative (non-decreasing)"
    )
    assert [(s.from_level, s.to_level) for s in mono.steps] == [("lo", "mid"), ("mid", "hi")]
    for step in mono.steps:
        assert step.difference == pytest.approx(-1.0, abs=1e-9)
        assert step.standard_error**2 == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert mono.statistic == pytest.approx(-math.sqrt(1.5), abs=1e-9)
    # p = min(1, 2 Phi(-sqrt(3/2))) = erfc(sqrt(3/4)): conservative union-intersection bound.
    assert mono.p_value == pytest.approx(math.erfc(math.sqrt(0.75)), abs=1e-9)
    assert mono.conservative is True
    assert mono.calibration == "unmeasured"
    assert any("failing to reject does not prove" in c for c in result.caveats)
    flipped = monotone(monotonicity=Direction.NON_INCREASING).monotonicity
    assert flipped is not None
    assert flipped.direction is Direction.NON_INCREASING
    assert flipped.null != mono.null
    assert flipped.statistic == pytest.approx(math.sqrt(1.5), abs=1e-9)
    assert flipped.p_value == pytest.approx(1.0, abs=1e-9)


def test_b4_categorical_calibration_is_unmeasured_and_says_so() -> None:
    result = abc()
    assert result.calibration == "unmeasured"
    assert result.inference_claim == "asymptotic_wald_calibration_unmeasured"
    assert any("calibration is unmeasured" in c for c in result.caveats)
    summary = result.to_dict()
    assert summary["calibration"] == "unmeasured"
    assert summary["table"][0]["level"] == "a"
    assert summary["table"][0]["estimate"] == 0.0
    assert [row["rows"] for row in summary["table"]] == [2, 3, 4]
    assert categorical_treatment.__doc__ is not None


def test_b4_categorical_sparse_absent_and_undeclared_levels_refuse() -> None:
    with pytest.raises(CategoricalTreatmentRefusal) as absent:
        abc(categories=["a", "b", "c", "d"])
    assert isinstance(absent.value, CausalUnsupportedError)
    assert absent.value.reason_code == "arm_not_populated"
    assert absent.value.detail == "categorical_treatment.absent_level"
    assert "`d`" in str(absent.value)
    assert_registered_refusal(absent.value)

    with pytest.raises(CategoricalTreatmentRefusal) as sparse:
        abc(min_level_rows=3)
    assert sparse.value.detail == "categorical_treatment.sparse_level"
    assert "`a`" in str(sparse.value)

    with pytest.raises(CategoricalTreatmentRefusal) as undeclared:
        abc(categories=["a", "b"], reference="a")
    assert undeclared.value.reason_code == "invalid_argument"
    assert undeclared.value.detail == "categorical_treatment.undeclared_level"

    with pytest.raises(CategoricalTreatmentRefusal) as reference:
        abc(reference="q")
    assert reference.value.detail == "categorical_treatment.unknown_reference"

    with pytest.raises(CategoricalTreatmentRefusal) as unordered:
        abc(monotonicity="non_decreasing")
    assert unordered.value.reason_code == "route_not_supported"
    assert unordered.value.detail == "categorical_treatment.monotonicity_requires_ordered"

    with pytest.raises(CategoricalTreatmentRefusal) as pair:
        abc(pairs=[("a", "zzz")])
    assert pair.value.detail == "categorical_treatment.unknown_level"


def test_b4_categorical_input_validation_is_typed() -> None:
    with pytest.raises(CausalTypeError):
        categorical_effects(OUTCOME, "abc")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        abc(covariance="cluster")
    with pytest.raises(CausalValueError):
        abc(monotonicity="sideways")
    with pytest.raises(CausalTypeError):
        abc(min_level_rows=2.5)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        consume_categorical_effects("not bytes")  # type: ignore[arg-type]


def test_b4_categorical_artifact_round_trip_recomputes_everything() -> None:
    result = monotone(pairs=[("lo", "hi")])
    again = consume_categorical_effects(result.export(), expected_identity=result.identity)
    assert again.effects == result.effects
    assert again.pairs == result.pairs
    assert again.omnibus == result.omnibus
    assert again.monotonicity == result.monotonicity
    assert again.counts == result.counts
    assert np.array_equal(again.covariance, result.covariance)
    assert again.identity == result.identity
    mapped = consume_categorical_effects(
        result.export(), expected_identity=result.to_dict()["identity"]
    )
    assert mapped.identity == result.identity


def test_b4_categorical_robust_covariance_embeds_the_rows_and_replays() -> None:
    model = abc()
    robust = abc(covariance="hc1")
    assert robust.replay == "rows"
    assert not np.allclose(robust.covariance, model.covariance)
    assert robust.effect("c").estimate == pytest.approx(7.0, abs=1e-9)
    again = consume_categorical_effects(robust.export(), expected_identity=robust.identity)
    assert np.array_equal(again.covariance, robust.covariance)


def test_b4_categorical_robust_replay_above_the_row_cap_is_refused() -> None:
    n = 4097
    index = np.arange(n)
    labels = np.array(["a", "b", "c"])[index % 3]
    outcome = (index % 5).astype(float)
    with pytest.raises(CategoricalTreatmentRefusal) as refused:
        categorical_effects(outcome, labels, reference="a", covariance="hc0")
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == "categorical_treatment.hc_replay_row_cap_exceeded"
    assert categorical_effects(outcome, labels, reference="a").replay == "summary"


def test_b4_categorical_resealed_change_is_refused_against_the_retained_identity() -> None:
    original = monotone()
    cases = {
        "level_scale": monotone(reference="mid"),
        "family": monotone(pairs=[("lo", "hi")]),
        "design": monotone(covariance="hc0"),
    }
    for offending, changed in cases.items():
        assert consume_categorical_effects(changed.export()).identity == changed.identity
        with pytest.raises(CategoricalTreatmentRefusal) as refused:
            consume_categorical_effects(changed.export(), expected_identity=original.identity)
        assert refused.value.reason_code == "route_not_supported"
        assert refused.value.detail == "categorical_treatment.wrong_contract"
        assert refused.value.offending == offending
        assert_registered_refusal(refused.value)


def test_b4_categorical_corrupt_artifact_raises_a_serialization_error() -> None:
    data = bytearray(abc().export())
    data[len(data) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_categorical_effects(bytes(data))
    with pytest.raises(CausalSerializationError):
        consume_categorical_effects(b"")
