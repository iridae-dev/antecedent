"""B4 joint vector-treatment coefficients through the Python facade.

Every expected number is hand algebra on the frozen dataset (n = 6, columns ``1, t1, t2``):
``t1 = [1,0,1,0,0,1]``, ``t2 = [1,1,1,0,0,0]``, ``y = [3,2,5,1,0,4]``. ``b = [2.75, 0.75]``,
``RSS = 3.25``, ``sigma^2 = 3.25 / 3`` and the treatment block of the model-based covariance is
``V = (3.25 / 36) [[9, -3], [-3, 9]]``. Calibration (coverage, Type I error, power) is
deliberately NOT measured here, and the result says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import vector_treatment
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.vector_treatment import (
    Contrast,
    Treatment,
    VectorTreatmentRefusal,
    consume_joint_effects,
    joint_effects,
)

from _refusal import assert_registered_refusal

X1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
X2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
W = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
Y = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0]
S = 3.25 / 36.0

DIFF = Contrast("diff", {"t1": 1.0, "t2": -1.0})
SUM = Contrast("sum", {"t1": 1.0, "t2": 1.0})


def fit(**kwargs):
    return joint_effects(Y, {"t1": X1, "t2": X2}, **kwargs)


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def test_b4_vector_coefficients_and_full_covariance_match_hand_algebra() -> None:
    result = fit()
    assert result.names == ("t1", "t2")
    assert result.coefficient("t1").estimate == pytest.approx(2.75, abs=1e-9)
    assert result.coefficient("t2").estimate == pytest.approx(0.75, abs=1e-9)
    assert result.estimates == pytest.approx([2.75, 0.75], abs=1e-9)
    assert result.covariance.shape == (2, 2)
    assert result.covariance[0, 0] == pytest.approx(9.0 * S, abs=1e-9)
    assert result.covariance[0, 1] == pytest.approx(-3.0 * S, abs=1e-9)
    assert result.covariance[1, 0] == pytest.approx(-3.0 * S, abs=1e-9)
    assert result.covariance[1, 1] == pytest.approx(9.0 * S, abs=1e-9)
    assert result.covariance_of("t1", "t2") == pytest.approx(-3.0 * S, abs=1e-9)
    assert not result.covariance.flags.writeable
    first = result.coefficient("t1")
    assert first.standard_error == pytest.approx(math.sqrt(9.0 * S), abs=1e-9)
    assert first.se == first.standard_error
    assert first.z == pytest.approx(2.75 / math.sqrt(9.0 * S), abs=1e-9)
    assert first.p_value == pytest.approx(normal_p(first.z), abs=1e-9)
    assert (result.n_rows, result.residual_df) == (6, 3)
    assert result.residual_variance == pytest.approx(3.25 / 3.0, abs=1e-9)
    assert result.covariance_kind == "model_based"
    assert result.replay == "summary"
    assert result.null == vector_treatment.NULL


def test_b4_vector_contrast_standard_error_uses_the_off_diagonal() -> None:
    result = fit(contrasts=[DIFF, SUM])
    diff = result.contrast("diff")
    assert diff.estimate == pytest.approx(2.0, abs=1e-9)
    # Var(b1 - b2) = V11 + V22 - 2 V12 = 24 s; the naive independent variance is 18 s.
    assert diff.standard_error**2 == pytest.approx(24.0 * S, abs=1e-9)
    assert diff.naive_independent_standard_error**2 == pytest.approx(18.0 * S, abs=1e-9)
    assert diff.standard_error**2 - diff.naive_independent_standard_error**2 == pytest.approx(
        6.0 * S, abs=1e-9
    )
    total = result.contrast("sum")
    assert total.estimate == pytest.approx(3.5, abs=1e-9)
    assert total.standard_error**2 == pytest.approx(12.0 * S, abs=1e-9)
    assert diff.z == pytest.approx(2.0 / math.sqrt(24.0 * S), abs=1e-9)
    assert diff.p_value == pytest.approx(normal_p(diff.z), abs=1e-9)


def test_b4_vector_joint_wald_and_holm_match_closed_forms() -> None:
    result = fit(contrasts=[DIFF, SUM])
    statistic = 85.5 / 6.5  # b' V^-1 b
    assert result.joint_wald.statistic == pytest.approx(statistic, abs=1e-9)
    assert result.joint_wald.degrees_of_freedom == 2
    assert result.joint_wald.p_value == pytest.approx(math.exp(-statistic / 2.0), abs=1e-9)
    raw = [c.p_value for c in result.contrasts]
    low, high = (0, 1) if raw[0] <= raw[1] else (1, 0)
    holm_low = min(1.0, 2.0 * raw[low])
    assert result.contrasts[low].p_holm == pytest.approx(holm_low, abs=1e-9)
    assert result.contrasts[high].p_holm == pytest.approx(max(holm_low, raw[high]), abs=1e-9)


def test_b4_vector_calibration_is_unmeasured_and_says_so() -> None:
    result = fit()
    assert result.calibration == "unmeasured"
    assert result.inference_claim == "asymptotic_wald_calibration_unmeasured"
    assert any("calibration is unmeasured" in c for c in result.caveats)
    assert any("no interval" in c for c in result.caveats)
    assert any("caller owns" in c for c in result.caveats)
    summary = result.to_dict()
    assert summary["calibration"] == "unmeasured"
    assert "artifact" not in summary
    assert summary["covariance"][0][1] == pytest.approx(-3.0 * S, abs=1e-9)


def test_b4_vector_coefficient_order_follows_the_mapping() -> None:
    forward = fit()
    reversed_fit = joint_effects(Y, {"t2": X2, "t1": X1})
    assert reversed_fit.names == ("t2", "t1")
    assert reversed_fit.estimates == pytest.approx([0.75, 2.75], abs=1e-9)
    assert reversed_fit.covariance_of("t1", "t2") == pytest.approx(-3.0 * S, abs=1e-9)
    assert reversed_fit.identity.treatment_id != forward.identity.treatment_id


def test_b4_vector_adjustment_columns_are_shared() -> None:
    result = fit(adjust={"w": W})
    assert result.adjustment == ("w",)
    assert result.identity.adjustment_set_id != fit().identity.adjustment_set_id


def test_b4_vector_incompatible_adjustment_set_snapshot_and_rows_refuse() -> None:
    with pytest.raises(VectorTreatmentRefusal) as other_set:
        joint_effects(
            Y,
            {"t1": Treatment(X1, adjustment_set=["w"]), "t2": Treatment(X2, adjustment_set=[])},
            adjust={"w": W},
        )
    assert isinstance(other_set.value, CausalUnsupportedError)
    assert other_set.value.reason_code == "route_not_supported"
    assert other_set.value.detail == "vector_treatment.adjustment_set_mismatch"
    assert_registered_refusal(other_set.value)

    with pytest.raises(VectorTreatmentRefusal) as other_snapshot:
        joint_effects(Y, {"t1": X1, "t2": Treatment(X2, snapshot="elsewhere")}, snapshot="snap")
    assert other_snapshot.value.detail == "vector_treatment.row_snapshot_mismatch"
    assert other_snapshot.value.reason_code == "route_not_supported"

    with pytest.raises(VectorTreatmentRefusal) as short:
        joint_effects(Y, {"t1": X1, "t2": X2[:-1]})
    assert short.value.detail == "vector_treatment.row_count_mismatch"


def test_b4_vector_degenerate_designs_refuse_with_typed_details() -> None:
    with pytest.raises(VectorTreatmentRefusal) as constant:
        joint_effects(Y, {"t1": X1, "t2": [1.0] * 6})
    assert constant.value.reason_code == "design_rank_deficient"
    assert constant.value.detail == "vector_treatment.treatment_without_variation"
    assert_registered_refusal(constant.value)

    with pytest.raises(VectorTreatmentRefusal) as collinear:
        joint_effects(Y, {"t1": X1, "t2": [2.0 * v for v in X1]})
    assert collinear.value.detail == "vector_treatment.collinear_treatments"

    with pytest.raises(VectorTreatmentRefusal) as single:
        joint_effects(Y, {"t1": X1})
    assert single.value.reason_code == "invalid_argument"
    assert single.value.detail == "vector_treatment.too_few_treatments"

    with pytest.raises(VectorTreatmentRefusal) as nan:
        joint_effects([float("nan"), *Y[1:]], {"t1": X1, "t2": X2})
    assert nan.value.detail == "vector_treatment.non_finite_value"

    with pytest.raises(VectorTreatmentRefusal) as unknown:
        fit(contrasts=[Contrast("bad", {"zzz": 1.0})])
    assert unknown.value.detail == "vector_treatment.unknown_contrast_coefficient"
    with pytest.raises(VectorTreatmentRefusal) as empty:
        fit(contrasts=[Contrast("empty", {})])
    assert empty.value.detail == "vector_treatment.empty_contrast"


def test_b4_vector_input_validation_is_typed() -> None:
    with pytest.raises(CausalTypeError):
        joint_effects(Y, [X1, X2])  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        fit(covariance="cluster")
    with pytest.raises(CausalValueError):
        joint_effects([Y, Y], {"t1": X1, "t2": X2})
    with pytest.raises(CausalTypeError):
        fit(contrasts=[42])  # type: ignore[list-item]
    with pytest.raises(VectorTreatmentRefusal) as bad_weight:
        fit(contrasts=[Contrast("inf", {"t1": float("inf")})])
    assert bad_weight.value.detail == "vector_treatment.non_finite_value"
    with pytest.raises(CausalTypeError):
        consume_joint_effects("not bytes")  # type: ignore[arg-type]


def test_b4_vector_artifact_round_trip_recomputes_everything() -> None:
    result = fit(contrasts=[DIFF, SUM], adjust={"w": W})
    again = consume_joint_effects(result.export(), expected=result.identity)
    assert again.names == result.names
    assert again.coefficients == result.coefficients
    assert again.contrasts == result.contrasts
    assert again.joint_wald == result.joint_wald
    assert np.array_equal(again.covariance, result.covariance)
    assert again.identity == result.identity
    # A mapping form of the retained identity is accepted too.
    mapped = consume_joint_effects(result.export(), expected=result.to_dict()["identity"])
    assert mapped.identity == result.identity


def test_b4_vector_robust_covariance_embeds_the_rows_and_replays() -> None:
    model = fit()
    robust = fit(covariance="hc1")
    assert robust.replay == "rows"
    assert robust.covariance_kind == "hc1"
    assert not np.allclose(robust.covariance, model.covariance)
    assert robust.estimates == pytest.approx(model.estimates, abs=1e-12)
    again = consume_joint_effects(robust.export(), expected=robust.identity)
    assert np.array_equal(again.covariance, robust.covariance)


def test_b4_vector_robust_replay_above_the_row_cap_is_refused() -> None:
    n = 4097
    index = np.arange(n)
    columns = {"t1": (index % 7).astype(float), "t2": ((index + 3) % 7).astype(float)}
    outcome = ((index + 2) % 7).astype(float)
    with pytest.raises(VectorTreatmentRefusal) as refused:
        joint_effects(outcome, columns, covariance="hc0")
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == "vector_treatment.hc_replay_row_cap_exceeded"
    assert_registered_refusal(refused.value)
    # The model-based covariance replays from a summary and has no row cap.
    assert joint_effects(outcome, columns).replay == "summary"


def test_b4_vector_resealed_change_is_refused_against_the_retained_identity() -> None:
    # One declared snapshot identity for all three: the default (a content digest of the
    # columns) would itself change with the data and be refused first as ``row_snapshot``.
    original = fit(contrasts=[DIFF], snapshot="snap")
    changed = joint_effects([3.5, *Y[1:]], {"t1": X1, "t2": X2}, contrasts=[DIFF], snapshot="snap")
    # Alone the changed artifact is internally consistent ...
    assert consume_joint_effects(changed.export()).identity == changed.identity
    # ... but against the identity the consumer retained it is refused.
    with pytest.raises(VectorTreatmentRefusal) as refused:
        consume_joint_effects(changed.export(), expected=original.identity)
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == "vector_treatment.wrong_contract"
    assert refused.value.offending == "design"
    assert_registered_refusal(refused.value)
    dropped = fit(snapshot="snap")
    with pytest.raises(VectorTreatmentRefusal) as contracts:
        consume_joint_effects(dropped.export(), expected=original.identity)
    assert contracts.value.offending == "contrasts"


def test_b4_vector_corrupt_artifact_raises_a_serialization_error() -> None:
    data = bytearray(fit().export())
    data[len(data) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_joint_effects(bytes(data))
    with pytest.raises(CausalSerializationError):
        consume_joint_effects(b"")
