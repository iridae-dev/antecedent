"""F18 ``EffectConstancy`` (2.3A): the frozen acceptance numbers, typed refusals, permutation
invariance and the portable artifact through the Python facade.

Every expected number is a closed form: Cochran's ``Q`` for independent partitions, the Wald
chi-square for a supplied covariance, and the chi-square survival functions ``erfc(sqrt(x / 2))``
(one degree of freedom) and ``exp(-x / 2)`` (two). Calibration (null Type I error and power)
is deliberately NOT measured here, and the result says so.
"""

from __future__ import annotations

import math

import pytest
from antecedent import temporal
from antecedent.errors import (
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.temporal import (
    ConstancyConclusion,
    EffectEstimand,
    Partition,
    TemporalRefusal,
    consume_effect_constancy_artifact,
    effect_constancy,
)

from _refusal import assert_registered_refusal

EFFECT = EffectEstimand("ate_difference", "outcome_units", "treat_vs_control", "all_observed_h2")
ERFC_1 = 0.157_299_207_050_285_13


def run(parts, **kwargs):
    kwargs.setdefault("estimand", EFFECT)
    return effect_constancy(parts, **kwargs)


def test_f18_frozen_acceptance_equal_effects_have_zero_statistic() -> None:
    # Effects 1 and 1, variance 1/4 each, zero covariance.
    result = run([("p1", 1.0, 0.5), ("p2", 1.0, 0.5)])
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.statistic_kind == "cochran_q"
    assert result.degrees_of_freedom == 1
    assert result.p_value == pytest.approx(1.0, abs=1e-12)
    assert result.conclusion is ConstancyConclusion.NOT_REJECTED
    assert result.pooled_effect == pytest.approx(1.0)
    assert result.null == temporal.NULL
    assert result.estimand == EFFECT


def test_f18_frozen_acceptance_one_versus_two_has_statistic_two() -> None:
    # Q = (2 - 1)^2 / (1/4 + 1/4) = 2 and sf = erfc(sqrt(2 / 2)) = erfc(1).
    result = run([("p1", 1.0, 0.5), ("p2", 2.0, 0.5)])
    assert result.statistic == pytest.approx(2.0, abs=1e-9)
    assert result.p_value == pytest.approx(ERFC_1, abs=1e-9)
    assert result.p_value == pytest.approx(math.erfc(1.0), abs=1e-9)
    assert result.conclusion is ConstancyConclusion.NOT_REJECTED
    assert result.pooled_effect == pytest.approx(1.5)
    (contrast,) = result.contrasts
    assert (contrast.left, contrast.right) == ("p1", "p2")
    assert contrast.difference == pytest.approx(-1.0)
    assert contrast.z == pytest.approx(-math.sqrt(2.0))
    assert contrast.p_value == pytest.approx(ERFC_1, abs=1e-9)
    # A single contrast: Holm leaves it unadjusted.
    assert contrast.p_holm == pytest.approx(ERFC_1, abs=1e-9)
    assert not contrast.rejected


def test_f18_calibration_is_unmeasured_and_non_rejection_is_not_proof() -> None:
    result = run([("p1", 1.0, 0.5), ("p2", 1.0, 0.5)])
    assert result.calibration == "unmeasured"
    assert result.inference_claim == "point_only"
    assert any("does not prove" in caveat for caveat in result.caveats)
    assert any("unmeasured" in caveat for caveat in result.caveats)
    assert result.to_dict()["calibration"] == "unmeasured"
    assert result.to_dict()["conclusion"] == "not_rejected"


def test_f18_three_partitions_df_two_and_holm_by_hand() -> None:
    # Effects 0, 1, 2, unit variance: pooled 1, Q = 1 + 0 + 1 = 2, df 2, p = exp(-1).
    result = run([("c", 2.0, 1.0), ("a", 0.0, 1.0), ("b", 1.0, 1.0)])
    assert [p.label for p in result.partitions] == ["a", "b", "c"]
    assert result.statistic == pytest.approx(2.0, abs=1e-9)
    assert result.degrees_of_freedom == 2
    assert result.p_value == pytest.approx(math.exp(-1.0), abs=1e-9)
    ab, ac, bc = result.contrasts
    assert (ab.left, ab.right, ac.left, ac.right, bc.left, bc.right) == (
        "a",
        "b",
        "a",
        "c",
        "b",
        "c",
    )
    # a-b and b-c: z = -1/sqrt(2), p = erfc(1/2); a-c: z = -sqrt(2), p = erfc(1).
    assert ab.p_value == pytest.approx(0.479_500_122_186_953_5, abs=1e-9)
    assert ac.p_value == pytest.approx(ERFC_1, abs=1e-9)
    # Holm (m = 3): a-c 3 * 0.1573 = 0.4719; next 2 * 0.4795 = 0.9590; last lifted to 0.9590.
    assert ac.p_holm == pytest.approx(0.471_897_621_150_855_4, abs=1e-9)
    assert ab.p_holm == pytest.approx(0.959_000_244_373_907, abs=1e-9)
    assert bc.p_holm == pytest.approx(0.959_000_244_373_907, abs=1e-9)
    assert not any(c.rejected for c in result.contrasts)


def test_f18_reference_family_is_reported() -> None:
    result = run([("a", 0.0, 1.0), ("b", 1.0, 1.0), ("c", 2.0, 1.0)], against="a")
    assert result.family == "against_reference"
    assert result.reference == "a"
    assert [(c.left, c.right) for c in result.contrasts] == [("b", "a"), ("c", "a")]
    # b-a: p = erfc(1/2); c-a: p = erfc(1). Holm (m = 2): c-a 2 * 0.1573; b-a max(0.4795, 0.3146).
    assert result.contrasts[0].p_holm == pytest.approx(0.479_500_122_186_953_5, abs=1e-9)
    assert result.contrasts[1].p_holm == pytest.approx(0.314_598_414_100_570_26, abs=1e-9)


def test_f18_dependent_covariance_gives_wald_four() -> None:
    # Estimates 1 and 2, variance 1/4 each, covariance 1/8: Var(e2 - e1) = 1/4, Wald = 4.
    parts = [("a", 1.0, 0.5), ("b", 2.0, 0.5)]
    dependent = run(parts, covariance=[[0.25, 0.125], [0.125, 0.25]])
    assert dependent.statistic_kind == "wald_chi_square"
    assert dependent.statistic == pytest.approx(4.0, abs=1e-9)
    assert dependent.p_value == pytest.approx(math.erfc(math.sqrt(2.0)), abs=1e-9)
    assert dependent.conclusion is ConstancyConclusion.REJECTED
    assert dependent.pooled_effect is None
    assert dependent.contrasts[0].se == pytest.approx(0.5)
    independent = run(parts)
    assert independent.statistic == pytest.approx(2.0, abs=1e-9)
    assert dependent.statistic / independent.statistic == pytest.approx(2.0)
    # A diagonal covariance is the same number as Cochran's Q.
    diagonal = run(parts, covariance=[[0.25, 0.0], [0.0, 0.25]])
    assert diagonal.statistic == pytest.approx(independent.statistic, abs=1e-12)


def test_f18_varying_effects_are_rejected() -> None:
    result = run([("a", 0.0, 0.5), ("b", 5.0, 0.5)])
    assert result.statistic == pytest.approx(50.0, abs=1e-9)
    assert result.p_value < 1e-9
    assert result.conclusion is ConstancyConclusion.REJECTED
    assert result.contrasts[0].rejected


def test_f18_per_partition_table_carries_effects_and_derived_coordinates() -> None:
    result = run(
        [("2024q1", 0.3, 0.4), Partition("2024q2", 1.1, 0.6, coordinate="period:Q2-2024")],
        coordinate="period",
    )
    assert result.table() == [
        {
            "label": "2024q1",
            "coordinate": "period:2024q1",
            "support": "supported",
            "effect": 0.3,
            "se": 0.4,
        },
        {
            "label": "2024q2",
            "coordinate": "period:Q2-2024",
            "support": "supported",
            "effect": 1.1,
            "se": 0.6,
        },
    ]
    mapped = run([{"label": "x", "effect": 1.0, "se": 0.5}, ("y", 1.0, 0.5)])
    assert [p.label for p in mapped.partitions] == ["x", "y"]


def test_f18_partition_permutation_gives_bit_identical_result() -> None:
    parts = [("a", 0.3, 0.4), ("b", 1.1, 0.6), ("c", -0.2, 0.5)]
    forward = run(parts)
    reversed_ = run(list(reversed(parts)))
    assert forward.to_dict() == reversed_.to_dict()
    assert forward.identity == reversed_.identity

    full = [[0.16, 0.05, -0.02], [0.05, 0.36, 0.03], [-0.02, 0.03, 0.25]]

    def dependent(order):
        covariance = [[full[i][j] for j in order] for i in order]
        return run([parts[i] for i in order], covariance=covariance)

    base = dependent([0, 1, 2])
    for order in ([2, 1, 0], [1, 2, 0], [0, 2, 1]):
        other = dependent(order)
        assert other.statistic == base.statistic, order
        assert other.p_value == base.p_value, order
        assert other.contrasts == base.contrasts, order
        assert other.partitions == base.partitions, order
        assert other.identity == base.identity, order


@pytest.mark.parametrize(
    "changed",
    [
        EffectEstimand("risk_ratio", "outcome_units", "treat_vs_control", "all_observed_h2"),
        EffectEstimand("ate_difference", "log_scale", "treat_vs_control", "all_observed_h2"),
        EffectEstimand("ate_difference", "outcome_units", "other_regime", "all_observed_h2"),
        EffectEstimand("ate_difference", "outcome_units", "treat_vs_control", "other_population"),
    ],
)
def test_f18_partitions_that_change_the_estimand_refuse(changed: EffectEstimand) -> None:
    parts = [Partition("a", 1.0, 0.5), Partition("b", 1.0, 0.5, estimand=changed)]
    with pytest.raises(TemporalRefusal) as raised:
        effect_constancy(parts, estimand=EFFECT)
    error = raised.value
    assert isinstance(error, CausalUnsupportedError)
    assert error.reason_code == "route_not_supported"
    assert error.detail == "effect_constancy.incompatible_partitions"
    assert "effect_constancy.incompatible_partitions" in str(error)
    assert_registered_refusal(error)


def test_f18_unsupported_coordinate_and_duplicate_labels_refuse() -> None:
    with pytest.raises(TemporalRefusal) as raised:
        run([Partition("a", 1.0, 0.5), Partition("b", 1.0, 0.5, support="unsupported")])
    assert raised.value.reason_code == "route_not_supported"
    assert raised.value.detail == "effect_constancy.unsupported_partition"
    with pytest.raises(TemporalRefusal) as raised:
        run([("a", 1.0, 0.5), ("a", 2.0, 0.5)])
    assert raised.value.detail == "effect_constancy.incompatible_partitions"
    # A partially supported coordinate is reported, not refused.
    partial = run([Partition("a", 1.0, 0.5), Partition("b", 1.0, 0.5, support="partial")])
    assert partial.partitions[1].support == "partial"


def test_f18_invalid_numbers_refuse_with_exact_details() -> None:
    def detail(parts, **kwargs) -> tuple[str, str]:
        with pytest.raises(TemporalRefusal) as raised:
            run(parts, **kwargs)
        return raised.value.reason_code, raised.value.detail

    assert detail([("a", 1.0, 0.5)]) == (
        "invalid_argument",
        "effect_constancy.too_few_partitions",
    )
    assert detail([("a", 1.0, 0.5), ("b", 1.0, 0.0)]) == (
        "invalid_argument",
        "effect_constancy.invalid_standard_error",
    )
    for covariance in (
        [[0.25, 0.1], [0.2, 0.25]],  # not symmetric
        [[0.25, 0.5], [0.5, 0.25]],  # not positive definite
        [[0.25, 0.25], [0.25, 0.25]],  # singular
        [[1.0, 0.0], [0.0, 0.25]],  # diagonal does not equal the squared standard error
        [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],  # wrong size
        [[0.25, float("nan")], [float("nan"), 0.25]],  # not finite
    ):
        assert detail([("a", 1.0, 0.5), ("b", 2.0, 0.5)], covariance=covariance) == (
            "invalid_argument",
            "effect_constancy.invalid_covariance",
        ), covariance
    assert detail([("a", 1.0, 0.5), ("b", 2.0, 0.5)], against="zzz") == (
        "invalid_argument",
        "effect_constancy.unknown_reference",
    )
    assert detail([("a", 1.0, 0.5), ("b", 2.0, 0.5)], alpha=1.5) == (
        "invalid_argument",
        "effect_constancy.invalid_alpha",
    )
    for bad in (float("nan"), float("inf")):
        with pytest.raises(TemporalRefusal) as raised:
            Partition("a", bad, 0.5)
        assert raised.value.detail == "effect_constancy.non_finite_estimate"


def test_f18_a_missing_shared_estimand_is_never_assumed() -> None:
    with pytest.raises(CausalValueError, match="estimand_missing"):
        effect_constancy([("a", 1.0, 0.5), ("b", 1.0, 0.5)])
    with pytest.raises(TypeError, match="partitions must be a sequence"):
        effect_constancy("ab")  # type: ignore[arg-type]


def test_f18_artifact_round_trip_retains_identities_covariance_and_calibration() -> None:
    result = run([("a", 1.0, 0.5), ("b", 2.0, 0.5)], covariance=[[0.25, 0.125], [0.125, 0.25]])
    artifact = result.export()
    assert isinstance(artifact, bytes)
    fresh = consume_effect_constancy_artifact(artifact, expected_identity=result.identity)
    assert fresh.statistic == result.statistic
    assert fresh.p_value == result.p_value
    assert fresh.contrasts == result.contrasts
    assert fresh.partitions == result.partitions
    assert fresh.identity == result.identity
    assert fresh.calibration == "unmeasured"
    assert fresh.dependence == "covariance"
    assert fresh.estimand == EFFECT
    assert fresh.export() == artifact
    # The identity may also be given as the plain mapping a consumer stored out-of-band.
    assert consume_effect_constancy_artifact(artifact, expected_identity=result.identity._wire())
    assert consume_effect_constancy_artifact(artifact).identity == result.identity


def test_f18_resealed_mutations_are_refused_against_the_retained_identity() -> None:
    base_parts = [("a", 1.0, 0.5), ("b", 2.0, 0.5)]
    covariance = [[0.25, 0.125], [0.125, 0.25]]
    original = run(base_parts, covariance=covariance)

    def refused(mutated, field: str) -> None:
        # Alone the mutated artifact is internally consistent ...
        assert consume_effect_constancy_artifact(mutated.export()).identity == mutated.identity
        # ... but against the identity the consumer retained it is refused.
        with pytest.raises(TemporalRefusal) as raised:
            consume_effect_constancy_artifact(mutated.export(), expected_identity=original.identity)
        assert raised.value.reason_code == "route_not_supported"
        assert raised.value.detail == "effect_constancy.wrong_contract"
        assert raised.value.offending == field, field

    refused(run(base_parts, covariance=[[0.25, 0.0625], [0.0625, 0.25]]), "covariance")
    refused(run(base_parts), "covariance")
    refused(
        run(
            base_parts,
            covariance=covariance,
            estimand=EffectEstimand(
                "ate_difference", "log_scale", "treat_vs_control", "all_observed_h2"
            ),
        ),
        "estimand",
    )
    refused(
        run(
            [Partition("a", 1.0, 0.5, coordinate="region:north"), ("b", 2.0, 0.5)],
            covariance=covariance,
        ),
        "partition_identity",
    )
    refused(run(base_parts, covariance=covariance, against="a"), "multiplicity_family")
    refused(run(base_parts, covariance=covariance, alpha=0.01), "multiplicity_family")
    refused(run([("a", 1.0, 0.5), ("b", 2.5, 0.5)], covariance=covariance), "evidence")


def test_f18_corrupt_truncated_and_foreign_artifacts_refuse() -> None:
    artifact = run([("a", 1.0, 0.5), ("b", 2.0, 0.5)]).export()
    corrupt = bytearray(artifact)
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_effect_constancy_artifact(bytes(corrupt))
    with pytest.raises(CausalSerializationError):
        consume_effect_constancy_artifact(artifact[:-5])
    with pytest.raises(CausalSerializationError):
        consume_effect_constancy_artifact(b"not an artifact")
    with pytest.raises(TypeError, match="artifact must be bytes"):
        consume_effect_constancy_artifact("text")  # type: ignore[arg-type]
