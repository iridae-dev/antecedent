"""2.3 B3: the source-target mechanism discrepancy diagnostic.

Hand datasets mirror ``crates/antecedent-estimate/tests/mechanism_discrepancy.rs``.

Dataset A (one parent ``x``, n = 4): ``x = [0, 1, 2, 3]``, ``y_s = [1, 3, 2, 4]``. ``X'X =
[[4, 6], [6, 14]]``, ``X'y = [10, 19]``, so ``b = [1.3, 0.8]`` with ``sigma^2 = 0.9`` and
``V = 0.9 (X'X)^-1 = [[0.63, -0.27], [-0.27, 0.18]]``. Adding ``c0 + c1 x`` to the outcome shifts
the OLS coefficients by exactly ``(c0, c1)`` and leaves ``V`` unchanged, so the summed
covariance is ``2 V``:

* slope shift 0.5 (intercept not compared): ``W = 0.5^2 / 0.36``, df 1, ``p = erfc(sqrt(W / 2))``;
* intercept and slope shift 0.5: ``W = 25 / 6``, df 2, ``p = exp(-25 / 12)``.

Dataset B (two parents, n = 6): ``a = [1, 0, 1, 0, 0, 1]``, ``b = [1, 1, 1, 0, 0, 0]``,
``y = [3, 2, 5, 1, 0, 4]``; ``b_hat = [0.75, 2.75, 0.75]``, ``sigma^2 = 3.25 / 3``.
"""

from __future__ import annotations

import math
import re

import pytest
from antecedent import mechanism_discrepancy as md
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError

X = [0.0, 1.0, 2.0, 3.0]
YS = [1.0, 3.0, 2.0, 4.0]
YT_SLOPE = [1.0, 3.5, 3.0, 5.5]  # YS + 0.5 x
YT_BOTH = [1.5, 4.0, 3.5, 6.0]  # YS + 0.5 + 0.5 x
YT_BIG = [1.0, 8.0, 12.0, 19.0]  # YS + 5 x

A = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
B = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
Y6 = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0]
Y6_TARGET = [4.5, 2.5, 6.5, 1.0, 0.0, 5.0]  # Y6 + 1 a + 0.5 b


def _measurement(
    parents: tuple[tuple[str, str], ...] = (("x", "cm"),),
    *,
    node: str = "V",
    node_unit: str = "mg",
    protocol_id: str = "protocol-1",
) -> md.Measurement:
    return md.Measurement(node, node_unit, parents=parents, protocol_id=protocol_id)


def _one_parent(label: str, y: list[float], **kwargs: object) -> md.Sample:
    return md.Sample(label, outcome=y, parents={"x": X}, **kwargs)  # type: ignore[arg-type]


def _run(
    source_y: list[float],
    target_y: list[float],
    *,
    compare_intercept: bool = True,
    **kwargs: object,
) -> md.MechanismDiscrepancyResult:
    return md.diagnose_mechanism_discrepancy(
        source=_one_parent("source", source_y),
        target=_one_parent("target", target_y),
        measurement=_measurement(),
        compare_intercept=compare_intercept,
        **kwargs,  # type: ignore[arg-type]
    )


def _two_parents(
    label: str, y: list[float], order: tuple[str, ...] = ("a", "b")
) -> tuple[md.Sample, md.Measurement]:
    columns = {"a": A, "b": B}
    sample = md.Sample(label, outcome=y, parents={name: columns[name] for name in order})
    return sample, _measurement(tuple((name, "u") for name in order))


def _refusal(call, code: str, detail: str) -> md.MechanismDiscrepancyRefusal:
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    error = caught.value
    assert isinstance(error, md.MechanismDiscrepancyRefusal)
    assert error.reason_code == code
    assert error.detail == f"mechanism_discrepancy.{detail}"
    assert detail in str(error)
    return error


# ------------------------------------------------------------- fits and statistic


def test_b3_discrepancy_ols_fits_and_standard_errors_match_hand_algebra():
    result = _run(YS, YT_BOTH)
    assert result.coefficient_names == ("(intercept)", "x")
    assert result.source.coefficients == pytest.approx((1.3, 0.8), abs=1e-9)
    assert result.target.coefficients == pytest.approx((1.8, 1.3), abs=1e-9)
    assert result.source.n == 4
    assert result.source.residual_df == 2
    assert result.source.residual_variance == pytest.approx(0.9, abs=1e-9)
    assert result.target.residual_variance == pytest.approx(0.9, abs=1e-9)
    assert result.source.standard_errors == pytest.approx(
        (math.sqrt(0.63), math.sqrt(0.18)), abs=1e-9
    )
    assert result.source.label == "source"
    assert result.target.label == "target"
    assert result.measurement.node == "V"
    assert result.measurement.parents == (("x", "cm"),)


def test_b3_discrepancy_identical_fits_give_statistic_zero_exactly():
    same = _run(YS, YS)
    assert same.statistic == 0.0
    assert same.p_value == 1.0
    assert all(c.difference == 0.0 and not c.rejected for c in same.coefficients)
    assert same.conclusion == "not_rejected"
    assert not same.rejected
    # The same OLS fit from the rows in reverse order has identical sufficient statistics.
    reversed_rows = md.Sample("target", outcome=list(reversed(YS)), parents={"x": X[::-1]})
    result = md.diagnose_mechanism_discrepancy(
        source=_one_parent("source", YS), target=reversed_rows, measurement=_measurement()
    )
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_b3_discrepancy_slope_shift_matches_df1_closed_form():
    result = _run(YS, YT_SLOPE, compare_intercept=False)
    # delta = 0.5, v_s + v_t = 0.36: W = delta^2 / 0.36.
    w = 0.25 / 0.36
    assert result.degrees_of_freedom == 1
    assert result.statistic == pytest.approx(w, abs=1e-9)
    # df = 1: P(chi2_1 > w) = erfc(sqrt(w / 2)).
    assert result.p_value == pytest.approx(math.erfc(math.sqrt(w / 2.0)), abs=1e-9)
    (coefficient,) = result.coefficients
    assert coefficient.name == "x"
    assert coefficient.difference == pytest.approx(0.5, abs=1e-9)
    assert coefficient.standard_error == pytest.approx(0.6, abs=1e-9)
    assert coefficient.z == pytest.approx(0.5 / 0.6, abs=1e-9)
    assert coefficient.p_value == pytest.approx(
        math.erfc(abs(0.5 / 0.6) / math.sqrt(2.0)), abs=1e-9
    )
    # One coefficient: Holm is the identity.
    assert coefficient.p_holm == pytest.approx(coefficient.p_value, abs=1e-12)
    assert result.conclusion == "not_rejected"
    assert result.compare_intercept is False


def test_b3_discrepancy_intercept_and_slope_shift_matches_df2_closed_form():
    result = _run(YS, YT_BOTH)
    # d = (0.5, 0.5); W = d' (2V)^-1 d = 0.25 (0.36 + 2 * 0.54 + 1.26) / 0.162 = 25 / 6.
    assert result.statistic == pytest.approx(25.0 / 6.0, abs=1e-9)
    assert result.degrees_of_freedom == 2
    # df = 2: P(chi2_2 > w) = exp(-w / 2).
    assert result.p_value == pytest.approx(math.exp(-25.0 / 12.0), abs=1e-9)
    intercept, slope = result.coefficients
    assert (intercept.name, slope.name) == ("(intercept)", "x")
    assert intercept.standard_error == pytest.approx(math.sqrt(1.26), abs=1e-9)
    assert slope.standard_error == pytest.approx(0.6, abs=1e-9)
    # Holm over two: the smaller raw p (slope) doubles; the other is at least that.
    assert slope.p_value < intercept.p_value
    assert slope.p_holm == pytest.approx(min(2.0 * slope.p_value, 1.0), abs=1e-12)
    assert intercept.p_holm == pytest.approx(max(slope.p_holm, intercept.p_value), abs=1e-12)


def test_b3_discrepancy_two_parent_statistic_matches_slope_block_algebra():
    source, measurement = _two_parents("source", Y6)
    target, _ = _two_parents("target", Y6_TARGET)
    result = md.diagnose_mechanism_discrepancy(
        source=source, target=target, measurement=measurement, compare_intercept=False
    )
    assert result.source.coefficients == pytest.approx((0.75, 2.75, 0.75), abs=1e-9)
    # d = (1, 0.5); summed block s [[9, -3], [-3, 9]], s = 3.25 / 18; inverse = [[9, 3], [3, 9]] / (72 s).
    w = (9.0 + 3.0 + 2.25) / (72.0 * 3.25 / 18.0)
    assert result.statistic == pytest.approx(w, abs=1e-9)
    assert result.degrees_of_freedom == 2
    assert result.p_value == pytest.approx(math.exp(-w / 2.0), abs=1e-9)
    assert [c.difference for c in result.coefficients] == pytest.approx([1.0, 0.5], abs=1e-9)
    assert result.coefficients[0].standard_error == pytest.approx(
        math.sqrt(9.0 * 3.25 / 18.0), abs=1e-9
    )


def test_b3_discrepancy_parent_order_is_irrelevant():
    source, measurement = _two_parents("source", Y6)
    target, _ = _two_parents("target", Y6_TARGET)
    base = md.diagnose_mechanism_discrepancy(source=source, target=target, measurement=measurement)
    permuted_source, permuted_measurement = _two_parents("source", Y6, ("b", "a"))
    permuted_target, _ = _two_parents("target", Y6_TARGET, ("b", "a"))
    # Each population may declare its own parent order; the test puts them in name order.
    mixed = md.diagnose_mechanism_discrepancy(
        source=md.Sample("source", outcome=Y6, parents={"a": A, "b": B}, measurement=measurement),
        target=md.Sample(
            "target",
            outcome=Y6_TARGET,
            parents={"b": B, "a": A},
            measurement=permuted_measurement,
        ),
    )
    assert mixed.coefficient_names == ("(intercept)", "a", "b")
    assert mixed.statistic == base.statistic
    assert mixed.coefficients == base.coefficients
    both = md.diagnose_mechanism_discrepancy(
        source=permuted_source, target=permuted_target, measurement=permuted_measurement
    )
    assert both.coefficient_names == ("(intercept)", "a", "b")
    assert both.statistic == base.statistic
    assert both.p_value == base.p_value
    assert both.coefficients == base.coefficients
    assert both.source.coefficients == base.source.coefficients
    assert both.measurement.parents == (("a", "u"), ("b", "u"))


# ----------------------------------------------------------------- power limit


def test_b3_discrepancy_minimal_detectable_difference_uses_normal_quantiles():
    result = _run(YS, YT_BOTH)
    factor = 1.959_964 + 0.841_621
    assert result.alpha == 0.05
    assert result.power == 0.8
    assert result.detectability_factor == pytest.approx(factor, abs=1e-6)
    for coefficient in result.coefficients:
        assert coefficient.minimal_detectable_difference == pytest.approx(
            factor * coefficient.standard_error, abs=1e-6
        )
    assert result.minimal_detectable_differences["x"] == pytest.approx(factor * 0.6, abs=1e-6)
    assert "not detectable" in result.power_statement
    assert "x:" in result.power_statement
    # A tighter level and a higher power make a larger difference undetectable.
    strict = _run(YS, YT_BOTH, alpha=0.01, power=0.9)
    assert strict.detectability_factor > result.detectability_factor
    assert strict.alpha == 0.01
    assert strict.power == 0.9


# ----------------------------------------------------- never certifies invariance


def test_b3_discrepancy_non_rejection_never_certifies_invariance():
    result = _run(YS, YT_SLOPE, compare_intercept=False)
    assert result.conclusion == "not_rejected"
    assert result.non_rejection_certifies_invariance is False
    assert any("does not certify" in caveat for caveat in result.caveats)
    assert any("unmeasured" in caveat for caveat in result.caveats)
    assert result.calibration == "unmeasured"
    assert result.inference_claim == "asymptotic_wald_calibration_unmeasured"
    assert result.informs_selection_on == ("V",)
    assert "non-rejection leaves a selection node" in result.alignment
    assert "cannot be excluded" in result.alignment
    assert "independent" in result.dependence_assumption
    # The minimal detectable slope difference exceeds the observed shift: the shift was not
    # detectable, which is exactly what the power limit reports.
    assert result.coefficients[0].minimal_detectable_difference > abs(
        result.coefficients[0].difference
    )
    text = result.explain()
    assert "does not certify" in text
    assert "not detectable" in text
    # "invariant" appears only in the negated alignment wording ("not invariant").
    assert re.search(r"(?<!not )invariant", text) is None

    big = _run(YS, YT_BIG, compare_intercept=False)
    assert big.conclusion == "rejected"
    assert big.rejected
    assert big.coefficients[0].rejected
    assert big.non_rejection_certifies_invariance is False
    assert big.informs_selection_on == ("V",)
    assert "cannot be excluded" in big.explain()
    assert re.search(r"(?<!not )invariant", big.explain()) is None


# ---------------------------------------------------------- summary-statistics input


def test_b3_discrepancy_summary_statistics_replay_the_raw_data_result():
    raw = _run(YS, YT_BOTH)
    # X'X = [[4, 6], [6, 14]], X'y = [10, 19], y'y = 30 for the source; the target adds 0.5 + 0.5 x.
    source = md.Sample.from_summary("source", n=4, xtx=[[4, 6], [6, 14]], xty=[10, 19], yty=30.0)
    target = md.Sample.from_summary("target", n=4, xtx=[4, 6, 6, 14], xty=[15, 29], yty=66.5)
    replay = md.diagnose_mechanism_discrepancy(
        source=source, target=target, measurement=_measurement()
    )
    assert replay.statistic == pytest.approx(raw.statistic, abs=1e-12)
    assert replay.p_value == pytest.approx(raw.p_value, abs=1e-12)
    for left, right in zip(replay.coefficients, raw.coefficients, strict=True):
        assert left.difference == pytest.approx(right.difference, abs=1e-12)
        assert left.p_holm == pytest.approx(right.p_holm, abs=1e-12)
    # An inconsistent summary (intercept sum of squares != n) refuses.
    broken = md.Sample.from_summary("target", n=4, xtx=[5, 6, 6, 14], xty=[15, 29], yty=66.5)
    _refusal(
        lambda: md.diagnose_mechanism_discrepancy(
            source=source, target=broken, measurement=_measurement()
        ),
        "invalid_argument",
        "inconsistent_summary",
    )


def test_b3_discrepancy_accepts_numpy_arrays():
    np = pytest.importorskip("numpy")
    result = md.diagnose_mechanism_discrepancy(
        source=md.Sample("source", outcome=np.array(YS), parents={"x": np.array(X)}),
        target=md.Sample("target", outcome=np.array(YT_BOTH), parents={"x": np.array(X)}),
        measurement=_measurement(),
    )
    assert result.statistic == pytest.approx(25.0 / 6.0, abs=1e-9)
    summary = md.Sample.from_summary(
        "source", n=4, xtx=np.array([[4.0, 6.0], [6.0, 14.0]]), xty=np.array([10.0, 19.0]), yty=30.0
    )
    assert summary.statistics is not None
    assert summary.statistics.xtx == (4.0, 6.0, 6.0, 14.0)


# --------------------------------------------------------------------- refusals


def test_b3_discrepancy_incomparable_measurements_refuse():
    source = _one_parent("source", YS)
    extra_column = [1.0, 0.0, 2.0, 5.0]
    cases = {
        "parent unit": md.Sample(
            "target",
            outcome=YT_BOTH,
            parents={"x": X},
            measurement=_measurement((("x", "m"),)),
        ),
        "node unit": md.Sample(
            "target", outcome=YT_BOTH, parents={"x": X}, measurement=_measurement(node_unit="g")
        ),
        "protocol": md.Sample(
            "target",
            outcome=YT_BOTH,
            parents={"x": X},
            measurement=_measurement(protocol_id="protocol-2"),
        ),
        "parent name": md.Sample(
            "target",
            outcome=YT_BOTH,
            parents={"w": X},
            measurement=_measurement((("w", "cm"),)),
        ),
        "extra parent": md.Sample(
            "target",
            outcome=YT_BOTH,
            parents={"x": X, "w": extra_column},
            measurement=_measurement((("x", "cm"), ("w", "cm"))),
        ),
        "blank unit": md.Sample(
            "target",
            outcome=YT_BOTH,
            parents={"x": X},
            measurement=_measurement((("x", ""),)),
        ),
        "other node": md.Sample(
            "target", outcome=YT_BOTH, parents={"x": X}, measurement=_measurement(node="W")
        ),
    }
    for name, target in cases.items():
        error = _refusal(
            lambda target=target: md.diagnose_mechanism_discrepancy(
                source=source, target=target, measurement=_measurement()
            ),
            "route_not_supported",
            "incomparable_measurements",
        )
        assert error.message, name


def test_b3_discrepancy_dependence_must_be_declared_independent():
    for dependence in ("unknown", "shared_units"):
        _refusal(
            lambda dependence=dependence: _run(YS, YT_BOTH, dependence=dependence),
            "route_not_supported",
            "dependence_unknown",
        )
    with pytest.raises(CausalValueError):
        _run(YS, YT_BOTH, dependence="paired")
    # A shared row identity refuses even when the declaration says independent.
    shared_source = _one_parent("source", YS, unit_ids=["u1", "u2", "u3", "u4"])
    shared_target = _one_parent("target", YT_BOTH, unit_ids=["u9", "u3", "u8", "u7"])
    _refusal(
        lambda: md.diagnose_mechanism_discrepancy(
            source=shared_source, target=shared_target, measurement=_measurement()
        ),
        "route_not_supported",
        "dependence_unknown",
    )
    # Disjoint ids are fine.
    disjoint = _one_parent("target", YT_BOTH, unit_ids=["t1", "t2", "t3", "t4"])
    result = md.diagnose_mechanism_discrepancy(
        source=shared_source, target=disjoint, measurement=_measurement()
    )
    assert result.dependence == "independent"


def test_b3_discrepancy_degenerate_inputs_refuse():
    invalid = "invalid_argument"
    source = _one_parent("source", YS)

    def against(target: md.Sample, **kwargs: object) -> md.MechanismDiscrepancyResult:
        return md.diagnose_mechanism_discrepancy(
            source=source,
            target=target,
            measurement=_measurement(),
            **kwargs,  # type: ignore[arg-type]
        )

    # Tiny sample: n = 3, p = 2 leaves one residual degree of freedom.
    tiny = md.Sample("target", outcome=[1.0, 3.0, 2.0], parents={"x": [0.0, 1.0, 2.0]})
    _refusal(lambda: against(tiny), invalid, "sample_too_small")
    # Rank deficient: a constant parent is collinear with the intercept.
    constant = md.Sample("target", outcome=YT_BOTH, parents={"x": [2.0] * 4})
    _refusal(lambda: against(constant), invalid, "rank_deficient_design")
    # Non-finite value.
    nan = md.Sample("target", outcome=[1.0, math.nan, 3.0, 4.0], parents={"x": X})
    _refusal(lambda: against(nan), invalid, "non_finite_value")
    # Row-count mismatch.
    ragged = md.Sample("target", outcome=YT_BOTH, parents={"x": [0.0, 1.0, 2.0]})
    _refusal(lambda: against(ragged), invalid, "row_count_mismatch")
    # Both populations fit exactly: the sampling variance is not estimable.
    _refusal(
        lambda: md.diagnose_mechanism_discrepancy(
            source=_one_parent("source", [0.0, 1.0, 2.0, 3.0]),
            target=_one_parent("target", [1.0, 3.0, 5.0, 7.0]),
            measurement=_measurement(),
        ),
        invalid,
        "degenerate_covariance",
    )
    # Levels.
    _refusal(lambda: against(source, alpha=1.0), invalid, "invalid_alpha")
    _refusal(lambda: against(source, power=0.3), invalid, "invalid_power")
    # A parent-free node with the intercept excluded compares nothing.
    bare = md.Measurement("V", "mg", parents=[], protocol_id="protocol-1")
    _refusal(
        lambda: md.diagnose_mechanism_discrepancy(
            source=md.Sample("source", outcome=YS, parents={}),
            target=md.Sample("target", outcome=YS, parents={}),
            measurement=bare,
            compare_intercept=False,
        ),
        invalid,
        "no_compared_coefficients",
    )


def test_b3_discrepancy_validates_its_arguments_as_typed_errors():
    measurement = _measurement()
    source = _one_parent("source", YS)
    with pytest.raises(CausalTypeError):
        md.diagnose_mechanism_discrepancy(source="s", target=source, measurement=measurement)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        md.diagnose_mechanism_discrepancy(source=source, target=source, measurement="m")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        md.diagnose_mechanism_discrepancy(
            source=source,
            target=source,
            measurement=measurement,
            compare_intercept=1,  # type: ignore[arg-type]
        )
    with pytest.raises(CausalTypeError):
        md.diagnose_mechanism_discrepancy(
            source=source, target=source, measurement=measurement, alpha="a"
        )  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        md.diagnose_mechanism_discrepancy(source=source, target=source)  # no measurement contract
    with pytest.raises(CausalValueError):
        md.diagnose_mechanism_discrepancy(
            source=source, target=source, measurement=measurement, alpha=math.nan
        )
    with pytest.raises(CausalValueError):
        md.Sample("   ", outcome=YS, parents={"x": X})
    with pytest.raises(CausalValueError):
        md.Sample("s")  # neither rows nor statistics
    with pytest.raises(CausalValueError):
        md.Sample("s", outcome=YS)  # rows need parents too
    with pytest.raises(CausalValueError):
        md.Sample(
            "s",
            outcome=YS,
            parents={"x": X},
            statistics=md.SufficientStatistics(4, [4, 6, 6, 14], [10, 19], 30.0),
        )
    with pytest.raises(CausalTypeError):
        md.Sample("s", outcome=YS, parents={"x": X}, unit_ids=[1, 2, 3, 4])  # type: ignore[list-item]
    with pytest.raises(CausalTypeError):
        md.Measurement("V", "mg", parents="x", protocol_id="p")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        md.SufficientStatistics(4.0, [4], [1], 1.0)  # type: ignore[arg-type]
    # Parent columns must match the declared parents.
    mismatch = md.Sample("target", outcome=YT_BOTH, parents={"z": X})
    with pytest.raises(CausalValueError):
        md.diagnose_mechanism_discrepancy(source=source, target=mismatch, measurement=measurement)


# ---------------------------------------------------------------------- artifact


def test_b3_discrepancy_artifact_round_trips_by_recomputation():
    result = _run(YS, YT_BOTH)
    data = result.export()
    assert isinstance(data, bytes)
    consumed = md.MechanismDiscrepancyResult.consume(data, expected_identity=result.identity)
    assert consumed.identity == result.identity
    assert consumed.statistic == result.statistic
    assert consumed.p_value == result.p_value
    assert consumed.coefficients == result.coefficients
    assert consumed.conclusion == result.conclusion
    assert consumed.non_rejection_certifies_invariance is False
    assert consumed.export() == data
    assert set(result.identity) == {
        "measurement_id",
        "source_evidence_id",
        "target_evidence_id",
        "null",
        "design_id",
        "digest",
    }
    # Without a retained identity the bytes are still recomputed.
    again = md.MechanismDiscrepancyResult.consume(data)
    assert again.identity == result.identity


def test_b3_discrepancy_artifact_refuses_a_resealed_mutation_and_corruption():
    result = _run(YS, YT_BOTH)
    other = _run(YS, YT_BIG)
    assert other.identity != result.identity
    # A consumer that retained the producer's identity refuses a different (even
    # consistently sealed) artifact: a changed summary statistic changes the identity.
    error = _refusal(
        lambda: md.MechanismDiscrepancyResult.consume(
            other.export(), expected_identity=result.identity
        ),
        "route_not_supported",
        "wrong_contract",
    )
    assert "changed" in error.message
    # Changing one retained field (a changed level, null or measurement contract) refuses too.
    forged = dict(result.identity)
    forged["design_id"] = "0" * len(forged["design_id"])
    _refusal(
        lambda: md.MechanismDiscrepancyResult.consume(result.export(), expected_identity=forged),
        "route_not_supported",
        "wrong_contract",
    )
    # Corruption and truncation do not decode.
    data = result.export()
    flipped = data[:-1] + bytes([data[-1] ^ 0xFF])
    for bad in (flipped, data[: len(data) // 2], b"not an artifact"):
        with pytest.raises(md.MechanismDiscrepancyRefusal) as caught:
            md.MechanismDiscrepancyResult.consume(bad)
        assert caught.value.detail.startswith("mechanism_discrepancy.")
    with pytest.raises(CausalTypeError):
        md.MechanismDiscrepancyResult.consume("bytes")  # type: ignore[arg-type]
