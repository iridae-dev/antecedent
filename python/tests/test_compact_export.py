"""B4 compact runtime export through the Python facade.

Hand values mirror the Rust core tests. Quadratic model ``y = 1 + 2 x - 0.5 x^2`` on ``x`` in
``[0, 10]`` with ``V = [[0.04, 0, -0.002], [0, 0.01, 0], [-0.002, 0, 0.0004]]``; one-hot model
``y = 1 + 2 x + 0.5 [g = b] - 1 [g = c]`` refusing ``x`` in ``[8, 10]`` with ``g = c``. The only
uncertainty the export reports is the model-based ``sqrt(phi' V phi)``; it is never an interval
and its calibration is unmeasured.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import compact_export as ce
from antecedent.compact_export import (
    Categorical,
    CompactExport,
    CompactExportRefusal,
    Intercept,
    Linear,
    MaskRegion,
    Numeric,
    OneHot,
    PointEstimate,
    Power,
    Quantity,
)
from antecedent.errors import CausalTypeError, CausalUnsupportedError

from _refusal import assert_registered_refusal

RESPONSE = Quantity("y", units="mmHg", role="outcome")
DOSE = Numeric("x", 0.0, 10.0, units="mg", role="treatment")
# Deliberately unsorted: build canonicalises.
GROUP = Categorical("g", ["c", "a", "b"])


def quadratic() -> CompactExport:
    return CompactExport.build(
        response=RESPONSE,
        inputs=[DOSE],
        terms=[Intercept(), Linear("x"), Power("x", 2)],
        coefficients=[1.0, 2.0, -0.5],
        covariance=[[0.04, 0.0, -0.002], [0.0, 0.01, 0.0], [-0.002, 0.0, 0.0004]],
    )


def one_hot() -> CompactExport:
    return CompactExport.build(
        response=RESPONSE,
        inputs=[DOSE, GROUP],
        terms=[Intercept(), Linear("x"), OneHot("g", "b"), OneHot("g", "c")],
        coefficients=[1.0, 2.0, 0.5, -1.0],
        covariance=np.diag([0.04, 0.01, 0.09, 0.16])
        + np.array([[0, 0, 0.01, 0], [0, 0, 0, 0], [0.01, 0, 0, 0], [0, 0, 0, 0]]),
        mask=[MaskRegion("high_dose_group_c", {"x": (8.0, 10.0), "g": {"c"}})],
    )


def refused(export: CompactExport, query: dict) -> CompactExportRefusal:
    with pytest.raises(CompactExportRefusal) as error:
        export.evaluate(query)
    assert isinstance(error.value, CausalUnsupportedError)
    assert_registered_refusal(error.value)
    return error.value


def test_b4_compact_quadratic_point_and_se_match_hand_values() -> None:
    export = quadratic()
    # x = 2: phi = (1, 2, 4); b'phi = 1 + 4 - 2 = 3; phi'V phi = 0.0704 = 0.0064 * 11.
    at_two = export.evaluate({"x": 2.0})
    assert isinstance(at_two, PointEstimate)
    assert at_two.point == pytest.approx(3.0, abs=1e-12)
    assert at_two.standard_error == pytest.approx(0.08 * math.sqrt(11.0), abs=1e-12)
    assert at_two.se == at_two.standard_error
    assert export.evaluate({"x": 0.0}).point == pytest.approx(1.0, abs=1e-12)
    assert export.evaluate({"x": 0.0}).standard_error == pytest.approx(0.2, abs=1e-12)
    # x = 10 (closed support bound): 1 + 20 - 50 = -29; phi'V phi = 4.64.
    at_ten = export.evaluate({"x": 10})
    assert at_ten.point == pytest.approx(-29.0, abs=1e-12)
    assert at_ten.standard_error == pytest.approx(math.sqrt(4.64), abs=1e-9)


def test_b4_compact_scope_states_point_and_model_based_se_only() -> None:
    export = quadratic()
    result = export.evaluate({"x": 2.0})
    assert result.se_basis == "model_based_sqrt_phi_v_phi"
    assert result.calibration == "calibration unmeasured"
    assert result.export_identity == export.identity
    scope = export.body["scope"]
    assert scope["point"].startswith("point prediction")
    assert "model-based standard error" in scope["uncertainty"]
    assert "not an interval" in scope["uncertainty"]
    assert scope["calibration"] == "calibration unmeasured"
    assert len(export.identity) == 64
    assert not any("interval" in field for field in PointEstimate.__slots__)


def test_b4_compact_one_hot_terms_select_the_level_with_hand_values() -> None:
    export = one_hot()
    b = export.evaluate({"x": 1.0, "g": "b"})  # phi = (1, 1, 1, 0); 0.04 + 0.01 + 0.09 + 0.02
    assert b.point == pytest.approx(3.5, abs=1e-12)
    assert b.standard_error == pytest.approx(0.4, abs=1e-12)
    c = export.evaluate({"x": 1.0, "g": "c"})  # phi = (1, 1, 0, 1); 0.04 + 0.01 + 0.16
    assert c.point == pytest.approx(2.0, abs=1e-12)
    assert c.standard_error == pytest.approx(math.sqrt(0.21), abs=1e-12)
    a = export.evaluate({"x": 1.0, "g": "a"})  # the reference level: 0.04 + 0.01
    assert a.point == pytest.approx(3.0, abs=1e-12)
    assert a.standard_error == pytest.approx(math.sqrt(0.05), abs=1e-12)
    # Levels were canonicalised even though supplied as c, a, b.
    group = next(i for i in export.body["inputs"] if i["quantity"]["variable_id"] == "g")
    assert group["support"] == {"levels": {"levels": ["a", "b", "c"]}}


def test_b4_compact_out_of_support_queries_refuse_with_exact_details() -> None:
    export = one_hot()
    expected = [
        ({"x": 10.5, "g": "a"}, "compact_export.out_of_support", "x"),
        ({"x": -0.001, "g": "a"}, "compact_export.out_of_support", "x"),
        ({"x": 1.0, "g": "d"}, "compact_export.out_of_support", "g"),
        ({"x": 1.0}, "compact_export.missing_quantity", "g"),
        ({"x": 1.0, "g": "a", "z": 3.0}, "compact_export.unknown_quantity", "z"),
        ({"x": float("nan"), "g": "a"}, "compact_export.non_finite_input", "x"),
        ({"x": float("inf"), "g": "a"}, "compact_export.non_finite_input", "x"),
        ({"x": "a", "g": "a"}, "compact_export.query_type_mismatch", "x"),
    ]
    for query, detail, subject in expected:
        error = refused(export, query)
        assert error.reason_code == "cell_not_licensed", query
        assert error.detail == detail, query
        assert error.offending == subject, query
    # The closed bounds themselves are inside the support.
    export.evaluate({"x": 0.0, "g": "a"})
    export.evaluate({"x": 10.0, "g": "a"})
    with pytest.raises(CausalTypeError):
        export.evaluate({"x": True, "g": "a"})


def test_b4_compact_masked_region_refuses_even_inside_the_support() -> None:
    export = one_hot()
    for x in (8.0, 9.0, 10.0):
        error = refused(export, {"x": x, "g": "c"})
        assert error.detail == "compact_export.masked_region"
        assert error.offending == "high_dose_group_c"
        assert error.reason_code == "cell_not_licensed"
    export.evaluate({"x": 7.999, "g": "c"})
    export.evaluate({"x": 9.0, "g": "b"})


def test_b4_compact_evaluate_many_returns_refusals_without_raising() -> None:
    export = one_hot()
    answers = export.evaluate_many(
        [{"x": 1.0, "g": "b"}, {"x": 9.0, "g": "c"}, {"x": 11.0, "g": "a"}]
    )
    assert isinstance(answers[0], PointEstimate)
    assert isinstance(answers[1], CompactExportRefusal)
    assert answers[1].detail == "compact_export.masked_region"
    assert isinstance(answers[2], CompactExportRefusal)
    assert answers[2].detail == "compact_export.out_of_support"


def test_b4_compact_verifier_round_trip_matches_the_producer() -> None:
    export = one_hot()
    consumed = CompactExport.consume(export.export(), expected_identity=export.identity)
    assert consumed.identity == export.identity
    assert consumed.body == export.body
    for query in ({"x": 1.0, "g": "a"}, {"x": 3.0, "g": "b"}, {"x": 7.5, "g": "c"}):
        assert consumed.evaluate(query) == export.evaluate(query)
    assert refused(consumed, {"x": 9.0, "g": "c"}).detail == "compact_export.masked_region"
    assert refused(consumed, {"x": 11.0, "g": "a"}).detail == "compact_export.out_of_support"


def test_b4_compact_wrong_expected_identity_is_refused() -> None:
    export = quadratic()
    with pytest.raises(CompactExportRefusal) as wrong:
        CompactExport.consume(export.export(), expected_identity="deadbeef")
    assert wrong.value.reason_code == "invalid_argument"
    assert wrong.value.detail == "compact_export.identity_unexpected"
    assert wrong.value.offending == "deadbeef"
    # A different model has a different identity and is refused under the first.
    other = CompactExport.build(
        response=RESPONSE,
        inputs=[DOSE],
        terms=[Intercept(), Linear("x"), Power("x", 2)],
        coefficients=[1.0 + 1e-9, 2.0, -0.5],
        covariance=[[0.04, 0.0, -0.002], [0.0, 0.01, 0.0], [-0.002, 0.0, 0.0004]],
    )
    assert other.identity != export.identity
    with pytest.raises(CompactExportRefusal) as swapped:
        CompactExport.consume(other.export(), expected_identity=export.identity)
    assert swapped.value.detail == "compact_export.identity_unexpected"
    # An export that evaluates under a swapped artifact keeps refusing: the retained identity
    # binds every evaluation.
    forged = CompactExport(identity=export.identity, body=other.body, artifact=other.export())
    with pytest.raises(CompactExportRefusal) as evaluated:
        forged.evaluate({"x": 2.0})
    assert evaluated.value.detail == "compact_export.identity_unexpected"


def test_b4_compact_corrupt_or_truncated_artifact_is_refused() -> None:
    export = quadratic()
    data = bytearray(export.export())
    data[len(data) // 2] ^= 0xFF
    with pytest.raises(CompactExportRefusal) as corrupt:
        CompactExport.consume(bytes(data), expected_identity=export.identity)
    assert corrupt.value.detail.startswith("compact_export.")
    assert corrupt.value.reason_code == "invalid_argument"
    with pytest.raises(CompactExportRefusal):
        CompactExport.consume(export.export()[:-5], expected_identity=export.identity)
    # The untouched artifact still verifies.
    assert CompactExport.consume(export.export(), expected_identity=export.identity)
    with pytest.raises(CausalTypeError):
        CompactExport.consume("not bytes", expected_identity="x")  # type: ignore[arg-type]


def test_b4_compact_malformed_specs_refuse_at_build() -> None:
    def build(**overrides):
        spec = {
            "response": RESPONSE,
            "inputs": [DOSE],
            "terms": [Intercept(), Linear("x")],
            "coefficients": [1.0, 2.0],
            "covariance": [[0.04, 0.0], [0.0, 0.01]],
        }
        spec.update(overrides)
        return CompactExport.build(**spec)

    assert build().identity
    cases = [
        (
            {"covariance": [[0.04, 0.0, 0.0], [0.0, 0.01, 0.0]]},
            "compact_export.dimension_mismatch",
        ),
        ({"covariance": [[0.04, 0.05], [0.0, 0.01]]}, "compact_export.covariance_asymmetric"),
        ({"covariance": [[-0.04, 0.0], [0.0, 0.01]]}, "compact_export.covariance_not_psd"),
        ({"covariance": [[0.04, 0.1], [0.1, 0.01]]}, "compact_export.covariance_not_psd"),
        ({"terms": [Intercept(), Linear("q")]}, "compact_export.unknown_quantity"),
        ({"terms": [Intercept(), Power("x", 9)]}, "compact_export.invalid_degree"),
        ({"terms": [Intercept(), Intercept()]}, "compact_export.duplicate_term"),
        ({"terms": [Intercept(), OneHot("x", "a")]}, "compact_export.term_support_mismatch"),
        ({"inputs": [Numeric("x", 5.0, 1.0)]}, "compact_export.invalid_support"),
        (
            {"mask": [MaskRegion("r", {"q": (0.0, 1.0)})]},
            "compact_export.unknown_quantity",
        ),
        (
            {"coefficients": [1.0, float("nan")]},
            "compact_export.non_finite_value",
        ),
    ]
    for overrides, detail in cases:
        with pytest.raises(CompactExportRefusal) as error:
            build(**overrides)
        assert error.value.detail == detail, overrides
        assert error.value.reason_code == "invalid_argument", overrides
        assert_registered_refusal(error.value)


def test_b4_compact_input_validation_is_typed() -> None:
    with pytest.raises(CausalTypeError):
        CompactExport.build(
            response="y",  # type: ignore[arg-type]
            inputs=[DOSE],
            terms=[Intercept()],
            coefficients=[1.0],
            covariance=[[1.0]],
        )
    with pytest.raises(CausalTypeError):
        Power("x", 2.5)._wire()  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        CompactExport.build(
            response=RESPONSE,
            inputs=["x"],  # type: ignore[list-item]
            terms=[Intercept()],
            coefficients=[1.0],
            covariance=[[1.0]],
        )
    assert ce.__doc__ is not None
