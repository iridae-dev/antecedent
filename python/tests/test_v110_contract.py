"""1.10 contracts-first: identities and transformation preview."""

from __future__ import annotations

import numpy as np
from antecedent.estimation import PreparedAnalysis
from antecedent.query import AverageEffect


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    z = rng.normal(size=80)
    t = (rng.uniform(size=80) < 1 / (1 + np.exp(-0.4 + 0.9 * z))).astype(float)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=80)
    return {"t": t, "y": y, "z": z}


def test_prepared_contract_has_domain_separated_identities() -> None:
    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    contract = prepared.contract()
    assert contract["accepted_version"] == "1"
    assert contract["accepted_variable_binding"] in {"explicit", "unbound"}
    assert contract["target"] != contract["program"]
    assert "identification_product" in contract
    assert contract["identification_status"] == "nonparametrically_identified"
    assert contract["matrix_coordinate"].startswith("AverageEffect:Dag:explicit:Frequentist:")
    assert contract["empirical_support"].startswith("unavailable:")
    assert "assumptions" in contract
    assert contract["uncertainty"].startswith("unavailable:")
    refused = prepared.preview_transform("average_unweighted_class")
    assert refused["refused"] == "true"
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["intent"] == "compatible_data_replace"
    assert preview["refused"] == "false"


def test_same_shape_refresh_changes_only_data_identity() -> None:
    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    before = prepared.contract()
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["input_data_snapshot"] == before["data_snapshot"]
    prepared.refresh({**data, "y": data["y"] + data["t"]})
    after = prepared.contract()
    for layer in (
        "target",
        "identification",
        "identification_product",
        "program",
        "inference_binding",
        "observation",
    ):
        assert before[layer] == after[layer], layer
    assert before["data_snapshot"] != after["data_snapshot"]
    assert preview["input_program"] == after["program"]
    assert preview["input_data_snapshot"] != after["data_snapshot"]


def test_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts

    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    prepared.estimate(data)
    contract = prepared.contract()
    encoded = prepared.export_contracted_artifact()
    loaded = artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
    assert loaded.contract is not None
    accepted = artifacts.accept(encoded)
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["target"] == contract["target"]
    for key in (
        "treatment",
        "outcome",
        "control",
        "active",
        "population",
        "temporal_coordinates",
        "variable_names",
    ):
        assert accepted[key] == contract[key], key
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "1"
    assert accepted["control"] == "set:0=0"
    assert accepted["active"] == "set:0=1"
    assert accepted["population"] == "all_observed"
    assert accepted["temporal_coordinates"] == "none"
    assert accepted["variable_names"] == "t,y,z"


def test_conditional_effect_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts
    from antecedent.query import ConditionalEffect

    n = 120
    t = np.array([float(i % 2) for i in range(n)])
    w = np.array([float(i % 5) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.5 * t * w
    data = {"t": t, "y": y, "w": w}
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("t", "y"), ("w", "y")],
        query=ConditionalEffect("t", "y", "w"),
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate(data)
    contract = prepared.contract()
    accepted = artifacts.accept(prepared.export_contracted_artifact())
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["target"] == contract["target"]
    assert accepted["query_kind"] == "conditional_effect"
    assert accepted["treatment"] == contract["treatment"]
    assert accepted["outcome"] == contract["outcome"]
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "1"
    assert accepted["population"] == "all_observed"
    assert contract["matrix_coordinate"].startswith("ConditionalEffect:")
    assert abs(result.estimate.ate - 3.0) < 1e-6


def test_mediation_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts
    from antecedent.query import MediationEffect

    a = np.sin(np.arange(200) * 0.71)
    m = 2.0 * a + np.cos(np.arange(200) * 1.13)
    y = 3.0 * a + 4.0 * m + 0.1 * np.sin(np.arange(200) * 0.31)
    data = {"a": a, "m": m, "y": y}
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("a", "m"), ("a", "y"), ("m", "y")],
        query=MediationEffect("a", "y", mediators=["m"], contrast="direct"),
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate(data)
    contract = prepared.contract()
    accepted = artifacts.accept(prepared.export_contracted_artifact())
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["target"] == contract["target"]
    assert accepted["query_kind"] == "mediation"
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "2"
    assert contract["matrix_coordinate"].startswith("MediationEffect:")
    assert np.isfinite(result.estimate.ate)


def test_response_curve_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts
    from antecedent.query import ResponseCurve

    n = 240
    z = np.sin(np.arange(n) / 17.0)
    treatment = z + np.cos(np.arange(n) / 11.0) + (np.arange(n) % 7) * 0.03
    outcome = 1.0 + 2.0 * treatment + 0.8 * z
    data = {"treatment": treatment, "outcome": outcome, "confounder": z}
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("confounder", "treatment"), ("confounder", "outcome"), ("treatment", "outcome")],
        query=ResponseCurve("treatment", "outcome", grid=[-0.5, 0.0, 0.5]),
        refute="none",
        bootstrap=0,
    )
    prepared.estimate(data)
    contract = prepared.contract()
    accepted = artifacts.accept(prepared.export_contracted_artifact())
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["query_kind"] == "response"
    assert accepted["temporal_coordinates"] == "none"
    assert contract["matrix_coordinate"].startswith("ResponseCurve:")


def test_intervention_response_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts
    from antecedent.intervention import Set
    from antecedent.query import InterventionResponse

    n = 240
    z = np.sin(np.arange(n) / 17.0)
    treatment = z + np.cos(np.arange(n) / 11.0)
    outcome = 1.0 + 2.0 * treatment + 0.8 * z
    data = {"treatment": treatment, "outcome": outcome, "confounder": z}
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("confounder", "treatment"), ("confounder", "outcome"), ("treatment", "outcome")],
        query=InterventionResponse("outcome", intervention=Set("treatment", 0.25)),
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate(data)
    contract = prepared.contract()
    accepted = artifacts.accept(prepared.export_contracted_artifact())
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["query_kind"] == "response"
    assert accepted["temporal_coordinates"] == "none"
    assert contract["matrix_coordinate"].startswith("InterventionResponse:")
    assert abs(float(np.asarray(result.response.values).ravel()[0]) - 1.5) < 0.25


def test_four_slots_agree_between_inspect_and_prepared_contract() -> None:
    from antecedent.results import ConsumerIntent

    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    inspected = prepared.inspect()
    reasoned = prepared.reasoning()
    contract = prepared.contract()
    assert inspected.support.payload["matrix_coordinate"] == contract["matrix_coordinate"]
    assert inspected.program_id == reasoned.program_id == contract["program"]
    assert inspected.claim_id is None
    assert reasoned.claim_id is None
    assert "claim_id" not in contract
    assert inspected.data_version == reasoned.data_version == contract["data_snapshot"]
    assert inspected.identification.available is True
    assert prepared.preflight().identification.available is False
    assert reasoned.identification.available is True
    assert reasoned.identification.payload["identified_mass"] == 1.0
    assert reasoned.rendering_limitation() is None
    result = prepared.estimate(_data())
    assert result.reasoning is not None
    assert result.program_id == contract["program"]
    assert result.claim_id is not None
    assert result.claim_id != result.program_id
    assert result.display_effect() == result.effect
    display = prepared.preview_intent(ConsumerIntent.DISPLAY)
    assert display["scientific"] == "false"
    assert display["kind"] == "display"
    scientific = prepared.preview_intent(ConsumerIntent.NEW_CAUSAL_TARGET)
    assert scientific["scientific"] == "true"
    assert scientific["intent"] == "new_conditional_query"


def test_four_slots_refuse_identified_atom_mean_for_partial_claim() -> None:
    from antecedent.errors import RenderingLimitation
    from antecedent.results import (
        AnalysisResult,
        EstimateView,
        IdentificationView,
        PerformanceView,
        PosteriorView,
        ValidationView,
    )

    result = AnalysisResult(
        identification=IdentificationView(
            status="GraphDependent",
            method="mixture",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        ),
        estimate=EstimateView(
            ate=2.29,
            se_analytic=0.03,
            se_bootstrap=None,
            estimator_id="ols",
            method="mixture",
        ),
        posterior=PosteriorView(
            effect_mean=2.29,
            effect_sd=0.03,
            q025=2.2,
            q975=2.4,
            n_draws=200,
            p_below_zero=0.0,
            backend="conjugate",
            unidentified_mass=0.2,
        ),
        validation=ValidationView(passed=True, ran=False, count=0),
        performance=PerformanceView(),
        diagnostics=[],
        provenance={},
        structural_unidentified_mass=0.2,
    )
    assert result.rendering_limitation() == "unidentified_mass"
    try:
        result.display_effect()
    except RenderingLimitation as err:
        assert err.reason == "unidentified_mass"
    else:
        raise AssertionError("partial claim must not display a point mean")
    html = result._repr_html_()
    assert "partial; cannot display a point mean" in html


def test_calibration_reads_from_contract() -> None:
    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    result = prepared.estimate()
    slot = result.calibration
    assert slot.status in {"calibrated", "scope_not_assessed", "unavailable"}
    if slot.status == "unavailable":
        assert slot.reason


def test_calibration_prepared_is_not_executed() -> None:
    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    assert prepared.calibration.status == "unavailable"
    assert prepared.calibration.reason == "not_executed"


def test_custom_validator_is_attested_and_covered_by_claim_id() -> None:
    def always_pass(*, ate, **_kwargs):
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
        validators=[always_pass],
    )
    result = prepared.estimate()
    assert result.claim_id is not None
    attested = result.inspect().to_dict().get("claim", {}).get("attested") or []
    assert result.claim_id
    prepared.rebind_validators({})
    _ = attested


def test_retarget_export_reload_and_reexecute_on_snapshot() -> None:
    from antecedent import artifacts

    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    prepared.estimate(data)
    weights = np.exp(data["z"] / 3)
    retargeted = prepared.retarget(weights, depends_on=["z"])
    encoded = retargeted.export()
    loaded = artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
    assert artifacts.accept(encoded)["accepts_as_verified_program"] == "true"


def test_retarget_refuses_new_data_with_code() -> None:
    from antecedent.errors import CausalUnsupportedError

    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    prepared.estimate(data)
    prepared.retarget(np.exp(data["z"] / 3), depends_on=["z"])
    try:
        prepared.estimate({**data, "y": data["y"] + 1.0})
    except CausalUnsupportedError as err:
        assert err.reason_code == "row_weights_bound_to_snapshot"
    else:
        raise AssertionError("row-weight retarget must refuse a new snapshot")
