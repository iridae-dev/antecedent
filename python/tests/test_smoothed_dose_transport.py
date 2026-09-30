"""Smoothed dose-response transport grid (2.2B cell X4)."""

import dataclasses

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import (
    CausalResourceError,
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced

DESIGNS = ("nested_cohort", "independent_samples")
H = 0.5
GRID = (1.0, 2.0, 3.0)


def density(a, z):
    """The known conditional dose density on [0, 4]."""
    return 0.25 * (1.0 + 0.6 * np.tanh(z) * (a - 2.0) / 2.0)


def truth(a, h=H, mean=0.6):
    """psi_h(a) = a^2 + h^2/5 + m_T (1 + a/2) for y = a^2 + z (1 + a/2) + e."""
    return a * a + h * h / 5.0 + mean * (1.0 + a / 2.0)


def fixture(
    design="independent_samples",
    shift=0.6,
    n_trial=2000,
    n_target=1500,
    seed=3,
    grid=GRID,
    provenance="known",
):
    rng = np.random.default_rng(seed)
    total = n_trial + n_target
    if design == "nested_cohort":
        source = rng.random(total) < n_trial / total
    else:
        source = np.arange(total) < n_trial
    z = np.where(source, rng.normal(size=total), shift + rng.normal(size=total))
    # Exact draws from pi(. | z) by rejection from the uniform law on [0, 4].
    a = np.empty(total)
    pending = np.arange(total)
    while pending.size:
        candidate = 4.0 * rng.random(pending.size)
        accept = rng.random(pending.size) * 0.4 < density(candidate, z[pending])
        a[pending[accept]] = candidate[accept]
        pending = pending[~accept]
    y = a * a + z * (1.0 + a / 2.0) + rng.normal(size=total)
    data = advanced.SmoothedDoseData(
        {"z": z},
        np.where(source, y, 0.0),
        np.where(source, a, 0.0),
        np.where(source, density(a, z), 0.0),
        [bool(v) for v in source],
        design,
    )
    graph = ac.Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    query = advanced.SmoothedDoseQuery(
        graph,
        advanced.SelectionDiagram("trial", "target", ["z"]),
        "a",
        "y",
        grid,
        H,
        (0.0, 4.0),
        density_provenance=provenance,
    )
    return query, data


def options(**kwargs):
    return advanced.SmoothedDoseOptions(folds=3, **kwargs)


@pytest.mark.parametrize("design", DESIGNS)
def test_known_truth_grid_under_each_design(design):
    query, data = fixture(design)
    result = advanced.prepare_smoothed_dose(query, data, options=options(), seed=5).estimate()
    assert [p.dose for p in result.grid] == list(GRID)
    for point in result.grid:
        assert point.estimate == pytest.approx(truth(point.dose), abs=0.3)
        assert point.quadrature["estimate_error"] < 1e-9
        assert point.estimate == pytest.approx(point.plug_in + point.augmentation, abs=1e-12)
    assert result.interval is None
    assert result.uncertainty["status"] == "point_only"
    assert result.sampling == design
    assert result.query["bandwidth"] == H and result.query["kernel"] == "epanechnikov"
    assert len(result.provenance) == 6
    assert result.folds["scheme"] == "stratified_round_robin_source_and_target"
    assert result.certificate["formula"] == "standardize"


def test_prepare_smoothed_dose_retains_a_checked_plan_after_builder_disposal():
    builder_query, builder_data = fixture()
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    del builder_query, builder_data
    # The retained plan holds the certificate and the query: estimating never identifies.
    plan = study.estimate()
    assert plan.certificate["formula"] == "standardize"
    assert plan.certificate["over"] == (0,)
    assert plan.query["grid"] == GRID
    assert plan[2.0].estimate == pytest.approx(truth(2.0), abs=0.3)


@pytest.mark.parametrize("design", DESIGNS)
def test_estimate_executes_the_retained_plan_after_builder_disposal(design):
    builder_query, builder_data = fixture(design)
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    del builder_query, builder_data
    plan = study.estimate()
    assert study.estimate().execution_id == plan.execution_id
    assert plan.sampling == design
    assert all(
        p.smoothing_bias["local_quadratic_bias"] == pytest.approx(H * H / 5, abs=0.05)
        for p in plan.grid
    )


def test_refresh_rebinds_rows_and_re_executes_the_retained_plan():
    builder_query, builder_data = fixture()
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    first = study.estimate()
    del builder_query, builder_data
    _, fresh = fixture(seed=9)
    study.refresh(fresh)
    plan = study.estimate()
    assert plan.execution_id != first.execution_id
    assert plan.premises_digest == first.premises_digest
    assert plan.certificate == first.certificate
    with pytest.raises((ValueError, CausalUnsupportedError), match="reprepare_required"):
        study.refresh(dataclasses.replace(fresh, sampling="nested_cohort"))
    with pytest.raises(TypeError, match="refresh requires SmoothedDoseData"):
        study.refresh(object())


@pytest.mark.parametrize("design", DESIGNS)
def test_prepare_once_estimate_export_and_independent_consume_after_builder_disposal(design):
    builder_query, builder_data = fixture(design)
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    del builder_query, builder_data
    result = study.estimate()
    artifact = study.export()
    plan = advanced.consume_smoothed_dose(artifact)
    assert plan.grid == result.grid
    assert plan.execution_id == result.execution_id
    assert plan.premises_digest == result.premises_digest
    assert plan.evidence_digest == result.evidence_digest
    assert plan.variable_names == ("z", "a", "y")
    assert plan.provenance == result.provenance


def test_consumer_replays_the_grid_after_builder_disposal():
    builder_query, builder_data = fixture()
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    result = study.estimate()
    artifact = study.export()
    del builder_query, builder_data, study
    # The artifact alone (certificate, rows, fold models) replays the plan's grid.
    plan = advanced.consume_smoothed_dose(artifact)
    assert [p.estimate for p in plan.grid] == [p.estimate for p in result.grid]
    assert plan.folds == result.folds
    assert plan.to_dict()["interval"] is None


def test_smoothed_dose_estimator_menu_reads_the_retained_plan_after_builder_disposal():
    builder_query, builder_data = fixture()
    menu = advanced.smoothed_dose_estimator_menu(builder_query)
    assert menu.selection == "manual"
    assert menu.eligible == ("smoothed_dose_transport_aipw",)
    assert (
        menu["kennedy_local_linear_point_curve"].refusal["detail"]
        == "dose_response.target_not_smoothed"
    )
    assert (
        menu["generalized_propensity_estimated_density"].refusal["detail"]
        == "dose_response.estimated_dose_density"
    )
    assert not any("recommend" in repr(entry).lower() for entry in menu.entries)
    tuned = advanced.smoothed_dose_estimator_menu(
        builder_query, options=options(quadrature_nodes=32)
    )
    assert any("32-node" in s for s in tuned["smoothed_dose_transport_aipw"].support_requirements)
    study = advanced.prepare_smoothed_dose(builder_query, builder_data, options=options(), seed=7)
    del builder_query, builder_data
    # The retained plan carries the certificate, query and options the menu is derived from.
    plan = study.estimator_menu()
    assert plan["smoothed_dose_transport_aipw"].refusal is None
    assert plan["smoothed_dose_transport_aipw"].sampling_design == ("independent_samples",)
    assert study.estimate()[2.0].estimate == pytest.approx(truth(2.0), abs=0.3)


def test_refusals_carry_their_reason_codes():
    query, data = fixture()
    with pytest.raises(
        CausalUnsupportedError, match="dose_response.grid_outside_dose_support"
    ) as info:
        advanced.prepare_smoothed_dose(
            dataclasses.replace(query, grid=(3.8,)), data, options=options()
        )
    assert info.value.reason_code == "transport_support_failure"
    with pytest.raises(
        CausalUnsupportedError, match="dose_response.estimated_dose_density"
    ) as info:
        advanced.prepare_smoothed_dose(
            dataclasses.replace(query, density_provenance="estimated"), data, options=options()
        )
    assert info.value.reason_code == "route_not_supported"
    for target, detail in (
        ("point_curve", "dose_response.target_not_smoothed"),
        ("derivative", "dose_response.target_not_smoothed"),
        ("simultaneous_band", "dose_response.cate_or_simultaneous"),
    ):
        with pytest.raises(CausalUnsupportedError, match=detail) as info:
            advanced.prepare_smoothed_dose(query, data, options=options(), target=target)
        assert info.value.reason_code == "route_not_supported"
    with pytest.raises(CausalUnsupportedError, match="dose_response.kernel_not_supported"):
        advanced.prepare_smoothed_dose(
            dataclasses.replace(query, kernel="gaussian"), data, options=options()
        )
    with pytest.raises(CausalUnsupportedError, match="dose_response.non_iid_design") as info:
        advanced.prepare_smoothed_dose(
            query, dataclasses.replace(data, sampling="clustered"), options=options()
        )
    assert info.value.reason_code == "sampling_dependence_unknown"
    with pytest.raises(CausalUnsupportedError, match="dose_response.bounds_exceeded"):
        advanced.prepare_smoothed_dose(query, data, options=options(quadrature_nodes=64))
    with pytest.raises(CausalValueError):
        advanced.SmoothedDoseOptions(min_membership_probability=0.7)
    # Weak membership overlap refuses rather than extrapolating.
    far_query, far = fixture(shift=3.0)
    study = advanced.prepare_smoothed_dose(far_query, far, options=options())
    with pytest.raises(CausalUnsupportedError, match="dose_response.membership_overlap") as info:
        study.estimate()
    assert info.value.reason_code == "transport_support_failure"
    # Below the replicate floor the point is kept and the interval withheld.
    below = advanced.prepare_smoothed_dose(
        query, data, options=options(bootstrap=50), seed=2
    ).estimate()
    assert below.uncertainty["detail"] == "dose_response.bootstrap_below_floor"


def test_a_tampered_artifact_or_relabelled_names_fail_consumption_with_typed_errors():
    query, data = fixture()
    study = advanced.prepare_smoothed_dose(query, data, options=options(), seed=7)
    with pytest.raises(CausalUnsupportedError, match="estimate before exporting"):
        study.export()
    study.estimate()
    artifact = study.export()
    with pytest.raises(CausalSerializationError):
        advanced.consume_smoothed_dose(artifact[:-40])
    with pytest.raises(CausalSerializationError):
        advanced.consume_smoothed_dose(b"not an artifact")
    tampered = bytearray(artifact)
    for offset in (len(artifact) // 2, len(artifact) // 3, len(artifact) - 200):
        tampered[offset] ^= 0x55
    with pytest.raises(CausalSerializationError):
        advanced.consume_smoothed_dose(bytes(tampered))
    with pytest.raises(CausalResourceError, match="row count"):
        advanced.consume_smoothed_dose(artifact, max_rows=10)
    # A foreign learned-continuous frame is not a smoothed-dose artifact.
    with pytest.raises(CausalSerializationError, match="smoothed dose"):
        advanced.consume_smoothed_dose(b"ANTECEDENT-LEARNED-CONTINUOUS\x01" + artifact[25:])


def test_the_unmeasured_interval_route_refuses_with_cell_not_licensed():
    query, data = fixture()
    study = advanced.prepare_smoothed_dose(query, data, options=options(bootstrap=199), seed=4)
    with pytest.raises(CausalUnsupportedError, match="dose_response.interval_withheld") as info:
        study.interval()
    assert info.value.reason_code == "cell_not_licensed"
    result = study.estimate()
    assert result.uncertainty["status"] == "withheld"
    assert result.uncertainty["reason"] == "cell_not_licensed"
    assert result.interval is None
    consumed = advanced.consume_smoothed_dose(study.export())
    assert consumed.uncertainty == result.uncertainty
    assert consumed.interval is None
