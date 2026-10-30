"""Independent diagnostic measurements and fit-free lifecycle integration.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

import dataclasses
import json

import antecedent as ant
import numpy as np
import pytest
from antecedent.data_diagnostics import DataDiagnostics, diagnose_data
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success


def data(n=200):
    t = np.arange(n, dtype=float) % 2
    return {"t": t, "y": 2 * t + np.sin(np.arange(n))}


def prepare(*, diagnostics=True, rows=None):
    return ant.prepare(
        data() if rows is None else rows,
        graph=[("t", "y")],
        query=ant.AverageEffect(treatment="t", outcome="y"),
        estimator="linear.adjustment.ate",
        refute=False,
        data_diagnostics=diagnostics,
    )


def test_standalone_population_moments_and_immutable_report():
    report = diagnose_data({"x": np.array([-2.0, -1.0, 0.0, 1.0, 2.0])})
    assert report.columns[0].skewness == pytest.approx(0, abs=1e-14)
    assert report.columns[0].excess_kurtosis == pytest.approx(-1.3)
    assert report.rows_complete == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.rows_total = 10
    with pytest.raises(TypeError, match="does not support item assignment"):
        report.context["new"] = 1
    assert json.loads(report.to_json())["identity"] == report.identity


def test_automatic_report_optout_explicit_evaluation_and_contract_invariance():
    a = prepare()
    b = prepare(diagnostics=False)
    assert a.data_diagnostics.rows_total == 200
    assert b.data_diagnostics is None
    assert a.data_diagnostics.identity == require_success(b.diagnose_data()).identity
    assert a.data_diagnostics.context["roles"]["t"] == "treatment"
    assert a.inspect().target_id == b.inspect().target_id
    assert a.inspect().program_id == b.inspect().program_id
    assert "data_reviews=" in repr(a)
    assert require_success(a.estimate()).ate == require_success(b.estimate()).ate


def test_rebind_updates_evidence_without_changing_original():
    a = prepare()
    b = a.rebind(data=data(100))
    assert a.data_diagnostics.rows_total == 200
    assert b.data_diagnostics.rows_total == 100
    assert a.data_diagnostics.identity != b.data_diagnostics.identity


def test_bounds_full_scan_counts_and_full_table_optin():
    values = np.arange(10000, dtype=float)
    values[5000:] = np.nan
    report = diagnose_data({"x": values})
    assert report.columns[0].non_finite == 5000
    assert report.sample_rows == 4096
    assert report.rows_complete == 5000
    full = diagnose_data({"x": values}, suite=DataDiagnostics(max_sample_rows=None))
    assert full.sample_rows == 10000
    assert full.identity != report.identity
    assert all(not c.sampled for c in full.checks)


def test_repeated_measurements_low_resolution_and_typed_equality():
    values = np.arange(2000, dtype=float) % 10
    report = diagnose_data({"x": values})
    assert any(f.code == "data.unexpected_discretization" for f in report.findings)
    integers = np.array([2**53, 2**53 + 1], dtype=np.int64)
    report = diagnose_data({"x": integers})
    assert report.columns[0].sample_distinct == 2
    assert report.duplicate_rows == 0
    zeros = diagnose_data({"x": np.array([0.0, -0.0, np.nan, np.inf])})
    assert zeros.columns[0].sample_distinct == 1
    assert zeros.columns[0].non_finite == 2


def test_suite_selection_and_threshold_boundaries():
    values = np.zeros(1000)
    values[990:] = 1
    report = diagnose_data({"x": values})
    assert any(f.code == "data.near_constant" for f in report.findings)
    quiet = diagnose_data(
        {"x": values}, suite=DataDiagnostics(distribution=False, duplicate_rows=False)
    )
    assert not quiet.findings
    assert any(c.reason == "disabled by suite" for c in quiet.checks)
    assert quiet.identity != report.identity


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"missing_share": float("nan")}, CausalValueError),
        ({"missing_share": 2}, CausalValueError),
        ({"min_observations": 1}, CausalValueError),
        ({"max_sample_rows": 0}, CausalValueError),
        ({"distribution": 1}, CausalTypeError),
    ],
)
def test_invalid_policy_uses_domain_errors(kwargs, error):
    with pytest.raises(error):
        DataDiagnostics(**kwargs)


def test_preparation_rejects_invalid_setting():
    with pytest.raises(CausalTypeError):
        prepare(diagnostics={"distribution": True})


def test_batch_reports_and_optout():
    from antecedent._batch import prepare_many

    queries = [ant.AverageEffect(treatment="t", outcome="y")]
    a = prepare_many(
        data(), graph=[("t", "y")], queries=queries, estimator="linear.adjustment.ate", refute=False
    )
    assert len(a.data_diagnostics) == 1
    assert a.data_diagnostics[0].rows_total == 200
    assert require_success(a.diagnose_data())[0].identity == a.data_diagnostics[0].identity
    b = prepare_many(
        data(),
        graph=[("t", "y")],
        queries=queries,
        estimator="linear.adjustment.ate",
        refute=False,
        data_diagnostics=False,
    )
    assert b.data_diagnostics == (None,)


def test_automatic_and_explicit_diagnostics_fit_no_models():
    from antecedent.execution_attempt import observe_native_attempts

    for operation in (lambda: prepare(), lambda: prepare().diagnose_data()):
        attempt = observe_native_attempts(operation)
        attempt.unwrap()
        assert attempt.report.is_complete
        for name in (
            "glm_fit",
            "adjusted_fit",
            "learner_fit",
            "posterior_fit",
            "posterior_draw",
            "score_construction",
        ):
            assert attempt.report.operations[name].attempted == 0


def test_series_values_and_roles_are_profiled_without_temporal_claims():
    from antecedent.graph import TemporalDag

    t = np.random.default_rng(37).normal(size=160)
    y = np.ones(160)
    y[1:] += 2 * t[:-1]
    graph = TemporalDag.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
    study = ant.prepare(
        {"t": t, "y": y},
        graph=graph,
        query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1, active_level=1.0),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    assert study.data_diagnostics.rows_total == 160
    assert study.data_diagnostics.context["roles"]["t"] == "treatment"
    assert any(
        c.code == "data.relationships" and c.status == "not_applicable"
        for c in study.data_diagnostics.checks
    )


def test_categorical_boolean_and_timestamp_storage_remain_typed():
    report = diagnose_data(
        {
            "label": np.array(["red", "blue", "red", None], dtype=object),
            "flag": np.array([True, False, True, False]),
            "when": np.array(
                ["2020-01-01", "2020-01-02", "2020-01-01", "NaT"], dtype="datetime64[ns]"
            ),
        }
    )
    assert [c.sample_distinct for c in report.columns] == [2, 2, 2]
    assert [c.non_finite for c in report.columns] == [1, 0, 1]
    assert report.duplicate_rows == 1
    assert report.columns[0].value_type == "categorical"
    assert report.columns[1].value_type == "binary"
    assert not any(f.code == "data.unexpected_discretization" for f in report.findings)


def test_declared_binary_treatment_does_not_warn_about_continuous_resolution():
    study = prepare(rows=data(2000))
    assert not any(
        f.columns == ("t",) and f.code in ("data.unexpected_discretization", "data.repeated_values")
        for f in study.data_diagnostics.findings
    )


def test_shared_conformance_fixture():
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2] / "conformance/data_diagnostics/basic/fixture.json"
        ).read_text()
    )
    report = diagnose_data(fixture["columns"])
    expected = fixture["expected"]
    assert report.rows_total == expected["rows_total"]
    assert report.rows_complete == expected["rows_complete"]
    assert report.duplicate_rows == expected["duplicate_rows"]
    for name in ("sample_distinct", "modal_share", "skewness", "excess_kurtosis"):
        assert getattr(report.columns[0], name) == pytest.approx(expected[name])


def test_registry_has_stable_codes_and_declared_applicability():
    checks = DataDiagnostics().checks
    assert len({check["code"] for check in checks}) == len(checks)
    assert {"data.near_constant", "data.skewness", "data.relationships"} <= {
        check["code"] for check in checks
    }
    assert all(check["family"] and check["applicability"] for check in checks)
    report = diagnose_data({"x": np.zeros(1000)})
    registered = {check["code"] for check in checks}
    assert {check.code for check in report.checks} <= registered
    assert {finding.code for finding in report.findings} <= registered


def test_event_values_are_profiled_and_panel_inputs_are_explicitly_unavailable():
    from antecedent.data import EventFrame, PanelFrame
    from antecedent.graph import TemporalDag

    t = np.random.default_rng(37).normal(size=160)
    y = np.ones(160)
    y[1:] += 2 * t[:-1]
    graph = TemporalDag.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
    query = ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1, active_level=1.0)
    events = EventFrame(["t", "y"], [t, y], np.arange(160, dtype=np.int64), 1)
    event_study = ant.prepare(
        events, graph=graph, query=query, inference=ant.Frequentist(bootstrap=0), refute=False
    )
    assert event_study.data_diagnostics.rows_total == 160
    panel = PanelFrame(["t", "y"], [[t, y], [t, y]], [0, 1])
    panel_study = ant.prepare(
        panel, graph=graph, query=query, inference=ant.Frequentist(bootstrap=0), refute=False
    )
    assert panel_study.data_diagnostics.snapshot_identity is None
    assert all(check.status == "not_evaluated" for check in panel_study.data_diagnostics.checks)


def test_conditional_effect_keeps_declared_treatment_support():
    rows = data(200)
    rows["z"] = np.sin(np.arange(200) * 0.37)
    study = ant.prepare(
        rows,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.ConditionalEffect("t", "y", "z"),
        refute=False,
    )
    assert study.data_diagnostics.treatment_arms["t"] == ((0.0, 100), (1.0, 100))
    assert not any(
        f.columns == ("t",) and f.code == "data.repeated_values"
        for f in study.data_diagnostics.findings
    )
