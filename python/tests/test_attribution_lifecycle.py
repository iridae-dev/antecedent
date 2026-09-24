"""AnomalyAttribution and ChangeAttribution on the golden Python lifecycle.

The two licensed GCM cells run on ``analyze`` and retain a study::

    result = ant.analyze(data, graph=Dag, query=query)
    study = result.study
    updated = study.refresh(new_data)
    report = result.inspect().to_dict()
    loaded = ant.load(result.export())

cheap/full, Bayesian, accepted, graph-posterior, and non-Dag stay refused.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis

from _repo_text import read_text

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads(
    read_text(ROOT / "conformance" / "estimate" / "staged_attribution" / "expected.json")
)


def _outlier_chain(n: int = 20, outlier_at: int = -1) -> dict[str, np.ndarray]:
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x.copy()
    y[outlier_at] = 200.0
    return {"x": x, "y": y}


def _two_period_chain(n: int = 80, comparison_intercept: float = 6.0) -> dict[str, np.ndarray]:
    x = np.array([(i % 40) * 0.1 for i in range(n)], dtype=np.float64)
    y = np.array(
        [
            (1.0 + 2.0 * (i % 40) * 0.1)
            if i < 40
            else (comparison_intercept + 2.0 * (i % 40) * 0.1)
            for i in range(n)
        ],
        dtype=np.float64,
    )
    return {"x": x, "y": y}


def _dag() -> ant.Dag:
    return ant.Dag.from_edges(["x", "y"], [("x", "y")])


def _anomaly_query() -> ant.AnomalyAttribution:
    return ant.AnomalyAttribution(["y"], max_units=100)


def _change_query() -> ant.ChangeAttribution:
    return ant.ChangeAttribution(
        "y", baseline_start=0, baseline_end=40, comparison_start=40, comparison_end=80
    )


def _golden(data, graph, query, new_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ant.analyze(data, graph=graph, query=query)
        study = result.study
        updated = study.refresh(new_data)
        report = result.inspect().to_dict()
        loaded = ant.load(result.export())
    return result, study, updated, report, loaded


def _assert_identities(result, study, updated, report, loaded, *, fresh, coordinate):
    assert isinstance(study, PreparedAnalysis)
    assert updated.program_id == result.program_id
    assert updated.data_snapshot_id != result.data_snapshot_id
    assert updated.data_snapshot_id == fresh.data_snapshot_id
    again = study.estimate()
    assert again.answer == updated.answer
    assert again.claim_id == updated.claim_id

    assert report["identification"]["available"] is True
    assert report["calibration"]["status"] in {"calibrated", "scope_not_assessed", "unavailable"}
    for identity in ("program_id", "claim_id", "data_snapshot_id", "identification_id"):
        assert report[identity], identity
    assert report["support"]["payload"]["matrix_coordinate"] == coordinate

    assert loaded.acceptance.verified
    assert loaded.artifact.payload_kind == "analysis_result"
    assert loaded.answer == result.answer
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    for name in ("status", "reason", "record_id", "observed_coverage"):
        assert getattr(loaded.calibration, name) == getattr(result.calibration, name), name
    assert loaded.export() == result.export()


def test_route_anomaly_tabular_explicit():
    data, new_data = _outlier_chain(), _outlier_chain(outlier_at=0)
    result, study, updated, report, loaded = _golden(data, _dag(), _anomaly_query(), new_data)
    fresh = ant.analyze(new_data, graph=_dag(), query=_anomaly_query())
    _assert_identities(
        result,
        study,
        updated,
        report,
        loaded,
        fresh=fresh,
        coordinate="AnomalyAttribution:Dag:explicit:Frequentist:none",
    )
    assert result.answer.kind == "structured"
    scores = result.anomaly
    assert scores is not None and len(scores) == 1
    assert scores[0].outcome == "y"
    assert scores[0].top_row == PIN["anomaly"]["top_row"]
    assert max(scores[0].scores) >= PIN["anomaly"]["score_min"]
    assert updated.anomaly[0].top_row == fresh.anomaly[0].top_row
    assert result.identification.method.startswith("gcm.parametric")
    assert result.estimate.estimator_id == "gcm.fit"


def test_route_change_tabular_explicit():
    data, new_data = _two_period_chain(), _two_period_chain(comparison_intercept=10.0)
    result, study, updated, report, loaded = _golden(data, _dag(), _change_query(), new_data)
    fresh = ant.analyze(new_data, graph=_dag(), query=_change_query())
    _assert_identities(
        result,
        study,
        updated,
        report,
        loaded,
        fresh=fresh,
        coordinate="ChangeAttribution:Dag:explicit:Frequentist:none",
    )
    assert result.answer.kind == "point"
    change = result.change_attribution
    assert change is not None
    assert abs(change.total_change - PIN["change"]["total_change"]) <= PIN["change"]["tolerance"]
    assert change.total_change >= PIN["change"]["total_change_min"]
    assert result.estimate.ate == change.total_change
    assert updated.change_attribution.total_change == fresh.change_attribution.total_change
    assert updated.estimate.ate != result.estimate.ate
    assert result.identification.method.startswith("gcm.parametric")
    assert result.estimate.estimator_id == "gcm.fit"


def test_anomaly_analyze_reproduces_the_conformance_pin():
    result = ant.analyze(_outlier_chain(), graph=_dag(), query=_anomaly_query())
    scores = result.anomaly[0]
    assert scores.top_row == PIN["anomaly"]["top_row"]
    assert max(scores.scores) >= PIN["anomaly"]["score_min"]


def test_change_analyze_reproduces_the_conformance_pin():
    result = ant.analyze(_two_period_chain(), graph=_dag(), query=_change_query())
    change = result.change_attribution
    assert abs(change.total_change - PIN["change"]["total_change"]) <= PIN["change"]["tolerance"]
    assert change.total_change >= PIN["change"]["total_change_min"]


def test_edge_list_graph_is_accepted():
    result = ant.analyze(_outlier_chain(), graph=[("x", "y")], query=_anomaly_query())
    assert result.anomaly[0].top_row == PIN["anomaly"]["top_row"]


@pytest.mark.parametrize("query", [_anomaly_query(), _change_query()], ids=["anomaly", "change"])
def test_second_click_refuter_suite_refuses(query):
    data = _outlier_chain() if isinstance(query, ant.AnomalyAttribution) else _two_period_chain()
    result = ant.analyze(data, graph=_dag(), query=query)
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.refute({"x": [0.0], "y": [0.0]}, suite="placebo")
    assert raised.value.reason_code == "option_not_applicable"


@pytest.mark.parametrize("query", [_anomaly_query(), _change_query()], ids=["anomaly", "change"])
def test_bootstrap_and_cheap_full_refuse(query):
    data = _outlier_chain() if isinstance(query, ant.AnomalyAttribution) else _two_period_chain()
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(data, graph=_dag(), query=query, bootstrap=50)
    assert raised.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(data, graph=_dag(), query=query, refute="full")
    assert raised.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(data, graph=_dag(), query=query, refute="cheap")
    assert raised.value.reason_code == "option_not_applicable"


def test_retarget_refuses():
    result = ant.analyze(_outlier_chain(), graph=_dag(), query=_anomaly_query())
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.retarget(np.ones(len(result.study._native.names)), [])
    assert raised.value.reason_code == "population_not_estimable"


def test_bayesian_anomaly_and_unlicensed_axes_on_analyze():
    bayesian = ant.analyze(
        _outlier_chain(),
        graph=_dag(),
        query=_anomaly_query(),
        inference=ant.Bayesian(n_draws=32),
    )
    assert bayesian.posterior is not None
    assert bayesian.posterior.backend == "gcm.attribution.shared_dirichlet_row_weights"
    with pytest.raises(CausalUnsupportedError):
        ant.analyze(
            _two_period_chain(),
            graph=ant.AcceptedGraph(_dag()),
            query=_change_query(),
        )
    with pytest.raises(
        (CausalUnsupportedError, ant.errors.CausalTypeError, ant.errors.CausalValueError)
    ):
        ant.analyze(
            _outlier_chain(),
            graph=ant.Admg.from_edges(["x", "y"], [("x", "y")]),
            query=_anomaly_query(),
        )
