"""AnomalyAttribution and ChangeAttribution on the golden Python lifecycle.

The two licensed GCM cells run on ``analyze`` and retain a study::

    result = ant.analyze(data, graph=Dag, query=query)
    study = result.study
    updated = study.refresh(new_data)
    report = result.inspect().to_dict()
    loaded = ant.load(result.export())

graph-posterior structures stay refused; cheap/full validation and accepted structures run the
functional refuter state and stay unlicensed.
"""

from __future__ import annotations

import json
import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, read_text
from _sealed_loads import assert_answer_kept

ROOT = REPO_ROOT
PIN = json.loads(
    read_text(ROOT / "conformance" / "estimate" / "staged_attribution" / "expected.json")
)


_NOISE = np.sin(np.arange(200, dtype=np.float64) * 1.7) * 0.5


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
        result = require_success(ant.analyze(data, graph=graph, query=query))
        study = result.study
        updated = require_success(study.refresh(new_data))
        report = result.inspect().to_dict()
        loaded = require_success(ant.load(bytes(result.export())))
    return result, study, updated, report, loaded


def _assert_identities(result, study, updated, report, loaded, *, fresh, coordinate):
    assert isinstance(study, PreparedAnalysis)
    assert updated.program_id == result.program_id
    assert updated.data_snapshot_id != result.data_snapshot_id
    assert updated.data_snapshot_id == fresh.data_snapshot_id
    again = require_success(study.estimate())
    assert again.answer == updated.answer
    assert again.claim_id == updated.claim_id

    assert report["identification"]["available"] is True
    assert report["calibration"]["status"] in {"calibrated", "scope_not_assessed", "unavailable"}
    for identity in ("program_id", "claim_id", "data_snapshot_id", "identification_id"):
        assert report[identity], identity
    assert report["support"]["payload"]["matrix_coordinate"] == coordinate

    assert_answer_kept(loaded)
    assert loaded.decoded_artifact.payload_kind == "analysis_result"
    assert loaded.answer == result.answer
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    for name in ("status", "reason", "record_id", "observed_coverage"):
        assert getattr(loaded.calibration, name) == getattr(result.calibration, name), name
    assert bytes(loaded.export()) == bytes(result.export())


def test_route_anomaly_tabular_explicit():
    data, new_data = _outlier_chain(), _outlier_chain(outlier_at=0)
    result, study, updated, report, loaded = _golden(data, _dag(), _anomaly_query(), new_data)
    fresh = require_success(ant.analyze(new_data, graph=_dag(), query=_anomaly_query()))
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
    fresh = require_success(ant.analyze(new_data, graph=_dag(), query=_change_query()))
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
    result = require_success(ant.analyze(_outlier_chain(), graph=_dag(), query=_anomaly_query()))
    scores = result.anomaly[0]
    assert scores.top_row == PIN["anomaly"]["top_row"]
    assert max(scores.scores) >= PIN["anomaly"]["score_min"]


def test_change_analyze_reproduces_the_conformance_pin():
    result = require_success(ant.analyze(_two_period_chain(), graph=_dag(), query=_change_query()))
    change = result.change_attribution
    assert abs(change.total_change - PIN["change"]["total_change"]) <= PIN["change"]["tolerance"]
    assert change.total_change >= PIN["change"]["total_change_min"]


def test_edge_list_graph_is_accepted():
    result = require_success(
        ant.analyze(_outlier_chain(), graph=[("x", "y")], query=_anomaly_query())
    )
    assert result.anomaly[0].top_row == PIN["anomaly"]["top_row"]


@pytest.mark.parametrize("query", [_anomaly_query(), _change_query()], ids=["anomaly", "change"])
def test_second_click_refuter_suite_refuses(query):
    data = _outlier_chain() if isinstance(query, ant.AnomalyAttribution) else _two_period_chain()
    result = require_success(ant.analyze(data, graph=_dag(), query=query))
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.refute({"x": [0.0], "y": [0.0]}, suite="placebo")
    assert raised.value.reason_code == "option_not_applicable"


@pytest.mark.parametrize("query", [_anomaly_query(), _change_query()], ids=["anomaly", "change"])
def test_bootstrap_runs_unlicensed_and_cheap_full_refuse(query):
    data = _outlier_chain() if isinstance(query, ant.AnomalyAttribution) else _two_period_chain()
    # A Frequentist replicate count runs the mechanism-refit row bootstrap: it publishes
    # a bootstrap standard error and the result is unlicensed.
    bootstrapped = require_success(
        ant.analyze(data, graph=_dag(), query=query, inference=ant.Frequentist(bootstrap=50))
    )
    assert bootstrapped.evidence_status != "licensed"
    assert any("gcm.frequentist_bootstrap" in note for note in bootstrapped.diagnostics)
    # The Bayesian cell keeps its Bayesian-bootstrap posterior and refuses bootstrap=.
    with pytest.raises(CausalUnsupportedError) as raised:
        require_success(
            ant.analyze(data, graph=_dag(), query=query, bootstrap=50, inference=ant.Bayesian())
        )
    assert raised.value.reason_code == "option_not_applicable"
    # Cheap and full validation refit the whole contribution vector on perturbed data: the
    # table needs noise for the refits to vary, and the result stays unlicensed.
    noisy = {name: values + _NOISE[: len(values)] for name, values in data.items()}
    for suite in ("cheap", "full"):
        validated = require_success(ant.analyze(noisy, graph=_dag(), query=query, refute=suite))
        assert validated.evidence_status != "licensed"


def test_retarget_refuses():
    result = require_success(ant.analyze(_outlier_chain(), graph=_dag(), query=_anomaly_query()))
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.retarget(np.ones(len(result.study._native.names)), [])
    assert raised.value.reason_code == "population_not_estimable"


def test_bayesian_anomaly_and_unlicensed_axes_on_analyze():
    bayesian = require_success(
        ant.analyze(
            _outlier_chain(),
            graph=_dag(),
            query=_anomaly_query(),
            inference=ant.Bayesian(n_draws=32),
        )
    )
    assert bayesian.posterior is not None
    assert bayesian.posterior.backend == "gcm.attribution.shared_dirichlet_row_weights"
    accepted = require_success(
        ant.analyze(
            _two_period_chain(),
            graph=ant.AcceptedGraph(_dag()),
            query=_change_query(),
        )
    )
    assert accepted.evidence_status != "licensed"
    with pytest.raises(ant.errors.CausalTypeError, match="requires graph=Dag"):
        require_success(
            ant.analyze(
                _outlier_chain(),
                graph=ant.Admg.from_edges(["x", "y"], [("x", "y")]),
                query=_anomaly_query(),
            )
        )


def test_fixed_anomaly_reference_runs_through_analyze_and_artifact():
    data = _outlier_chain()
    query = ant.AnomalyAttribution(
        ["y"], max_units=100, reference=ant.AnomalyReference(center=0.0, scale=10.0)
    )
    result = require_success(
        ant.analyze(
            data,
            graph=_dag(),
            query=query,
            inference=ant.Bayesian(n_draws=32),
            return_posterior_artifact=True,
        )
    )
    empirical = require_success(
        ant.analyze(data, graph=_dag(), query=_anomaly_query(), inference=ant.Bayesian(n_draws=32))
    )
    assert result.anomaly[0].scores != empirical.anomaly[0].scores
    assert result.posterior is not None
    posterior = ant.inference.decode_posterior_artifact(result.posterior.artifact)
    assert posterior.quantity_names == ["mean_anomaly_score[1]"]
    assert posterior.q025[0] < posterior.q975[0]
    loaded = require_success(ant.load(bytes(result.export())))
    assert loaded.answer == result.answer
    assert bytes(loaded.export()) == bytes(result.export())


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_fixed_anomaly_reference_rejects_invalid_scale(scale):
    with pytest.raises(
        ant.errors.CausalValueError, match="finite center and a finite positive scale"
    ):
        ant.AnomalyReference(center=0.0, scale=scale)
