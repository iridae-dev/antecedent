"""Identification continues through one immutable preparation and execution path."""

import importlib
import inspect
from dataclasses import FrozenInstanceError

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.execution import Execution
from antecedent.graph import TieredBackground
from antecedent.inference import Frequentist
from antecedent.outcomes import Outcome, require_success
from antecedent.prepared import PreparedAnalysis


def observations():
    rng = np.random.default_rng(201)
    z = rng.normal(size=300)
    t = 0.4 * z + rng.normal(size=300)
    return {"z": z, "t": t, "y": 2 * t + z + rng.normal(scale=0.2, size=300)}


def identification(query=None):
    return ac.identify(
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect("t", "y") if query is None else query,
        names=["z", "t", "y"],
    )


def test_stage_preparation_owns_rows_and_preserves_common_lifecycle():
    stage = identification()
    rows = observations()
    execution = Execution(seed=7, threads=1)
    study = stage.prepare(rows, inference=Frequentist(bootstrap=0), execution=execution)
    baseline = stage.prepare(
        observations(), inference=Frequentist(bootstrap=0), execution=execution
    )
    assert isinstance(study, PreparedAnalysis)
    assert study.execution is execution
    assert study._declarations["refute"] is False
    assert study.inspect().stage == "prepared"
    assert study.reuse.preparation == "reprepared"
    assert "identification" in study.reuse.invalidated
    assert "identification" not in study.reuse.retained
    rows["y"][:] = 1e8
    result = ac.estimate(study)
    assert isinstance(result, Outcome)
    assert require_success(result).ate == pytest.approx(require_success(ac.estimate(baseline)).ate)
    assert study.inspect().stage == "prepared"
    with pytest.raises(FrozenInstanceError):
        study.execution = Execution()


def test_convenience_estimate_matches_preparation_and_common_analysis():
    stage = identification()
    rows = observations()
    options = {"inference": Frequentist(bootstrap=0), "execution": Execution(seed=9)}
    staged = require_success(stage.estimate(rows, **options))
    prepared = require_success(ac.estimate(stage.prepare(rows, **options)))
    direct = require_success(ac.analyze(rows, graph=stage.graph, query=stage.query, **options))
    assert staged.ate == prepared.ate == direct.ate
    assert staged.identification == direct.identification


def test_tiered_identify_remains_data_free_and_estimates_its_resolved_strategy(monkeypatch):
    preparation = importlib.import_module("antecedent.prepared")
    original = preparation.prepare

    def forbidden(*args, **kwargs):
        raise AssertionError("identify-only must not prepare")

    monkeypatch.setattr(preparation, "prepare", forbidden)
    graph = TieredBackground([["z"], ["t"], ["y"]])
    stage = ac.identify(graph=graph, query=ac.AverageEffect("t", "y"))
    assert stage.certificate is not None
    monkeypatch.setattr(preparation, "prepare", original)
    rows = observations()
    options = {"inference": Frequentist(bootstrap=0), "execution": Execution(seed=1)}
    study = stage.prepare(rows, **options)
    assert study._declarations["identifier"] is None
    result = require_success(ac.estimate(study))
    direct = require_success(ac.analyze(rows, graph=graph, query=stage.query, **options))
    assert result.ate == direct.ate
    assert result.identification == direct.identification


def test_response_journey_retains_grid_and_rejects_inapplicable_refutation():
    stage = identification(ac.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5]))
    options = {"inference": Frequentist(bootstrap=0), "execution": Execution(seed=3)}
    study = stage.prepare(observations(), **options)
    response = require_success(ac.estimate(study)).response
    assert [list(point) for point in response.points] == [[-0.5], [0.0], [0.5]]
    with pytest.raises(CausalUnsupportedError) as invalid:
        stage.estimate(observations(), refute="cheap", **options)
    assert invalid.value.reason_code == "refutation_not_applicable"


def test_transport_journey_uses_common_preparation():
    from antecedent import transport

    graph = ac.Admg.from_edges(["x", "y"], [("x", "y")])
    query = transport.Transport(
        ac.AverageEffect("x", "y"),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source", kind="experimental", interventions=["x"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )
    rows = transport.ExactTransportData(
        tuple(
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                probabilities,
                "v1",
                interventions=(("x", level),),
            )
            for level, probabilities in ((0.0, (0.5, 0.5)), (1.0, (0.2, 0.8)))
        )
    )
    stage = ac.identify(graph=graph, query=query)
    study = stage.prepare(rows, execution=Execution(seed=5))
    assert isinstance(study, PreparedAnalysis)
    assert study._declarations["refute"] is None
    with pytest.raises(CausalUnsupportedError) as invalid:
        stage.prepare(rows, refute=False)
    assert invalid.value.reason_code == "option_not_applicable"
    result = ac.estimate(study)
    direct = ac.analyze(rows, graph=graph, query=query, execution=Execution(seed=5))
    assert result.accepted == direct.accepted
    assert result.answer == direct.answer


@pytest.mark.parametrize("keyword", ["seed", "threads", "bootstrap", "latency"])
@pytest.mark.parametrize("operation", ["prepare", "estimate"])
def test_operational_legacy_keywords_are_rejected(keyword, operation):
    with pytest.raises(TypeError, match="unexpected keyword"):
        getattr(identification(), operation)(observations(), **{keyword: 1})


def test_invalid_execution_and_duplicate_public_estimate_are_rejected():
    stage = identification()
    with pytest.raises(CausalTypeError, match="Execution"):
        stage.prepare(observations(), execution={"seed": 1})
    module = importlib.import_module("antecedent.identify")
    assert not hasattr(module, "estimate")
    assert "execution" in inspect.signature(stage.estimate).parameters


def test_edge_list_stage_keeps_declared_isolated_variables_and_order():
    stage = ac.identify(
        graph=[("t", "y")], query=ac.AverageEffect("t", "y"), names=["isolated", "y", "t"]
    )
    rows = observations()
    rows["isolated"] = np.arange(len(rows["t"]), dtype=float)
    study = stage.prepare(rows, inference=Frequentist(bootstrap=0))
    assert list(study._declarations["graph"].nodes()) == ["isolated", "y", "t"]
    assert study._declarations["identifier"] == stage.identifier
