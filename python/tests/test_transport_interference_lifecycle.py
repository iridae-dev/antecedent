"""TransportQuery and InterferenceQuery on the golden Python lifecycle.

Both licensed design cells run on ``analyze`` and retain a study exactly like
every other licensed cell::

    result = ant.analyze(data, graph=graph, query=query)
    study = result.study
    updated = study.refresh(new_data)
    report = result.inspect().to_dict()
    loaded = ant.load(result.export())

with no warnings. The retained study re-executes the same program, a refresh
executes on the refreshed data, the export verifies and its answer is the live
answer, and the calibration slot carries the match key the coverage harness
binds records under. Operations with no meaning for a design-defined estimand
(retarget, a second-click refuter suite, a bootstrap) refuse with a registered
reason code, and so does a construction outside the licensed cell. The
unlicensed utilities ``transport.estimate_trial_effect`` and
``interference.estimate`` keep that unlicensed behaviour and call the same Rust
estimator, so on a licensed construction they report the analysis's point numbers.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.transport import advanced as transport
from antecedent.transport.advanced import TransportQuery

from _repo_text import read_text

ROOT = Path(__file__).resolve().parents[2]
TRANSPORT_PIN = json.loads(
    read_text(ROOT / "conformance" / "estimate" / "staged_transport" / "expected.json")
)
INTERFERENCE_PIN = json.loads(
    read_text(ROOT / "conformance" / "response" / "randomized_interference" / "expected.json")
)

TRANSPORT_NAMES = ["a", "y", "trial", "s", "e", "x"]


def _transport_data(n: int, seed: int) -> dict[str, np.ndarray]:
    """Trial membership ``S | x ~ Bern(σ(x/2))``, randomized ``a`` in the trial."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    s = 1.0 / (1.0 + np.exp(-0.5 * x))
    trial = rng.uniform(size=n) < s
    a = np.where(trial, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
    y = np.where(trial, 1.0 + a * (1.0 + x) + x + rng.normal(size=n), 0.0)
    return {"a": a, "y": y, "trial": trial.astype(float), "s": s, "e": np.full(n, 0.5), "x": x}


def _transport_graph() -> ant.Admg:
    return ant.Admg.from_edges(TRANSPORT_NAMES, [("a", "y"), ("x", "y")])


def _transport_query(**overrides: object) -> TransportQuery:
    fields: dict[str, object] = {
        "trial": "trial",
        "selection_probability": "s",
        "treatment_probability": "e",
    }
    fields.update(overrides)
    return TransportQuery(
        ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", ["x"]),
        source_experiments=["a"],
        **fields,  # type: ignore[arg-type]
    )


UNITS = 40


def _ring() -> list[tuple[int, int]]:
    return [((i + 1) % UNITS, i) for i in range(UNITS)] + [
        ((i - 1) % UNITS, i) for i in range(UNITS)
    ]


def _interference_design(seed: int = 3) -> tuple[list[bool], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    assignment = rng.uniform(size=UNITS) < 0.5
    neighbors = np.array(
        [int(assignment[(i + 1) % UNITS]) + int(assignment[(i - 1) % UNITS]) for i in range(UNITS)]
    )
    y = 1.0 + 2.0 * assignment + 0.5 * neighbors + rng.normal(size=UNITS)
    return [bool(value) for value in assignment], {"y": y}


def _interference_query(assignment: list[bool], **overrides: object) -> ant.InterferenceQuery:
    fields: dict[str, object] = {"network": _ring(), "realized_assignment": assignment}
    fields.update(overrides)
    return ant.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0), interference.ExposureLevel(1.0)
        ),
        **fields,  # type: ignore[arg-type]
    )


def _golden(data, graph, query, new_data):
    """The five golden lines, under an error filter for every warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ant.analyze(data, graph=graph, query=query)
        study = result.study
        updated = study.refresh(new_data)
        report = result.inspect().to_dict()
        loaded = ant.load(result.export())
    return result, study, updated, report, loaded


def _assert_lifecycle(result, study, updated, report, loaded, *, fresh, coordinate, key):
    assert isinstance(study, PreparedAnalysis)
    assert result.answer.kind == "point"

    # refresh re-executes the same program on the new data.
    assert updated.program_id == result.program_id
    assert updated.data_snapshot_id != result.data_snapshot_id
    assert updated.answer == fresh.answer
    assert updated.data_snapshot_id == fresh.data_snapshot_id
    assert updated.estimate.ate != result.estimate.ate
    # ...and the refreshed study keeps executing the refreshed data.
    again = study.estimate()
    assert again.answer == updated.answer
    assert again.claim_id == updated.claim_id

    # inspect() reports identification, the answer, the calibration slot and identities.
    assert report["identification"]["available"] is True
    assert report["answer"]["kind"] == "point"
    assert report["answer"]["value"] == result.answer.value
    assert report["calibration"]["status"] in {"calibrated", "scope_not_assessed", "unavailable"}
    for identity in ("program_id", "claim_id", "data_snapshot_id", "identification_id"):
        assert report[identity], identity
    assert report["support"]["payload"]["matrix_coordinate"] == coordinate

    # export -> load verifies and round-trips the same answer and identities.
    assert loaded.acceptance.verified
    assert loaded.artifact.payload_kind == "analysis_result"
    assert loaded.answer == result.answer
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    for name in ("status", "reason", "record_id", "observed_coverage"):
        assert getattr(loaded.calibration, name) == getattr(result.calibration, name), name
    assert loaded.export() == result.export()

    # The calibration slot carries the key a harness-measured record binds under.
    basis = loaded.artifact.contract["claim"]["calibration"]["basis"]["key"]
    assert {name: basis[name] for name in key} == key
    assert basis["level"] == pytest.approx(0.95)


def test_route_transport_tabular_explicit():
    """`parity/python_products.toml` route row: TransportQuery retains its study."""
    data, new_data = _transport_data(400, 1), _transport_data(400, 2)
    result, study, updated, report, loaded = _golden(
        data, _transport_graph(), _transport_query(), new_data
    )
    fresh = ant.analyze(new_data, graph=_transport_graph(), query=_transport_query())
    _assert_lifecycle(
        result,
        study,
        updated,
        report,
        loaded,
        fresh=fresh,
        coordinate="TransportQuery:Admg:explicit:Frequentist:none",
        key={
            "query": "TransportQuery",
            "graph_class": "Admg",
            "structure": "fixed",
            "modality": "tabular",
            "inference": "Frequentist",
            "estimator": "transport.trial_ipw",
            "interval_method": "analytic_se",
            "dependence": "iid",
            "identification": "point",
        },
    )
    assert result.identification.status == "NonparametricallyIdentified"
    assert result.identification.method.startswith("transport.sid")
    assert result.estimate.estimator_id == "transport.trial_ipw"
    overlap = result.transport_overlap
    assert overlap is not None
    assert overlap.treatment.probability_min == overlap.treatment.probability_max == 0.5
    trial_rows = data["trial"] != 0.0
    assert overlap.selection.probability_min == pytest.approx(float(np.min(data["s"][trial_rows])))


def test_route_interference_tabular_explicit():
    """`parity/python_products.toml` route row: InterferenceQuery retains its study."""
    assignment, data = _interference_design()
    new_data = {"y": data["y"] + np.linspace(0.0, 1.0, UNITS)}
    query = _interference_query(assignment)
    result, study, updated, report, loaded = _golden(data, [], query, new_data)
    fresh = ant.analyze(new_data, graph=[], query=query)
    _assert_lifecycle(
        result,
        study,
        updated,
        report,
        loaded,
        fresh=fresh,
        coordinate="InterferenceQuery:Dag:explicit:Frequentist:none",
        key={
            "query": "InterferenceQuery",
            "graph_class": "Dag",
            "structure": "fixed",
            "modality": "tabular",
            "inference": "Frequentist",
            "estimator": "interference.ht_hajek",
            "interval_method": "analytic_se",
            "dependence": "iid",
            "identification": "point",
        },
    )
    assert result.identification.status == "NonparametricallyIdentified"
    assert result.identification.method == "interference.design"
    contrast = result.interference
    assert contrast is not None
    assert result.estimate.ate == contrast.contrast.horvitz_thompson
    assert result.estimate.se_analytic == pytest.approx(
        np.sqrt(contrast.contrast.conservative_variance)
    )
    assert updated.interference.contrast == fresh.interference.contrast


def test_transport_analyze_reproduces_the_conformance_pin():
    names = ["a", "y", "trial", "s", "e"]
    data = {
        "a": [1.0, 0.0, 0.0, 0.0],
        "y": [3.0, 1.0, 0.0, 0.0],
        "trial": [1.0, 1.0, 0.0, 0.0],
        "s": [0.5] * 4,
        "e": [0.5] * 4,
    }
    result = ant.analyze(
        data,
        graph=ant.Admg.from_edges(names, [("a", "y")]),
        query=TransportQuery(
            ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
            transport.SelectionDiagram("trial", "target", []),
            source_experiments=["a"],
            trial="trial",
            selection_probability="s",
            treatment_probability="e",
        ),
    )
    assert abs(result.estimate.ate - TRANSPORT_PIN["ipw"]) <= TRANSPORT_PIN["tolerance"]


def test_interference_analyze_reproduces_the_conformance_pin():
    pin = INTERFERENCE_PIN
    query = ant.InterferenceQuery(
        interference.BernoulliAssignment(pin["design"]["probability"]),
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y",
            interference.ExposureLevel(pin["from"]["own"], pin["from"]["neighbors"]),
            interference.ExposureLevel(pin["to"]["own"], pin["to"]["neighbors"]),
        ),
        network=[tuple(edge) for edge in pin["directed_edges"]],
        realized_assignment=pin["assignment"],
    )
    result = ant.analyze({"y": pin["outcomes"]}, graph=[], query=query)
    expected, atol = pin["expected"], pin["tolerance"]["atol"]
    contrast = result.interference.contrast
    assert abs(contrast.horvitz_thompson - expected["horvitz_thompson_contrast"]) <= atol
    assert abs(contrast.hajek - expected["hajek_contrast"]) <= atol
    assert abs(contrast.conservative_variance - expected["conservative_variance"]) <= atol
    assert result.interference.from_probability_method == expected["probability_method"]


# --- the unlicensed utilities share the estimator with the licensed path -----------


def test_trial_effect_utility_and_analysis_share_the_estimator():
    data = _transport_data(300, 5)
    identification = transport.identify(
        graph=_transport_graph(),
        query=_transport_query(trial=None, selection_probability=None, treatment_probability=None),
    )
    utility = transport.estimate_trial_effect(
        identification,
        data["a"] != 0.0,
        data["y"],
        data["trial"] != 0.0,
        data["s"],
        data["e"],
    )
    analysis = ant.analyze(data, graph=_transport_graph(), query=_transport_query())
    assert utility.ipw == analysis.estimate.ate
    assert utility.overlap == analysis.transport_overlap
    assert utility.rule == identification.certificate.rule


def test_interference_utility_and_analysis_share_the_estimator():
    """An exactly enumerated design, so neither side draws a Monte Carlo stream."""
    pin = INTERFERENCE_PIN
    design = interference.BernoulliAssignment(pin["design"]["probability"])
    contrast = interference.ExposureContrast(
        "y",
        interference.ExposureLevel(pin["from"]["own"], pin["from"]["neighbors"]),
        interference.ExposureLevel(pin["to"]["own"], pin["to"]["neighbors"]),
    )
    edges = [tuple(edge) for edge in pin["directed_edges"]]
    utility = interference.estimate(
        {"y": pin["outcomes"]},
        assignment=pin["assignment"],
        edges=edges,
        query=ant.InterferenceQuery(design, interference.NeighborCount(), contrast),
    )
    analysis = ant.analyze(
        {"y": pin["outcomes"]},
        graph=[],
        query=ant.InterferenceQuery(
            design,
            interference.NeighborCount(),
            contrast,
            network=edges,
            realized_assignment=pin["assignment"],
        ),
    )
    assert utility.contrast == analysis.interference.contrast
    assert utility.from_probability_method == analysis.interference.from_probability_method


# --- operations with no meaning for a design-defined estimand refuse --------------


def _transport_result():
    return ant.analyze(_transport_data(200, 7), graph=_transport_graph(), query=_transport_query())


def _interference_result():
    assignment, data = _interference_design()
    return ant.analyze(data, graph=[], query=_interference_query(assignment))


@pytest.mark.parametrize(
    "run", [_transport_result, _interference_result], ids=["transport", "interference"]
)
def test_retarget_refuses_a_design_defined_population(run):
    result = run()
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.retarget(np.ones(len(result.study._native.names)), [])
    assert raised.value.reason_code == "population_not_estimable"
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.reexecute_retarget(result.export())
    assert raised.value.reason_code == "population_not_estimable"


@pytest.mark.parametrize(
    "run", [_transport_result, _interference_result], ids=["transport", "interference"]
)
def test_second_click_refuter_suite_refuses(run):
    result = run()
    with pytest.raises(CausalUnsupportedError) as raised:
        result.study.refute({"y": [0.0]}, suite="placebo")
    assert raised.value.reason_code == "option_not_applicable"


def test_bootstrap_refuses_on_the_design_cells():
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(
            _transport_data(200, 7),
            graph=_transport_graph(),
            query=_transport_query(),
            bootstrap=50,
        )
    assert raised.value.reason_code == "option_not_applicable"
    assignment, data = _interference_design()
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(data, graph=[], query=_interference_query(assignment), bootstrap=50)
    assert raised.value.reason_code == "option_not_applicable"


def test_constructions_outside_the_licensed_cells_refuse():
    assignment, data = _interference_design()
    complete = ant.InterferenceQuery(
        interference.CompleteRandomization(UNITS // 2),
        interference.NeighborCount(),
        interference.ExposureContrast(
            "y", interference.ExposureLevel(0.0), interference.ExposureLevel(1.0)
        ),
        network=_ring(),
        realized_assignment=assignment,
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(data, graph=[], query=complete)
    assert raised.value.reason_code == "construction_not_licensed"

    derivative = TransportQuery(
        ant.PointDerivative("a", "y", at=0.5),
        transport.SelectionDiagram("trial", "target", ["x"]),
        source_experiments=["a"],
        trial="trial",
        selection_probability="s",
        treatment_probability="e",
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        ant.analyze(_transport_data(200, 7), graph=_transport_graph(), query=derivative)
    assert raised.value.reason_code == "construction_not_licensed"


def test_unlicensed_axes_refuse_on_analyze():
    with pytest.raises(CausalUnsupportedError):
        ant.analyze(
            _transport_data(200, 7),
            graph=_transport_graph(),
            query=_transport_query(),
            inference=ant.Bayesian(n_draws=32),
        )
    assignment, data = _interference_design()
    with pytest.raises(CausalUnsupportedError):
        ant.analyze(data, graph=[], query=_interference_query(assignment), refute="full")
    with pytest.raises(CausalUnsupportedError):
        ant.analyze(
            data,
            graph=ant.AcceptedGraph(ant.Dag.from_edges(["y"], [])),
            query=_interference_query(assignment),
        )


def test_analyze_requires_the_design_facts():
    with pytest.raises(ant.errors.CausalValueError, match="trial columns"):
        ant.analyze(
            _transport_data(50, 1),
            graph=_transport_graph(),
            query=_transport_query(
                trial=None, selection_probability=None, treatment_probability=None
            ),
        )
    assignment, data = _interference_design()
    with pytest.raises(ant.errors.CausalValueError, match="network"):
        ant.analyze(
            data,
            graph=[],
            query=_interference_query(assignment, network=None, realized_assignment=None),
        )
    with pytest.raises(ant.errors.CausalValueError):
        _transport_query(treatment_probability=None)
    with pytest.raises(ant.errors.CausalValueError):
        _interference_query(assignment, network=None)


def test_root_query_classes_are_the_stage_module_classes():
    assert TransportQuery is transport.TransportQuery
    assert ant.InterferenceQuery is interference.InterferenceQuery
    assert _transport_query().kind == "transport"
    assert _interference_query([True] * UNITS).kind == "interference"


def test_transport_catalog_survives_prepared_execution_and_export() -> None:
    data = _transport_data(300, 22)
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "randomized-a",
                "trial",
                kind="experimental",
                interventions=["a"],
                measured=["y", "x"],
            ),
            transport.EvidenceRegime("target-x", "target", measured=["x"]),
        ],
        target_sampling="representative_sample",
    )
    result = ant.analyze(
        data,
        graph=_transport_graph(),
        query=_transport_query(catalog=catalog),
        bootstrap=0,
        refute="none",
    )
    loaded = ant.load(result.export())
    restored_catalog = loaded.artifact.payload["query"]["transport"]["catalog"]
    assert [r["label"] for r in restored_catalog["regimes"]] == ["randomized-a", "target-x"]
    assert restored_catalog["target_sampling"] == "representative_sample"
    assert restored_catalog["regimes"][0]["interventions"] == [0]
    assert loaded.as_point() == result.as_point()
