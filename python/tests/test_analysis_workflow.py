"""One-call analysis, retained studies, frozen executions, and descriptive reports."""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _sealed_loads import assert_answer_kept


def sample(seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=128)
    t = (rng.uniform(size=128) < 0.5).astype(float)
    return {"t": t, "y": 2 * t + z + rng.normal(scale=0.1, size=128), "z": z}


GRAPH = [("t", "y"), ("z", "y")]
QUERY = ant.AverageEffect("t", "y")


def test_one_call_study_and_frozen_export():
    data = sample()
    first = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            execution=ant.Execution(seed=19),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    assert first.study.inspect().identification.available
    before = bytes(first.export())
    assert ant.artifacts.accept(before)["accepts_as_verified_program"] == "true"
    assert ant.artifacts.loads(before).contract["execution"]["seed"] == 19
    old = first.inspect()
    changed = {**data, "y": data["y"] + data["t"]}
    branch = first.study.rebind(data=changed)
    second = require_success(branch.estimate())
    assert second.effect == pytest.approx(first.effect + 1.0)
    assert bytes(first.export()) == before
    assert first.inspect() == old
    assert second.data_snapshot_id != first.data_snapshot_id
    assert ant.artifacts.accept(bytes(second.export()))["accepts_as_verified_program"] == "true"
    assert require_success(first.study.estimate()).effect == pytest.approx(first.effect)
    assert first.study.inspect().data_snapshot_id == first.data_snapshot_id
    via_result = require_success(first.study.rebind(data=changed).estimate())
    assert via_result.effect == pytest.approx(second.effect)


def test_estimate_other_data_does_not_rebind_study():
    data = sample()
    study = ant.prepare(
        data, graph=GRAPH, query=QUERY, inference=ant.Frequentist(bootstrap=0), refute="none"
    )
    before = study.inspect()
    baseline = require_success(study.estimate())
    branch = study.rebind(data={**data, "y": data["y"] + data["t"]})
    branch_before = branch.inspect()
    other = require_success(branch.estimate())
    assert other.data_snapshot_id != baseline.data_snapshot_id
    # Native snapshot identities include sealed fit moments. Deferred studies
    # retain their original projections; completed studies retain actual fits.
    assert study.inspect() == before
    assert branch.inspect() == branch_before
    assert baseline.study.inspect().data_snapshot_id == baseline.data_snapshot_id
    assert other.study.inspect().data_snapshot_id == other.data_snapshot_id
    assert study.inspect().stage == branch.inspect().stage == "prepared"
    assert ant.artifacts.accept(bytes(other.export()))["accepts_as_verified_program"] == "true"
    assert require_success(study.estimate()).effect == pytest.approx(baseline.effect)


def test_missing_row_padding_does_not_upgrade_calibration():
    rng = np.random.default_rng(42)
    t = rng.normal(size=100)
    y = 2 * t + rng.normal(size=100)
    labels = []
    effects = []
    ses = []
    for total in (100, 500, 1000):
        missing = np.full(total - 100, np.nan)
        result = require_success(
            ant.analyze(
                {"t": np.r_[t, missing], "y": np.r_[y, missing]},
                graph=[("t", "y")],
                query=ant.AverageEffect("t", "y"),
                refute="none",
                inference=ant.Frequentist(bootstrap=0),
            )
        )
        labels.append(result.calibration.describe())
        effects.append(result.effect)
        ses.append(result.estimate.se_analytic)
    assert effects[0] == pytest.approx(effects[1])
    assert effects[0] == pytest.approx(effects[2])
    assert ses[0] == pytest.approx(ses[1])
    assert ses[0] == pytest.approx(ses[2])
    assert labels[0] == labels[1] == labels[2]


def test_report_has_json_types_and_explicit_calibration_scope():
    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    report = result.inspect()
    assert report.answer.kind == "point"
    assert report.answer.value == result.effect
    # A record measured this exact construction (Dag / Frequentist /
    # analytic_se at 0.95), but over a sample-size grid of 150 to 600 rows and
    # this study runs 128, so the scope is not assessed and the slot names which
    # bound put it outside and which record it would otherwise cite. That is the explicit
    # scope the test is about: an unqualified "unavailable" would hide it.
    assert report.calibration.status == "scope_not_assessed"
    assert report.calibration.reason == "sample_size_outside_measured_range"
    assert report.calibration.record_id is not None
    assert report.uncertainty.available
    json.dumps(report.to_dict(), allow_nan=False)


def test_response_retains_study_and_exports_own_execution():
    data = sample()
    data["t"] = np.random.default_rng(8).normal(size=128)
    data["y"] = 2 * data["t"] + data["z"]
    result = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=ant.ResponseCurve("t", "y", grid=[0.0, 0.5, 1.0]),
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
        )
    )
    assert result.study is not None
    assert result.answer.kind == "response"
    encoded = bytes(result.export())
    require_success(result.study.rebind(data={**data, "y": data["y"] + 1.0}).estimate())
    assert bytes(result.export()) == encoded
    refreshed = require_success(result.study.rebind(data={**data, "y": data["y"] + 1.0}).estimate())
    assert refreshed.answer.kind == "response"
    assert bytes(result.export()) == encoded
    loaded = require_success(ant.load(encoded))
    assert_answer_kept(loaded)
    assert loaded.decoded_artifact.to_dict()["payload"]["response"] is not None
    assert loaded.inspect().uncertainty.available


def test_refusal_retains_description_without_changing_exception_type():
    with pytest.raises(ant.CausalError, match="graph") as caught:
        ant.prepare(sample(), graph=None, query=QUERY)
    report = caught.value.report
    assert report.operation == "prepare"
    assert "graph" in report.message
    assert report.query == QUERY
    assert report.query is not QUERY
    assert report.identification == "unavailable"
    json.dumps(report.to_dict())


def test_load_verified_execution_and_forward_uncontracted_body():
    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    encoded = bytes(result.export())
    loaded = require_success(ant.load(encoded))
    assert loaded.acceptance.verified
    assert loaded.answer.value == result.effect
    assert loaded.inspect().program_id == result.program_id == loaded.program_id
    assert loaded.inspect().claim_id == result.claim_id == loaded.claim_id
    assert result.program_id
    assert result.claim_id
    assert result.claim_id != result.program_id
    assert loaded.inspect().data_snapshot_id == result.data_snapshot_id
    assert (
        loaded.inspect().uncertainty.payload["components"]
        == result.inspect().uncertainty.payload["components"]
    )
    assert bytes(loaded.export()) == encoded
    json.dumps(loaded.inspect().to_dict(), allow_nan=False)
    with pytest.raises(ant.errors.CausalUnsupportedError, match="not_executed"):
        _ = loaded.study

    body_only = ant.artifacts.dumps(
        "analysis_result",
        loaded.decoded_artifact.to_dict()["payload"],
        variable_names=loaded.decoded_artifact.variable_names,
        artifact_id="body-only",
    )
    unverified = require_success(ant.load(body_only))
    assert not unverified.acceptance.verified
    assert not unverified.inspect().identification.available
    assert unverified.answer.kind == "unavailable"
    assert bytes(unverified.export()) == body_only


def test_load_in_an_independent_python_process(tmp_path):
    import subprocess
    import sys

    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            execution=ant.Execution(seed=23),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    path = tmp_path / "result.ant"
    path.write_bytes(bytes(result.export()))
    code = """import antecedent as ant, json, pathlib, sys
r = ant.load(pathlib.Path(sys.argv[1]).read_bytes())
assert r.acceptance.verified
print(json.dumps(r.inspect().to_dict(), allow_nan=False))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)], check=True, text=True, capture_output=True
    )
    report = json.loads(completed.stdout)
    assert report["answer"]["value"] == result.effect
    assert report["data_snapshot_id"] == result.data_snapshot_id


@pytest.mark.parametrize("latency", [None, "interactive", "standard", "report"])
def test_one_call_and_explicit_prepare_have_same_defaults(latency):
    # Native latency.rs resolves validation suites by profile; the original
    # explicit four-replicate budget overrides every profile's bootstrap count.
    suite, suite_id, count = {
        None: ("placebo", "placebo+rcc", 2),
        "interactive": ("cheap", "overlap+evalue", 2),
        "standard": ("placebo", "placebo+rcc", 2),
        "report": ("full", "validation.full", 14),
    }[latency]
    kwargs = dict(
        graph=GRAPH,
        query=QUERY,
        execution=ant.Execution(seed=19),
        inference=ant.Frequentist(bootstrap=4),
        refute=suite,
    )
    one = require_success(ant.analyze(sample(), **kwargs))
    two = require_success(ant.prepare(sample(), **kwargs).estimate())
    assert one.effect == two.effect
    assert one.estimate.se_bootstrap == two.estimate.se_bootstrap
    assert one.plan.validation_suite == two.plan.validation_suite
    assert one.validation.count == two.validation.count == count
    assert one.plan.validation_suite == suite_id
    assert one.performance.bootstrap_replicates_requested == 4
    assert two.performance.bootstrap_replicates_requested == 4


def test_failed_branch_estimate_retains_previous_binding_and_error_context():
    study = ant.prepare(
        sample(), graph=GRAPH, query=QUERY, inference=ant.Frequentist(bootstrap=0), refute="none"
    )
    before = study.inspect()
    result = require_success(study.estimate())
    branch = study.rebind(data={"t": np.zeros(2), "y": np.zeros(2), "z": np.zeros(2)})
    with pytest.raises(ant.errors.CausalEstimateError, match="rank deficient") as caught:
        require_success(branch.estimate())
    assert caught.value.reason_code == "design_rank_deficient"
    assert caught.value.study is branch
    assert caught.value.report.operation == "estimate"
    assert study.inspect() == before
    assert result.study.inspect().data_snapshot_id == result.data_snapshot_id
    assert require_success(study.estimate()).effect == result.effect


def test_partial_and_missing_evidence_are_not_scalar_success():
    from dataclasses import replace

    from antecedent.results._report import copy_model

    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    ident = replace(
        result.reasoning.identification,
        payload={**result.reasoning.identification.payload, "unevaluable_mass": 0.25},
    )
    partial = copy_model(result, reasoning=replace(result.reasoning, identification=ident))
    assert partial.answer.kind == "partial"
    assert partial.answer.value is None
    assert "partial" in repr(partial)
    assert "unevaluable_mass" in partial._repr_html_()
    empty = copy_model(
        result,
        reasoning=None,
        estimate=copy_model(result.estimate, se_analytic=float("nan")),
        assumptions=None,
    )
    # `.inspect()` reports the same uncertainty/assumptions availability as
    # `load(export()).inspect()`: whenever a native execution is retained, it
    # re-derives both from that execution's own exported contract rather than
    # from these Python-level fields (`_with_portable_record`). `_execution` is
    # a `PrivateAttr`, invisible to `copy_model`'s `model_copy(update=...)`
    # (pydantic stores it outside `__dict__`/extras), so exercising "missing
    # evidence" here means clearing it directly, the same way
    # `test_result_html.py` mutates a frozen result's fields for a test fixture.
    object.__setattr__(empty, "_execution", None)
    assert not empty.inspect().uncertainty.available
    assert not empty.inspect().assumptions.available
    json.dumps(empty.inspect().to_dict(), allow_nan=False)


def test_refuting_an_old_result_does_not_change_its_export():
    data = sample()
    first = require_success(
        ant.analyze(
            data, graph=GRAPH, query=QUERY, inference=ant.Frequentist(bootstrap=0), refute="none"
        )
    )
    before = bytes(first.export())
    require_success(first.study.rebind(data={**data, "y": data["y"] + data["t"]}).estimate())
    checked = first.refute(data, suite="cheap")
    assert checked.effect == first.effect
    assert checked.data_snapshot_id == first.data_snapshot_id
    assert bytes(first.export()) == before
    assert require_success(ant.load(bytes(checked.export()))).acceptance.verified


def test_contracted_posterior_keeps_draws_and_native_uncertainty_target():
    result = require_success(
        ant.analyze(
            sample(),
            graph=GRAPH,
            query=QUERY,
            refute="none",
            inference=ant.Bayesian(n_draws=32, backend="conjugate"),
        )
    )
    loaded = require_success(ant.load(bytes(result.export())))
    assert_answer_kept(loaded)
    assert loaded.decoded_artifact.to_dict()["payload"]["posterior_artifact"]
    assert any(
        c["source"] == "parameter" and c["target"] == "posterior"
        for c in loaded.inspect().uncertainty.payload["components"]
    )
    json.dumps(result.inspect().to_dict(), allow_nan=False)


def test_retarget_exports_target_weight_identity():
    import re

    hex64 = re.compile(r"[0-9a-f]{64}")
    data = sample()
    result = require_success(
        ant.analyze(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=QUERY,
            estimator="aipw",
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    retargeted = require_success(result.study.retarget(np.exp(data["z"] / 3), depends_on=["z"]))
    loaded = require_success(ant.load(bytes(retargeted.export())))
    assert_answer_kept(loaded)
    weights_id = loaded.inspect().target_weights_id
    assert isinstance(weights_id, str) and hex64.fullmatch(weights_id)
    assert retargeted.inspect().target_weights_id == weights_id
    assert retargeted.inspect().claim_id == loaded.inspect().claim_id
    original = require_success(ant.load(bytes(result.export())))
    assert_answer_kept(original)
    assert original.inspect().target_weights_id is None
    assert loaded.inspect().target_id != original.inspect().target_id
    assert loaded.inspect().identification_id == original.inspect().identification_id
    other = require_success(
        ant.load(
            bytes(
                require_success(
                    result.study.retarget(np.exp(data["z"] / 2), depends_on=["z"])
                ).export()
            )
        )
    )
    assert_answer_kept(other)
    assert hex64.fullmatch(other.inspect().target_weights_id)
    assert other.inspect().target_weights_id != weights_id
    uniform = require_success(
        ant.load(
            bytes(
                require_success(
                    result.study.retarget(np.ones(len(data["t"])), depends_on=[])
                ).export()
            )
        )
    )
    assert_answer_kept(uniform)
    assert uniform.inspect().target_weights_id is None
    assert uniform.inspect().target_id == original.inspect().target_id


def test_prepare_entry_points_share_omitted_defaults():
    data = sample()
    one_shot = ant.prepare(
        data, graph=GRAPH, query=QUERY, inference=ant.Frequentist(bootstrap=0), refute="none"
    )
    from antecedent.prepared import prepare

    prepared = prepare(
        data, graph=GRAPH, query=QUERY, inference=ant.Frequentist(bootstrap=0), refute="none"
    )
    assert one_shot.inspect().identification.available
    assert prepared.inspect().identification.available


def test_analyze_is_prepare_then_estimate():
    data = sample()
    analyzed = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            execution=ant.Execution(seed=19),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    prepared = ant.prepare(
        data,
        graph=GRAPH,
        query=QUERY,
        execution=ant.Execution(seed=1),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    branch = prepared.reconfigure(execution=ant.Execution(seed=19))
    estimated = require_success(branch.estimate())
    assert prepared.execution.seed == 1
    assert branch.execution.seed == 19
    assert analyzed.inspect().claim_id == estimated.inspect().claim_id


def test_workflow_signatures_remain_inspectable():
    import inspect
    import typing

    for fn in (ant.analyze, ant.prepare, ant.load, ant.PreparedAnalysis.estimate):
        assert typing.get_type_hints(fn)["return"] is not None
        assert "kwargs" not in inspect.signature(fn).parameters


def test_response_envelope_does_not_render_as_an_unrestricted_curve():
    t = np.random.default_rng(31).integers(0, 2, size=160).astype(float)
    data = {"t": t, "y": 2 * t + np.random.default_rng(32).normal(size=160)}
    graph = ant.Cpdag.from_directed_undirected(["t", "y"], directed=[], undirected=[("t", "y")])
    result = require_success(
        ant.analyze(
            data,
            graph=graph,
            query=ant.InterventionResponse("y", intervention=ant.intervention.Set("t", 1.0)),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    assert result.envelope is not None
    assert result.answer.kind == "partial"
    assert "partial" in repr(result)
    assert result.answer.value is None


def test_cpdag_partial_identification_with_full_mass_reports_its_identified_set(monkeypatch):
    z = np.linspace(0.0, 1.0, 200)
    t = (z > 0.5).astype(float)
    graph = ant.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    result = require_success(
        ant.analyze(
            {"t": t, "y": 1 + 2 * t + 3 * z, "z": z},
            graph=graph,
            query=QUERY,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
        )
    )
    assert result.inspect().identification.payload["identified_mass"] == 1.0
    # Completions disagree, so the answer is the set they span, never a scalar.
    assert result.answer.kind == "bounds"
    assert result.answer.value is None
    lower, upper = result.answer.bounds
    assert lower < upper, "the two completions disagree, so the set is non-degenerate"
    assert result.answer.bounds == result.structural_identified_set
    assert result.answer.detail == "identified_set"
    with pytest.warns(UserWarning, match="result.answer is the safe"):
        assert result.effect is not None
    monkeypatch.setenv("ANTECEDENT_STRICT_ANSWER", "1")
    with pytest.raises(ant.errors.CausalUnsupportedError, match="ANTECEDENT_STRICT_ANSWER"):
        _ = result.effect


@pytest.mark.parametrize("kind", ["cpdag", "pag"])
def test_class_posterior_bounds_answer_survives_loading(kind):
    """Practitioner S: degenerate bounds retain the identified-set disclosure."""
    names = ["t", "y", "z"]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    graph = (
        ant.Cpdag.from_directed_undirected(names, edges, [])
        if kind == "cpdag"
        else ant.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in edges])
    )
    if kind == "pag":
        # z is a visibility witness for t -> y; z is not adjacent to y.
        graph = ant.Pag.from_marked_edges(
            names, [("z", "t", "tail", "arrow"), ("t", "y", "tail", "arrow")]
        )
    data = sample()
    if kind == "pag":
        data["y"] = data["y"] - data["z"]
    posterior = ant.discovery.GraphPosterior.from_graphs(names, [0.4, 0.6], [graph, graph])
    result = require_success(
        ant.analyze(
            data,
            discovery=posterior,
            query=QUERY,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=731),
        )
    )
    assert result.answer.kind == "bounds"
    assert result.answer.detail == "identified_set"
    loaded = require_success(ant.load(bytes(result.export())))
    assert_answer_kept(loaded)
    assert loaded.answer == result.answer
