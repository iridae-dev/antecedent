"""Native identification read views reached through canonical preparations.

These exercise the native foundation, not the still separate public
``PreparedAnalysis.identification`` adapter. Explicit materialization is tested
for scientific completeness; its allocation admission remains separate work.
"""

from __future__ import annotations

import gc
import json
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import GraphPosterior
from antecedent.errors import CausalValueError
from antecedent.graph import TemporalCpdag, TemporalDag
from antecedent.outcomes import require_success

_FAMILIES = (
    "static",
    "pag",
    "cpdag",
    "temporal",
    "temporal_class",
    "graph_posterior",
    "dbn_posterior",
    "temporal_class_posterior",
)


def _static_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(91)
    z = rng.normal(size=160)
    t = (rng.uniform(size=160) < 1 / (1 + np.exp(-0.4 * z))).astype(float)
    return {"t": t, "y": 1 + 2 * t + 0.7 * z, "z": z}


def _pag_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(91)
    z = rng.normal(size=160)
    r = rng.normal(size=160)
    t = (rng.uniform(size=160) < 1 / (1 + np.exp(-0.4 * z - 0.3 * r))).astype(float)
    return {"t": t, "y": 1 + 2 * t + 0.7 * z, "z": z, "r": r}


def _temporal_data() -> dict[str, np.ndarray]:
    t = np.random.default_rng(37).normal(size=160)
    y = np.ones(160)
    y[1:] += 2 * t[:-1]
    return {"t": t, "y": y}


def _prepare(family: str, *, curve: bool = False) -> Any:
    options: dict[str, Any] = {
        "inference": ant.Frequentist(bootstrap=0),
        "execution": ant.Execution(seed=19, threads=1),
        "refute": "none",
    }
    if family in {"temporal", "temporal_class", "dbn_posterior", "temporal_class_posterior"}:
        temporal_class = family in {"temporal_class", "temporal_class_posterior"}
        graph_type = TemporalCpdag if temporal_class else TemporalDag
        graph = graph_type.from_lagged_edges(["t", "y"], [("t", 1, "y", 0)])
        query = (
            ant.ResponseCurve(
                "t", "y", grid=[0.0, 1.0], horizons=[1, 2], policy="pulse", treatment_lag=1
            )
            if curve
            else ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1, active_level=1.0)
        )
        data = _temporal_data()
    else:
        names = ["t", "y", "z"]
        edges = [("z", "t"), ("z", "y"), ("t", "y")]
        if family == "pag":
            # R is nonadjacent to Y, making T→Y visible in MAG semantics.
            # Z still confounds T and Y and must remain in the adjustment set.
            names.append("r")
            edges.append(("r", "t"))
            graph = ant.Pag.from_marked_edges(
                names, [(source, target, "tail", "arrow") for source, target in edges]
            )
        elif family == "cpdag":
            graph = ant.Cpdag.from_directed_undirected(names, edges, [])
        else:
            graph = ant.Dag.from_edges(names, edges)
        query = ant.AverageEffect("t", "y")
        data = _pag_data() if family == "pag" else _static_data()
    if family in {"graph_posterior", "dbn_posterior", "temporal_class_posterior"}:
        options["discovery"] = GraphPosterior.from_graphs(list(data), [0.7, 0.3], [graph, graph])
    else:
        options["graph"] = graph
    return ant.prepare(ant.data.table(data), query=query, **options)


def _summary(view: Any) -> dict[str, Any]:
    return json.loads(view.summary_json())


def _cache(study: Any, family: str) -> Any:
    view = study._engine._native.identification_view()
    assert _summary(view)["cache_kind"] == family
    # The lowered layer also retains caches. Select the installed cache directly,
    # avoiding duplicate proof counts across the layer and the study cache.
    return next(child for child in view.children() if _summary(child)["kind"] == family)


def _walk(view: Any) -> Any:
    yield view
    count = _summary(view)["children"]
    for offset in range(0, count, 2):
        for child in view.children(offset=offset, limit=2):
            yield from _walk(child)


def _nodes(view: Any, kind: str) -> list[Any]:
    return [node for node in _walk(view) if _summary(node)["kind"] == kind]


def _wire_nodes(wire: dict[str, Any]) -> Any:
    yield wire
    for child in wire["children"]:
        yield from _wire_nodes(child)


@pytest.mark.parametrize("family", _FAMILIES)
def test_native_readview_preserves_each_prepared_cache_and_complete_proofs(family):
    study = _prepare(family)
    view = _cache(study, family)
    proofs = _nodes(view, "proof")
    assert proofs, f"{family} lost every retained identification proof"
    for proof in proofs:
        summary = _summary(proof)
        assert summary["status"] == "nonparametrically_identified"
        assert summary["estimand_count"] >= 1
        # Aggregate class proofs retain a merged arena and assumptions, with
        # their original empty derivation/performance counters. Per-case proofs
        # retain their own evidence; inspection must not invent aggregate work.
        assert summary["sets_returned"] >= 0
        assert summary["derivation_step_count"] >= 0

    wire = json.loads(view.materialize_json(seed=19, threads=1))
    converted = [node for node in _wire_nodes(wire) if "identification" in node]
    assert len(converted) == len(proofs)
    for source, converted_proof in zip(proofs, converted, strict=True):
        proof = converted_proof["identification"]
        summary = _summary(source)
        assert proof["status"] == summary["status"]
        assert proof["query"]
        assert proof["arena"]
        assert len(proof["estimands"]) == summary["estimand_count"]
        assert len(proof["derivation"]) == summary["derivation_step_count"]
        assert len(proof["required_assumptions"]) == summary["assumption_count"]
        assert proof["candidates_examined"] == summary["candidates_examined"]
    assert study._engine.last_result is None


@pytest.mark.parametrize("family", _FAMILIES)
def test_reads_and_repr_do_not_enter_execution_or_legacy_identification_conversion(
    family, monkeypatch
):
    from antecedent.estimation import PreparedAnalysis as Engine
    from antecedent.identify import Identification

    study = _prepare(family)

    def forbidden(*args, **kwargs):
        pytest.fail("bounded identification inspection entered execution or legacy conversion")

    monkeypatch.setattr(ant.Execution, "_call", forbidden)
    monkeypatch.setattr(Engine, "estimate", forbidden)
    monkeypatch.setattr(Engine, "identification", property(forbidden))
    monkeypatch.setattr(Identification, "from_prepared", forbidden)
    root = study._engine._native.identification_view()
    summary = _summary(root)
    assert summary["kind"] == "identification"
    assert summary["retained"]["kind"] == family
    assert len(root.summary_json()) < 4096
    assert "PreparedIdentificationView" in repr(root)
    for node in _walk(root):
        assert "children=" in repr(node)
        assert "identification" not in _summary(node)
    assert study._engine.last_result is None


@pytest.mark.parametrize("family", ["cpdag", "pag", "temporal_class"])
def test_completion_cases_retain_mass_graphs_and_invariants(family):
    view = _cache(_prepare(family), family)
    envelopes = _nodes(view, "completion_envelope")
    assert envelopes
    for envelope in envelopes:
        summary = _summary(envelope)
        assert summary["identified_weight"] == pytest.approx(1)
        assert summary["unidentified_weight"] == pytest.approx(0)
        assert summary["truncated_completions"] == 0
        assert summary["case_count"] == summary["children"] == 1
        wire = json.loads(envelope.materialize_json())
        assert wire["invariant"] is not None
        case = wire["children"][0]
        assert case["summary"]["weight"] == pytest.approx(1)
        assert case["graph"]
        assert case["children"][0]["identification"]["status"] == "nonparametrically_identified"


@pytest.mark.parametrize("family", ["graph_posterior", "dbn_posterior", "temporal_class_posterior"])
def test_duplicate_posterior_samples_keep_original_order_and_mass(family):
    view = _cache(_prepare(family), family)
    summary = _summary(view)
    assert summary["status"] is None
    assert summary["mass_available"] == "per_sample"
    assert summary["sample_count"] == 2
    samples = _nodes(view, "posterior_sample")
    assert [_summary(sample)["weight"] for sample in samples] == [0.7, 0.3]
    assert all(_summary(sample)["identified"] is True for sample in samples)
    atom_kind = "class_atom" if family == "temporal_class_posterior" else "identified_atom"
    atoms = _nodes(view, atom_kind)
    assert {_summary(atom)["key"] for atom in atoms} == {
        _summary(sample)["key"] for sample in samples
    }
    assert len(atoms) == (1 if family == "graph_posterior" else 2)
    if family == "temporal_class_posterior":
        for atom in atoms:
            assert _summary(atom)["identified_weight"] == pytest.approx(1)
            assert _summary(atom)["truncated_completions"] == 0


def test_partial_posterior_keeps_unidentified_first_sample_and_identified_later_atom():
    names = ["t", "a", "b", "y"]
    directed = [("t", "a"), ("a", "b"), ("b", "y")]
    # This bow-free graph is one C-component. After removing T, ID factors
    # into {A,Y} and {B}; the {A,Y} recursion reaches the single-component
    # hedge failure (general.id.line5). No packed edge has two meanings.
    hedge = ant.Admg.from_edges(
        names, directed=directed, bidirected=[("t", "b"), ("t", "y"), ("a", "y")]
    )
    identified = ant.Admg.from_edges(names, directed=directed, bidirected=[])
    posterior = GraphPosterior.from_graphs(names, [0.3, 0.7], [hedge, identified])
    rng = np.random.default_rng(91)
    t = rng.integers(0, 2, size=160).astype(float)
    a = t + rng.normal(size=160)
    b = a + rng.normal(size=160)
    rows = {"t": t, "a": a, "b": b, "y": b + rng.normal(size=160)}
    study = ant.prepare(
        ant.data.table({name: rows[name] for name in names}),
        discovery=posterior,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=19, threads=1),
        refute="none",
    )
    view = _cache(study, "graph_posterior")
    samples = [_summary(sample) for sample in _nodes(view, "posterior_sample")]
    assert [sample["weight"] for sample in samples] == [0.3, 0.7]
    assert [sample["identified"] for sample in samples] == [False, True]
    assert sum(sample["weight"] for sample in samples if sample["identified"]) == pytest.approx(0.7)
    atoms = _nodes(view, "identified_atom")
    assert len(atoms) == 1
    assert _summary(atoms[0])["key"] == samples[1]["key"]
    assert _summary(view)["status"] is None
    wire = json.loads(view.materialize_json())
    materialized_samples = [
        node["summary"]
        for node in _wire_nodes(wire)
        if node["summary"]["kind"] == "posterior_sample"
    ]
    assert materialized_samples == samples


@pytest.mark.parametrize("edge", [("t", "y"), ("y", "t")])
def test_posterior_rejects_bows_without_changing_explicit_admg(edge):
    names = ["t", "y"]
    bow = ant.Admg.from_edges(names, directed=[edge], bidirected=[("t", "y")])
    assert bow.children(edge[0]) == [edge[1]]
    assert bow.bidirected_neighbors("t") == ["y"]
    with pytest.raises(CausalValueError, match="cannot represent a bow"):
        GraphPosterior.from_graphs(names, [1.0], [bow])
    # Rejecting the posterior encoding does not narrow the explicit native ADMG
    # graph interface: its actual bow has a hedge, not an identified zero effect.
    proof = ant.identify(graph=bow, query=ant.AverageEffect(edge[0], edge[1]))
    assert proof.status == "NotIdentified"
    assert bow.children(edge[0]) == [edge[1]]
    assert bow.bidirected_neighbors("t") == ["y"]


def test_pag_confounded_triangle_retains_native_nonidentification():
    names = ["t", "y", "z"]
    graph = ant.Pag.from_marked_edges(
        names,
        [
            (source, target, "tail", "arrow")
            for source, target in [("z", "t"), ("z", "y"), ("t", "y")]
        ],
    )
    study = ant.prepare(
        ant.data.table(_static_data()),
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=19, threads=1),
        refute="none",
    )
    view = _cache(study, "pag")
    envelopes = _nodes(view, "completion_envelope")
    assert envelopes
    for envelope in envelopes:
        summary = _summary(envelope)
        assert summary["status"] == "not_identified"
        assert summary["identified_weight"] == pytest.approx(0)
        assert summary["unidentified_weight"] == pytest.approx(1)
        assert summary["truncated_completions"] == 0
    proofs = _nodes(view, "proof")
    assert proofs
    assert all(_summary(proof)["status"] == "not_identified" for proof in proofs)
    assert all(_summary(proof)["estimand_count"] == 0 for proof in proofs)
    assert study._engine.last_result is None


@pytest.mark.parametrize("family", ["cpdag", "pag"])
def test_static_class_posterior_preserves_every_case_and_selected_estimand(family):
    explicit = _prepare(family)
    graph = explicit._declarations["graph"]
    data = _pag_data() if family == "pag" else _static_data()
    posterior = GraphPosterior.from_graphs(list(data), [0.7, 0.3], [graph, graph])
    study = ant.prepare(
        ant.data.table(data),
        discovery=posterior,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=19, threads=1),
        refute="none",
    )
    view = _cache(study, "graph_posterior")
    samples = [_summary(sample) for sample in _nodes(view, "posterior_sample")]
    assert [sample["weight"] for sample in samples] == [0.7, 0.3]
    atoms = _nodes(view, "class_atom")
    assert {_summary(atom)["key"] for atom in atoms} == {sample["key"] for sample in samples}
    assert not _nodes(view, "identified_atom")
    for atom in atoms:
        summary = _summary(atom)
        assert summary["identified_weight"] == pytest.approx(1)
        assert summary["truncated_completions"] == 0
        wire = json.loads(atom.materialize_json())
        assert wire["invariant"] is not None
        cases = [node for node in _wire_nodes(wire) if node["summary"]["kind"] == "completion_case"]
        assert len(cases) == 1
        assert cases[0]["summary"]["weight"] == pytest.approx(1)
        assert cases[0]["selected_estimand"] == wire["invariant"]
        # Every completion retains its own full proof as well as the selected
        # estimand. Its local derivation is not replaced by the aggregate proof.
        case_proof = cases[0]["children"][0]["identification"]
        assert case_proof["status"] == cases[0]["summary"]["status"]
        assert case_proof["arena"]
        assert any(
            all(candidate.get(key) == value for key, value in cases[0]["selected_estimand"].items())
            for candidate in case_proof["estimands"]
        )
        assert case_proof["derivation"]


@pytest.mark.parametrize("family", ["temporal", "temporal_class", "dbn_posterior"])
def test_multiple_horizons_preserve_order_and_each_proof_namespace(family):
    view = _cache(_prepare(family, curve=True), family)
    groups = _nodes(view, "horizons") + _nodes(view, "horizon_envelopes")
    assert groups
    for group in groups:
        horizons = group.children()
        assert [_summary(horizon)["horizon"] for horizon in horizons] == [1, 2]
        for horizon in horizons:
            expected = _summary(horizon)["horizon"]
            proofs = _nodes(horizon, "proof")
            assert proofs
            for proof in proofs:
                namespace = _summary(proof)["temporal_namespace"]
                assert namespace["horizon"] == expected
                assert namespace["variable_count"] == 2
                assert namespace["history"] >= 1


def test_paging_bounds_and_child_owner_lifetime():
    study = _prepare("graph_posterior")
    root = study._engine._native.identification_view()
    view = _cache(study, "graph_posterior")
    samples = _nodes(view, "posterior_samples")[0]
    assert len(samples.children(offset=0, limit=1)) == 1
    assert _summary(samples.children(offset=1, limit=1)[0])["weight"] == 0.3
    assert samples.children(offset=2, limit=1) == []
    assert samples.children(offset=0, limit=0) == []
    with pytest.raises(CausalValueError, match="at most 256"):
        samples.children(limit=257)
    with pytest.raises(CausalValueError, match="out of range"):
        samples.child(2)
    child = samples.child(1)
    before = json.loads(view.materialize_json())
    del study, root, samples
    gc.collect()
    assert _summary(child)["weight"] == 0.3
    assert json.loads(view.materialize_json()) == before


def test_graph_branch_preserves_earlier_adjustment_proof():
    study = _prepare("static")
    original = _cache(study, "static")
    before = json.loads(original.materialize_json())
    original_names = original.schema_names()
    adjustment = before["selected_estimand"]["adjustment_set"]
    assert [original_names[index] for index in adjustment] == ["z"]
    alternative = ant.reconfigure(
        study,
        graph=ant.Dag.from_edges(["t", "y", "z"], [("t", "y")]),
    )
    after = json.loads(_cache(alternative, "static").materialize_json())
    assert after["selected_estimand"]["adjustment_set"] == []
    assert json.loads(original.materialize_json()) == before
    assert study._engine.last_result is None and alternative._engine.last_result is None


@pytest.mark.parametrize("family", ["static", "pag", "temporal"])
def test_inspection_then_explicit_execution_retains_known_effect(family):
    study = _prepare(family)
    view = _cache(study, family)
    before = json.loads(view.materialize_json())
    result = require_success(ant.estimate(study))
    np.testing.assert_allclose(result.estimate.ate, 2.0, atol=1e-8, rtol=0)
    assert json.loads(view.materialize_json()) == before
