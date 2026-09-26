"""Exact-law stages have native authority and make no sampling-coverage claim."""

from dataclasses import replace

import pytest
from antecedent import Admg
from antecedent.errors import CausalCancelledError, CausalResourceError, CausalUnsupportedError
from antecedent.transport import advanced as transport


def fixture():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identification = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime("obs", "target", measured=["x", "y"]),
        ]
    )
    law = transport.ExactDiscreteLaw(
        "target",
        "obs",
        (("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.4, 0.1, 0.15, 0.35),
        "snapshot-1",
    )
    return identification, catalog, transport.ExactTransportData((law,))


def test_exact_full_distribution_and_native_display_authority():
    identified, catalog, data = fixture()
    assert identified.outcome == "identified"
    assert identified.formula
    result = transport.evaluate_exact(identified, catalog, data, at={"x": 1.0})
    assert result.probabilities == pytest.approx((0.3, 0.7))
    assert result.mean("y") == pytest.approx(0.7)
    assert result.uncertainty is None
    edited = replace(identified, outcomes=("unrelated",), outcome="not_certified", formula="forged")
    result = transport.evaluate_exact(edited, catalog, data, at={"x": 1.0})
    assert result.outcomes == ("y",)
    assert result.probabilities == pytest.approx((0.3, 0.7))


def test_exact_invalid_law_and_budgets():
    identified, catalog, data = fixture()
    bad = transport.ExactTransportData((replace(data.laws[0], probabilities=(0.0,) * 4),))
    with pytest.raises(ValueError, match="unnormalized_law"):
        transport.evaluate_exact(identified, catalog, bad, at={"x": 1.0})
    with pytest.raises(CausalResourceError, match="budget"):
        transport.evaluate_exact(identified, catalog, data, at={"x": 1.0}, max_operations=1)
    with pytest.raises(CausalResourceError, match="memory[ _]budget"):
        transport.evaluate_exact(identified, catalog, data, at={"x": 1.0}, memory_bytes=1)


def test_classical_negative_is_scoped_to_selected_source():
    graph = Admg.from_edges(["x", "y"], [("x", "y")], [("x", "y")])
    result = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    assert result.outcome == "proven_non_transportable"
    assert result.formula is None


def test_catalog_reordering_preserves_bound_expression():
    identified, catalog, data = fixture()
    unused = transport.EvidenceRegime("unused", "source", measured=["x"])
    forward = replace(catalog, regimes=(*catalog.regimes, unused))
    reverse = replace(catalog, regimes=(unused, *catalog.regimes))
    a = transport.evaluate_exact(identified, forward, data, at={"x": 0.0})
    b = transport.evaluate_exact(identified, reverse, data, at={"x": 0.0})
    assert a == b


def test_catalog_search_uses_available_source_when_target_formula_cannot_bind():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "experiment", "source", kind="experimental", interventions=["x"], measured=["y"]
            ),
        ]
    )
    law = transport.ExactDiscreteLaw(
        "source",
        "experiment",
        (("y", (0.0, 1.0)),),
        (0.2, 0.8),
        "source-1",
        interventions=(("x", 1.0),),
    )
    result = transport.evaluate_exact(
        identified, catalog, transport.ExactTransportData((law,)), at={"x": 1.0}
    )
    assert result.probabilities == pytest.approx((0.2, 0.8))
    assert result.rules == ("transport.direct",)
    assert result.formula != identified.formula


def test_recursive_frontdoor_exact_distribution_matches_latent_enumeration():
    graph = Admg.from_edges(["x", "m", "y"], [("x", "m"), ("m", "y")], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    assert "sid.line8" in identified.rules

    def mass(level, probability):
        return probability if level else 1.0 - probability

    probabilities = tuple(
        sum(
            mass(u, 0.4)
            * mass(x, 0.2 + 0.6 * u)
            * mass(m, 0.15 + 0.55 * x)
            * mass(y, 0.1 + 0.45 * m + 0.2 * u)
            for u in (0, 1)
        )
        for x in (0, 1)
        for m in (0, 1)
        for y in (0, 1)
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime("observational", "target", measured=["x", "m", "y"]),
        ]
    )
    law = transport.ExactDiscreteLaw(
        "target",
        "observational",
        tuple((v, (0.0, 1.0)) for v in ("x", "m", "y")),
        probabilities,
        "latent-oracle",
    )
    result = transport.evaluate_exact(
        identified, catalog, transport.ExactTransportData((law,)), at={"x": 1.0}
    )
    assert result.probabilities == pytest.approx((0.505, 0.495))
    assert result.mean("y") == pytest.approx(0.495)
    assert result.uncertainty is None


def test_checked_exact_transport_execution_survives_builder_disposal():
    """The prepared exact evaluator retains its checked target and replays independently."""
    from antecedent import Admg

    graph_builder = Admg.from_edges(["x", "y"], [("x", "y")])
    builder = transport.identify_classical(
        graph_builder,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    identification = builder
    catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("observational", "target", measured=["x", "y"])],
    )
    law = transport.ExactDiscreteLaw(
        "target",
        "observational",
        (("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.4, 0.1, 0.15, 0.35),
        "oracle-law",
    )
    program = identification.formula
    assert program
    del builder, graph_builder
    prepared = transport.prepare_exact(
        identification,
        catalog,
        transport.ExactTransportData((law,)),
        at={"x": 1.0},
    )
    del identification
    retained_plan = prepared.inspect()
    assert retained_plan.identification.available
    result = prepared.estimate()
    # Independent conditional-probability calculation: .35 / (.15 + .35) = .7.
    assert result.probabilities == pytest.approx((0.3, 0.7))
    consumer = transport.consume_exact(prepared.export())
    assert consumer.inspect().program_id == retained_plan.program_id
    assert consumer.estimate().probabilities == pytest.approx((0.3, 0.7))


def test_inspect_does_not_evaluate_or_fit(monkeypatch):
    from antecedent import prepare

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("evaluate")
        raise AssertionError("inspect must not evaluate or fit")

    monkeypatch.setattr(transport, "evaluate_exact", boom)
    report = prepared.inspect()
    assert report.identification.available
    assert not report.uncertainty.available
    assert calls == []


def test_causaleffect_standardize_formula_matches_enumerated_scm():
    graph = Admg.from_edges(["X", "Z", "Y"], [("Z", "Y"), ("X", "Y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["Z"]),
        outcomes=["Y"],
        treatments=["X"],
    )
    assert identified.outcome == "identified"
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "src",
                "source",
                kind="experimental",
                interventions=["X"],
                measured=["Z", "Y"],
            ),
            transport.EvidenceRegime("tgt", "target", measured=["Z"]),
        ]
    )
    source = transport.ExactDiscreteLaw(
        "source",
        "src",
        (("Z", (0.0, 1.0)), ("Y", (0.0, 1.0))),
        (0.48, 0.12, 0.08, 0.32),
        "src-v1",
        interventions=(("X", 1.0),),
    )
    target = transport.ExactDiscreteLaw(
        "target",
        "tgt",
        (("Z", (0.0, 1.0)),),
        (0.6, 0.4),
        "tgt-v1",
    )
    result = transport.evaluate_exact(
        identified, catalog, transport.ExactTransportData((source, target)), at={"X": 1.0}
    )
    assert result.mean("Y") == pytest.approx(0.6 * 0.2 + 0.4 * 0.8)


def test_common_prepared_lifecycle_export_consume_atomic_refresh():
    from antecedent import load, prepare
    from antecedent.estimation import PreparedAnalysis

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    assert isinstance(prepared, PreparedAnalysis)
    before = prepared.inspect()
    assert before.identification.available
    assert not before.support.available
    result = prepared.estimate()
    assert result.mean("y") == pytest.approx(0.7)
    assert not prepared.inspect().uncertainty.available
    assert load(prepared.export()).probabilities == result.probabilities
    restored = transport.consume_exact(prepared.export())
    assert restored.inspect().program_id == before.program_id
    assert restored.estimate() == result
    unsupported = transport.ExactTransportData(
        (replace(data.laws[0], probabilities=(0.8, 0.2, 0.0, 0.0)),)
    )
    with pytest.raises(ValueError, match="zero denominator"):
        prepared.refresh(unsupported)
    assert prepared.inspect().execution_id == before.execution_id
    updated = transport.ExactTransportData(
        (replace(data.laws[0], probabilities=(0.1, 0.4, 0.4, 0.1), snapshot_identity="two"),)
    )
    prepared.replace_snapshot(updated)
    assert prepared.inspect().program_id == before.program_id
    assert prepared.inspect().execution_id != before.execution_id
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        prepared.export()
    with pytest.raises(CausalUnsupportedError, match="stale_request"):
        prepared._native.estimate(execution=before.execution_id)
    assert prepared.refresh(data) == result


def test_changed_evidence_contract_cannot_use_stale_preparation():
    from antecedent import prepare

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    result = prepared.estimate()
    before = prepared.inspect()
    # The same regime name now denotes an experiment on x; its law is a different object.
    experimental = replace(
        catalog,
        regimes=(
            transport.EvidenceRegime(
                "obs", "target", kind="experimental", interventions=["x"], measured=["y"]
            ),
        ),
    )
    world = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("y", (0.0, 1.0)),),
                (0.3, 0.7),
                "snapshot-1",
                interventions=(("x", 1.0),),
            ),
        )
    )
    with pytest.raises(ValueError, match="evidence regime"):
        prepared.refresh(world)
    assert prepared.inspect().execution_id == before.execution_id
    assert prepared.refresh(data) == result
    # The old identification does not bind under the changed contract either.
    with pytest.raises(CausalUnsupportedError, match="missing_evidence"):
        prepare(world, query=transport.ExactTransportQuery(identified, experimental, {"x": 1.0}))


def test_catalog_standardization_uses_only_supplied_target_covariate_marginal():
    graph = Admg.from_edges(["z", "x", "y"], [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["z"]),
        outcomes=["y"],
        treatments=["x"],
    )
    del graph
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("covariates", "target", measured=["z"]),
        ]
    )
    data = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "trial",
                (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
                (0.32, 0.08, 0.12, 0.48),
                "trial",
                interventions=(("x", 1.0),),
            ),
            transport.ExactDiscreteLaw(
                "target", "covariates", (("z", (0.0, 1.0)),), (0.75, 0.25), "target"
            ),
        )
    )
    prepared = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    result = prepared.estimate()
    assert result.mean("y") == pytest.approx(0.35)
    assert "transport.pretreatment_standardize" in result.rules
    assert transport.consume_exact(prepared.export()).estimate() == result


def test_common_prepare_entry_and_exact_contrast():
    from antecedent import prepare

    identified, catalog, data = fixture()
    a = prepare(
        data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0})
    ).estimate()
    b = prepare(
        data, query=transport.ExactTransportQuery(identified, catalog, {"x": 0.0})
    ).estimate()
    assert a.contrast(b, "y") == pytest.approx(0.5)


def test_checked_irrelevant_zero_in_standardization_and_relevant_zero_failure():
    graph = Admg.from_edges(["z", "x", "y"], [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["z"]),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("covariates", "target", measured=["z"]),
        ]
    )
    source = transport.ExactDiscreteLaw(
        "source",
        "trial",
        (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.7, 0.3, 0.0, 0.0),
        "trial",
        interventions=(("x", 1.0),),
    )
    target = transport.ExactDiscreteLaw(
        "target", "covariates", (("z", (0.0, 1.0)),), (1.0, 0.0), "target"
    )
    prepared = transport.prepare_exact(
        identified, catalog, transport.ExactTransportData((source, target)), at={"x": 1.0}
    )
    result = prepared.estimate()
    assert result.mean("y") == pytest.approx(0.3)
    assert transport.consume_exact(prepared.export()).estimate() == result
    with pytest.raises(ValueError, match="zero_conditioning_mass"):
        prepared.refresh(
            transport.ExactTransportData((source, replace(target, probabilities=(0.5, 0.5))))
        )


def test_result_reasoning_and_export_retain_native_authority_after_display_edits():
    identified, catalog, data = fixture()
    result = transport.evaluate_exact(identified, catalog, data, at={"x": 1.0})
    report = result.inspect()
    assert report.identification.available
    assert report.support.available
    assert report.support.payload["factor_support"]
    assert not report.uncertainty.available
    assert report.assumptions.available
    assert set(result.to_dict()["reasoning"]) >= {
        "identification",
        "support",
        "uncertainty",
        "assumptions",
    }
    edited = replace(result, probabilities=(0.99, 0.01), formula="edited")
    assert transport.consume_exact(edited.export()).estimate().probabilities == result.probabilities


def test_snapshot_axis_permutation_and_shared_transform_contract():
    identified, catalog, data = fixture()
    prepared = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    before = prepared.inspect()
    permuted = replace(
        data.laws[0],
        axes=tuple(reversed(data.laws[0].axes)),
        probabilities=(0.4, 0.15, 0.1, 0.35),
        snapshot_identity="permuted",
    )
    prepared.replace_snapshot(transport.ExactTransportData((permuted,)))
    assert prepared.inspect().program_id == before.program_id
    assert prepared.estimate().mean("y") == pytest.approx(0.7)
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["refused"] == "false"
    assert preview["input_identification"] == before.identification_id
    assert prepared.preview_transform("change_graph")["refused"] == "true"


def test_cancelled_exact_click_and_prepare_publish_no_claim():
    from antecedent import prepare
    from antecedent.state import CancellationToken

    identified, catalog, data = fixture()
    prepared = transport.prepare_exact(identified, catalog, data, at={"x": 1.0})
    token = CancellationToken()
    retained = transport.prepare_exact(identified, catalog, data, at={"x": 1.0}, cancel=token)
    prepared.estimate()
    artifact = prepared.export()
    identity = prepared.inspect().data_snapshot_id
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        retained.estimate()
    with pytest.raises(CausalCancelledError, match="cancel"):
        prepared.replace_snapshot(data, cancel=token)
    assert prepared.inspect().data_snapshot_id == identity
    assert prepared.export() == artifact
    with pytest.raises(CausalCancelledError, match="cancel"):
        transport.consume_exact(artifact, cancel=token)
    with pytest.raises(CausalCancelledError, match="cancel"):
        prepared.estimate(cancel=token)
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        prepared.export()
    with pytest.raises(CausalCancelledError, match="cancel"):
        prepare(
            data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}), cancel=token
        )


def test_catalog_diagnostics_distinguish_missing_evidence_and_future_experiments():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph, transport.SelectionDiagram("source", "target", []), outcomes=["y"], treatments=["x"]
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "future",
                "source",
                kind="experimental",
                interventions=["x"],
                measured=["y"],
                evidence_kind="proposed",
            ),
        ]
    )
    report = transport.inspect_catalog(identified, catalog)
    assert report["outcome"] == "missing_evidence"
    assert report["exhausted"]
    assert not report["finite_catalog_complete"]
    assert report["future_experiments"] == ["future"]
    assert report["missing_factors"]


def test_checked_proof_graph_names_factor_binding_failure():
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph, transport.SelectionDiagram("source", "target", []), outcomes=["y"], treatments=["x"]
    )
    catalog = transport.EvidenceCatalog(regimes=[])
    view = transport.inspect_proof_graph(identified, catalog)
    assert view["steps"]
    assert view["factors"]
    assert any(leaf["binding_failure"] for leaf in view["factors"])
    assert all(leaf["supplied_by"] is None for leaf in view["factors"])


def test_classical_checked_program_and_catalog_proof_survive_graph_builder_disposal():
    """Identification owns a checked expression and a proof of its factor bindings."""
    graph = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    builder = identified
    del graph
    catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("obs", "target", measured=["x", "y"])],
    )
    proof = transport.inspect_proof_graph(builder, catalog)
    assert proof["steps"]
    assert proof["factors"]
    assert all(leaf["supplied_by"] == 0 for leaf in proof["factors"])
    report = transport.inspect_catalog(builder, catalog)
    assert report["outcome"] == "identified"
    assert report["missing_factors"] == []
    program = builder.formula
    assert program is not None

    # The native checked derivation and executable wire are retained independently
    # of the graph/query objects used to ask for identification.
    data = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
                (0.4, 0.1, 0.15, 0.35),
                "proof-bound-law",
            ),
        )
    )
    prepared = transport.prepare_exact(builder, catalog, data, at={"x": 1.0})
    del builder, identified
    assert prepared.estimate().probabilities == pytest.approx((0.3, 0.7))
    assert transport.consume_exact(prepared.export()).estimate().probabilities == pytest.approx(
        (0.3, 0.7)
    )


def test_hypothetical_catalog_delta_does_not_change_supplied_evidence():
    base = transport.EvidenceCatalog.empty()
    proposal = transport.EvidenceCatalogDelta(
        [
            transport.EvidenceRegime(
                "future",
                "source",
                kind="experimental",
                evidence_kind="proposed",
                interventions=["z"],
                measured=["x", "y"],
            )
        ]
    )
    preview = proposal.preview(base)
    assert not base.has_available_experiment("source", ["z"])
    assert preview.has_available_experiment("source", ["z"])
    assert proposal.proposed_regimes[0].evidence_kind == "proposed"
