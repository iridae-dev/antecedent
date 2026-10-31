"""Exact-law stages have native authority and make no sampling-coverage claim."""

from dataclasses import replace

import antecedent as _ant_api
import pytest
from antecedent import Admg
from antecedent.errors import CausalCancelledError, CausalResourceError, CausalUnsupportedError
from antecedent.outcomes import require_success
from antecedent.transport import TransportEvaluation
from antecedent.transport import advanced as transport


def exact_study(identified, catalog, data, at, **options):
    """Prepare the typed exact question on the common immutable lifecycle."""
    return _ant_api.prepare(
        data, query=transport.ExactTransportQuery(identified, catalog, at), **options
    )


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


def test_prepared_fixed_graph_root_mechanism_sensitivity_smoke():
    graph = Admg.from_edges(["u", "x", "y"], [("u", "x"), ("u", "y"), ("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["x"]),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["u", "y"]
            ),
            transport.EvidenceRegime("population", "target", measured=["u"]),
        ],
        bindings=[
            transport.RegimeBinding(
                "trial", "source-snapshot", schema_names=["u", "y"], sampling="independent"
            ),
            transport.RegimeBinding(
                "population", "target-snapshot", schema_names=["u"], sampling="independent"
            ),
        ],
        target_sampling="supplied_population_law",
    )
    source = transport.ExactDiscreteLaw(
        "source",
        "trial",
        (("u", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.6, 0.15, 0.175, 0.075),
        "source-snapshot",
        interventions=(("x", 1.0),),
    )
    target = transport.ExactDiscreteLaw(
        "target", "population", (("u", (0.0, 1.0)),), (0.75, 0.25), "target-snapshot"
    )
    prepared = exact_study(
        identified, catalog, transport.ExactTransportData((source, target)), {"x": 1.0}
    )
    report = prepared.mechanism_sensitivity(
        outcome_values=[0.0, 1.0],
        parent_cardinalities=[2, 2],
        treatment_levels=(0, 1),
        max_fraction=0.2,
        source_kernel_regime=1,
        source_kernel_snapshot="source-snapshot",
        target_parent_regime=0,
        target_parent_snapshot="target-snapshot",
        source_kernel=[
            ([0, 0], [0.8, 0.2]),
            ([0, 1], [0.2, 0.8]),
            ([1, 0], [0.7, 0.3]),
            ([1, 1], [0.3, 0.7]),
        ],
        source_parent_law=[([0], 0.75), ([1], 0.25)],
        target_parent_law=[
            (0, [0], 0.75),
            (0, [1], 0.25),
            (1, [0], 0.75),
            (1, [1], 0.25),
        ],
        decision_threshold=0.53,
        perturbed_conditional_mechanism="u",
    )
    original = prepared.inspect().execution_id
    report = require_success(report)
    assert prepared.inspect().execution_id == original
    assert report["perturbed_conditional_mechanism"] == "u"
    assert report["baseline"] == pytest.approx(0.55)
    assert report["assumption_range"] == pytest.approx({"minimum": 0.52, "maximum": 0.56})
    assert report["tipping_fraction"] == pytest.approx(2 / 15)


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


def test_evaluate_exact_executes_the_retained_program_after_builder_disposal():
    """Exact evaluation binds the retained checked derivation, not the graph or display fields."""
    graph_builder = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph_builder,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    _, catalog, data = fixture()
    program = identified.formula
    assert program
    del graph_builder
    result = transport.evaluate_exact(identified, catalog, data, at={"x": 1.0})
    # Independent conditional-probability calculation: .35 / (.15 + .35) = .7.
    assert result.probabilities == pytest.approx((0.3, 0.7))
    assert result.mean("y") == pytest.approx(0.7)
    assert result.uncertainty is None
    plan = result.inspect()
    assert plan.identification.available
    assert not plan.uncertainty.available
    # Forged display fields cannot steer execution; the native derivation is authoritative.
    edited = replace(identified, outcomes=("unrelated",), outcome="not_certified", formula="forged")
    del identified
    again = transport.evaluate_exact(edited, catalog, data, at={"x": 1.0})
    assert again.outcomes == ("y",)
    assert again.probabilities == pytest.approx((0.3, 0.7))
    assert again.inspect().program_id == plan.program_id


def test_exact_invalid_law_and_budgets():
    identified, catalog, data = fixture()
    bad = transport.ExactTransportData((replace(data.laws[0], probabilities=(0.0,) * 4),))
    with pytest.raises(ValueError, match="unnormalized_law"):
        transport.evaluate_exact(identified, catalog, bad, at={"x": 1.0})
    with pytest.raises(CausalResourceError, match="budget"):
        transport.evaluate_exact(
            identified,
            catalog,
            data,
            at={"x": 1.0},
            evaluation=TransportEvaluation(max_operations=1),
        )
    with pytest.raises(CausalResourceError, match="budget|memory|admission"):
        transport.evaluate_exact(
            identified,
            catalog,
            data,
            at={"x": 1.0},
            execution=_ant_api.Execution(limits=_ant_api.ResourceLimits(memory_bytes=1)),
        )


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
    prepared = exact_study(
        identification, catalog, transport.ExactTransportData((law,)), {"x": 1.0}
    )
    del identification
    retained_plan = prepared.inspect()
    assert retained_plan.identification.available
    outcome = _ant_api.estimate(prepared)
    result = require_success(outcome)
    # Independent conditional-probability calculation: .35 / (.15 + .35) = .7.
    assert result.probabilities == pytest.approx((0.3, 0.7))
    consumer = transport.consume_exact(bytes(outcome.export()))
    assert consumer.inspect().program_id == retained_plan.program_id
    assert consumer.probabilities == pytest.approx((0.3, 0.7))


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


def test_exact_inspection_publishes_formula_scope_factors_and_bindings_without_evaluating():
    from antecedent import prepare

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    report = prepared.inspect()
    identification = report.identification.payload
    assert identification["formula"] and identification["theorem_scope"]
    assert identification["classical_family"] and identification["catalog_family"]
    assert identification["rules"], "the consumed derivation rules are listed"
    # Support is execution-specific: nothing is evaluated by inspecting, so none is claimed.
    assert report.support.available is False
    assert report.support.summary == "execution_specific"
    factors = report.support.payload["required_factors"]
    assert factors and all(f["population"] and "variables" in f for f in factors)
    assert len(report.support.payload["bindings"]) == len(data.laws)
    assert report.support.payload["factor_support"] == ()
    # The identity layers are distinct hex digests of the program, the snapshot and the run.
    ids = {report.program_id, report.data_snapshot_id, report.execution_id}
    assert None not in ids and len(ids) == 3
    assert prepared.inspect().execution_id == report.execution_id


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


def test_common_prepared_lifecycle_export_load_and_rebind_branches():
    from antecedent import estimate, load, prepare, rebind
    from antecedent.prepared import PreparedAnalysis

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    assert isinstance(prepared, PreparedAnalysis)
    before = prepared.inspect()
    assert before.identification.available
    assert not before.support.available
    outcome = estimate(prepared)
    result = require_success(outcome)
    assert result.mean("y") == pytest.approx(0.7)
    assert not prepared.inspect().uncertainty.available
    wire = bytes(outcome.export())
    assert require_success(load(wire)).probabilities == result.probabilities
    restored = transport.consume_exact(wire)
    assert restored.inspect().program_id == before.program_id
    assert restored == result
    unsupported = transport.ExactTransportData(
        (replace(data.laws[0], probabilities=(0.8, 0.2, 0.0, 0.0)),)
    )
    with pytest.raises(ValueError, match="zero denominator"):
        require_success(estimate(rebind(prepared, data=unsupported)))
    assert prepared.inspect().execution_id == before.execution_id
    updated = transport.ExactTransportData(
        (replace(data.laws[0], probabilities=(0.1, 0.4, 0.4, 0.1), snapshot_identity="two"),)
    )
    branch = rebind(prepared, data=updated)
    assert branch.inspect().program_id == before.program_id
    assert branch.inspect().execution_id != before.execution_id
    # The source study and its completed outcome are not changed by the branch.
    assert prepared.inspect().execution_id == before.execution_id
    assert bytes(outcome.export()) == wire
    # The branch has produced no claim, and a request for the old execution is stale.
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        branch._engine._native.export()
    with pytest.raises(CausalUnsupportedError, match="stale_request"):
        branch._engine._native.estimate(execution=before.execution_id)
    assert require_success(estimate(rebind(branch, data=data))) == result


def test_consume_exact_estimates_from_the_embedded_law_after_builder_disposal():
    """The consumer rechecks the embedded exact-law claim with no producer alive."""
    from antecedent import estimate, load, prepare

    identified, catalog, data = fixture()
    producer_builder = prepare(
        data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0})
    )
    plan = producer_builder.inspect()
    assert plan.identification.available
    outcome = estimate(producer_builder)
    result = require_success(outcome)
    wire = bytes(outcome.export())
    del producer_builder, outcome, identified, catalog, data
    consumed = transport.consume_exact(wire)
    assert consumed.inspect().program_id == plan.program_id
    assert consumed.inspect().identification.available
    assert not consumed.inspect().uncertainty.available
    # Independent conditional-probability calculation: .35 / (.15 + .35) = .7.
    assert consumed.probabilities == pytest.approx((0.3, 0.7))
    assert consumed == result
    assert consumed.acceptance.verified
    assert not consumed.acceptance.resumable
    assert bytes(consumed.export()) == wire
    assert require_success(load(wire)).probabilities == result.probabilities


def test_changed_evidence_contract_cannot_use_stale_preparation():
    from antecedent import estimate, prepare, rebind

    identified, catalog, data = fixture()
    prepared = prepare(data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}))
    result = require_success(estimate(prepared))
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
        require_success(estimate(rebind(prepared, data=world)))
    assert prepared.inspect().execution_id == before.execution_id
    assert require_success(estimate(rebind(prepared, data=data))) == result
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
    outcome = _ant_api.estimate(exact_study(identified, catalog, data, {"x": 1.0}))
    result = require_success(outcome)
    assert result.mean("y") == pytest.approx(0.35)
    assert "transport.pretreatment_standardize" in result.rules
    assert transport.consume_exact(bytes(outcome.export())) == result


def test_common_prepare_entry_and_exact_contrast():
    from antecedent import prepare

    identified, catalog, data = fixture()
    a = require_success(
        prepare(
            data, query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0})
        ).estimate()
    )
    b = require_success(
        prepare(
            data, query=transport.ExactTransportQuery(identified, catalog, {"x": 0.0})
        ).estimate()
    )
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
    prepared = exact_study(
        identified, catalog, transport.ExactTransportData((source, target)), {"x": 1.0}
    )
    outcome = _ant_api.estimate(prepared)
    result = require_success(outcome)
    assert result.mean("y") == pytest.approx(0.3)
    assert transport.consume_exact(bytes(outcome.export())) == result
    with pytest.raises(ValueError, match="zero_conditioning_mass"):
        require_success(
            _ant_api.estimate(
                _ant_api.rebind(
                    prepared,
                    data=transport.ExactTransportData(
                        (source, replace(target, probabilities=(0.5, 0.5)))
                    ),
                )
            )
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
    assert transport.consume_exact(bytes(edited.export())).probabilities == result.probabilities


def test_snapshot_axis_permutation_and_shared_transform_contract():
    identified, catalog, data = fixture()
    prepared = exact_study(identified, catalog, data, {"x": 1.0})
    before = prepared.inspect()
    permuted = replace(
        data.laws[0],
        axes=tuple(reversed(data.laws[0].axes)),
        probabilities=(0.4, 0.15, 0.1, 0.35),
        snapshot_identity="permuted",
    )
    branch = _ant_api.rebind(prepared, data=transport.ExactTransportData((permuted,)))
    assert branch.inspect().program_id == before.program_id
    assert require_success(_ant_api.estimate(branch)).mean("y") == pytest.approx(0.7)
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["refused"] == "false"
    assert preview["input_identification"] == before.identification_id
    assert prepared.preview_transform("change_graph")["refused"] == "true"


def test_cancelled_exact_click_and_prepare_publish_no_claim():
    from antecedent import estimate, prepare, rebind, reconfigure
    from antecedent.state import CancellationToken

    identified, catalog, data = fixture()
    token = CancellationToken()
    prepared = exact_study(identified, catalog, data, {"x": 1.0})
    retained = exact_study(
        identified, catalog, data, {"x": 1.0}, execution=_ant_api.Execution(cancel=token)
    )
    cancelled = reconfigure(prepared, execution=_ant_api.Execution(cancel=token))
    outcome = estimate(prepared)
    artifact = bytes(outcome.export())
    identity = prepared.inspect().data_snapshot_id
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        estimate(retained)
    with pytest.raises(CausalCancelledError, match="cancel"):
        rebind(cancelled, data=data)
    assert prepared.inspect().data_snapshot_id == identity
    assert bytes(outcome.export()) == artifact
    with pytest.raises(CausalCancelledError, match="cancel"):
        transport.consume_exact(artifact, execution=_ant_api.Execution(cancel=token))
    with pytest.raises(CausalCancelledError, match="cancel"):
        estimate(cancelled)
    # A cancelled branch publishes no claim and leaves the original study usable.
    assert bytes(estimate(prepared).export()) == artifact
    with pytest.raises(CausalCancelledError, match="cancel"):
        prepare(
            data,
            query=transport.ExactTransportQuery(identified, catalog, {"x": 1.0}),
            execution=_ant_api.Execution(cancel=token),
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


def test_inspect_catalog_diagnoses_the_retained_proof_after_builder_disposal():
    """Catalog diagnostics read the retained proof; the same proof then executes when bound."""
    graph_builder = Admg.from_edges(["x", "y"], [("x", "y")])
    identified = transport.identify_classical(
        graph_builder,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    program = identified.formula
    assert program
    del graph_builder
    proposed = transport.EvidenceCatalog(
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
    report = transport.inspect_catalog(identified, proposed)
    assert report["outcome"] == "missing_evidence"
    assert report["exhausted"]
    assert not report["finite_catalog_complete"]
    assert report["future_experiments"] == ["future"]
    assert report["missing_factors"]
    _, catalog, data = fixture()
    bound = transport.inspect_catalog(identified, catalog)
    assert bound["outcome"] == "identified"
    assert bound["missing_factors"] == []
    assert bound["future_experiments"] == []
    plan = exact_study(identified, catalog, data, {"x": 1.0})
    del identified
    assert plan.inspect().identification.available
    # Independent conditional-probability calculation: .35 / (.15 + .35) = .7.
    assert require_success(plan.estimate()).probabilities == pytest.approx((0.3, 0.7))


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
    prepared = exact_study(builder, catalog, data, {"x": 1.0})
    del builder, identified
    outcome = _ant_api.estimate(prepared)
    assert require_success(outcome).probabilities == pytest.approx((0.3, 0.7))
    assert transport.consume_exact(bytes(outcome.export())).probabilities == pytest.approx(
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


def test_the_checked_proof_graph_of_a_frontdoor_derivation_is_a_closed_rooted_rule_graph():
    graph = Admg.from_edges(["x", "m", "y"], [("x", "m"), ("m", "y")], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["y"]),
        outcomes=["y"],
        treatments=["x"],
    )
    assert "sid.line8" in identified.rules
    view = transport.inspect_proof_graph(identified, transport.EvidenceCatalog(regimes=[]))
    steps = {step["index"]: step for step in view["steps"]}
    assert len(steps) == len(view["steps"]) >= 2
    referenced = set()
    for index, step in steps.items():
        assert step["rule"]
        # Every dependency is a step of the same graph that was derived before it.
        assert set(step["children"]) <= set(steps)
        assert all(child < index for child in step["children"])
        referenced.update(step["children"])
    # Exactly one step is nobody's dependency: the root.
    assert len(set(steps) - referenced) == 1
    factor_nodes = {factor["expression_node"] for factor in view["factors"]}
    assert factor_nodes
    assert all(set(step["factor_nodes"]) <= factor_nodes for step in steps.values())
    # With no evidence in the catalog every required factor fails to bind.
    assert all(factor["binding_failure"] for factor in view["factors"])
    assert all(factor["supplied_by"] is None for factor in view["factors"])


def _experimental_source_study():
    graph = Admg.from_edges(["u", "x", "y"], [("u", "x"), ("u", "y"), ("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["x"]),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["u", "y"]
            ),
            transport.EvidenceRegime("population", "target", measured=["u"]),
        ],
        bindings=[
            transport.RegimeBinding(
                "trial", "source-snapshot", schema_names=["u", "y"], sampling="independent"
            ),
            transport.RegimeBinding(
                "population", "target-snapshot", schema_names=["u"], sampling="independent"
            ),
        ],
        target_sampling="supplied_population_law",
    )
    source = transport.ExactDiscreteLaw(
        "source",
        "trial",
        (("u", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.6, 0.15, 0.175, 0.075),
        "source-snapshot",
        interventions=(("x", 1.0),),
    )
    target = transport.ExactDiscreteLaw(
        "target", "population", (("u", (0.0, 1.0)),), (0.75, 0.25), "target-snapshot"
    )
    data = transport.ExactTransportData((source, target))
    return exact_study(identified, catalog, data, {"x": 1.0})


def test_the_required_factors_name_their_population_regime_axes_and_interventions():
    """Each required distribution leaf of the bound formula is published before any
    evaluation: the population and regime it is read from, its measured axes, conditioning and
    intervened axes. Execution then reports the support it checked for exactly those leaves."""
    # Observational target law: the leaf is the joint P(x, y) of regime 0, nothing intervened.
    identified, catalog, data = fixture()
    study = exact_study(identified, catalog, data, {"x": 1.0})
    (factor,) = study.inspect().support.payload["required_factors"]
    assert (factor["population"], factor["regime"]) == ("target", 0)
    assert tuple(factor["variables"]) == ("x", "y")
    assert tuple(factor["conditioned_on"]) == () and tuple(factor["interventions"]) == ()

    # Experimental source law: the leaf is P(y | do(x)) of the source regime.
    study = _experimental_source_study()
    before = study.inspect()
    (factor,) = before.support.payload["required_factors"]
    assert (factor["population"], factor["regime"]) == ("source", 1)
    assert tuple(factor["variables"]) == ("y",)
    assert tuple(factor["interventions"]) == ("x",)
    assert tuple(factor["conditioned_on"]) == ()
    assert [binding[:2] for binding in before.support.payload["bindings"]] == [
        ("source", 1),
        ("target", 0),
    ]
    assert before.support.payload["factor_support"] == ()

    outcome = _ant_api.estimate(study)
    result = require_success(outcome)
    # Independent oracle: P(y=1 | do(x=1)) is the y=1 mass of the supplied source law.
    assert result.mean("y") == pytest.approx(0.15 + 0.075)
    after = outcome.study.inspect()
    assert after.support.available and after.support.summary == "exact_factor_support_checked"
    assert tuple(after.support.payload["required_factors"]) == tuple(
        before.support.payload["required_factors"]
    )
    checked = after.support.payload["factor_support"]
    assert {row["expression"] for row in checked} == {factor["expression"]}
    assert all(row["status"] == "checked" for row in checked)
    assert {tuple(row["assignment"].items()) for row in checked} == {
        (("x", 1.0), ("y", 0.0)),
        (("x", 1.0), ("y", 1.0)),
    }


def test_raising_regime_property_propagates():
    armed = []

    class Regime(transport.EvidenceRegime):
        @property
        def selected_on(self):
            if armed:
                raise RuntimeError("selected_on failed")
            return getattr(self, "_selected_on", ())

        @selected_on.setter
        def selected_on(self, value):
            object.__setattr__(self, "_selected_on", value)

    identified, _, data = fixture()
    catalog = transport.EvidenceCatalog(
        regimes=[Regime("obs", "target", measured=["x", "y"])]
    )
    armed.append(True)
    with pytest.raises(RuntimeError, match="selected_on failed"):
        transport.evaluate_exact(identified, catalog, data, at={"x": 1.0})
