"""T1 catalog fixtures: environments, regimes, and typed identify outcomes."""

from __future__ import annotations

import antecedent
import pytest
from antecedent.errors import CausalValueError
from antecedent.transport import advanced as transport


def _mean_curve() -> antecedent.ResponseCurve:
    return antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0])


def test_single_and_joint_experiments_are_distinct() -> None:
    singles = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime("do_a", "trial", kind="experimental", interventions=["a"]),
            transport.EvidenceRegime("do_b", "trial", kind="experimental", interventions=["b"]),
        ]
    )
    joint = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "do_ab", "trial", kind="experimental", interventions=["a", "b"]
            ),
        ]
    )
    assert singles.has_available_experiment("trial", ["a"])
    assert singles.has_available_experiment("trial", ["b"])
    assert not singles.has_available_experiment("trial", ["a", "b"])
    assert joint.has_available_experiment("trial", ["a", "b"])


def test_manipulable_is_not_available_evidence() -> None:
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "do_a",
                "trial",
                kind="experimental",
                evidence_kind="manipulable",
                interventions=["a"],
            )
        ]
    )
    assert catalog.source_experiment_variables("trial") == ()
    assert not catalog.has_available_experiment("trial", ["a"])


def test_separate_marginals_never_imply_a_joint() -> None:
    regime = transport.EvidenceRegime(
        "obs",
        "target",
        measured=["x", "z"],
        distribution="separate_marginals",
    )
    assert regime.distribution == "separate_marginals"
    assert regime.distribution != "joint"


def test_convenience_sample_is_not_the_target_law() -> None:
    convenience = transport.EvidenceCatalog(target_sampling="convenience_sample")
    law = transport.EvidenceCatalog(target_sampling="supplied_population_law")
    assert convenience.target_sampling == "convenience_sample"
    assert law.target_sampling == "supplied_population_law"
    assert convenience.target_sampling != law.target_sampling


def test_same_named_incompatible_variables_are_rejected() -> None:
    with pytest.raises(CausalValueError, match="incompatible"):
        transport.EvidenceCatalog(
            environments=[
                transport.Environment(
                    "trial",
                    [transport.VariableCoordinate("a", domain="binary", unit="dose")],
                ),
                transport.Environment(
                    "target",
                    [transport.VariableCoordinate("a", domain="continuous", unit="mg")],
                ),
            ]
        )


def test_empty_catalog_declares_no_target_law() -> None:
    """A catalog is an evidence declaration: silence about the target is not availability."""
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog.empty(),
    )
    result = transport.identify(graph=graph, query=query)
    assert not result.transportable
    assert result.outcome == "missing_evidence"
    assert isinstance(result.certificate, transport.MissingEvidenceCertificate)
    assert result.formula is None


def test_declared_target_regime_binds_target_only_identify() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog(
            regimes=[transport.EvidenceRegime("target_obs", "target", measured=["a", "y"])]
        ),
    )
    result = transport.identify(graph=graph, query=query)
    assert result.transportable
    assert result.outcome == "identified"
    assert isinstance(result.formula, transport.RecursiveFactorizationFormula)
    assert all(f.population == "target" and not f.interventions for f in result.formula.factors)
    assert all(f.regime == 0 for f in result.formula.factors)


def test_missing_experiment_is_not_not_certified() -> None:
    graph = antecedent.graph.Admg.from_edges(
        ["a", "z", "y"], [("a", "y"), ("z", "y")], [("a", "y")]
    )
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", ["z"]),
        catalog=transport.EvidenceCatalog.empty(),
    )
    result = transport.identify(graph=graph, query=query)
    assert not result.transportable
    assert result.outcome == "missing_evidence"
    assert isinstance(result.certificate, transport.MissingEvidenceCertificate)
    assert result.outcome != "not_certified"
    assert result.outcome != "proven_non_transportable"


def test_source_experiments_must_agree_with_catalog() -> None:
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime("do_a", "trial", kind="experimental", interventions=["a"]),
        ]
    )
    with pytest.raises(CausalValueError, match="disagree"):
        transport.TransportQuery(
            _mean_curve(),
            transport.SelectionDiagram("trial", "target", []),
            source_experiments=["b"],
            catalog=catalog,
        )


def test_identifier_preserves_measured_regime_and_sampling_constraints() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")], [("a", "y")])
    diagram = transport.SelectionDiagram("trial", "target", [])
    for measured, expected in [([], "missing_evidence"), (["y"], "identified")]:
        catalog = transport.EvidenceCatalog(
            regimes=[
                transport.EvidenceRegime(
                    "trial_a",
                    "trial",
                    kind="experimental",
                    interventions=["a"],
                    measured=measured,
                )
            ]
        )
        result = transport.identify(
            graph=graph,
            query=transport.TransportQuery(
                _mean_curve(),
                diagram,
                catalog=catalog,
            ),
        )
        assert result.outcome == expected
        if expected == "identified":
            assert result._native.leaf_regimes == [0]


def test_unspecified_coordinate_does_not_mask_later_conflict() -> None:
    with pytest.raises(CausalValueError, match="incompatible"):
        transport.EvidenceCatalog(
            environments=[
                transport.Environment("unspecified", [transport.VariableCoordinate("x")]),
                transport.Environment("binary", [transport.VariableCoordinate("x", "binary")]),
                transport.Environment(
                    "continuous", [transport.VariableCoordinate("x", "continuous")]
                ),
            ]
        )
    with pytest.raises(CausalValueError, match="incompatible"):
        transport.EvidenceCatalog(
            environments=[
                transport.Environment(
                    "a", [transport.VariableCoordinate("x", "categorical", cardinality=2)]
                ),
                transport.Environment(
                    "b", [transport.VariableCoordinate("x", "categorical", cardinality=3)]
                ),
            ]
        )


def test_target_sampling_is_checked_by_identifier() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog(target_sampling="convenience_sample"),
    )
    assert transport.identify(graph=graph, query=query).outcome == "missing_evidence"


def test_restricted_intervention_cannot_supply_a_whole_curve() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog(
            regimes=[
                transport.EvidenceRegime(
                    "one_arm",
                    "trial",
                    kind="experimental",
                    interventions=["a"],
                    measured=["y"],
                    intervention_values={"a": 1.0},
                )
            ]
        ),
    )
    assert transport.identify(graph=graph, query=query).outcome == "missing_evidence"


def test_target_only_formula_does_not_require_irrelevant_measurements() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y", "irrelevant"], [])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog(
            regimes=[
                transport.EvidenceRegime(
                    "target_y",
                    "target",
                    measured=["y"],
                    distribution="separate_marginals",
                )
            ]
        ),
    )
    result = transport.identify(graph=graph, query=query)
    assert result.outcome == "identified"
    assert result.formula.factors[0].variables == ["y"]
    assert result.formula.factors[0].regime == 0


def test_target_only_confounded_effect_is_never_identified_without_an_experiment() -> None:
    # a <-> y with a -> y: the target observational law alone has an s-hedge for
    # P(y | do(a)); no source experiment exists, so no formula may be produced.
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog(
            regimes=[transport.EvidenceRegime("target_obs", "target", measured=["a", "y"])]
        ),
    )
    result = transport.identify(graph=graph, query=query)
    assert result.outcome != "identified"
    assert result.formula is None


def test_identify_proof_executes_through_retained_checked_plan_after_builder_disposal() -> None:
    """The identify stage's certified proof is the program the retained study executes."""
    names = ["a", "y", "trial", "s", "e"]
    graph_builder = antecedent.Admg.from_edges(names, [("a", "y")])
    query_builder = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
        trial="trial",
        selection_probability="s",
        treatment_probability="e",
    )
    identification = transport.identify(graph=graph_builder, query=query_builder)
    assert identification.outcome == "identified"
    assert identification.transportable
    program = transport.reload_lowered_program(identification)
    assert program.source_root == program.executable_root
    # conformance/estimate/staged_transport: two trial rows (A,Y)=(1,3),(0,1) with
    # s=e=0.5 give ipw_sum = 2*3 - 2*1 = 4 over two target rows, so the contrast is 2.
    data = {
        "a": [1.0, 0.0, 0.0, 0.0],
        "y": [3.0, 1.0, 0.0, 0.0],
        "trial": [1.0, 1.0, 0.0, 0.0],
        "s": [0.5] * 4,
        "e": [0.5] * 4,
    }
    study = antecedent.prepare(data, graph=graph_builder, query=query_builder)
    del graph_builder, query_builder
    plan = study.plan
    assert plan.plan_id
    result = study.estimate()
    assert result.estimate.ate == pytest.approx(2.0, abs=1e-12)
    assert result.identification.method == "transport.sid.direct"
    assert result.estimate.estimator_id == "transport.trial_ipw"
    refreshed = study.refresh(data)
    assert refreshed.estimate.ate == pytest.approx(result.estimate.ate, abs=1e-12)
    assert refreshed.program_id == result.program_id
    # Independent consumption names the retained operation it cannot replay
    # instead of accepting a program-less replay as verified.
    loaded = antecedent.load(result.export())
    assert not loaded.acceptance.verified
    assert (
        "dependencies.checked_transport_trial_operation" in loaded.acceptance.details["unresolved"]
    )
