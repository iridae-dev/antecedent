"""Complementary sources and retained grids preserve mathematical and native authority."""

from dataclasses import replace

import pytest
from antecedent import Admg, load, prepare
from antecedent.errors import CausalCancelledError, CausalResourceError, CausalUnsupportedError
from antecedent.transport import advanced as transport


def fixture(missing=False, statistical=False):
    graph = Admg.from_edges(
        ["x", "z", "y"], [("x", "z"), ("z", "y")], bidirected=[("x", "z"), ("x", "y")]
    )
    variables = [transport.VariableCoordinate(name, "binary") for name in ("x", "z", "y")]
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("a", variables, selection_targets=["y"]),
            transport.Environment("b", variables, selection_targets=["z"]),
            transport.Environment("target", variables),
        ],
        regimes=[
            transport.EvidenceRegime(
                "a_trial", "a", kind="experimental", interventions=["x"], measured=["z"]
            ),
            transport.EvidenceRegime(
                "b_trial", "b", kind="experimental", interventions=["z"], measured=["y"]
            ),
        ],
        bindings=[
            transport.RegimeBinding(
                "a_trial", "a1", sampling="independent", dependence="independent_studies"
            ),
            transport.RegimeBinding(
                "b_trial", "b1", sampling="independent", dependence="independent_studies"
            ),
        ],
    )
    identified = transport.identify_meta(
        graph, catalog, target="target", outcomes=["y"], treatments=["x"]
    )
    laws = []
    samples = []
    for population, regime, axis, treatment, probabilities in [
        ("a", "a_trial", "z", "x", (0.2, 0.8)),
        ("b", "b_trial", "y", "z", (0.1, 0.9)),
    ]:
        for value, probability in enumerate(probabilities):
            if missing and population == "a" and value == 0:
                continue
            laws.append(
                transport.ExactDiscreteLaw(
                    population,
                    regime,
                    ((axis, (0.0, 1.0)),),
                    (1 - probability, probability),
                    population + "1",
                    interventions=((treatment, float(value)),),
                )
            )
            samples.append(
                transport.RegimeSample(
                    population,
                    regime,
                    population + "1",
                    {
                        axis: [0.0] * round(100 * (1 - probability))
                        + [1.0] * round(100 * probability)
                    },
                    interventions=((treatment, float(value)),),
                )
            )
    data = (
        transport.StatisticalTransportData(samples=tuple(samples))
        if statistical
        else transport.ExactTransportData(tuple(laws))
    )
    return graph, identified, catalog, data


def test_complementary_sources_exact_grid_and_independent_consume():
    graph, identified, catalog, data = fixture()
    assert identified.outcome == "identified"
    for source in ("a", "b"):
        reduced = replace(
            catalog,
            environments=tuple(e for e in catalog.environments if e.identity in (source, "target")),
            regimes=tuple(r for r in catalog.regimes if r.population == source),
            bindings=tuple(b for b in catalog.bindings if b.regime.startswith(source)),
        )
        assert (
            transport.identify_meta(
                graph, reduced, target="target", outcomes=["y"], treatments=["x"]
            ).outcome
            == "proven_non_transportable"
        )
    study = prepare(
        data,
        query=transport.TransportResponseGridQuery(identified, catalog, ({"x": 0.0}, {"x": 1.0})),
    )
    assert study.inspect().identification.available
    result = study.estimate()
    assert [result.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74])
    assert result.contrast(1, 0, "y")["estimate"] == pytest.approx(0.48)
    assert result.contrast(1, 0, "y")["interval"] is None
    restored = load(result.export())
    assert restored.execution_id == result.execution_id
    assert restored.points == result.points
    assert restored.export() == result.export()


def test_identify_meta_proof_executes_through_retained_checked_plan_after_builder_disposal():
    """The meta proof is the checked program the retained grid plan executes."""
    graph_builder, identification_builder, catalog, data = fixture()
    assert identification_builder.outcome == "identified"
    program = identification_builder.formula
    assert program
    # Neither source identifies alone; the proof needs both complementary sources.
    for source in ("a", "b"):
        reduced = replace(
            catalog,
            environments=tuple(e for e in catalog.environments if e.identity in (source, "target")),
            regimes=tuple(r for r in catalog.regimes if r.population == source),
            bindings=tuple(b for b in catalog.bindings if b.regime.startswith(source)),
        )
        assert (
            transport.identify_meta(
                graph_builder, reduced, target="target", outcomes=["y"], treatments=["x"]
            ).outcome
            == "proven_non_transportable"
        )
    plan = prepare(
        data,
        query=transport.TransportResponseGridQuery(
            identification_builder, catalog, ({"x": 0.0}, {"x": 1.0})
        ),
    )
    del graph_builder, identification_builder
    assert plan.inspect().identification.available
    family = plan.estimate()
    # Independent arithmetic: P(y | do(x)) = sum_z P(z | do(x)) P(y | do(z)).
    assert [family.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74])
    assert family.contrast(1, 0, "y")["estimate"] == pytest.approx(0.48)
    assert family.contrast(1, 0, "y")["interval"] is None
    assert plan.refresh(data).points == family.points


@pytest.mark.parametrize("statistical", [False, True])
def test_partial_grid_retains_missing_first_point(statistical):
    _, identified, catalog, data = fixture(missing=True, statistical=statistical)
    study = transport.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=19
    )
    result = study.estimate()
    assert result.points[0]["status"] == "missing_evidence"
    assert result.mean(1, "y") == pytest.approx(0.74)
    with pytest.raises(ValueError, match="executable"):
        result.contrast(1, 0, "y")
    restored = load(result.export())
    assert restored.points == result.points
    if statistical:
        with pytest.raises(CausalUnsupportedError, match="samples_not_embedded"):
            transport.consume_response_grid(result.export()).estimate()
    assert study.refresh(data).points == result.points


@pytest.mark.parametrize("statistical", [False, True])
def test_evaluate_response_grid_retains_local_failures_after_builder_disposal(statistical):
    """Grid evaluation runs from the retained plan and keeps point-local failures."""
    graph_builder, identification_builder, catalog, data = fixture(
        missing=True, statistical=statistical
    )
    plan = transport.prepare_response_grid(
        identification_builder, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40
    )
    del graph_builder, identification_builder
    assert plan.inspect().identification.available
    family = plan.estimate()
    # The missing first point is retained as a local failure, not dropped.
    assert family.points[0]["status"] == "missing_evidence"
    assert family.points[1]["status"] == "available"
    assert family.mean(1, "y") == pytest.approx(0.74)
    with pytest.raises(ValueError, match="executable"):
        family.contrast(1, 0, "y")
    if statistical:
        assert family.points[1]["uncertainty"]["replicate_ids"]
    else:
        assert family.points[1]["uncertainty"]["available"] is False
        assert (
            family.points[1]["uncertainty"]["reason"]
            == "exact_supplied_law_no_sampling_uncertainty"
        )
    assert plan.refresh(data).points == family.points
    assert load(family.export()).points == family.points


def test_statistical_grid_paired_contrast_and_catalog_permutation():
    _, identified, catalog, data = fixture(statistical=True)
    # A nominal 0.95 percentile interval needs B >= 40 (PERCENTILE_95_MIN_REPLICATES);
    # fewer replicates run cleanly but withhold the contrast interval as unlicensed.
    study = transport.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40, seed=7
    )
    result = study.estimate()
    assert (
        result.points[0]["uncertainty"]["replicate_ids"]
        == result.points[1]["uncertainty"]["replicate_ids"]
    )
    assert result.contrast(0, 0, "y")["interval"] == (0.0, 0.0)
    assert load(result.export()).contrast(1, 0, "y") == result.contrast(1, 0, "y")
    changed = replace(
        catalog,
        environments=tuple(reversed(catalog.environments)),
        regimes=tuple(reversed(catalog.regimes)),
        bindings=tuple(reversed(catalog.bindings)),
    )
    again = transport.prepare_response_grid(
        identified, changed, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40, seed=7
    ).estimate()
    assert again.points == result.points
    assert again.execution_id == result.execution_id


def test_grid_budgets_invalid_values_and_display_edits():
    _, identified, catalog, data = fixture()
    with pytest.raises(ValueError, match="finite domain"):
        transport.prepare_response_grid(identified, catalog, data, at=[{"x": 0.5}])
    with pytest.raises(CausalResourceError, match="budget"):
        transport.prepare_response_grid(
            identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], max_operations=1
        )
    result = transport.prepare_response_grid(identified, catalog, data, at=[{"x": 1.0}]).estimate()
    edited = replace(result, execution_id="forged", points=())
    assert edited.export() == result.export()


def test_structural_positive_and_negative_round_trips():
    graph, identified, catalog, _ = fixture()
    restored = transport.consume_identification(identified.export())
    assert restored.formula == identified.formula
    assert restored.inspect() == identified.inspect()
    source = replace(
        catalog,
        environments=tuple(e for e in catalog.environments if e.identity != "b"),
        regimes=(catalog.regimes[0],),
        bindings=(catalog.bindings[0],),
    )
    negative = transport.identify_meta(
        graph, source, target="target", outcomes=["y"], treatments=["x"]
    )
    assert negative.outcome == "proven_non_transportable"
    assert load(negative.export()).outcome == negative.outcome
    assert load(negative.export()).inspect() == negative.inspect()


def test_checked_meta_statistical_plan_survives_builder_and_independent_consume():
    # The two sources jointly identify P(y | do(x)); neither source does alone.
    graph, identification_builder, catalog, data = fixture(statistical=True)
    proof_wire = identification_builder.export()
    del identification_builder
    proof = transport.consume_identification(proof_wire)
    assert proof.outcome == "identified"
    assert proof.inspect().identification.available

    # A new theorem-stage compilation has the same checked target and formula.
    compiled = transport.identify_meta(
        graph, catalog, target="target", outcomes=["y"], treatments=["x"]
    )
    assert compiled.formula == proof.formula
    assert compiled.outcomes == proof.outcomes

    builder = transport.StatisticalTransportQuery(compiled, catalog, {"x": 1.0}, bootstrap=40)
    plan = transport.prepare(builder, data)
    assert plan.inspect().identification.available
    del builder, graph, compiled
    result = plan.estimate()
    # Independent arithmetic: 0.8 P(y=1 | do(z=1)) + 0.2 P(y=1 | do(z=0)).
    assert result.mean("y") == pytest.approx(0.8 * 0.9 + 0.2 * 0.1)
    direct_plan = transport.prepare_statistical(proof, catalog, data, at={"x": 1.0}, bootstrap=40)
    assert direct_plan.inspect().identification.available
    assert direct_plan.estimate().mean("y") == pytest.approx(result.mean("y"))
    consumed = transport.consume_statistical(result.export())
    assert consumed.inspect().identification.available
    with pytest.raises(CausalUnsupportedError, match="samples_not_embedded"):
        consumed.estimate()
    assert consumed.refresh(data).mean("y") == pytest.approx(result.mean("y"))

    grid = transport.evaluate_statistical_grid(
        proof, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40
    )
    assert [point.mean("y") for point in grid] == pytest.approx([0.26, 0.74])
    assert grid[1].contrast(grid[0], "y")["estimate"] == pytest.approx(0.48)

    grid_plan = transport.prepare_response_grid(
        proof, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40
    )
    assert grid_plan.inspect().identification.available
    family = grid_plan.estimate()
    assert [family.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74])
    assert family.contrast(1, 0, "y")["estimate"] == pytest.approx(0.48)
    restored_family = transport.consume_response_grid(family.export())
    assert restored_family.inspect().identification.available
    with pytest.raises(CausalUnsupportedError, match="samples_not_embedded"):
        restored_family.estimate()
    assert restored_family.refresh(data).points == family.points


def test_grid_snapshot_refresh_and_scalar_loss_receipt():
    _, identified, catalog, data = fixture(statistical=True)
    study = transport.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=19, seed=2
    )
    original = study.estimate()
    updated = replace(
        data,
        samples=tuple(
            replace(sample, snapshot_identity=sample.snapshot_identity + "next")
            for sample in data.samples
        ),
    )
    result = study.refresh(updated)
    assert result.execution_id != original.execution_id
    assert load(result.export()).points == result.points
    scalar = result.scalar_projection(0, "y")
    assert scalar["value"] == result.mean(0, "y")
    assert not scalar["equivalent_claim"]
    assert "accept_as_claim" in scalar["unavailable_operations"]
    study.replace_snapshot(data)
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        study.export()
    assert study.estimate().execution_id == original.execution_id


def test_consume_response_grid_refreshes_from_the_consumed_family_after_builder_disposal():
    """The consumer rechecks the embedded grid claim and replays only from supplied samples."""
    graph_builder, identification_builder, catalog, data = fixture(statistical=True)
    producer_builder = transport.prepare_response_grid(
        identification_builder, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=40, seed=2
    )
    family = producer_builder.estimate()
    family_wire = family.export()
    plan = producer_builder.inspect()
    assert plan.identification.available
    expected_points = family.points
    del producer_builder, identification_builder, graph_builder, family
    consumed = transport.consume_response_grid(family_wire)
    assert consumed.inspect().program_id == plan.program_id
    assert consumed.inspect().identification.available
    # No fitting, fetching or resampling on load: raw samples are an explicit dependency.
    with pytest.raises(CausalUnsupportedError, match="samples_not_embedded"):
        consumed.estimate()
    replay = consumed.refresh(data)
    assert replay.points == expected_points
    assert [replay.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74])
    assert replay.contrast(1, 0, "y")["estimate"] == pytest.approx(0.48)
    # A scalar projection of the consumed family carries its loss receipt.
    scalar = replay.scalar_projection(0, "y")
    assert scalar["value"] == replay.mean(0, "y")
    assert not scalar["equivalent_claim"]
    assert "accept_as_claim" in scalar["unavailable_operations"]
    assert load(family_wire).points == expected_points


def test_forwarded_dataset_aliases_share_resampling_and_reject_conflicts():
    _, identified, catalog, data = fixture(statistical=True)
    catalog = replace(
        catalog,
        bindings=tuple(
            replace(binding, dataset_identity=binding.regime) for binding in catalog.bindings
        ),
    )
    kwargs = dict(at=[{"x": 0.0}, {"x": 1.0}], bootstrap=29, seed=9)
    original = transport.prepare_response_grid(identified, catalog, data, **kwargs).estimate()
    alias_regime = replace(catalog.regimes[0], id="forwarded_a")
    alias_binding = replace(catalog.bindings[0], regime="forwarded_a")
    copies = tuple(
        replace(sample, regime="forwarded_a") for sample in data.samples if sample.population == "a"
    )
    alias_catalog = replace(
        catalog,
        regimes=(*catalog.regimes, alias_regime),
        bindings=(*catalog.bindings, alias_binding),
    )
    aliases = replace(data, samples=(*data.samples, *copies))
    forwarded = transport.prepare_response_grid(
        identified, alias_catalog, aliases, **kwargs
    ).estimate()
    for left, right in zip(original.points, forwarded.points, strict=True):
        assert left["probabilities"] == right["probabilities"]
        assert left["uncertainty"] == right["uncertainty"]
    assert load(forwarded.export()).points == forwarded.points
    bad = replace(copies[0], columns={"z": [0.0] * 100})
    with pytest.raises(ValueError, match="conflicting forwarded"):
        transport.prepare_response_grid(
            identified,
            alias_catalog,
            replace(data, samples=(*data.samples, bad, copies[1])),
            **kwargs,
        )


def test_all_missing_providers_and_sample_order_are_retained():
    _, identified, catalog, data = fixture(statistical=True)
    empty_source = replace(data, samples=tuple(s for s in data.samples if s.population == "b"))
    missing = transport.prepare_response_grid(
        identified, catalog, empty_source, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=7
    ).estimate()
    assert [p["status"] for p in missing.points] == ["missing_evidence"] * 2
    assert missing.points[0]["factor_diagnostic"]["bindings"]
    assert load(missing.export()).points == missing.points
    first = transport.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=7
    ).estimate()
    reversed_data = replace(data, samples=tuple(reversed(data.samples)))
    second = transport.prepare_response_grid(
        identified, catalog, reversed_data, at=[{"x": 0.0}, {"x": 1.0}], bootstrap=7
    ).estimate()
    assert first.points == second.points
    assert first.execution_id == second.execution_id


def test_grid_family_identity_atomic_refresh_and_preview():
    _, identified, catalog, data = fixture()
    study = transport.prepare_response_grid(identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}])
    result = study.estimate()
    larger = transport.prepare_response_grid(
        identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}, {"x": 0.0}]
    ).estimate()
    assert larger.execution_id != result.execution_id
    malformed = replace(
        data, laws=(replace(data.laws[0], probabilities=(0.2, 0.2)), *data.laws[1:])
    )
    with pytest.raises(ValueError, match="unnormalized_law"):
        study.refresh(malformed)
    assert study.estimate().execution_id == result.execution_id
    preview = study.preview_transform("compatible_data_replace")
    assert preview is not None
    with pytest.raises(ValueError, match="retained graph"):
        prepare(
            data,
            query=transport.TransportResponseGridQuery(identified, catalog, ({"x": 0.0},)),
            threads=2,
        )


def test_prepare_response_grid_retains_family_plan_after_builder_disposal():
    """The prepared family keeps its checked plan, refreshes atomically and invalidates identity."""
    graph_builder, identification_builder, catalog, data = fixture()
    query_builder = transport.TransportResponseGridQuery(
        identification_builder, catalog, ({"x": 0.0}, {"x": 1.0})
    )
    plan = prepare(data, query=query_builder)
    del query_builder, identification_builder, graph_builder
    before = plan.inspect()
    assert before.identification.available
    family = plan.estimate()
    assert [family.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74])
    assert plan.inspect().execution_id == family.execution_id
    # A malformed refresh is rejected atomically: the previous execution stands.
    malformed = replace(
        data, laws=(replace(data.laws[0], probabilities=(0.2, 0.2)), *data.laws[1:])
    )
    with pytest.raises(ValueError, match="unnormalized_law"):
        plan.refresh(malformed)
    assert plan.inspect().execution_id == family.execution_id
    assert plan.estimate().points == family.points
    # A new snapshot identity invalidates the execution claim until it is re-run.
    renamed = replace(
        data,
        laws=tuple(
            replace(law, snapshot_identity=law.snapshot_identity + "next") for law in data.laws
        ),
    )
    plan.replace_snapshot(renamed)
    assert plan.inspect().program_id == before.program_id
    assert plan.inspect().execution_id != family.execution_id
    with pytest.raises(CausalUnsupportedError, match="no_execution_claim"):
        plan.export()
    assert plan.refresh(data).points == family.points
    assert plan.inspect().execution_id == family.execution_id


def calibration_case(seed, parameter=0, bootstrap=40):
    # A nominal 0.95 percentile interval needs B >= 40 (PERCENTILE_95_MIN_REPLICATES);
    # fewer replicates run cleanly but withhold the contrast interval as unlicensed.
    """Reproducible unequal-size Bernoulli arms; exact means follow by total probability."""
    import random

    _, identified, catalog, data = fixture(statistical=True)
    pz = (0.2 + 0.05 * parameter, 0.8 - 0.03 * parameter)
    py = (0.1 + 0.02 * parameter, 0.9 - 0.04 * parameter)
    rng = random.Random(seed)
    samples = []
    for index, sample in enumerate(data.samples):
        treatment = int(sample.interventions[0][1])
        probability = (pz if sample.population == "a" else py)[treatment]
        axis = "z" if sample.population == "a" else "y"
        n = (160, 220, 180, 260)[index]
        samples.append(
            replace(sample, columns={axis: [float(rng.random() < probability) for _ in range(n)]})
        )
    result = transport.prepare_response_grid(
        identified,
        catalog,
        replace(data, samples=tuple(samples)),
        at=[{"x": 0.0}, {"x": 1.0}],
        bootstrap=bootstrap,
        seed=seed,
    ).estimate()
    truth = [py[0] * (1 - p) + py[1] * p for p in pz]
    return result, truth


@pytest.mark.parametrize("parameter", [0, 1])
def test_bounded_multisource_calibration_fixture(parameter):
    # This checks reproducibility and inferential plumbing, not empirical coverage.
    result, truth = calibration_case(41, parameter)
    again, same_truth = calibration_case(41, parameter)
    assert result.points == again.points
    assert truth == same_truth
    assert all(0 < p < 1 for p in truth)
    assert (
        result.points[0]["uncertainty"]["replicate_ids"]
        == result.points[1]["uncertainty"]["replicate_ids"]
    )
    contrast = result.contrast(1, 0, "y")
    assert contrast["replicates_ok"] == 40
    assert contrast["calibration_status"] == "not_bound_to_this_execution"
    assert contrast["interval"][0] <= contrast["interval"][1]


def test_joint_regimes_do_not_come_from_separate_experiments():
    graph = Admg.from_edges(
        ["x", "w", "y"], [("x", "y"), ("w", "y")], bidirected=[("x", "y"), ("w", "y")]
    )
    variables = [transport.VariableCoordinate(name, "binary") for name in ("x", "w", "y")]
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("source", variables, selection_targets=["x", "w"]),
            transport.Environment("target", variables),
        ],
        regimes=[
            transport.EvidenceRegime(
                "joint", "source", kind="experimental", interventions=["x", "w"], measured=["y"]
            )
        ],
    )
    identified = transport.identify_meta(
        graph, catalog, target="target", outcomes=["y"], treatments=["x", "w"]
    )
    assert identified.outcome == "identified"
    laws = tuple(
        transport.ExactDiscreteLaw(
            "source",
            "joint",
            (("y", (0.0, 1.0)),),
            (1 - p, p),
            "joint1",
            interventions=(("x", x), ("w", w)),
        )
        for x, w, p in [(0.0, 0.0, 0.1), (0.0, 1.0, 0.5), (1.0, 0.0, 0.4)]
    )
    grid = [{"x": x, "w": w} for x in (0.0, 1.0) for w in (0.0, 1.0)]
    result = transport.prepare_response_grid(
        identified, catalog, transport.ExactTransportData(laws), at=grid
    ).estimate()
    assert [p["status"] for p in result.points] == ["available"] * 3 + ["missing_evidence"]
    assert [result.mean(i, "y") for i in range(3)] == pytest.approx([0.1, 0.5, 0.4])
    assert load(result.export()).points == result.points
    separate = replace(
        catalog,
        regimes=[
            transport.EvidenceRegime(
                "x", "source", kind="experimental", interventions=["x"], measured=["w", "y"]
            ),
            transport.EvidenceRegime(
                "w", "source", kind="experimental", interventions=["w"], measured=["x", "y"]
            ),
        ],
    )
    with pytest.raises(CausalUnsupportedError, match="catalog"):
        transport.prepare_response_grid(
            identified, separate, transport.ExactTransportData(()), at=grid
        )


def test_irrelevant_source_deterministic_alternative_and_wrong_selection():
    graph, identified, catalog, data = fixture()
    variables = catalog.environments[0].variables
    irrelevant = replace(
        catalog,
        environments=(
            *catalog.environments,
            transport.Environment("zero", variables, selection_targets=["x", "z", "y"]),
        ),
    )
    other = transport.identify_meta(
        graph, irrelevant, target="target", outcomes=["y"], treatments=["x"]
    )
    result = transport.prepare_response_grid(other, irrelevant, data, at=[{"x": 1.0}]).estimate()
    assert result.mean(0, "y") == pytest.approx(0.74)
    # A lexically earlier admissible source has no providers. Catalog-aware search must try a.
    alternative = replace(
        catalog,
        environments=(
            transport.Environment("0a", variables, selection_targets=["y"]),
            *catalog.environments,
        ),
    )
    alternate_id = transport.identify_meta(
        graph, alternative, target="target", outcomes=["y"], treatments=["x"]
    )
    assert transport.prepare_response_grid(
        alternate_id, alternative, data, at=[{"x": 1.0}]
    ).estimate().mean(0, "y") == pytest.approx(0.74)
    conflicting = replace(
        catalog,
        environments=(
            replace(catalog.environments[0], selection_targets=["z"]),
            *catalog.environments[1:],
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="catalog mechanism selections disagree"):
        transport.prepare_response_grid(identified, conflicting, data, at=[{"x": 1.0}])


def test_retained_support_failure_and_checked_irrelevant_zero():
    graph = Admg.from_edges(["z", "x", "y"], [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["z"]),
        outcomes=["y"],
        treatments=["x"],
    )
    variables = [transport.VariableCoordinate(name, "binary") for name in ("z", "x", "y")]
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("source", variables, selection_targets=["z"]),
            transport.Environment("target", variables),
        ],
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("covariates", "target", measured=["z"]),
        ],
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
        "target", "covariates", (("z", (0.0, 1.0)),), (0.5, 0.5), "target"
    )
    result = transport.prepare_response_grid(
        identified, catalog, transport.ExactTransportData((source, target)), at=[{"x": 1.0}]
    ).estimate()
    assert result.points[0]["status"] == "support_failure"
    assert result.points[0]["factor_diagnostic"]["assignment"]
    assert load(result.export()).points == result.points
    valid = transport.prepare_response_grid(
        identified,
        catalog,
        transport.ExactTransportData((source, replace(target, probabilities=(1.0, 0.0)))),
        at=[{"x": 1.0}],
    ).estimate()
    assert valid.mean(0, "y") == pytest.approx(0.3)
    assert valid.points[0]["factor_support"]


def test_grid_cancellation_preserves_previous_execution():
    from antecedent.state import CancellationToken

    _, identified, catalog, data = fixture()
    study = transport.prepare_response_grid(identified, catalog, data, at=[{"x": 0.0}, {"x": 1.0}])
    result = study.estimate()
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        study.estimate(cancel=token)
    assert load(study.export()).execution_id == result.execution_id


def test_consume_identification_executes_from_the_consumed_proof_after_builder_disposal():
    from pathlib import Path

    # The identification builder is released before the consumed proof drives a
    # checked statistical plan; consumption re-verifies the embedded claim.
    graph, identification_builder, catalog, data = fixture(statistical=True)
    proof_wire = identification_builder.export()
    expected = identification_builder.inspect()
    del identification_builder, graph
    proof = transport.consume_identification(proof_wire)
    assert proof.outcome == "identified"
    assert proof.inspect() == expected
    plan = transport.prepare_statistical(proof, catalog, data, at={"x": 1.0}, bootstrap=40)
    assert plan.inspect().identification.available
    # Independent arithmetic: 0.8 P(y=1 | do(z=1)) + 0.2 P(y=1 | do(z=0)).
    assert plan.estimate().mean("y") == pytest.approx(0.8 * 0.9 + 0.2 * 0.1)

    # A frozen NotCertified claim is consumed as it was recorded: never upgraded
    # to a proof or to an obstruction.
    payload = (
        Path(__file__).parent / "fixtures" / "transport" / "not_certified_v1.cbor"
    ).read_bytes()
    frozen = load(payload)
    assert frozen.outcome == "not_certified"
    assert load(frozen.export()).inspect() == frozen.inspect()


def test_frozen_rust_not_certified_consumes_without_upgrade():
    from pathlib import Path

    payload = (
        Path(__file__).parent / "fixtures" / "transport" / "not_certified_v1.cbor"
    ).read_bytes()
    result = load(payload)
    assert result.outcome == "not_certified"
    assert result.outcomes == ("y",)
    assert load(result.export()).inspect() == result.inspect()
    report = result.inspect()
    assert report.identification.summary == "not_certified"
    assert {name for name in ("identification", "support", "uncertainty", "assumptions")} <= set(
        report.to_dict()
    )


def test_invalid_source_coordinates_and_continuous_provider_contracts():
    graph, identified, catalog, data = fixture()
    wrong_target = replace(
        catalog,
        environments=(
            *catalog.environments[:2],
            replace(catalog.environments[2], selection_targets=["y"]),
        ),
    )
    with pytest.raises(ValueError, match="target environment"):
        transport.identify_meta(
            graph, wrong_target, target="target", outcomes=["y"], treatments=["x"]
        )
    continuous = replace(
        catalog,
        environments=tuple(
            replace(
                env,
                variables=tuple(
                    replace(v, domain="continuous") if v.name == "z" else v for v in env.variables
                ),
            )
            for env in catalog.environments
        ),
    )
    with pytest.raises(ValueError, match="finite discrete"):
        transport.prepare_response_grid(identified, continuous, data, at=[{"x": 1.0}])
